import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.cfg import py2cfg
from train_supervision import Supervision_Train
from tools.metric import Evaluator
from geoseg.utils.boundary_metrics import BoundaryEvaluator
from geoseg.datasets.sen2_lulc_dataset import CLASSES, COLOR_MAP


def label2rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        rgb[mask == class_id] = color
    return rgb


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate model on SEN-2 LULC with semantic and boundary metrics.")
    parser.add_argument("-c", "--config_path", type=str, default="config/sen2_lulc/unetformer.py", help="Path to config")
    parser.add_argument("-w", "--ckpt_path", type=str, default=None, help="Path to model checkpoint (.ckpt). Defaults to best checkpoint in config.")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="Path to save visual predictions (optional)")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of validation samples to evaluate")
    parser.add_argument("--boundary_tolerance", type=int, default=2, help="Pixel tolerance for boundary F1/IoU")
    parser.add_argument("--rgb", action="store_true", help="Save RGB colorized masks instead of raw label indices")
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)

    ckpt_path = args.ckpt_path
    if not ckpt_path:
        default_ckpt = os.path.join(config.weights_path, f"{config.weights_name}.ckpt")
        last_ckpt = os.path.join(config.weights_path, "last.ckpt")
        if os.path.exists(default_ckpt):
            ckpt_path = default_ckpt
        elif os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
        else:
            raise FileNotFoundError(f"No checkpoint found at {default_ckpt} or {last_ckpt}")

    print(f"\n[EVAL] Loading checkpoint: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using device: {device}")

    model = Supervision_Train.load_from_checkpoint(ckpt_path, config=config)
    model.to(device)
    model.eval()

    val_dataset = config.val_dataset
    if args.max_samples and hasattr(val_dataset, 'samples') and args.max_samples < len(val_dataset.samples):
        val_dataset.samples = val_dataset.samples[:args.max_samples]
        print(f"[EVAL] Limiting evaluation to {args.max_samples} validation samples")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    boundary_evaluator = BoundaryEvaluator(
        num_classes=config.num_classes,
        tolerance=args.boundary_tolerance,
        dilation_width=args.boundary_tolerance,
        ignore_index=getattr(config, 'ignore_index', 255),
    )

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    sample_counter = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            imgs = batch['img'].to(device)
            masks = batch['gt_semantic_seg']

            preds = model(imgs)
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            preds = nn.Softmax(dim=1)(preds).argmax(dim=1).cpu().numpy()
            gts = masks.cpu().numpy()

            for i in range(preds.shape[0]):
                pred_i = preds[i]
                gt_i = gts[i]
                evaluator.add_batch(pre_image=pred_i, gt_image=gt_i)
                boundary_evaluator.add_batch(gt_image=gt_i, pre_image=pred_i)

                if args.output_path:
                    import cv2
                    out_name = f"sample_{sample_counter:05d}.png"
                    save_p = os.path.join(args.output_path, out_name)
                    if args.rgb:
                        rgb = label2rgb(pred_i)
                        cv2.imwrite(save_p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(save_p, pred_i.astype(np.uint8))
                    sample_counter += 1

    # Segmentation metrics
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    oa = evaluator.OA()
    miou = np.nanmean(iou_per_class)
    mf1 = np.nanmean(f1_per_class)

    print("\n" + "=" * 75)
    print("           SEMANTIC SEGMENTATION RESULTS (SEN-2 LULC)           ")
    print("=" * 75)
    print(f"{'Class':<20} | {'IoU':<12} | {'F1-Score':<12}")
    print("-" * 75)
    for c_name, ciou, cf1 in zip(config.classes, iou_per_class, f1_per_class):
        print(f"{c_name:<20} | {ciou*100:>10.2f}% | {cf1*100:>10.2f}%")
    print("-" * 75)
    print(f"{'Overall Mean':<20} | {miou*100:>10.2f}% | {mf1*100:>10.2f}%")
    print(f"{'Overall Accuracy':<20} | {oa*100:>10.2f}%")
    print("=" * 75)

    # Boundary metrics
    bnd_summary = boundary_evaluator.summary(class_names=list(config.classes))
    print("\n" + "=" * 75)
    print(f"       BOUNDARY EVALUATION METRICS (Tolerance = {args.boundary_tolerance} px)       ")
    print("=" * 75)
    print(f"Overall Boundary Precision : {bnd_summary['overall_precision']*100:.2f}%")
    print(f"Overall Boundary Recall    : {bnd_summary['overall_recall']*100:.2f}%")
    print(f"Overall Boundary F1 (BF1)  : {bnd_summary['overall_f1']*100:.2f}%")
    print(f"Mean Boundary IoU (mBIoU)  : {bnd_summary['mean_boundary_iou']*100:.2f}%")
    print(f"Mean Boundary F1 (mBF1)    : {bnd_summary['mean_boundary_f1']*100:.2f}%")
    print("-" * 75)
    print(f"{'Class':<20} | {'BIoU':<10} | {'BF1':<10} | {'B-Prec':<10} | {'B-Recall':<10}")
    print("-" * 75)
    for c_name, biou, bf1, bp, br in zip(
        config.classes,
        bnd_summary['boundary_iou_per_class'],
        bnd_summary['boundary_f1_per_class'],
        bnd_summary['boundary_precision_per_class'],
        bnd_summary['boundary_recall_per_class'],
    ):
        biou_s = f"{biou*100:.2f}%" if not np.isnan(biou) else "N/A"
        bf1_s = f"{bf1*100:.2f}%" if not np.isnan(bf1) else "N/A"
        bp_s = f"{bp*100:.2f}%" if not np.isnan(bp) else "N/A"
        br_s = f"{br*100:.2f}%" if not np.isnan(br) else "N/A"
        print(f"{c_name:<20} | {biou_s:>10} | {bf1_s:>10} | {bp_s:>10} | {br_s:>10}")
    print("-" * 75)
    print("Boundary-Distance Accuracy (Accuracy vs Distance to Edge):")
    print(f"{'Distance Range':<22} | {'Pixel Ratio (%)':<16} | {'Accuracy (%)':<14}")
    print("-" * 75)
    for bucket_label, b_data in bnd_summary['distance_accuracy'].items():
        acc = b_data['accuracy']
        acc_s = f"{acc*100:>12.2f}%" if not np.isnan(acc) else "         N/A"
        print(f"{bucket_label:<22} | {b_data['pixel_pct']:>14.2f}% | {acc_s}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
