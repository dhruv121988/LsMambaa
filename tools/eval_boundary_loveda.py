#!/usr/bin/env python3
"""
Evaluate boundary metrics on LoveDA validation set with high-throughput async pipeline.
"""
import os
import sys
import json
import queue
import threading
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from tools.cfg import py2cfg
from train_supervision import Supervision_Train
from geoseg.utils.boundary_metrics import BoundaryEvaluator
from tools.metric import Evaluator


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate boundary metrics on LoveDA")
    parser.add_argument("-c", "--config", type=str, default="config/loveda/boundary_vmamba_unet.py")
    parser.add_argument("--ckpt", type=str, default="model_weights/loveda/boundary_vmamba_unet-multiplicative-aux0.4-bnd0.3-epoch16/last.ckpt")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", help="Enable FP16 AMP for faster inference")
    parser.add_argument("--limit_batches", type=int, default=None, help="Optionally limit evaluation to first N batches")
    parser.add_argument("--save_json", type=str, default=None, help="Path to save evaluation summary JSON")
    args = parser.parse_args()

    config_path = args.config
    ckpt_path = args.ckpt

    print(f"Loading config: {config_path}")
    print(f"Loading checkpoint: {ckpt_path}")

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    config = py2cfg(config_path)
    model = Supervision_Train.load_from_checkpoint(ckpt_path, config=config)
    model.cuda()
    model.eval()

    val_dataset = config.val_dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    ignore_index = getattr(config, 'ignore_index', len(config.classes))
    bnd_evaluator = BoundaryEvaluator(
        num_classes=config.num_classes,
        tolerance=2,
        dilation_width=2,
        ignore_index=ignore_index
    )

    # High-throughput asynchronous consumer pipeline
    eval_queue = queue.Queue(maxsize=16)

    def worker_loop():
        while True:
            item = eval_queue.get()
            if item is None:
                eval_queue.task_done()
                break
            batch_mask, batch_pred = item
            evaluator.add_batch(batch_mask, batch_pred)
            bnd_evaluator.add_batch(gt_image=batch_mask, pre_image=batch_pred)
            eval_queue.task_done()

    consumer_thread = threading.Thread(target=worker_loop, daemon=True)
    consumer_thread.start()

    total_samples = len(val_dataset) if not args.limit_batches else min(len(val_dataset), args.limit_batches * args.batch_size)
    print(f"Evaluating LoveDA on {total_samples} validation samples (Batch size={args.batch_size}, AMP={args.amp})...")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="LoveDA Val", total=args.limit_batches if args.limit_batches else len(val_loader))):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break
            img = batch["img"].cuda(non_blocking=True)
            mask = batch["gt_semantic_seg"].numpy()  # (B, H, W)

            if args.amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    output = model(img)
            else:
                output = model(img)

            if isinstance(output, (tuple, list)):
                output = output[0]
            elif isinstance(output, dict):
                output = output.get("seg_logits", output.get("pred", output))

            # argmax directly on logits avoids redundant softmax calculation
            pred = output.argmax(dim=1).cpu().numpy()  # (B, H, W)

            # Asynchronously enqueue to consumer thread without stalling GPU
            eval_queue.put((mask, pred))

    # Signal worker thread to terminate and wait for all metrics to be calculated
    eval_queue.put(None)
    consumer_thread.join()

    bnd_summary = bnd_evaluator.summary()
    region_miou = np.nanmean(evaluator.Intersection_over_Union()) * 100
    region_f1 = np.nanmean(evaluator.F1()) * 100
    region_oa = evaluator.OA() * 100

    print("\n" + "=" * 75)
    print("        LOVEDA - BOUNDARY EVALUATION REPORT (PROPOSED MODEL)        ")
    print("=" * 75)
    print(f"Region mIoU: {region_miou:.2f}% | Region F1: {region_f1:.2f}% | Region OA: {region_oa:.2f}%")
    print("-" * 75)
    print("                     CVPR 2021 BOUNDARY BENCHMARK                     ")
    print("-" * 75)
    print(f"Mean Boundary IoU (mBIoU): {bnd_summary['mean_boundary_iou'] * 100:.2f}%")
    print(f"Mean Boundary F1 (mBF1)  : {bnd_summary['mean_boundary_f1'] * 100:.2f}%")
    print(f"Overall Boundary Precision: {bnd_summary['overall_precision'] * 100:.2f}%")
    print(f"Overall Boundary Recall   : {bnd_summary['overall_recall'] * 100:.2f}%")
    print(f"Overall Boundary F1       : {bnd_summary['overall_f1'] * 100:.2f}%")
    print("-" * 75)
    print(f"{'Class':<20} | {'BIoU':<10} | {'BF1':<10} | {'Precision':<10} | {'Recall':<10}")
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
        print(f"{c_name:<20} | {biou_s:<10} | {bf1_s:<10} | {bp_s:<10} | {br_s:<10}")

    print("-" * 75)
    print("Accuracy vs Distance to Nearest Boundary:")
    print("-" * 75)
    print(f"{'Distance Range':<22} | {'Pixel Ratio (%)':<16} | {'Accuracy (%)':<14}")
    print("-" * 75)
    for label, b_data in bnd_summary['distance_accuracy'].items():
        acc_s = f"{b_data['accuracy'] * 100:.2f}%" if not np.isnan(b_data['accuracy']) else "N/A"
        print(f"{label:<22} | {b_data['pixel_pct']:>13.2f}% | {acc_s:>12}")
    print("=" * 75 + "\n")

    # Save summary JSON for show_results.py and paper tables
    save_path = args.save_json
    if not save_path and os.path.exists(os.path.dirname(ckpt_path)):
        save_path = os.path.join(os.path.dirname(ckpt_path), "boundary_eval_results.json")

    if save_path:
        out_data = {
            "checkpoint": ckpt_path,
            "region_miou": float(region_miou),
            "region_f1": float(region_f1),
            "region_oa": float(region_oa),
            "mean_boundary_iou": float(bnd_summary['mean_boundary_iou']),
            "mean_boundary_f1": float(bnd_summary['mean_boundary_f1']),
            "overall_precision": float(bnd_summary['overall_precision']),
            "overall_recall": float(bnd_summary['overall_recall']),
            "overall_f1": float(bnd_summary['overall_f1']),
            "classes": config.classes,
            "boundary_iou_per_class": [float(x) if not np.isnan(x) else None for x in bnd_summary['boundary_iou_per_class']],
            "boundary_f1_per_class": [float(x) if not np.isnan(x) else None for x in bnd_summary['boundary_f1_per_class']],
        }
        with open(save_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Boundary evaluation summary saved to: {save_path}")


if __name__ == "__main__":
    main()

