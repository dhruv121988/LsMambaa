import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]
    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        mask_tif = cv2.cvtColor(mask_tif, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


from geoseg.utils.boundary_metrics import BoundaryEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"]) ## lr is flip TTA, d4 is multi-scale TTA
    arg("--rgb", help="whether output rgb masks", action='store_true')
    arg("--val", help="whether eval validation set", action='store_true')
    arg("--boundary_tolerance", type=int, default=2, help="Tolerance width in pixels for boundary metrics (default: 2)")
    arg("--no_boundary", action='store_true', help="Disable boundary metrics evaluation")
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)
    model.cuda()
    model.eval()
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset
    if args.val:
        evaluator = Evaluator(num_class=config.num_classes)
        evaluator.reset()
        test_dataset = config.val_dataset
        ignore_index = getattr(config, 'ignore_index', len(config.classes))
        boundary_evaluator = BoundaryEvaluator(
            num_classes=config.num_classes,
            tolerance=args.boundary_tolerance,
            dilation_width=args.boundary_tolerance,
            ignore_index=ignore_index,
        )

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].cuda())

            image_ids = input["img_id"]
            if args.val:
                masks_true = input['gt_semantic_seg']

            img_type = input['img_type']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                mask_name = image_ids[i]
                mask_type = img_type[i]
                if args.val:
                    if not os.path.exists(os.path.join(args.output_path, mask_type)):
                        os.mkdir(os.path.join(args.output_path, mask_type))
                    gt_mask = masks_true[i].cpu().numpy()
                    evaluator.add_batch(pre_image=mask, gt_image=gt_mask)
                    if not args.no_boundary:
                        boundary_evaluator.add_batch(gt_image=gt_mask, pre_image=mask)
                    results.append((mask, str(args.output_path / mask_type / mask_name), args.rgb))
                else:
                    results.append((mask, str(args.output_path / mask_name), args.rgb))
    if args.val:
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()
        for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
            print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
        print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA))

        if not args.no_boundary:
            bnd_summary = boundary_evaluator.summary(class_names=config.classes)
            print('\n' + '=' * 65)
            print('Boundary Evaluation Metrics (Tolerance = {} px)'.format(args.boundary_tolerance))
            print('=' * 65)
            print('Boundary Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}'.format(
                bnd_summary['overall_precision'],
                bnd_summary['overall_recall'],
                bnd_summary['overall_f1'],
            ))
            print('Mean Boundary IoU (mBIoU): {:.4f} | Mean Boundary F1 (mBF1): {:.4f}'.format(
                bnd_summary['mean_boundary_iou'],
                bnd_summary['mean_boundary_f1'],
            ))
            print('-' * 65)
            print(f"{'Class':<16} {'BIoU':<10} {'BF1':<10} {'B-Precision':<14} {'B-Recall':<10}")
            print('-' * 65)
            for c_name, biou, bf1, bp, br in zip(
                config.classes,
                bnd_summary['boundary_iou_per_class'],
                bnd_summary['boundary_f1_per_class'],
                bnd_summary['boundary_precision_per_class'],
                bnd_summary['boundary_recall_per_class'],
            ):
                biou_s = f"{biou:.4f}" if not np.isnan(biou) else "N/A"
                bf1_s = f"{bf1:.4f}" if not np.isnan(bf1) else "N/A"
                bp_s = f"{bp:.4f}" if not np.isnan(bp) else "N/A"
                br_s = f"{br:.4f}" if not np.isnan(br) else "N/A"
                print(f"{c_name:<16} {biou_s:<10} {bf1_s:<10} {bp_s:<14} {br_s:<10}")

            print('-' * 65)
            print('Boundary-Distance Accuracy (Accuracy vs Distance to Edge):')
            print('-' * 65)
            print(f"{'Distance Range':<20} {'Pixel Ratio (%)':<18} {'Accuracy (%)':<15}")
            print('-' * 65)
            for bucket_label, b_data in bnd_summary['distance_accuracy'].items():
                acc_s = f"{b_data['accuracy'] * 100:.2f}%" if not np.isnan(b_data['accuracy']) else "N/A"
                print(f"{bucket_label:<20} {b_data['pixel_pct']:>6.2f}%            {acc_s:>10}")
            print('=' * 65 + '\n')

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()
