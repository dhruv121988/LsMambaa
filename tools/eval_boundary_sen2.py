#!/usr/bin/env python3
"""
Evaluate boundary metrics on SEN-2 LULC (Indian dataset) using BoundaryEvaluator.
"""
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Ensure path
sys.path.insert(0, os.path.abspath("."))

from tools.cfg import py2cfg
from train_supervision import Supervision_Train
from geoseg.utils.boundary_metrics import BoundaryEvaluator
from tools.metric import Evaluator


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate boundary metrics on SEN-2 LULC")
    parser.add_argument("-c", "--config", type=str, default="config/sen2_lulc/boundary_vmamba_unet.py")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    config_path = args.config
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = "model_weights/sen2_lulc/boundary_vmamba_unet-sen2_lulc-epoch16/boundary_vmamba_unet-sen2_lulc-epoch16.ckpt"
        if not os.path.exists(ckpt_path):
            ckpt_path = "model_weights/sen2_lulc/boundary_vmamba_unet-sen2_lulc-epoch16/last.ckpt"

    print(f"Loading config: {config_path}")
    print(f"Loading checkpoint: {ckpt_path}")

    config = py2cfg(config_path)
    model = Supervision_Train.load_from_checkpoint(ckpt_path, config=config)
    model.cuda()
    model.eval()

    # Load validation samples
    from geoseg.datasets.sen2_lulc_dataset import SEN2LULCDataset
    val_data_root = getattr(config, 'data_root', None)
    if val_data_root is None:
        if hasattr(config, 'val_dataset') and hasattr(config.val_dataset, 'data_root'):
            val_data_root = config.val_dataset.data_root
        else:
            val_data_root = "/home/admin/Downloads/SEN-2 LULC"

    target_size = getattr(config, 'target_size', None)
    if target_size is None:
        if hasattr(config, 'val_dataset') and hasattr(config.val_dataset, 'target_size'):
            target_size = config.val_dataset.target_size
        else:
            target_size = 128

    val_dataset = SEN2LULCDataset(
        data_root=val_data_root,
        split="val",
        target_size=target_size,
        max_samples=args.max_samples
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    bnd_evaluator = BoundaryEvaluator(
        num_classes=config.num_classes,
        tolerance=2,
        dilation_width=2,
        ignore_index=None
    )

    print(f"Evaluating on {len(val_dataset)} validation samples...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            img = batch["img"].cuda()
            mask = batch["gt_semantic_seg"].numpy() # (B, H, W)
            
            output = model(img)
            if isinstance(output, (tuple, list)):
                output = output[0]
            elif isinstance(output, dict):
                output = output.get("seg_logits", output.get("pred", output))
            pred = torch.softmax(output, dim=1).argmax(dim=1).cpu().numpy() # (B, H, W)

            # Accumulate region metrics
            evaluator.add_batch(mask, pred)

            # Accumulate boundary metrics
            bnd_evaluator.add_batch(gt_image=mask, pre_image=pred)

    bnd_summary = bnd_evaluator.summary()
    region_miou = np.nanmean(evaluator.Intersection_over_Union()) * 100
    region_f1 = np.nanmean(evaluator.F1()) * 100
    region_oa = evaluator.OA() * 100

    print("\n" + "=" * 75)
    print("      SEN-2 LULC (INDIAN DATASET) - EVALUATION REPORT (EPOCH 16)      ")
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


if __name__ == "__main__":
    main()
