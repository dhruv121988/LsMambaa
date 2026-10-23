#!/usr/bin/env python3
"""
Publication Figure Generator: Side-by-Side Qualitative Visual Comparisons
Format: [Original Aerial Image] | [Ground Truth Mask] | [Proposed Model Prediction]
With publication-grade horizontal color legend at the bottom.
"""
import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath("."))

from tools.cfg import py2cfg
from train_supervision import Supervision_Train

# =====================================================================
# Publication Color Palette for LoveDA (Official Benchmark 7-Classes)
# Matching Remote Sensing Publications (e.g., ISPRS / IEEE TGRS format)
# =====================================================================
CLASSES_LEGEND = [
    ("Background",   [100, 116, 139]),  # #64748b - Slate gray
    ("Building",     [239, 68, 68]),    # #ef4444 - Vibrant Red
    ("Road",         [250, 204, 21]),   # #facc15 - Road Yellow
    ("Water",        [6, 182, 212]),    # #06b6d4 - Cyan / River Blue
    ("Barren",       [168, 85, 247]),   # #a855f7 - Purple
    ("Forest",       [34, 197, 94]),    # #22c55e - Emerald Green
    ("Agricultural", [132, 204, 22]),   # #84cc16 - Vibrant field lime/crop
]

PALETTE = np.array([
    [100, 116, 139],
    [239, 68, 68],
    [250, 204, 21],
    [6, 182, 212],
    [168, 85, 247],
    [34, 197, 94],
    [132, 204, 22]
], dtype=np.uint8)

def colorize_mask(mask_2d):
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(PALETTE)):
        rgb[mask_2d == c] = PALETTE[c]
    return rgb

def main():
    config_path = "config/loveda/boundary_vmamba_unet.py"
    ckpt_path = "model_weights/loveda/boundary_vmamba_unet-multiplicative-aux0.4-bnd0.3-epoch16/last.ckpt"
    output_png = "paper_qualitative_comparison.png"

    print(f"Loading config: {config_path}")
    print(f"Loading checkpoint: {ckpt_path}")

    cfg = py2cfg(config_path)
    model = Supervision_Train.load_from_checkpoint(ckpt_path, config=cfg).cuda().eval()

    # Select representative test scenes (e.g. 1 Urban, 1 Rural)
    scenes = [
        {
            "title": "Urban Scene #3515",
            "img_path": "data/LoveDA/Val/Val/Urban/images_png/3515.png",
            "gt_path": "data/LoveDA/Val/Val/Urban/masks_png/3515.png"
        },
        {
            "title": "Rural Scene #2522",
            "img_path": "data/LoveDA/Val/Val/Rural/images_png/2522.png",
            "gt_path": "data/LoveDA/Val/Val/Rural/masks_png/2522.png"
        }
    ]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    num_scenes = len(scenes)
    fig, axes = plt.subplots(num_scenes, 3, figsize=(12, 4 * num_scenes), dpi=300)
    plt.subplots_adjust(wspace=0.03, hspace=0.05, bottom=0.15, top=0.96, left=0.02, right=0.98)

    col_headers = ["(a) Aerial Image (RGB)", "(b) Ground Truth", "(c) BoundaryVMambaUNet (Proposed)"]

    for row_idx, scene in enumerate(scenes):
        orig_img = Image.open(scene["img_path"]).convert("RGB")
        orig_w, orig_h = orig_img.size
        gt_mask = np.array(Image.open(scene["gt_path"]))

        # Model Inference at 512x512 then resized to original
        img_512 = orig_img.resize((512, 512), Image.BILINEAR)
        img_np = np.array(img_512, dtype=np.float32) / 255.0
        img_norm = (img_np - mean) / std
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().cuda()

        with torch.no_grad():
            res1, res2, res3, res4 = model.net.backbone(tensor)
            out = model.net.decoder(res1, res2, res3, res4, h=512, w=512)
            pred = torch.argmax(out['seg_logits'], dim=1).squeeze().cpu().numpy()

        # Resize prediction back to original size for crisp alignment
        pred_full = np.array(Image.fromarray(pred.astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST))

        # Convert raw LoveDA gt_mask (1-indexed) to 0-indexed (0: Background ... 6: Agricultural)
        gt_0idx = np.zeros_like(gt_mask, dtype=np.uint8)
        for raw_v in range(1, 8):
            gt_0idx[gt_mask == raw_v] = raw_v - 1

        gt_color = colorize_mask(gt_0idx)
        pred_color = colorize_mask(pred_full)

        # Plot 3 columns
        ax_img = axes[row_idx, 0]
        ax_gt = axes[row_idx, 1]
        ax_pred = axes[row_idx, 2]

        ax_img.imshow(orig_img)
        ax_gt.imshow(gt_color)
        ax_pred.imshow(pred_color)

        for ax in [ax_img, ax_gt, ax_pred]:
            ax.set_xticks([])
            ax.set_yticks([])

        if row_idx == 0:
            for ax, title in zip([ax_img, ax_gt, ax_pred], col_headers):
                ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    # Add Horizontal Publication Legend at the Bottom
    # Like in the user's reference figure
    legend_ax = fig.add_axes([0.05, 0.02, 0.90, 0.08])
    legend_ax.axis('off')

    num_items = len(CLASSES_LEGEND)
    box_w = 0.08
    box_h = 0.40
    spacing = 1.0 / num_items

    for i, (name, color_rgb) in enumerate(CLASSES_LEGEND):
        norm_color = [c / 255.0 for c in color_rgb]
        x_start = i * spacing + 0.015
        rect = mpatches.Rectangle(
            (x_start, 0.45), box_w, box_h,
            facecolor=norm_color, edgecolor='black', linewidth=1.0,
            transform=legend_ax.transAxes
        )
        legend_ax.add_patch(rect)
        legend_ax.text(
            x_start + box_w / 2, 0.15, name,
            ha='center', va='center', fontsize=9.5, fontfamily='serif',
            transform=legend_ax.transAxes
        )

    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    print(f"Publication-grade comparison figure successfully saved to: {output_png}")

if __name__ == "__main__":
    main()
