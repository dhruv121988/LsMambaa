#!/usr/bin/env python3
import os
import sys
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from tools.cfg import py2cfg
from train_supervision import Supervision_Train
from geoseg.utils.boundary_metrics import BoundaryEvaluator
from tools.metric import Evaluator

def main():
    config_path = "config/loveda/boundary_vmamba_unet.py"
    ckpt_path = "model_weights/loveda/boundary_vmamba_unet-multiplicative-aux0.4-bnd0.3-epoch100/last.ckpt"

    config = py2cfg(config_path)
    model = Supervision_Train.load_from_checkpoint(ckpt_path, config=config).cuda().eval()

    val_dataset = Subset(config.val_dataset, list(range(100)))
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    ignore_index = getattr(config, 'ignore_index', len(config.classes))
    bnd_evaluator = BoundaryEvaluator(
        num_classes=config.num_classes,
        tolerance=2,
        dilation_width=2,
        ignore_index=ignore_index
    )

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="LoveDA 100 samples"):
            img = batch["img"].cuda()
            mask = batch["gt_semantic_seg"].numpy()
            pred = torch.softmax(model(img), dim=1).argmax(dim=1).cpu().numpy()

            evaluator.add_batch(mask, pred)
            bnd_evaluator.add_batch(gt_image=mask, pre_image=pred)

    bnd_summary = bnd_evaluator.summary()
    print("mIoU:", np.nanmean(evaluator.Intersection_over_Union()) * 100)
    print("F1:", np.nanmean(evaluator.F1()) * 100)
    print("OA:", evaluator.OA() * 100)
    print("mBIoU:", bnd_summary['mean_boundary_iou'] * 100)
    print("mBF1:", bnd_summary['mean_boundary_f1'] * 100)
    print("BIoU per class:", [f"{c}: {b*100:.2f}%" for c, b in zip(config.classes, bnd_summary['boundary_iou_per_class'])])

if __name__ == "__main__":
    main()
