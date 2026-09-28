"""
Baseline Plain VMamba-UNet configuration for SEN-2 LULC dataset
==============================================================
Plain VMamba-UNet baseline with:
- VMamba-Tiny encoder backbone (pretrained on ImageNet-1k)
- Standard UNet-style decoder with direct skip connections (NO boundary gating)
- NO boundary heads
- NO boundary loss (standard JointLoss: SoftCE + Dice)
- use_aux_loss = False
"""

import argparse
import os
import os.path as osp
import sys
import torch
from torch.utils.data import DataLoader

from geoseg.losses import *
from geoseg.datasets.sen2_lulc_dataset import CLASSES, SEN2LULCDataset
from geoseg.models.BoundaryVMambaUNet import BoundaryVMambaUNet
from tools.utils import Lookahead, process_model_params


# CLI Argument Parser
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--batch_size", type=int, default=64)
_parser.add_argument("--val_batch_size", type=int, default=64)
_parser.add_argument("--epochs", type=int, default=16)
_parser.add_argument("--backbone_lr", type=float, default=1e-5)
_parser.add_argument("--decoder_lr", type=float, default=2e-4)
_parser.add_argument("--data_root", type=str, default="/home/admin/Downloads/SEN-2 LULC")
_parser.add_argument("--target_size", type=int, default=128)
_parser.add_argument("--max_train_samples", type=int, default=8000)
_parser.add_argument("--max_val_samples", type=int, default=2000)
_parser.add_argument("--num_workers", type=int, default=8)
_parser.add_argument("--check_val_every_n_epoch", type=int, default=4)
_parser.add_argument("--resume_ckpt_path", "--resume-ckpt-path", type=str, default=None)
_parser.add_argument("--resume", action="store_true", default=False)
_parser.add_argument("--pretrained_backbone_path", type=str, default="model_weights/vmamba_tiny_e292.pth")

_cli_args, _ = _parser.parse_known_args()

# Hyperparameters
max_epoch = _cli_args.epochs
train_batch_size = _cli_args.batch_size
val_batch_size = _cli_args.val_batch_size
data_root = _cli_args.data_root
target_size = _cli_args.target_size
check_val_every_n_epoch = _cli_args.check_val_every_n_epoch
ignore_index = 255

num_classes = len(CLASSES)
classes = CLASSES

weights_name = f"baseline_plain_vmamba_unet-sen2_lulc-epoch{max_epoch}"
weights_path = f"model_weights/sen2_lulc/{weights_name}"
log_name = f"sen2_lulc/{weights_name}"
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
gpus = "auto"
accelerator = "gpu"
devices = 1

resume_ckpt_path = _cli_args.resume_ckpt_path or os.environ.get("RESUME_CKPT_PATH", None)
if resume_ckpt_path is None and (_cli_args.resume or os.environ.get("AUTO_RESUME", "0") == "1"):
    last_ckpt = os.path.join(weights_path, "last.ckpt")
    if os.path.isfile(last_ckpt):
        resume_ckpt_path = last_ckpt
        print(f"\n[AUTO-RESUME] Found existing checkpoint, resuming from: {last_ckpt}\n")

pretrained_ckpt_path = None
pretrained_backbone_path = _cli_args.pretrained_backbone_path

# Model definition: Plain VMamba-UNet (No Boundary Heads, No Gating)
net = BoundaryVMambaUNet(
    num_classes=num_classes,
    decoder_channels=128,
    use_boundary_heads=False,
    gated_levels=(),
    dropout=0.1,
    use_checkpoint=True,
    pretrained_backbone_path=pretrained_backbone_path,
)

# Standard JointLoss (Soft Cross-Entropy + Dice)
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False

# Optimizer & Scheduler
backbone_lr = _cli_args.backbone_lr
decoder_lr = _cli_args.decoder_lr

layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=1e-2),
}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=decoder_lr, weight_decay=1e-2)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=1e-8
)

# Datasets & Loaders
train_dataset = SEN2LULCDataset(
    data_root=data_root,
    split="train",
    target_size=target_size,
    max_samples=_cli_args.max_train_samples,
)
val_dataset = SEN2LULCDataset(
    data_root=data_root,
    split="val",
    target_size=target_size,
    max_samples=_cli_args.max_val_samples,
)

num_workers = _cli_args.num_workers
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    shuffle=False,
    drop_last=False,
)
