"""
BoundaryVMambaUNet configuration for SEN-2 LULC dataset
======================================================
Architecture:
- VMamba-Tiny encoder with 4-way Cross-Scan Triton kernels
- Progressive Boundary-Gated Decoder (multiplicative gating)
- Multi-Task Boundary Loss: SoftCE + Dice + Boundary BCE
- 7 LULC Classes: Water, Dense Forest, Sparse Forest, Barren Land, Built-up, Agriculture, Fallow
"""

import argparse
import os
import os.path as osp
import sys
import torch
from torch.utils.data import DataLoader

from geoseg.losses import *
from geoseg.losses.boundary_loss import BoundaryVMambaLoss
from geoseg.datasets.sen2_lulc_dataset import CLASSES, SEN2LULCDataset
from geoseg.models.BoundaryVMambaUNet import BoundaryVMambaUNet
from tools.utils import Lookahead, process_model_params


# CLI Argument Parser
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--batch_size", type=int, default=32)
_parser.add_argument("--val_batch_size", type=int, default=32)
_parser.add_argument("--epochs", type=int, default=16)
_parser.add_argument("--backbone_lr", type=float, default=1e-5)
_parser.add_argument("--decoder_lr", type=float, default=2e-4)
_parser.add_argument("--data_root", type=str, default="/home/admin/Downloads/SEN-2 LULC")
_parser.add_argument("--target_size", type=int, default=128)
_parser.add_argument("--max_train_samples", type=int, default=None)
_parser.add_argument("--max_val_samples", type=int, default=None)
_parser.add_argument("--num_workers", type=int, default=8)
_parser.add_argument("--check_val_every_n_epoch", type=int, default=1)

_cli_args, _ = _parser.parse_known_args()

# Hyperparameters
max_epoch = _cli_args.epochs
train_batch_size = _cli_args.batch_size
val_batch_size = _cli_args.val_batch_size
data_root = _cli_args.data_root
target_size = _cli_args.target_size
check_val_every_n_epoch = _cli_args.check_val_every_n_epoch

num_classes = len(CLASSES)
classes = CLASSES

weights_name = f"boundary_vmamba_unet-sen2_lulc-epoch{max_epoch}"
weights_path = f"model_weights/sen2_lulc/{weights_name}"
log_name = f"sen2_lulc/{weights_name}"
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
gpus = "auto"
accelerator = "gpu"
devices = 1

resume_ckpt_path = None
pretrained_ckpt_path = None
pretrained_backbone_path = "model_weights/vmamba_tiny_e292.pth"

# Model definition
net = BoundaryVMambaUNet(
    num_classes=num_classes,
    decoder_channels=128,
    gate_type="multiplicative",
    alpha=1.0,
    beta=0.5,
    dropout=0.1,
    pretrained_backbone_path=pretrained_backbone_path,
)

loss = BoundaryVMambaLoss(
    ignore_index=255,
    label_smoothing=0.1,
    lambda_aux=0.4,
    lambda_bnd=0.3,
    boundary_loss_type="bce",
)
use_aux_loss = True

layerwise_params = {
    "backbone.*": dict(lr=1e-5, weight_decay=1e-2),
}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=1e-4, weight_decay=1e-2)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=1e-8
)

# Datasets & Loaders
train_dataset = SEN2LULCDataset(
    data_root=data_root,
    split="train",
    target_size=target_size,
    max_samples=_cli_args.max_train_samples
)
val_dataset = SEN2LULCDataset(
    data_root=data_root,
    split="val",
    target_size=target_size,
    max_samples=_cli_args.max_val_samples
)

num_workers = _cli_args.num_workers
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    shuffle=False,
    drop_last=False
)
