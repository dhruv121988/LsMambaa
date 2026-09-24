"""
Single-Stage Gate VMamba-UNet configuration for LoveDA dataset
==============================================================

Ablation study (Section V-B/C of the paper):
VMamba-UNet with:
- VMamba-Tiny encoder backbone
- Boundary gating applied ONLY at Level 3 (res3 gated by boundary4)
- Levels 2 and 1 use standard pass-through skip connections (no gating)
- Boundary heads at levels 4, 3, 2 and final decoder output
- Multi-task boundary loss: seg_loss + lambda_aux * aux_loss + lambda_bnd * boundary_loss
- use_aux_loss = True
"""

import argparse
import os
import os.path as osp
import sys

import torch
from torch.utils.data import DataLoader

from geoseg.losses import *
from geoseg.losses.boundary_loss import BoundaryVMambaLoss
from geoseg.datasets.loveda_dataset import *
from geoseg.models.BoundaryVMambaUNet import BoundaryVMambaUNet
from tools.utils import Lookahead, process_model_params


# =====================================================================
# CLI & Environment Variable Overrides
# =====================================================================

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--gate_type", "--gate-type",
    type=str,
    default=None,
    choices=["multiplicative", "residual", "bounded_residual"],
    help="Boundary gate variant for Level 3: multiplicative, residual, bounded_residual",
)
_parser.add_argument(
    "--lambda_aux", "--lambda-aux",
    type=float,
    default=None,
    help="Auxiliary loss weight lambda_aux (default: 0.4)",
)
_parser.add_argument(
    "--lambda_bnd", "--lambda-bnd",
    type=float,
    default=None,
    help="Boundary loss weight lambda_bnd (default: 0.3)",
)
_parser.add_argument(
    "--boundary_loss_type", "--boundary-loss-type",
    type=str,
    default=None,
    choices=["bce", "focal"],
    help="Boundary loss variant: bce or focal (default: bce)",
)
_parser.add_argument(
    "--batch_size", "--batch-size",
    type=int,
    default=None,
    help="Batch size for training and validation (default: 8)",
)
_parser.add_argument(
    "--epochs", "--max_epoch",
    type=int,
    default=None,
    help="Total training epochs (default: 100)",
)
_parser.add_argument(
    "--backbone_lr", "--backbone-lr",
    type=float,
    default=None,
    help="Learning rate for VMamba backbone (default: 1e-6, 1/100 of decoder_lr)",
)
_parser.add_argument(
    "--decoder_lr", "--decoder-lr", "--lr",
    type=float,
    default=None,
    help="Learning rate for decoder (default: 1e-4)",
)
_parser.add_argument(
    "--grad_clip", "--gradient_clip_val",
    type=float,
    default=None,
    help="Gradient clipping maximum norm (default: 1.0)",
)
_parser.add_argument(
    "--pretrained_backbone_path", "--pretrained-backbone-path",
    type=str,
    default=None,
    help="Path to VMamba-Tiny ImageNet checkpoint",
)
_parser.add_argument(
    "--data_root", "--data-root",
    type=str,
    default=None,
    help="Root directory for LoveDA training split (default: data/LoveDA/Train)",
)
_parser.add_argument(
    "--weights_name", "--weights-name",
    type=str,
    default=None,
    help="Custom name for checkpoint saving and logging",
)
_parser.add_argument(
    "--val_batch_size", "--val-batch-size",
    type=int,
    default=None,
    help="Validation batch size (default: 1 for 1024x1024 images)",
)

_cli_args, _ = _parser.parse_known_args()


# =====================================================================
# Hyperparameter Resolution (CLI > ENV > Defaults)
# =====================================================================

gate_type = _cli_args.gate_type or os.environ.get("GATE_TYPE", "multiplicative")
lambda_aux = (
    _cli_args.lambda_aux
    if _cli_args.lambda_aux is not None
    else float(os.environ.get("LAMBDA_AUX", 0.4))
)
lambda_bnd = (
    _cli_args.lambda_bnd
    if _cli_args.lambda_bnd is not None
    else float(os.environ.get("LAMBDA_BND", 0.3))
)
boundary_loss_type = (
    _cli_args.boundary_loss_type or os.environ.get("BOUNDARY_LOSS_TYPE", "bce")
)

# Training epochs & batch sizes
max_epoch = (
    _cli_args.epochs
    if _cli_args.epochs is not None
    else int(os.environ.get("EPOCHS", 100))
)
train_batch_size = (
    _cli_args.batch_size
    if _cli_args.batch_size is not None
    else int(os.environ.get("BATCH_SIZE", 2))
)
val_batch_size = (
    _cli_args.val_batch_size
    if _cli_args.val_batch_size is not None
    else int(os.environ.get("VAL_BATCH_SIZE", 1))
)

# Learning rates (backbone LR split vs decoder LR)
decoder_lr = (
    _cli_args.decoder_lr
    if _cli_args.decoder_lr is not None
    else float(os.environ.get("DECODER_LR", 1e-4))
)
lr = decoder_lr
backbone_lr = (
    _cli_args.backbone_lr
    if _cli_args.backbone_lr is not None
    else float(os.environ.get("BACKBONE_LR", 1e-6))
)

weight_decay = 0.01
backbone_weight_decay = 0.01

# Gradient clipping
gradient_clip_val = (
    _cli_args.grad_clip
    if _cli_args.grad_clip is not None
    else float(os.environ.get("GRAD_CLIP", 1.0))
)
gradient_clip = gradient_clip_val
grad_clip = gradient_clip_val

# Class metadata
num_classes = len(CLASSES)
classes = CLASSES
ignore_index = len(CLASSES)

# Model weight paths and logging names (auto-labeled by ablation config)
default_weights_name = (
    f"single_stage_gate-{gate_type}-aux{lambda_aux}-bnd{lambda_bnd}-epoch{max_epoch}"
)
weights_name = _cli_args.weights_name or os.environ.get("WEIGHTS_NAME", default_weights_name)
weights_path = f"model_weights/loveda/{weights_name}"
test_weights_name = weights_name
log_name = f"loveda/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
gpus = "auto"
resume_ckpt_path = None
pretrained_ckpt_path = None

# Pretrained backbone path
default_pretrained_backbone = "model_weights/vmamba_tiny_e292.pth"
pretrained_backbone_path = (
    _cli_args.pretrained_backbone_path or
    os.environ.get("PRETRAINED_BACKBONE_PATH", default_pretrained_backbone)
)


# =====================================================================
# Network Definition: Gating ONLY at Level 3 (res3 gated by boundary4)
# =====================================================================

net = BoundaryVMambaUNet(
    num_classes=num_classes,
    decoder_channels=128,
    gate_type=gate_type,
    alpha=1.0,
    beta=0.5,
    dropout=0.1,
    use_boundary_heads=True,
    gated_levels=(3,),  # Single-stage gate at Level 3 only
    pretrained_backbone_path=pretrained_backbone_path,
)


# =====================================================================
# Multi-Task Loss Definition
# (seg_loss + lambda_aux * aux_loss + lambda_bnd * boundary_loss)
# =====================================================================

loss = BoundaryVMambaLoss(
    ignore_index=ignore_index,
    label_smoothing=0.1,
    lambda_aux=lambda_aux,
    lambda_bnd=lambda_bnd,
    boundary_loss_type=boundary_loss_type,
)
use_aux_loss = True


# =====================================================================
# Dataloaders
# =====================================================================

data_root = (
    _cli_args.data_root
    if getattr(_cli_args, 'data_root', None) is not None
    else os.environ.get("DATA_ROOT", "data/LoveDA/Train")
)

train_dataset = LoveDATrainDataset(
    transform=train_aug,
    data_root=data_root,
)

val_data_root = data_root.replace("Train", "Val") if "Train" in data_root else "data/LoveDA/Val"
val_dataset = LoveDATrainDataset(data_root=val_data_root, mosaic_ratio=0.0, transform=val_aug)
test_dataset = LoveDATestDataset()

_has_train_data = len(train_dataset) > 0
_has_val_data = len(val_dataset) > 0

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=_has_train_data,
    drop_last=len(train_dataset) >= train_batch_size if _has_train_data else False,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)


# =====================================================================
# Optimizer & Learning Rate Scheduler
# (Layerwise LR: backbone gets backbone_lr, decoder gets decoder_lr)
# =====================================================================

layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay),
}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=1e-8
)
