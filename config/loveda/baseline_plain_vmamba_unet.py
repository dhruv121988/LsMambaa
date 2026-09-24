"""
Baseline Plain VMamba-UNet configuration for LoveDA dataset
============================================================

Ablation study (Section V-B/C of the paper):
Plain VMamba-UNet baseline with:
- VMamba-Tiny encoder backbone
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
from geoseg.datasets.loveda_dataset import *
from geoseg.models.BoundaryVMambaUNet import BoundaryVMambaUNet
from tools.utils import Lookahead, process_model_params


# =====================================================================
# CLI & Environment Variable Overrides
# =====================================================================

_parser = argparse.ArgumentParser(add_help=False)
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
    "--max_train_samples", "--max-train-samples",
    type=int,
    default=None,
    help="Maximum number of training samples to load (subsampling)",
)
_parser.add_argument(
    "--max_val_samples", "--max-val-samples",
    type=int,
    default=None,
    help="Maximum number of validation samples to load (subsampling)",
)
_parser.add_argument(
    "--accelerator",
    type=str,
    default=None,
    choices=["auto", "gpu", "cuda", "cpu"],
    help="Accelerator device: auto, gpu, cuda, or cpu",
)
_parser.add_argument(
    "--precision",
    type=str,
    default=None,
    help="Precision for PyTorch Lightning Trainer (e.g. 16-mixed, bf16-mixed, 32-true)",
)
_parser.add_argument(
    "--val_batch_size", "--val-batch-size",
    type=int,
    default=None,
    help="Validation batch size (default: 1 for 1024x1024 images)",
)
_parser.add_argument(
    "--accumulate_grad_batches", "--accumulate-grad-batches", "--grad_accum",
    type=int,
    default=None,
    help="Number of batches to accumulate gradients (default: 1)",
)
_parser.add_argument(
    "--check_val_every_n_epoch", "--check-val-every-n-epoch",
    type=int,
    default=None,
    help="Run validation evaluation every N epochs (default: 1)",
)
_parser.add_argument(
    "--num_workers", "--num-workers",
    type=int,
    default=None,
    help="DataLoader worker subprocesses (default: 8)",
)
_parser.add_argument(
    "--resume_ckpt_path", "--resume-ckpt-path",
    type=str,
    default=None,
    help="Path to checkpoint file to resume training from",
)
_parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Automatically resume from weights_path/last.ckpt if it exists",
)

_cli_args, _ = _parser.parse_known_args()


# =====================================================================
# Hyperparameter Resolution (CLI > ENV > Defaults)
# =====================================================================

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
accumulate_grad_batches = (
    _cli_args.accumulate_grad_batches
    if _cli_args.accumulate_grad_batches is not None
    else int(os.environ.get("ACCUMULATE_GRAD_BATCHES", 1))
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
default_weights_name = f"baseline_plain_vmamba_unet-epoch{max_epoch}"
weights_name = _cli_args.weights_name or os.environ.get("WEIGHTS_NAME", default_weights_name)
weights_path = f"model_weights/loveda/{weights_name}"
test_weights_name = weights_name
log_name = f"loveda/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = (
    _cli_args.check_val_every_n_epoch
    if getattr(_cli_args, 'check_val_every_n_epoch', None) is not None
    else int(os.environ.get("CHECK_VAL_EVERY_N_EPOCH", 1))
)
gpus = "auto"
accelerator = _cli_args.accelerator or os.environ.get("ACCELERATOR", "auto")
if accelerator == "cuda":
    accelerator = "gpu"
devices = 1 if accelerator in ["gpu", "cuda"] else ("auto" if accelerator == "auto" else 1)
resume_ckpt_path = _cli_args.resume_ckpt_path or os.environ.get("RESUME_CKPT_PATH", None)
if resume_ckpt_path is None and (_cli_args.resume or os.environ.get("AUTO_RESUME", "0") == "1"):
    last_ckpt = osp.join(weights_path, "last.ckpt")
    if osp.isfile(last_ckpt):
        resume_ckpt_path = last_ckpt
        print(f"\n[AUTO-RESUME] Found existing checkpoint, resuming from: {last_ckpt}\n")

pretrained_ckpt_path = None

# Pretrained backbone path
default_pretrained_backbone = "model_weights/vmamba_tiny_e292.pth"
pretrained_backbone_path = (
    _cli_args.pretrained_backbone_path or
    os.environ.get("PRETRAINED_BACKBONE_PATH", default_pretrained_backbone)
)


# =====================================================================
# Network Definition: Plain VMamba-UNet (No Boundary Heads, No Gating)
# =====================================================================

net = BoundaryVMambaUNet(
    num_classes=num_classes,
    decoder_channels=128,
    use_boundary_heads=False,
    gated_levels=(),
    dropout=0.1,
    use_checkpoint=True,
    pretrained_backbone_path=pretrained_backbone_path,
)


# =====================================================================
# Standard Segmentation Loss (No Boundary Loss)
# =====================================================================

loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False


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

# Subsampling for smoke tests / ablations
max_train_samples = (
    _cli_args.max_train_samples
    if _cli_args.max_train_samples is not None
    else (int(os.environ["MAX_TRAIN_SAMPLES"]) if "MAX_TRAIN_SAMPLES" in os.environ else None)
)
max_val_samples = (
    _cli_args.max_val_samples
    if _cli_args.max_val_samples is not None
    else (int(os.environ["MAX_VAL_SAMPLES"]) if "MAX_VAL_SAMPLES" in os.environ else None)
)

if max_train_samples is not None and len(train_dataset.img_ids) > max_train_samples:
    train_dataset.img_ids = train_dataset.img_ids[:max_train_samples]

if max_val_samples is not None and len(val_dataset.img_ids) > max_val_samples:
    val_dataset.img_ids = val_dataset.img_ids[:max_val_samples]

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
