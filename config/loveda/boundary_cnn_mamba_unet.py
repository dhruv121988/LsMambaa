"""
BoundaryCNNMambaUNet configuration for LoveDA dataset
=====================================================

Swapped architecture:
- Encoder: CNN Backbone (ResNet-34 / ResNet-50) capturing high-resolution local features
- Decoder: BoundaryMambaDecoder (Progressive Boundary-Gated SS2D / Mamba)
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
from geoseg.models.BoundaryCNNMambaUNet import BoundaryCNNMambaUNet
from tools.utils import Lookahead, process_model_params


# =====================================================================
# CLI Overrides
# =====================================================================

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--backbone_name", "--backbone-name", type=str, default="resnet34",
                     help="CNN backbone (e.g. resnet18, resnet34, resnet50)")
_parser.add_argument("--pretrained_backbone", "--pretrained-backbone", action="store_true", default=True)
_parser.add_argument("--gate_type", "--gate-type", type=str, default="multiplicative",
                     choices=["multiplicative", "residual", "bounded_residual"])
_parser.add_argument("--lambda_aux", "--lambda-aux", type=float, default=0.4)
_parser.add_argument("--lambda_bnd", "--lambda-bnd", type=float, default=0.3)
_parser.add_argument("--boundary_loss_type", "--boundary-loss-type", type=str, default="bce",
                     choices=["bce", "focal"])
_parser.add_argument("--batch_size", "--batch-size", type=int, default=2)
_parser.add_argument("--val_batch_size", "--val-batch-size", type=int, default=1)
_parser.add_argument("--epochs", "--max_epoch", type=int, default=85)
_parser.add_argument("--backbone_lr", "--backbone-lr", type=float, default=1e-4)
_parser.add_argument("--decoder_lr", "--decoder-lr", "--lr", type=float, default=2e-4)
_parser.add_argument("--grad_clip", "--gradient_clip_val", type=float, default=1.0)
_parser.add_argument("--data_root", "--data-root", type=str, default=None)
_parser.add_argument("--weights_name", "--weights-name", type=str, default=None)
_parser.add_argument("--accumulate_grad_batches", type=int, default=1)
_parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
_parser.add_argument("--num_workers", type=int, default=4)
_parser.add_argument("--resume_ckpt_path", type=str, default=None)
_parser.add_argument("--resume", action="store_true", default=False)
_parser.add_argument("--limit_train_batches", type=float, default=None)
_parser.add_argument("--limit_val_batches", type=float, default=None)

_cli_args, _ = _parser.parse_known_args()

limit_train_batches = (
    _cli_args.limit_train_batches
    if _cli_args.limit_train_batches is not None
    else float(os.environ.get("LIMIT_TRAIN_BATCHES", 1.0))
)
limit_val_batches = (
    _cli_args.limit_val_batches
    if _cli_args.limit_val_batches is not None
    else float(os.environ.get("LIMIT_VAL_BATCHES", 1.0))
)

# Hyperparameters
backbone_name = _cli_args.backbone_name or os.environ.get("BACKBONE_NAME", "resnet34")
pretrained_backbone = _cli_args.pretrained_backbone
gate_type = _cli_args.gate_type or os.environ.get("GATE_TYPE", "multiplicative")
lambda_aux = _cli_args.lambda_aux if _cli_args.lambda_aux is not None else float(os.environ.get("LAMBDA_AUX", 0.4))
lambda_bnd = _cli_args.lambda_bnd if _cli_args.lambda_bnd is not None else float(os.environ.get("LAMBDA_BND", 0.3))
boundary_loss_type = _cli_args.boundary_loss_type or os.environ.get("BOUNDARY_LOSS_TYPE", "bce")

max_epoch = _cli_args.epochs if _cli_args.epochs is not None else int(os.environ.get("EPOCHS", 85))
train_batch_size = _cli_args.batch_size if _cli_args.batch_size is not None else int(os.environ.get("BATCH_SIZE", 4))
val_batch_size = _cli_args.val_batch_size if _cli_args.val_batch_size is not None else int(os.environ.get("VAL_BATCH_SIZE", 4))
num_workers = _cli_args.num_workers if _cli_args.num_workers is not None else int(os.environ.get("NUM_WORKERS", 4))
accumulate_grad_batches = _cli_args.accumulate_grad_batches if _cli_args.accumulate_grad_batches is not None else int(os.environ.get("ACCUMULATE_GRAD_BATCHES", 1))

decoder_lr = _cli_args.decoder_lr if _cli_args.decoder_lr is not None else float(os.environ.get("DECODER_LR", 2e-4))
lr = decoder_lr
backbone_lr = _cli_args.backbone_lr if _cli_args.backbone_lr is not None else float(os.environ.get("BACKBONE_LR", 1e-4))
weight_decay = 0.01
backbone_weight_decay = 0.01

gradient_clip_val = _cli_args.grad_clip if _cli_args.grad_clip is not None else float(os.environ.get("GRAD_CLIP", 1.0))
gradient_clip = gradient_clip_val
grad_clip = gradient_clip_val

num_classes = len(CLASSES)
classes = CLASSES
ignore_index = len(CLASSES)

default_weights_name = f"boundary_cnn_mamba_unet_{backbone_name}-{gate_type}-epoch{max_epoch}"
weights_name = _cli_args.weights_name or os.environ.get("WEIGHTS_NAME", default_weights_name)
weights_path = f"model_weights/loveda/{weights_name}"
test_weights_name = weights_name
log_name = f"loveda/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = _cli_args.check_val_every_n_epoch if _cli_args.check_val_every_n_epoch is not None else int(os.environ.get("CHECK_VAL_EVERY_N_EPOCH", 1))
gpus = "auto"

resume_ckpt_path = _cli_args.resume_ckpt_path or os.environ.get("RESUME_CKPT_PATH", None)
if resume_ckpt_path is None and (_cli_args.resume or os.environ.get("AUTO_RESUME", "0") == "1"):
    last_ckpt = osp.join(weights_path, "last.ckpt")
    if osp.isfile(last_ckpt):
        resume_ckpt_path = last_ckpt

pretrained_ckpt_path = None

# =====================================================================
# Model Definition
# =====================================================================

net = BoundaryCNNMambaUNet(
    num_classes=num_classes,
    backbone_name=backbone_name,
    pretrained=pretrained_backbone,
    decoder_channels=128,
    gate_type=gate_type,
    alpha=1.0,
    beta=0.5,
    dropout=0.1,
    use_boundary_heads=True,
)

# =====================================================================
# Loss Definition
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
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    shuffle=_has_train_data,
    drop_last=len(train_dataset) >= train_batch_size if _has_train_data else False,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    drop_last=False,
)

# =====================================================================
# Optimizer & Learning Rate Scheduler
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
