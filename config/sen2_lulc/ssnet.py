import argparse
import os
import torch
from torch.utils.data import DataLoader

from geoseg.losses import *
from geoseg.datasets.sen2_lulc_dataset import CLASSES, SEN2LULCDataset
from geoseg.models.SSNet import SSNet
from tools.utils import Lookahead, process_model_params

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--batch_size", type=int, default=64)
_parser.add_argument("--val_batch_size", type=int, default=64)
_parser.add_argument("--epochs", type=int, default=16)
_parser.add_argument("--lr", type=float, default=2e-4)
_parser.add_argument("--data_root", type=str, default="/home/admin/Downloads/SEN-2 LULC")
_parser.add_argument("--target_size", type=int, default=128)
_parser.add_argument("--max_train_samples", type=int, default=None)
_parser.add_argument("--max_val_samples", type=int, default=None)
_parser.add_argument("--num_workers", type=int, default=8)
_parser.add_argument("--check_val_every_n_epoch", type=int, default=4)
_cli_args, _ = _parser.parse_known_args()

max_epoch = _cli_args.epochs
num_classes = len(CLASSES)
classes = CLASSES
ignore_index = 255

weights_name = f"ssnet-sen2_lulc-epoch{max_epoch}"
weights_path = f"model_weights/sen2_lulc/{weights_name}"
log_name = f"sen2_lulc/{weights_name}"
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = _cli_args.check_val_every_n_epoch
pretrained_ckpt_path = None
gpus = 'auto'
resume_ckpt_path = None

net = SSNet(num_classes=num_classes, patch_size=16, vit_dim=768, num_layers=8, dropout=0.1)
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
use_aux_loss = False

train_dataset = SEN2LULCDataset(data_root=_cli_args.data_root, split="train", target_size=_cli_args.target_size, max_samples=_cli_args.max_train_samples)
val_dataset = SEN2LULCDataset(data_root=_cli_args.data_root, split="val", target_size=_cli_args.target_size, max_samples=_cli_args.max_val_samples)

num_workers = _cli_args.num_workers
train_loader = DataLoader(train_dataset, batch_size=_cli_args.batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0), shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=_cli_args.val_batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0), shuffle=False, drop_last=False)

base_optimizer = torch.optim.AdamW(net.parameters(), lr=_cli_args.lr, weight_decay=1e-2)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
