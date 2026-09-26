import argparse
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader

from geoseg.losses import *
from geoseg.datasets.loveda_dataset import *
from geoseg.models.TransUNet import TransUNet
from tools.utils import Lookahead, process_model_params

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--batch_size", "--batch-size", type=int, default=None)
_parser.add_argument("--val_batch_size", "--val-batch-size", type=int, default=None)
_parser.add_argument("--epochs", "--max_epoch", type=int, default=None)
_parser.add_argument("--lr", type=float, default=None)
_parser.add_argument("--check_val_every_n_epoch", type=int, default=None)
_parser.add_argument("--num_workers", type=int, default=None)
_parser.add_argument("--data_root", type=str, default=None)
_parser.add_argument("--weights_name", type=str, default=None)
_cli_args, _ = _parser.parse_known_args()

max_epoch = _cli_args.epochs if _cli_args.epochs is not None else 16
ignore_index = len(CLASSES)
train_batch_size = _cli_args.batch_size if _cli_args.batch_size is not None else 12
val_batch_size = _cli_args.val_batch_size if _cli_args.val_batch_size is not None else 4
lr = _cli_args.lr if _cli_args.lr is not None else 3e-4
weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

default_weights_name = f"transunet-r50_vitb-epoch{max_epoch}"
weights_name = _cli_args.weights_name or os.environ.get("WEIGHTS_NAME", default_weights_name)
weights_path = f"model_weights/loveda/{weights_name}"
test_weights_name = "last"
log_name = f"loveda/{weights_name}"
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = _cli_args.check_val_every_n_epoch or 1
num_workers = _cli_args.num_workers or 4
pretrained_ckpt_path = None
gpus = 'auto'
resume_ckpt_path = None

net = TransUNet(num_classes=num_classes, img_size=512, pretrained=True, dropout=0.1)
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
use_aux_loss = False

data_root = _cli_args.data_root or os.environ.get("DATA_ROOT", "data/LoveDA/Train")
train_dataset = LoveDATrainDataset(transform=train_aug, data_root=data_root)
val_data_root = data_root.replace("Train", "Val") if "Train" in data_root else "data/LoveDA/Val"
val_dataset = LoveDAValDataset(transform=val_aug, data_root=val_data_root)

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
    shuffle=False,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    drop_last=False,
)

layerwise_params = {"conv1.*": dict(lr=lr*0.1), "layer1.*": dict(lr=lr*0.1), "layer2.*": dict(lr=lr*0.1), "layer3.*": dict(lr=lr*0.1)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
