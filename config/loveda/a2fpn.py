import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as albu

from geoseg.losses import *
from geoseg.datasets.loveda_dataset import *
from geoseg.models.A2FPN import A2FPN
from tools.utils import Lookahead, process_model_params


# =====================================================================
# CLI Overrides
# =====================================================================
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--batch_size", "--batch-size", type=int, default=None)
_parser.add_argument("--val_batch_size", "--val-batch-size", type=int, default=None)
_parser.add_argument("--epochs", "--max_epoch", type=int, default=None)
_parser.add_argument("--lr", type=float, default=None)
_parser.add_argument("--check_val_every_n_epoch", "--check-val-every-n-epoch", type=int, default=None)
_parser.add_argument("--num_workers", "--num-workers", type=int, default=None)
_parser.add_argument("--data_root", "--data-root", type=str, default=None)
_parser.add_argument("--weights_name", "--weights-name", type=str, default=None)
_cli_args, _ = _parser.parse_known_args()

# training hparam
max_epoch = _cli_args.epochs if _cli_args.epochs is not None else 16
ignore_index = len(CLASSES)
train_batch_size = _cli_args.batch_size if _cli_args.batch_size is not None else 16
val_batch_size = _cli_args.val_batch_size if _cli_args.val_batch_size is not None else 4
lr = _cli_args.lr if _cli_args.lr is not None else 1e-3
weight_decay = 0.01
backbone_lr = 1e-4
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

default_weights_name = f"a2fpn-r18-epoch{max_epoch}"
weights_name = _cli_args.weights_name or os.environ.get("WEIGHTS_NAME", default_weights_name)
weights_path = f"model_weights/loveda/{weights_name}"
test_weights_name = "last"
log_name = f"loveda/{weights_name}"
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = (
    _cli_args.check_val_every_n_epoch
    if getattr(_cli_args, 'check_val_every_n_epoch', None) is not None
    else int(os.environ.get("CHECK_VAL_EVERY_N_EPOCH", 1))
)
num_workers = _cli_args.num_workers if _cli_args.num_workers is not None else 4
pretrained_ckpt_path = None
gpus = 'auto'
resume_ckpt_path = None

# define the network (A2FPN CNN baseline)
net = A2FPN(band=3, class_num=num_classes)

# define the loss
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False

# define dataloaders
def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)

def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

data_root = _cli_args.data_root or os.environ.get("DATA_ROOT", "data/LoveDA/Train")
val_data_root = data_root.replace("Train", "Val") if "Train" in data_root else "data/LoveDA/Val"

train_dataset = LoveDATrainDataset(transform=train_aug, data_root=data_root)
val_dataset = LoveDATrainDataset(data_root=val_data_root, mosaic_ratio=0.0, transform=val_aug)
test_dataset = LoveDATestDataset()

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

# define optimizer
layerwise_params = {"base_model.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
