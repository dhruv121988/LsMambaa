import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    args, _ = parser.parse_known_args()
    return args


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net

        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

        if getattr(self.net, "pretrained_loaded", False):
            print("\n" + "=" * 65)
            print("  Training Backbone Status: PRETRAINED")
            print("=" * 65 + "\n")
        else:
            print("\n" + "=" * 65)
            print("  Training Backbone Status: RANDOM INIT — WARNING")
            print("=" * 65 + "\n")

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self.net(img)
        loss = self.loss(prediction, mask)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        with torch.no_grad():
            mask_flat = mask.view(-1)
            pre_flat = pre_mask.view(-1)
            valid = (mask_flat >= 0) & (mask_flat < self.config.num_classes)
            if valid.any():
                idx = self.config.num_classes * mask_flat[valid].long() + pre_flat[valid].long()
                cm = torch.bincount(idx, minlength=self.config.num_classes ** 2).view(self.config.num_classes, self.config.num_classes)
                self.metrics_train.confusion_matrix += cm.cpu().numpy()

        return {"loss": loss}

    def on_train_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        with torch.no_grad():
            mask_flat = mask.view(-1)
            pre_flat = pre_mask.view(-1)
            valid = (mask_flat >= 0) & (mask_flat < self.config.num_classes)
            if valid.any():
                idx = self.config.num_classes * mask_flat[valid].long() + pre_flat[valid].long()
                cm = torch.bincount(idx, minlength=self.config.num_classes ** 2).view(self.config.num_classes, self.config.num_classes)
                self.metrics_val.confusion_matrix += cm.cpu().numpy()

        loss_val = self.loss(prediction, mask)
        self.log("val_loss", loss_val, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        save_last=False,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name,
    )
    last_checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=0,
        dirpath=config.weights_path,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )
    logger = CSVLogger('lightning_logs', name=config.log_name)

    callbacks = [best_checkpoint_callback, last_checkpoint_callback]
    early_stopping_patience = getattr(config, 'early_stopping_patience', 10)
    if early_stopping_patience and early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            monitor=config.monitor,
            mode=config.monitor_mode,
            patience=early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stopping)

    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    grad_clip = getattr(config, 'gradient_clip_val', getattr(config, 'grad_clip', None))
    accumulate_grad = getattr(config, 'accumulate_grad_batches', 1)
    accelerator = getattr(config, 'accelerator', 'auto')
    devices = getattr(config, 'devices', config.gpus)
    precision = getattr(config, 'precision', '16-mixed' if torch.cuda.is_available() and accelerator != 'cpu' else '32-true')
    if accelerator == 'cpu':
        devices = 1
        precision = '32-true'
    limit_train_batches = getattr(config, 'limit_train_batches', 1.0)
    limit_val_batches = getattr(config, 'limit_val_batches', 1.0)
    trainer = pl.Trainer(devices=devices, max_epochs=config.max_epoch, accelerator=accelerator,
                         precision=precision,
                         gradient_clip_val=grad_clip,
                         accumulate_grad_batches=accumulate_grad,
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         limit_train_batches=limit_train_batches,
                         limit_val_batches=limit_val_batches,
                         callbacks=callbacks, strategy='auto',
                         logger=logger)
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
   main()
