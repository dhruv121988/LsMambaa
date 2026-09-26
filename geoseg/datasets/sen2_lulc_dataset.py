"""
SEN-2 LULC Dataset implementation for GeoSeg / LSMamba
======================================================
Paper: Sawant, S., Garg, R. D., Meshram, V., & Mistry, S. (2023).
"Sen-2 LULC: Land use land cover dataset for deep learning approaches."
Data in Brief, 51, 109724.

7 LULC Classes:
1. Water
2. Dense Forest
3. Sparse Forest
4. Barren Land
5. Built-up
6. Agriculture Land
7. Fallow Land
"""

import os
import os.path as osp
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import albumentations as albu
from albumentations.pytorch import ToTensorV2


CLASSES = (
    'Water',
    'Dense Forest',
    'Sparse Forest',
    'Barren Land',
    'Built-up',
    'Agriculture Land',
    'Fallow Land'
)

COLOR_MAP = {
    0: (0, 0, 255),       # Water - Blue
    1: (0, 100, 0),       # Dense Forest - Dark Green
    2: (144, 238, 144),   # Sparse Forest - Light Green
    3: (165, 42, 42),     # Barren Land - Brown
    4: (255, 0, 0),       # Built-up - Red
    5: (255, 255, 0),     # Agriculture Land - Yellow
    6: (210, 180, 140)    # Fallow Land - Tan
}


def mask_to_boundary(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Generate 1-channel binary boundary map dynamically from class label mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, width))
    gradient = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    return (gradient > 0).astype(np.float32)


def get_training_transform(target_size=128):
    return albu.Compose([
        albu.Resize(target_size, target_size, interpolation=cv2.INTER_LINEAR),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'boundary': 'mask'})


def get_val_transform(target_size=128):
    return albu.Compose([
        albu.Resize(target_size, target_size, interpolation=cv2.INTER_LINEAR),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'boundary': 'mask'})


class SEN2LULCDataset(Dataset):
    """
    Dataset loader for SEN-2 LULC with dynamic on-the-fly boundary generation.
    """
    def __init__(
        self,
        data_root: str = "/home/admin/Downloads/SEN-2 LULC",
        split: str = "train",
        target_size: int = 128,
        transform=None,
        max_samples: int = None
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.target_size = target_size
        self.transform = transform or (get_training_transform(target_size) if split == "train" else get_val_transform(target_size))

        img_dir = osp.join(data_root, f"{split}_images", split)
        mask_dir = osp.join(data_root, f"{split}_masks", split)

        # Collect sorted matching pairs
        all_imgs = sorted(glob.glob(osp.join(img_dir, "*.png")))
        self.samples = []
        for img_p in all_imgs:
            base_id = osp.splitext(osp.basename(img_p))[0]
            mask_p = osp.join(mask_dir, f"{base_id}.tif")
            if osp.exists(mask_p):
                self.samples.append((img_p, mask_p))

        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        print(f"[SEN-2 LULC] Loaded {len(self.samples)} valid (image, mask) pairs for split '{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load RGB image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask (values 1 to 7 -> shift to 0 to 6)
        mask_raw = Image.open(mask_path)
        mask = np.array(mask_raw, dtype=np.int64)
        mask = np.clip(mask - 1, 0, 6)

        # Dynamic boundary extraction
        boundary = mask_to_boundary(mask, width=3)

        # Apply augmentation
        augmented = self.transform(image=img, mask=mask, boundary=boundary)
        img_t = augmented['image']
        mask_t = augmented['mask'].long()
        boundary_t = augmented['boundary'].float()

        return {
            'img': img_t,
            'gt_semantic_seg': mask_t,
            'gt_boundary': boundary_t
        }
