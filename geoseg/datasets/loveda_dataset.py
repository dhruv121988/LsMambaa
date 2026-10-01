import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import matplotlib.patches as mpatches
from PIL import Image, ImageOps
import random
import numbers

from .transform import *


CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest',
           'agricultural')

PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
           [159, 129, 183], [0, 255, 0], [255, 195, 128]]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)


# =====================================================================
# Triple (img, mask, boundary) Custom Transforms
# =====================================================================

class ComposeTriple(object):
    """Applies a sequence of transforms jointly to (img, mask, boundary) triples."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, boundary):
        assert img.size == mask.size == boundary.size, (
            f"Size mismatch: img {img.size}, mask {mask.size}, boundary {boundary.size}"
        )
        for t in self.transforms:
            img, mask, boundary = t(img, mask, boundary)
        return img, mask, boundary


class RandomScaleTriple(object):
    """Multi-scale resize applied consistently to image, mask, and boundary."""

    def __init__(self, scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'):
        self.scale_list = scale_list
        self.mode = mode

    def __call__(self, img, mask, boundary):
        assert img.size == mask.size == boundary.size
        oh, ow = img.size
        scale_amt = 1.0
        if self.mode == 'value':
            scale_amt = float(np.random.choice(self.scale_list, 1)[0])
        elif self.mode == 'range':
            scale_amt = random.uniform(self.scale_list[0], self.scale_list[-1])

        h = int(scale_amt * oh)
        w = int(scale_amt * ow)
        return (
            img.resize((w, h), Image.BICUBIC),
            mask.resize((w, h), Image.NEAREST),
            boundary.resize((w, h), Image.NEAREST)
        )


class SmartCropV1Triple(object):
    """Crop applied identically to image, mask, and boundary map with rejection sampling."""

    def __init__(self, crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.max_ratio = max_ratio
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def _crop_triple(self, img, mask, boundary):
        assert img.size == mask.size == boundary.size
        w, h = img.size
        th, tw = self.crop_size
        if w == tw and h == th:
            return img, mask, boundary

        if self.nopad:
            if th > h or tw > w:
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            pad_h = (th - h) // 2 + 1 if th > h else 0
            pad_w = (tw - w) // 2 + 1 if tw > w else 0
            if pad_h or pad_w:
                border = (pad_w, pad_h, pad_w, pad_h)
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                boundary = ImageOps.expand(boundary, border=border, fill=0)
                w, h = img.size

        x1 = 0 if w == tw else random.randint(0, w - tw)
        y1 = 0 if h == th else random.randint(0, h - th)
        box = (x1, y1, x1 + tw, y1 + th)
        return img.crop(box), mask.crop(box), boundary.crop(box)

    def __call__(self, img, mask, boundary):
        assert img.size == mask.size == boundary.size
        count = 0
        while True:
            img_crop, mask_crop, bnd_crop = self._crop_triple(img.copy(), mask.copy(), boundary.copy())
            count += 1
            labels, cnt = np.unique(np.array(mask_crop), return_counts=True)
            cnt = cnt[labels != self.ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.max_ratio:
                break
            if count > 10:
                break

        return img_crop, mask_crop, bnd_crop


# =====================================================================
# Augmentation Pipelines (Joint Crop, Flip, Color Jitter)
# =====================================================================

def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform, additional_targets={'boundary': 'mask'})


def train_aug(img, mask, boundary=None):
    """Joint augmentation for image, semantic mask, and boundary map.

    Spatial transforms (random scale, smart crop, horizontal/vertical flip)
    are applied identically across all three targets.
    Color-jitter (brightness/contrast) and normalization are applied strictly
    to the image.
    Supports backwards compatibility when called with 2 arguments (img, mask).
    """
    if boundary is None:
        crop_aug = Compose([
            RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
            SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False)
        ])
        img, mask = crop_aug(img, mask)
        img, mask = np.array(img), np.array(mask)
        aug = get_training_transform()(image=img.copy(), mask=mask.copy())
        return aug['image'], aug['mask']

    crop_aug = ComposeTriple([
        RandomScaleTriple(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
        SmartCropV1Triple(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False)
    ])
    img, mask, boundary = crop_aug(img, mask, boundary)

    img, mask, boundary = np.array(img), np.array(mask), np.array(boundary)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy(), boundary=boundary.copy())
    return aug['image'], aug['mask'], aug['boundary']


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform, additional_targets={'boundary': 'mask'})


def val_aug(img, mask, boundary=None):
    """Validation transform supporting both (img, mask) and (img, mask, boundary)."""
    if boundary is None:
        img, mask = np.array(img), np.array(mask)
        aug = get_val_transform()(image=img.copy(), mask=mask.copy())
        return aug['image'], aug['mask']

    img, mask, boundary = np.array(img), np.array(mask), np.array(boundary)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy(), boundary=boundary.copy())
    return aug['image'], aug['mask'], aug['boundary']


train_aug_boundary = train_aug
val_aug_boundary = val_aug


# =====================================================================
# LoveDA Dataset Classes
# =====================================================================

class LoveDATrainDataset(Dataset):
    """LoveDA dataset extended with boundary ground-truth map support.

    Loads images, converted semantic segmentation masks (0-indexed, ignore=7/255),
    and boundary maps (generated by tools/generate_boundary_maps.py).
    Applies joint crop, flip, and color-jitter transforms across image,
    mask, and boundary map.

    __getitem__ returns:
        'img': (3, H, W) float tensor
        'gt_semantic_seg': (H, W) long tensor
        'gt_boundary': (H, W) float tensor (0.0 to 1.0)
        'img_id': str
        'img_type': str ('Urban' or 'Rural')
    """

    def __init__(self, data_root='data/LoveDA/Train', img_dir='images_png', mosaic_ratio=0.25,
                 mask_dir='masks_png_convert', boundary_dir='masks_png_boundary',
                 img_suffix='.png', mask_suffix='.png',
                 transform=train_aug, img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.boundary_dir = boundary_dir
        self.mosaic_ratio = mosaic_ratio

        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio < self.mosaic_ratio:
            img, mask, boundary = self.load_mosaic_img_mask_boundary(index)
        else:
            img, mask, boundary = self.load_img_mask_boundary(index)

        if self.transform:
            try:
                img, mask, boundary = self.transform(img, mask, boundary)
            except TypeError:
                # In case a 2-argument transform function was provided
                img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        boundary = torch.from_numpy(np.array(boundary)).float()
        if boundary.max() > 1.0:
            boundary = boundary / 255.0

        img_id, img_type = self.img_ids[index]
        results = {
            'img': img,
            'gt_semantic_seg': mask,
            'gt_boundary': boundary,
            'img_id': img_id,
            'img_type': img_type
        }
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        # Auto-detect nested structure (e.g. data/LoveDA/Train/Train/Urban)
        if not osp.exists(osp.join(data_root, 'Urban', img_dir)):
            sub_name = osp.basename(osp.normpath(data_root))
            nested = osp.join(data_root, sub_name)
            if osp.exists(osp.join(nested, 'Urban', img_dir)):
                self.data_root = nested
                data_root = nested

        # Auto-detect mask directory (fallback to masks_png if masks_png_convert does not exist)
        if not osp.exists(osp.join(data_root, 'Urban', mask_dir)) and osp.exists(osp.join(data_root, 'Urban', 'masks_png')):
            self.mask_dir = 'masks_png'
            mask_dir = 'masks_png'

        urban_img_dir = osp.join(data_root, 'Urban', img_dir)
        urban_mask_dir = osp.join(data_root, 'Urban', mask_dir)
        urban_img_ids = []
        if osp.exists(urban_img_dir) and osp.exists(urban_mask_dir):
            urban_img_filename_list = os.listdir(urban_img_dir)
            urban_mask_filename_list = os.listdir(urban_mask_dir)
            assert len(urban_img_filename_list) == len(urban_mask_filename_list), (
                f"Urban image/mask count mismatch: {len(urban_img_filename_list)} vs {len(urban_mask_filename_list)}"
            )
            urban_img_ids = [(str(id.split('.')[0]), 'Urban') for id in urban_img_filename_list]

        rural_img_dir = osp.join(data_root, 'Rural', img_dir)
        rural_mask_dir = osp.join(data_root, 'Rural', mask_dir)
        rural_img_ids = []
        if osp.exists(rural_img_dir) and osp.exists(rural_mask_dir):
            rural_img_filename_list = os.listdir(rural_img_dir)
            rural_mask_filename_list = os.listdir(rural_mask_dir)
            assert len(rural_img_filename_list) == len(rural_mask_filename_list), (
                f"Rural image/mask count mismatch: {len(rural_img_filename_list)} vs {len(rural_mask_filename_list)}"
            )
            rural_img_ids = [(str(id.split('.')[0]), 'Rural') for id in rural_img_filename_list]

        return urban_img_ids + rural_img_ids

    def load_img_and_mask(self, index):
        """Loads (img, mask) for backward compatibility."""
        img, mask, _ = self.load_img_mask_boundary(index)
        return img, mask

    def load_img_mask_boundary(self, index):
        """Loads image, segmentation mask, and boundary ground-truth map."""
        img_id, img_type = self.img_ids[index]
        img_name = osp.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, img_type, self.mask_dir, img_id + self.mask_suffix)
        boundary_name = osp.join(self.data_root, img_type, self.boundary_dir, img_id + self.mask_suffix)

        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        if osp.isfile(boundary_name):
            boundary = Image.open(boundary_name).convert('L')
        else:
            # Fallback to zero boundary map of matching spatial size if not yet generated
            boundary = Image.fromarray(np.zeros((mask.size[1], mask.size[0]), dtype=np.uint8))

        return img, mask, boundary

    def load_mosaic_img_and_mask(self, index):
        """Loads mosaic (img, mask) for backward compatibility."""
        img, mask, _ = self.load_mosaic_img_mask_boundary(index)
        return img, mask

    def load_mosaic_img_mask_boundary(self, index):
        """Mosaic augmentation for (image, mask, boundary) triples."""
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        data = [self.load_img_mask_boundary(i) for i in indexes]

        imgs = [np.array(d[0]) for d in data]
        masks = [np.array(d[1]) for d in data]
        bnds = [np.array(d[2]) for d in data]

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_sizes = [
            (offset_x, offset_y),
            (w - offset_x, offset_y),
            (offset_x, h - offset_y),
            (w - offset_x, h - offset_y),
        ]

        img_crops, mask_crops, bnd_crops = [], [], []
        for i in range(4):
            cw, ch = crop_sizes[i]
            rc = albu.Compose(
                [albu.RandomCrop(width=cw, height=ch)],
                additional_targets={'boundary': 'mask'}
            )
            cropped = rc(image=imgs[i].copy(), mask=masks[i].copy(), boundary=bnds[i].copy())
            img_crops.append(cropped['image'])
            mask_crops.append(cropped['mask'])
            bnd_crops.append(cropped['boundary'])

        top_img = np.concatenate((img_crops[0], img_crops[1]), axis=1)
        bottom_img = np.concatenate((img_crops[2], img_crops[3]), axis=1)
        img = np.ascontiguousarray(np.concatenate((top_img, bottom_img), axis=0))

        top_mask = np.concatenate((mask_crops[0], mask_crops[1]), axis=1)
        bottom_mask = np.concatenate((mask_crops[2], mask_crops[3]), axis=1)
        mask = np.ascontiguousarray(np.concatenate((top_mask, bottom_mask), axis=0))

        top_bnd = np.concatenate((bnd_crops[0], bnd_crops[1]), axis=1)
        bottom_bnd = np.concatenate((bnd_crops[2], bnd_crops[3]), axis=1)
        boundary = np.ascontiguousarray(np.concatenate((top_bnd, bottom_bnd), axis=0))

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        boundary = Image.fromarray(boundary)

        return img, mask, boundary


loveda_val_dataset = LoveDATrainDataset(data_root='data/LoveDA/Val', mosaic_ratio=0.0,
                                        transform=val_aug)


class LoveDABoundaryTrainDataset(LoveDATrainDataset):
    """Explicit alias for boundary-conditioned training."""
    pass


class LoveDABoundaryValDataset(LoveDATrainDataset):
    """Explicit boundary-conditioned validation dataset."""

    def __init__(self, data_root='data/LoveDA/Val', img_dir='images_png',
                 mask_dir='masks_png_convert', boundary_dir='masks_png_boundary',
                 img_suffix='.png', mask_suffix='.png',
                 transform=val_aug, img_size=ORIGIN_IMG_SIZE):
        super().__init__(
            data_root=data_root,
            img_dir=img_dir,
            mask_dir=mask_dir,
            boundary_dir=boundary_dir,
            mosaic_ratio=0.0,
            img_suffix=img_suffix,
            mask_suffix=mask_suffix,
            transform=transform,
            img_size=img_size
        )


class LoveDATestDataset(Dataset):
    def __init__(self, data_root='data/LoveDA/Test', img_dir='images_png',
                 img_suffix='.png',  mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir

        self.img_suffix = img_suffix
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir)

    def __getitem__(self, index):
        img = self.load_img(index)

        img = np.array(img)
        aug = albu.Normalize()(image=img)
        img = aug['image']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img_id, img_type = self.img_ids[index]

        results = {'img': img, 'img_id': img_id, 'img_type': img_type}

        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir):
        urban_img_dir = osp.join(data_root, 'Urban', img_dir)
        urban_img_ids = []
        if osp.exists(urban_img_dir):
            urban_img_filename_list = os.listdir(urban_img_dir)
            urban_img_ids = [(str(id.split('.')[0]), 'Urban') for id in urban_img_filename_list]

        rural_img_dir = osp.join(data_root, 'Rural', img_dir)
        rural_img_ids = []
        if osp.exists(rural_img_dir):
            rural_img_filename_list = os.listdir(rural_img_dir)
            rural_img_ids = [(str(id.split('.')[0]), 'Rural') for id in rural_img_filename_list]

        return urban_img_ids + rural_img_ids

    def load_img(self, index):
        img_id, img_type = self.img_ids[index]
        img_name = osp.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)
        img = Image.open(img_name).convert('RGB')

        return img


# =====================================================================
# Visualization Helpers
# =====================================================================

def show_img_mask_seg(seg_path, img_path, mask_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        mask = cv2.imread(f'{mask_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).convert('P')
        mask.putpalette(np.array(PALETTE, dtype=np.uint8))
        mask = np.array(mask.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE ' + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title('Mask True ' + seg_id)
        ax[i, 2].set_axis_off()
        ax[i, 2].imshow(img_seg)
        ax[i, 2].set_title('Mask Predict ' + seg_id)
        ax[i, 2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_seg(seg_path, img_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE '+img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(img_seg)
        ax[i, 1].set_title('Seg IMAGE '+seg_id)
        ax[i, 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_mask(img, mask, img_id):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert('RGB'))
    ax1.imshow(img)
    ax1.set_title('RS IMAGE ' + str(img_id)+'.png')
    ax2.imshow(mask)
    ax2.set_title('Mask ' + str(img_id)+'.png')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')
