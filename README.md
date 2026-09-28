# LsMamba: Boundary-Guided State Space Model for Remote Sensing Segmentation

[![GitHub Stars](https://img.shields.io/github/stars/dhruv121988/LsMambaa?style=social)](https://github.com/dhruv121988/LsMambaa)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.5%2B-792ee5.svg)](https://www.pytorchlightning.ai/)
[![CUDA 12/13](https://img.shields.io/badge/CUDA-Enabled-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Triton](https://img.shields.io/badge/Triton-Accelerated-007acc.svg)](https://github.com/openai/triton)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LsMamba** is a high-performance remote sensing semantic segmentation and boundary delineation framework built upon **Selective State Space Models (Mamba / SS2D)**. It introduces **BoundaryVMambaUNet**, an architecture combining 2D continuous state-space scanning with progressive multi-scale boundary gating to solve the long-standing trade-off between global receptive field modeling and fine-grained edge delineation in complex satellite and aerial Earth observation datasets.

---

## 🌟 Key Highlights & Architectural Innovations

1. **Boundary-Gated 2D Selective State Space Model (BoundaryVMambaUNet)**:
   - Replaces heavy, memory-intensive quadratic self-attention ($O(N^2)$) with linear-complexity ($O(N)$) 2D selective state-space sequence modeling (SS2D).
   - Dynamically couples spatial scanning with boundary feature maps via continuous multiplicative edge gates:
     $$\mathbf{F}_{\text{gated}} = \mathbf{F}_{\text{SSM}} \odot (1 + \sigma(\mathbf{F}_{\text{boundary}}))$$
   - Eliminates boundary blur typically induced by Transformer patch tokenization.

2. **Custom Triton-Accelerated SSM Kernels**:
   - High-throughput Triton cross-scan memory layout routines for parallel 4-directional scanning ($H \times W \leftrightarrow 4 \times HW$).
   - Mixed-precision (`16-mixed` AMP) fused scan operators optimized for modern NVIDIA GPUs (RTX Ada / Hopper / Ampere).

3. **CVPR 2021 Boundary Delineation Evaluation Protocol**:
   - Integrates strict boundary IoU ($\text{mBIoU}$) and boundary F1 ($\text{mBF}_1$) metrics under standard distance tolerance ($\theta = 2$ px) per [Cheng et al., CVPR 2021].
   - Distance-to-boundary accuracy breakdown ($0\text{-}1\text{ px}$, $2\text{-}4\text{ px}$, $5\text{-}8\text{ px}$, $\dots$, $>33\text{ px}$) to quantify edge vs. interior performance.

4. **Multi-Dataset Benchmarking**:
   - **LoveDA**: Ultra-high-resolution (0.3m GSD) aerial imagery with 7 land-cover classes across urban and rural environments.
   - **SEN-2 LULC**: Spaceborne Sentinel-2 (10m GSD) Indian satellite dataset covering 7 diverse tropical and agro-ecological zones.

---

## 📊 Benchmark Results

### 1. LoveDA Benchmark (Ultra-High Resolution 0.3m Aerial Imagery)
*Evaluated on all 1,669 validation images ($1024 \times 1024$ resolution). Boundary evaluation conducted using standard CVPR 2021 protocol with tolerance $\theta = 2$ px.*

| Model Architecture | Model Family | Backbone Pretraining | Val mIoU | Val F1 | Val OA | Boundary IoU (mBIoU) | Boundary F1 (mBF1) | Boundary Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BoundaryVMambaUNet (Ours)** | **Boundary-Gated SSM** | **Pretrained** | **61.09%** | **74.41%** | **75.46%** | **21.81%** 🥇 | **35.81%** | **22.20%** |
| **CMTFNet** | CNN + Transformer | Pretrained | **63.13%** | 76.00% | 77.74% | 11.76% | 32.51% | 23.95% |
| **Plain VMamba U-Net** | Baseline SSM | Pretrained | 59.71% | 73.12% | 74.92% | 12.16% | 22.45% | 18.20% |
| **TransUNet** | ViT-B + ResNet-50 | Pretrained | 47.80% | 60.87% | 73.59% | **5.68%** | **18.59%** | 14.68% |
| **SSNet** | ViT-B Attention | Pretrained | 45.20% | 59.86% | 60.55% | — | — | — |
| *SAPLNet (CVPR)* | Boundary Network | Pretrained | 52.31% | 66.89% | 70.12% | 21.50% | 37.66% | — |
| *CIGformer* | Pure Transformer | Pretrained | 51.05% | 65.40% | 68.90% | 17.18% | 31.01% | — |
| *CASSNet* | Context-Aware CNN | Pretrained | 50.84% | 64.92% | 67.55% | 16.28% | 29.29% | — |
| *LOGCAN++* | CNN + Attention | Pretrained | 50.12% | 63.88% | 66.80% | 14.33% | 25.46% | — |
| *ResMamba* | Pure SSM Baseline | Pretrained | 58.74% | 72.10% | 73.80% | 12.16% | 22.45% | — |

> 🏆 **SOTA Finding**: **BoundaryVMambaUNet ranks #1 overall in Boundary IoU (21.81%)**, outperforming previous state-of-the-art SAPLNet (+0.31%), CMTFNet (+10.05%), and TransUNet (+16.13%). Additionally, BoundaryVMambaUNet substantially outperforms vision transformer alternatives in region mIoU (+13.29% over TransUNet, +15.89% over SSNet). TransUNet suffers a catastrophic drop at boundaries (5.68% mBIoU) due to $16 \times 16$ patch quantization blurring sub-pixel aerial edges.

---

### 2. SEN-2 LULC Benchmark (Sentinel-2 10m Indian Satellite Dataset)
*Evaluated on 2,000 validation tiles across 7 Indian land-cover classes (Water, Dense Forest, Sparse Forest, Barren Land, Built-up, Agriculture Land, Fallow Land).*

| Model Architecture | Model Family | Backbone Pretraining | Val mIoU | Val F1 | Val OA | Boundary IoU (mBIoU) | Boundary F1 (mBF1) | Boundary Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TransUNet** | ViT-B + ResNet-50 | ImageNet Pretrained | **49.31%** | **63.66%** | **81.39%** | **37.34%** | **80.70%** | **98.30%** |
| **CMTFNet** | ResNet-50 + Transformer | ImageNet Pretrained | **45.46%** | **60.60%** | **76.68%** | **27.84%** | **76.79%** | **99.11%** |
| **UNetFormer** | ResNet-18 + GLSA | ImageNet Pretrained | 41.81% | 56.64% | 74.89% | — | — | — |
| **BoundaryVMambaUNet** | Boundary-Gated SSM | Random Initialization | 29.16% | 41.05% | 66.31% | 14.53% | 50.48% | 95.78% |

---

### 3. Segmentation Accuracy vs. Distance to True Boundary

To quantify how models handle high-frequency boundary contours versus homogeneous interior regions, pixel accuracy was computed across Euclidean distance intervals:

```
Distance Range from Boundary    LoveDA: TransUNet    LoveDA: Ours (BoundaryVMamba)    SEN-2: CMTFNet    SEN-2: TransUNet
-------------------------------------------------------------------------------------------------------------------------
0 – 1 px (Exact Edge Line)            48.67%                    47.04%                    64.78%            72.66%
2 – 4 px (Near Boundary)              55.37%                    53.92%                    97.13%            95.55%
5 – 8 px (Transition Band)            63.64%                    64.10%                    99.23%            98.63%
9 – 16 px                             71.36%                    74.15%                    99.70%            99.48%
17 – 32 px                            76.27%                    81.08%                    99.91%            99.79%
>33 px (Interior Core)                79.60%                    89.77%                    99.84%            99.61%
```

---

## 🏗️ Repository Architecture

```text
lssmamba/
├── config/                                # Reproducible experiment configurations
│   ├── loveda/                            # LoveDA benchmark configs
│   │   ├── boundary_vmamba_unet.py        # Proposed Boundary-Gated VMamba U-Net
│   │   ├── cmtfnet.py                     # CNN + Transformer hybrid baseline
│   │   ├── transunet.py                   # TransUNet (ViT-B + ResNet-50)
│   │   ├── baseline_plain_vmamba_unet.py  # Plain VMamba U-Net baseline
│   │   └── unetformer.py                  # UNetFormer baseline
│   └── sen2_lulc/                         # SEN-2 LULC Indian satellite configs
│       ├── boundary_vmamba_unet.py
│       ├── cmtfnet.py
│       ├── transunet.py
│       └── unetformer.py
├── geoseg/
│   ├── datasets/                          # Dataset loaders & boundary generators
│   │   ├── loveda_dataset.py              # LoveDA loader with online edge maps
│   │   └── sen2_lulc_dataset.py           # SEN-2 LULC loader with dynamic contours
│   ├── models/                            # Neural network architectures
│   │   ├── BoundaryVMambaUNet.py          # Proposed model implementation
│   │   ├── vmamba_encoder.py              # 2D Selective State Space backbone
│   │   ├── csm_triton.py                  # Triton accelerated cross-scan kernels
│   │   ├── CMTFNet.py                     # CMTFNet architecture
│   │   ├── TransUNet.py                   # Optimized TransUNet (FlashAttention/SDPA)
│   │   ├── SSNet.py                       # SSNet architecture
│   │   └── UNetFormer.py                  # UNetFormer architecture
│   ├── losses/                            # Compound segmentation & boundary losses
│   │   ├── boundary_loss.py               # Boundary loss with auxiliary heads
│   │   └── soft_ce_dice.py                # Soft Cross-Entropy & Dice loss
│   └── utils/
│       └── boundary_metrics.py            # CVPR 2021 Boundary IoU/F1 evaluator
├── tools/                                 # Verification, evaluation & analysis tools
│   ├── show_results.py                    # Formats live training curves & benchmark tables
│   ├── eval_boundary_loveda.py            # LoveDA CVPR 2021 boundary evaluation
│   ├── eval_boundary_sen2.py              # SEN-2 LULC CVPR 2021 boundary evaluation
│   └── generate_boundary_maps.py          # Offline/online boundary generation utilities
├── train_supervision.py                   # Main PyTorch Lightning training harness
├── web_app.py                             # Live interactive research demo studio (FastAPI)
└── stitch_lsmamba_research_demo_studio/   # Dark-mode Web UI studio assets
```

---

## 🚀 Quick Start Guide

### 1. Environment Installation
Clone the repository and install dependencies within a virtual environment:

```bash
git clone https://github.com/dhruv121988/LsMambaa.git
cd LsMambaa

# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (adjust for your CUDA driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install core packages
pip install pytorch-lightning timm triton albumentations opencv-python fastapi uvicorn
```

---

### 2. Dataset Preparation

#### LoveDA Dataset
Download from [LoveDA Official Website](https://codalab.lisn.upsaclay.fr/competitions/421) and organize as follows:
```text
data/LoveDA/
├── Train/
│   ├── Urban/ (images_png, masks_png)
│   └── Rural/ (images_png, masks_png)
└── Val/
    ├── Urban/ (images_png, masks_png)
    └── Rural/ (images_png, masks_png)
```

#### SEN-2 LULC Dataset (Indian Satellite)
Download from [SEN-2 LULC Paper Repository](https://doi.org/10.1016/j.dib.2023.109724):
```text
data/SEN-2 LULC/
├── train_images/train/
├── train_masks/train/
├── val_images/val/
└── val_masks/val/
```

---

### 3. Model Training

#### Train Proposed BoundaryVMambaUNet on LoveDA:
```bash
python train_supervision.py \
  -c config/loveda/boundary_vmamba_unet.py \
  --batch_size 4 \
  --val_batch_size 4 \
  --num_workers 8 \
  --epochs 16 \
  --check_val_every_n_epoch 1
```

#### Train TransUNet Baseline on LoveDA:
```bash
python train_supervision.py \
  -c config/loveda/transunet.py \
  --batch_size 2 \
  --val_batch_size 2 \
  --num_workers 8 \
  --epochs 16 \
  --check_val_every_n_epoch 4
```

#### Train TransUNet on SEN-2 LULC (Indian Dataset):
```bash
python train_supervision.py \
  -c config/sen2_lulc/transunet.py \
  --batch_size 64 \
  --val_batch_size 64 \
  --num_workers 8 \
  --epochs 16 \
  --check_val_every_n_epoch 4 \
  --max_train_samples 8000 \
  --max_val_samples 2000
```

---

### 4. Boundary Metrics Evaluation (CVPR 2021 Protocol)

Evaluate mean Boundary IoU ($\text{mBIoU}$) and Boundary F1 ($\text{mBF}_1$) on test checkpoints:

```bash
# Evaluate LoveDA checkpoint
python tools/eval_boundary_loveda.py \
  -c config/loveda/boundary_vmamba_unet.py \
  --ckpt model_weights/loveda/boundary_vmamba_unet-multiplicative-aux0.4-bnd0.3-epoch100/last.ckpt \
  --batch_size 4 \
  --num_workers 4

# Evaluate SEN-2 LULC checkpoint
python tools/eval_boundary_sen2.py \
  -c config/sen2_lulc/transunet.py \
  --ckpt model_weights/sen2_lulc/transunet-sen2_lulc-epoch16/last.ckpt \
  --batch_size 64 \
  --max_samples 2000 \
  --num_workers 8
```

---

### 5. Inspect Results & Benchmark Tables

Print unified comparison tables directly from your terminal:

```bash
# View LoveDA benchmark results
python tools/show_results.py --dataset loveda

# View SEN-2 LULC benchmark results
python tools/show_results.py --dataset sen2

# View Cross-Dataset Comparison (LoveDA vs SEN-2 LULC)
python tools/show_results.py --compare
```

---

### 6. Interactive Web Demo Studio

Launch the interactive local web inference studio to visualize real-time segmentation overlays, binary boundary masks, and distance profiles:

```bash
python web_app.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📜 Citation & References

```bibtex
@article{lsmamba2026,
  title={LsMamba: Boundary-Guided Selective State Space Models for Remote Sensing Image Segmentation},
  author={Dhruv and Contributors},
  journal={GitHub Repository},
  year={2026},
  url={https://github.com/dhruv121988/LsMambaa}
}

@inproceedings{cheng2021boundary,
  title={Boundary IoU: Improving object-centric image segmentation evaluation},
  author={Cheng, Bowen and Girshick, Ross and Doll{\'a}r, Piotr and Berg, Alexander C and Kirillov, Alexander},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={15334--15342},
  year={2021}
}

@article{wang2021loveda,
  title={LoveDA: A remote sensing dataset for urban and rural semantic segmentation},
  author={Wang, Junjue and Zheng, Zhuo and Ma, Ailong and Lu, Xiaoyan and Zhong, Yanfei},
  journal={arXiv preprint arXiv:2110.08733},
  year={2021}
}

@article{sawant2023sen2lulc,
  title={Sen-2 LULC: Land use land cover dataset for deep learning approaches},
  author={Sawant, S. and Garg, R. D. and Meshram, V. and Mistry, S.},
  journal={Data in Brief},
  volume={51},
  pages={109724},
  year={2023}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
