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

| Model Architecture | Model Family | Backbone Pretraining | Val mIoU | Val F1 | Val OA | Boundary IoU (mBIoU) | Boundary F1 (mBF1) | Boundary Precision | Boundary Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BoundaryVMambaUNet (Proposed)** | Boundary-Gated SSM | ImageNet-1k | **62.19%** | **75.20%** | **76.93%** | **10.25%** | **29.46%** | 14.46% | 23.68% |
| **Plain VMamba U-Net** | Baseline SSM | ImageNet-1k | 59.71% | 73.12% | 74.92% | 9.42% | 25.56% | 9.91% | 23.93% |
| **SETR-Naive** | Pure Vision Transformer | ImageNet-1k | 58.70% | 71.87% | 75.58% | 9.85% | 26.51% | 17.20% | 18.01% |
| **BoundaryCNNMambaUNet** | CNN + Boundary-Mamba | ImageNet-1k | 57.79% | 71.46% | 68.53% | 8.01% | 23.97% | 11.93% | 25.32% |
| **FCN-8s** | Classical Pure CNN | ImageNet-1k | 55.19% | 69.42% | 68.83% | 9.21% | 25.24% | 10.26% | 26.01% |

> **Key 16-Epoch Finding**: Under the exact same 16-epoch training budget, BoundaryVMambaUNet (**62.19% mIoU, 10.25% mBIoU**) achieves the highest performance on LoveDA. It delivers a +2.48% mIoU / +0.83% mBIoU improvement over Plain VMamba U-Net (59.71% mIoU, 9.42% mBIoU), confirming that progressive boundary gating enhances both region and edge features. It significantly outperforms pure Transformer baselines like SETR-Naive (+3.49% mIoU, +0.40% mBIoU), hybrid models like BoundaryCNNMambaUNet (+4.40% mIoU, +2.24% mBIoU), and classical CNN baselines like FCN-8s (+7.00% mIoU, +1.04% mBIoU), while maintaining linear complexity through 2D selective state-space scanning.

---

### 2. SEN-2 LULC Benchmark (Sentinel-2 10m Indian Satellite Dataset)
*Evaluated on 2,000 validation tiles across 7 Indian land-cover classes (Water, Dense Forest, Sparse Forest, Barren Land, Built-up, Agriculture Land, Fallow Land) under 16-epoch training. Boundary tolerance $\theta = 2$ px (CVPR 2021 protocol).*

| Model Architecture | Model Family | Backbone Pretraining | Val mIoU | Val F1 | Val OA | Boundary IoU (mBIoU) | Boundary F1 (mBF1) | Boundary Precision | Boundary Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BoundaryCNNMambaUNet** | CNN + Boundary-Mamba | ResNet-34 (Pretrained) | **37.25%** | **51.70%** | **70.77%** | **20.56%** | **65.17%** | **97.57%** | **80.45%** |
| **SETR-Naive** | Pure Vision Transformer | ViT-B (Pretrained) | 29.67% | 43.27% | 62.81% | 7.12% | 31.18% | 94.43% | 39.80% |
| **BoundaryVMambaUNet (Proposed)** | Boundary-Gated SSM | VMamba-Tiny (Random Init) | 29.16% | 41.05% | 66.31% | 14.53% | 50.48% | 95.78% | 72.70% |
| **FCN-8s** | Classical Pure CNN | VGG-16 (Pretrained) | 28.32% | 40.89% | 63.32% | 10.65% | 42.14% | 95.01% | 58.89% |
| **Plain VMamba U-Net** | Baseline SSM | VMamba-Tiny (Random Init) | 28.25% | 40.67% | 63.15% | 17.60% | 57.87% | 96.13% | 79.67% |

> **Cross-Architecture Insights on Spaceborne Imagery**:
> - **Pretrained Inductive Bias**: `BoundaryCNNMambaUNet` achieves the highest overall accuracy (**37.25% mIoU, 20.56% BIoU**) because pretrained ResNet-34 features provide strong multi-scale feature representations for 10m Sentinel-2 pixels, while the Boundary-Gated Mamba decoder sharpens regional transitions.
> - **SSM vs. Vision Transformer**: Pure Vision Transformers (`SETR-Naive`) collapse in boundary resolution down to **7.12% BIoU** (31.18% BF1) due to $16 \times 16$ patch smoothing. In contrast, 2D State-Space scanning (`Plain VMamba` at 17.60% BIoU and `BoundaryVMambaUNet` at 14.53% BIoU) retains sharp geometric delineation with over **95.7%+ boundary precision**.

---

### 3. Model Efficiency & Computational Complexity Benchmark
*Benchmarked on NVIDIA RTX 2000 Ada (16GB VRAM) under PyTorch 2.6 + CUDA 12.4 with `torch.autocast(fp16)`. GFLOPs measured at $512 \times 512$ and $1024 \times 1024$ resolutions with Triton SS2D analytical correction ($36 \times L \times D \times N$).*

| Model Architecture | Parameters | GFLOPs ($512^2$) | GFLOPs ($1024^2$) | Throughput (FPS, $512^2$ B1) | Throughput (FPS, $512^2$ B8) | Throughput (FPS, $1024^2$ B1) | Peak VRAM ($512^2$ B8) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plain VMamba U-Net** | 24.23M | **4.98G** | **19.93G** | 5.3 | 5.3 | 1.3 | 7,556 MB |
| **BoundaryVMambaUNet (Proposed)** | 24.53M | **4.98G** | **19.93G** | 5.6 | 5.4 | 1.3 | 7,460 MB |
| **SETR-Naive** | 88.56M | 114.20G | 456.80G | 62.4 | 43.1 | 12.8 | 3,512 MB |
| **FCN-8s** | 134.31M | 158.40G | 633.60G | 78.5 | 51.2 | 14.1 | 5,058 MB |

> 📌 **Efficiency Highlights**:
> - **Theoretical Complexity**: VMamba's 2D selective state-space model achieves the lowest theoretical FLOPs (**4.98 GFLOPs** at $512^2$, a **$23\times$ to $31\times$ reduction** compared to heavy Transformer and CNN alternatives like SETR-Naive at 114.2 GFLOPs and FCN-8s at 158.4 GFLOPs).
> - **Compact Parameter Footprint**: BoundaryVMambaUNet requires only **24.53M parameters**, representing a **$3.6\times$ reduction** versus SETR-Naive (88.56M) and **$5.5\times$ reduction** versus FCN-8s (134.31M).

---

### 4. Segmentation Accuracy vs. Distance to True Boundary

To quantify how models handle high-frequency boundary contours versus homogeneous interior regions, pixel accuracy was computed across Euclidean distance intervals:

```
Distance Range from Boundary    FCN-8s (VGG-16)    SETR-Naive (ViT-B)    BoundaryVMambaUNet (Proposed)
------------------------------------------------------------------------------------------------------
0 – 1 px (Exact Edge Line)          46.44%               47.41%                     47.04%
2 – 4 px (Near Boundary)            50.90%               52.23%                     53.92%
5 – 8 px (Transition Band)          57.73%               59.81%                     64.10%
9 – 16 px                           65.74%               68.34%                     74.20%
17 – 32 px                          72.87%               75.18%                     81.50%
>33 px (Interior Core)              74.56%               85.64%                     89.77%
```


---

## 🏗️ Repository Architecture

```text
lssmamba/
├── config/                                # Reproducible experiment configurations
│   ├── loveda/                            # LoveDA benchmark configs
│   │   ├── boundary_vmamba_unet.py        # Proposed Boundary-Gated VMamba U-Net
│   │   ├── baseline_plain_vmamba_unet.py  # Plain VMamba U-Net baseline
│   │   ├── setr_naive.py                  # SETR-Naive baseline (Pure ViT)
│   │   ├── boundary_cnn_mamba_unet.py     # Hybrid CNN + Mamba baseline
│   │   └── fcn8s.py                       # FCN-8s baseline (Pure CNN)
│   └── sen2_lulc/                         # SEN-2 LULC Indian satellite configs
│       ├── boundary_vmamba_unet.py        # Proposed Boundary-Gated VMamba U-Net
│       ├── baseline_plain_vmamba_unet.py  # Plain VMamba U-Net baseline
│       ├── setr_naive.py                  # SETR-Naive baseline (Pure ViT)
│       ├── boundary_cnn_mamba_unet.py     # Hybrid CNN + Mamba baseline
│       └── fcn8s.py                       # FCN-8s baseline (Pure CNN)
├── geoseg/
│   ├── datasets/                          # Dataset loaders & boundary generators
│   │   ├── loveda_dataset.py              # LoveDA loader with online edge maps
│   │   └── sen2_lulc_dataset.py           # SEN-2 LULC loader with dynamic contours
│   ├── models/                            # Neural network architectures
│   │   ├── BoundaryVMambaUNet.py          # Proposed model implementation
│   │   ├── vmamba_encoder.py              # 2D Selective State Space backbone
│   │   ├── csm_triton.py                  # Triton accelerated cross-scan kernels
│   │   ├── SETRNaive.py                   # Pure Vision Transformer (SETR-Naive)
│   │   ├── FCN8s.py                       # Fully Convolutional Network (FCN-8s)
│   │   └── BoundaryCNNMambaUNet.py        # CNN-Mamba hybrid architecture
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

#### Train Baseline Plain VMamba U-Net on LoveDA:
```bash
python train_supervision.py \
  -c config/loveda/baseline_plain_vmamba_unet.py \
  --batch_size 4 \
  --val_batch_size 4 \
  --num_workers 8 \
  --epochs 16 \
  --check_val_every_n_epoch 1
```

#### Train Proposed BoundaryVMambaUNet on SEN-2 LULC (Indian Dataset):
```bash
python train_supervision.py \
  -c config/sen2_lulc/boundary_vmamba_unet.py \
  --batch_size 64 \
  --val_batch_size 64 \
  --num_workers 8 \
  --epochs 16 \
  --check_val_every_n_epoch 1 \
  --max_train_samples 8000 \
  --max_val_samples 2000
```

---

### 4. Boundary Metrics Evaluation (CVPR 2021 Protocol)

Evaluate mean Boundary IoU ($\text{mBIoU}$) and Boundary F1 ($\text{mBF}_1$) on test checkpoints:

```bash
# Evaluate LoveDA checkpoint (High-throughput async evaluation)
python tools/eval_boundary_loveda.py \
  -c config/loveda/boundary_vmamba_unet.py \
  --ckpt model_weights/loveda/boundary_vmamba_unet-multiplicative-aux0.4-bnd0.3-epoch16/last.ckpt \
  --batch_size 2 \
  --num_workers 8 \
  --amp

# Evaluate SEN-2 LULC checkpoint
python tools/eval_boundary_sen2.py \
  -c config/sen2_lulc/boundary_vmamba_unet.py \
  --ckpt model_weights/sen2_lulc/boundary_vmamba_unet-sen2_lulc-epoch16/last.ckpt \
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
  author={Sharma, Dhruv},
  journal={arXiv preprint},
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
