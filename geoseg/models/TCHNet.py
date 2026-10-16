"""
Transformer and CNN Hybrid Deep Neural Network (TCHNet)
=======================================================
Reference:
  Zhang et al., "Transformer and CNN Hybrid Deep Neural Network for Semantic
  Segmentation of Very-High-Resolution Remote Sensing Imagery", IEEE TGRS, 2022.

Architecture Overview:
  1. Encoder: Swin Transformer (Swin-T) backbone extracting hierarchical multiscale
     features across 4 stages (strides 4, 8, 16, 32).
  2. Bottleneck: Separable Atrous Spatial Pyramid Pooling (SASPP) using depthwise
     separable convolutions with diverse dilation rates to capture multiscale context.
  3. Decoder: CNN-based U-shaped progressive decoder with skip connections from Swin stages.
  4. Attention: Squeeze-and-Excitation (SE) channel attention block before segmentation head.
  5. Auxiliary Boundary Branch: Dedicated edge detection head providing explicit boundary guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention Block.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 16), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 16), channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution with dilation for Atrous Spatial Pyramid Pooling.
    """
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=dilation, dilation=dilation, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class SASPP(nn.Module):
    """
    Spatial Atrous Spatial Pyramid Pooling (SASPP) based on depthwise separable convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int = 256, atrous_rates=(6, 12, 18)):
        super().__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b1 = DepthwiseSeparableConv(in_channels, out_channels, dilation=atrous_rates[0])
        self.b2 = DepthwiseSeparableConv(in_channels, out_channels, dilation=atrous_rates[1])
        self.b3 = DepthwiseSeparableConv(in_channels, out_channels, dilation=atrous_rates[2])
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        f0 = self.b0(x)
        f1 = self.b1(x)
        f2 = self.b2(x)
        f3 = self.b3(x)
        f4 = F.interpolate(self.b4(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.project(torch.cat([f0, f1, f2, f3, f4], dim=1))


class TCHNet(nn.Module):
    """
    Transformer and CNN Hybrid Deep Neural Network (Zhang et al., IEEE TGRS 2022).
    """
    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        aux_boundary: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_loaded = pretrained
        self.aux_boundary = aux_boundary

        # 1. Swin-T Encoder (ImageNet-1k pretrained)
        weights = tv_models.Swin_T_Weights.DEFAULT if pretrained else None
        swin = tv_models.swin_t(weights=weights)
        self.encoder = swin.features

        # 2. SASPP Bottleneck on Stage 4 (768 channels, stride 32)
        self.saspp = SASPP(in_channels=768, out_channels=256)

        # 3. CNN-based U-Shaped Decoder
        # Stage 3 skip: 384 ch. Input from SASPP: 256 ch
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Stage 2 skip: 192 ch. Input from dec3: 256 ch
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 192, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        # Stage 1 skip: 96 ch. Input from dec2: 192 ch
        self.dec1 = nn.Sequential(
            nn.Conv2d(192 + 96, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 4. Squeeze-and-Excitation (SE) Channel Attention
        self.se = SEBlock(channels=128, reduction=16)

        # 5. Auxiliary Boundary Detection Branch
        self.bnd_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        # 6. Main Semantic Segmentation Head
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        h, w = x.shape[2:]

        # Extract Swin Transformer hierarchical features
        feats = []
        out = x
        for i, layer in enumerate(self.encoder):
            out = layer(out)
            if i in [1, 3, 5, 7]:
                feats.append(out.permute(0, 3, 1, 2))
        c1, c2, c3, c4 = feats  # c1: 96, c2: 192, c3: 384, c4: 768

        # Bottleneck SASPP
        bottleneck = self.saspp(c4)  # [B, 256, H/32, W/32]

        # U-shaped Decoder with multi-scale skip connections
        d3 = F.interpolate(bottleneck, size=c3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, c3], dim=1))

        d2 = F.interpolate(d3, size=c2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, c2], dim=1))

        d1 = F.interpolate(d2, size=c1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, c1], dim=1))

        # SE Channel Attention
        d1 = self.se(d1)

        # Final Segmentation Output
        seg_out = self.seg_head(d1)
        seg_out = F.interpolate(seg_out, size=(h, w), mode="bilinear", align_corners=False)

        # Auxiliary Boundary Output (if enabled during training)
        if self.training and self.aux_boundary:
            bnd_out = self.bnd_head(d1)
            bnd_out = F.interpolate(bnd_out, size=(h, w), mode="bilinear", align_corners=False)
            return seg_out, bnd_out

        return seg_out
