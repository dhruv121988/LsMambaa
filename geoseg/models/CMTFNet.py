"""
CMTFNet: Convolution and Multiscale Transformer Fusion Network for Remote Sensing
================================================================================
Architecture:
- Backbone: ResNet-50 (extracts features at stages 1, 2, 3, 4: 256, 512, 1024, 2048 channels)
- Multiscale Transformer Fusion (MTF) blocks to model long-range context
- Hierarchical feature aggregation decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class MultiscaleTransformerFusionBlock(nn.Module):
    """
    Multiscale Transformer Fusion (MTF) block that projects CNN features into tokens,
    computes multi-head cross-scale attention, and projects back to spatial feature maps.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)
        norm_x = self.norm1(x_flat)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        out = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return out


class CMTFNet(nn.Module):
    """
    CMTFNet with ResNet-50 backbone.
    """
    def __init__(self, num_classes=7, pretrained=True, dropout=0.1):
        super().__init__()
        # Backbone: ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256, 1/4
        self.layer2 = resnet.layer2  # 512, 1/8
        self.layer3 = resnet.layer3  # 1024, 1/16
        self.layer4 = resnet.layer4  # 2048, 1/32

        # Project all channels to uniform dim
        unified_dim = 256
        self.proj4 = nn.Conv2d(2048, unified_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(1024, unified_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(512, unified_dim, kernel_size=1)
        self.proj1 = nn.Conv2d(256, unified_dim, kernel_size=1)

        # Transformer Fusion Blocks on deep stages
        self.tf_block4 = MultiscaleTransformerFusionBlock(unified_dim, num_heads=8, dropout=dropout)
        self.tf_block3 = MultiscaleTransformerFusionBlock(unified_dim, num_heads=8, dropout=dropout)

        # Decoder fusion layers
        self.fuse3 = ConvBNReLU(unified_dim * 2, unified_dim)
        self.fuse2 = ConvBNReLU(unified_dim * 2, unified_dim)
        self.fuse1 = ConvBNReLU(unified_dim * 2, unified_dim)

        self.head = nn.Sequential(
            ConvBNReLU(unified_dim, 128),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # ResNet-50 Encoder
        c1 = self.conv1(x)
        p1 = self.maxpool(c1)
        e1 = self.layer1(p1)  # (B, 256, H/4, W/4)
        e2 = self.layer2(e1)  # (B, 512, H/8, W/8)
        e3 = self.layer3(e2)  # (B, 1024, H/16, W/16)
        e4 = self.layer4(e3)  # (B, 2048, H/32, W/32)

        # Projections
        f4 = self.proj4(e4)
        f3 = self.proj3(e3)
        f2 = self.proj2(e2)
        f1 = self.proj1(e1)

        # Transformer contextualization
        f4 = self.tf_block4(f4)
        f3 = self.tf_block3(f3)

        # Decoder with progressive upsampling
        up4 = F.interpolate(f4, size=f3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.fuse3(torch.cat([up4, f3], dim=1))

        up3 = F.interpolate(d3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.fuse2(torch.cat([up3, f2], dim=1))

        up2 = F.interpolate(d2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.fuse1(torch.cat([up2, f1], dim=1))

        out = self.head(d1)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out
