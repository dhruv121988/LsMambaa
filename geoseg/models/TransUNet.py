"""
TransUNet: Transformers Make Strong Encoders for Medical & Remote Sensing Segmentation
======================================================================================
Paper: Chen et al., 2021 (arXiv:2102.04306)
Architecture:
- Hybrid R50-ViT-B Encoder: ResNet-50 extracts spatial feature maps, Vision Transformer (ViT-B)
  encodes long-range global context with 12 layers and 768 embedding dimensions.
- Cascaded Upsampler (CUP) U-Net Decoder: Upsamples features and fuses skip connections
  from ResNet-50 stages to preserve fine spatial details.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ViTBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TransUNet(nn.Module):
    """
    TransUNet with R50-ViT-B backbone.
    """
    def __init__(self, num_classes=7, img_size=512, pretrained=True, num_vit_layers=8, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        vit_dim = 768

        # 1. ResNet-50 Convolutional Stem
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256, 1/4
        self.layer2 = resnet.layer2  # 512, 1/8
        self.layer3 = resnet.layer3  # 1024, 1/16

        # 2. Token Embedding to ViT-B (from 1/16 stage)
        self.embedding = nn.Conv2d(1024, vit_dim, kernel_size=1)

        # 3. Vision Transformer Layers (ViT-Base)
        self.transformer_layers = nn.ModuleList([
            ViTBlock(dim=vit_dim, num_heads=12, dropout=dropout)
            for _ in range(num_vit_layers)
        ])
        self.vit_norm = nn.LayerNorm(vit_dim)
        self.proj_back = ConvBNReLU(vit_dim, 512)

        # 4. Cascaded Upsampler (CUP) Decoder
        self.dec3 = DecoderBlock(in_channels=512, out_channels=256, skip_channels=512)   # fuses layer2 (512)
        self.dec2 = DecoderBlock(in_channels=256, out_channels=128, skip_channels=256)   # fuses layer1 (256)
        self.dec1 = DecoderBlock(in_channels=128, out_channels=64, skip_channels=64)     # fuses conv1 (64)

        # 5. Output Head
        self.head = nn.Sequential(
            ConvBNReLU(64, 32),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # ResNet-50 Stem (extract multi-scale skips)
        x_c1 = self.conv1(x)         # (B, 64, H/2, W/2)
        x_pool = self.pool(x_c1)
        x_l1 = self.layer1(x_pool)   # (B, 256, H/4, W/4)
        x_l2 = self.layer2(x_l1)     # (B, 512, H/8, W/8)
        x_l3 = self.layer3(x_l2)     # (B, 1024, H/16, W/16)

        # Tokenize and project to ViT-B
        B, C, H_tok, W_tok = x_l3.shape
        tokens = self.embedding(x_l3).flatten(2).transpose(1, 2)  # (B, N, 768)

        # ViT Transformer encoding
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.vit_norm(tokens)

        # Reshape back to 2D
        vit_feat = tokens.transpose(1, 2).reshape(B, 768, H_tok, W_tok)
        d_in = self.proj_back(vit_feat)  # (B, 512, H/16, W/16)

        # Cascaded Decoder with skip connections
        d3 = self.dec3(d_in, x_l2)       # (B, 256, H/8, W/8)
        d2 = self.dec2(d3, x_l1)         # (B, 128, H/4, W/4)
        d1 = self.dec1(d2, x_c1)         # (B, 64, H/2, W/2)

        out = self.head(d1)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out
