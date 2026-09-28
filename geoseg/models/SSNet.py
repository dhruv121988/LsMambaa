"""
SSNet: Novel Transformer and CNN Hybrid Network for Remote Sensing Semantic Segmentation
=======================================================================================
Architecture:
- Backbone: ViT-B (Vision Transformer Base with 768 hidden dimensions)
- Feature Fusion Module (FFM): Captures spatial and channel dependencies across stages
- Feature Injection Module (FIM): Condenses global transformer context and injects into spatial decoder
- Remote sensing segmentation head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module (FFM): Computes channel-spatial attention to blend multi-scale representations.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv2(self.conv1(x))
        c_att = self.channel_gate(feat)
        s_att = self.spatial_gate(feat)
        return feat * c_att * s_att + feat


class FeatureInjectionModule(nn.Module):
    """
    Feature Injection Module (FIM): Condenses global transformer features and injects them
    into local CNN decoder features.
    """
    def __init__(self, cnn_dim, vit_dim):
        super().__init__()
        self.proj_vit = nn.Sequential(
            nn.Conv2d(vit_dim, cnn_dim, kernel_size=1),
            nn.BatchNorm2d(cnn_dim),
            nn.ReLU(inplace=True)
        )
        self.fuse = ConvBNReLU(cnn_dim * 2, cnn_dim)

    def forward(self, cnn_feat, vit_feat):
        vit_proj = self.proj_vit(vit_feat)
        if vit_proj.shape[2:] != cnn_feat.shape[2:]:
            vit_proj = F.interpolate(vit_proj, size=cnn_feat.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([cnn_feat, vit_proj], dim=1))


class ViTBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SSNet(nn.Module):
    """
    SSNet with ViT-B encoder and FFM + FIM decoder.
    """
    def __init__(self, num_classes=7, patch_size=16, vit_dim=768, num_layers=8, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.vit_dim = vit_dim

        # 1. Patch Embedding (ViT-B)
        self.patch_embed = nn.Conv2d(3, vit_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Local Convolutional Branch (for shallow spatial details)
        self.stem1 = ConvBNReLU(3, 64, kernel_size=3, stride=2, padding=1)    # 1/2
        self.stem2 = ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1)  # 1/4
        self.stem3 = ConvBNReLU(128, 256, kernel_size=3, stride=2, padding=1) # 1/8

        # 3. Vision Transformer Layers (ViT-B)
        self.transformer_blocks = nn.ModuleList([
            ViTBlock(dim=vit_dim, num_heads=12, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.vit_norm = nn.LayerNorm(vit_dim)

        # 4. FIM (Feature Injection Modules) & FFM (Feature Fusion Modules)
        self.fim3 = FeatureInjectionModule(cnn_dim=256, vit_dim=vit_dim)
        self.proj_up3 = ConvBNReLU(256, 128)

        self.fim2 = FeatureInjectionModule(cnn_dim=128, vit_dim=vit_dim)
        self.proj_up2 = ConvBNReLU(128, 64)

        self.fim1 = FeatureInjectionModule(cnn_dim=64, vit_dim=vit_dim)

        self.ffm = FeatureFusionModule(in_channels=64, out_channels=128)

        # 5. Output Head
        self.head = nn.Sequential(
            ConvBNReLU(128, 64),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # Local branch
        s1 = self.stem1(x)    # (B, 64, H/2, W/2)
        s2 = self.stem2(s1)   # (B, 128, H/4, W/4)
        s3 = self.stem3(s2)   # (B, 256, H/8, W/8)

        # ViT-B Patch embedding
        p = self.patch_embed(x)  # (B, 768, H/16, W/16)
        B, C, H_p, W_p = p.shape
        tokens = p.flatten(2).transpose(1, 2)

        for blk in self.transformer_blocks:
            tokens = blk(tokens)
        tokens = self.vit_norm(tokens)
        vit_feat = tokens.transpose(1, 2).reshape(B, C, H_p, W_p)

        # Feature Injection & Upsampling
        inj3 = self.fim3(s3, vit_feat)  # (B, 256, H/8, W/8)
        up3 = self.proj_up3(F.interpolate(inj3, size=s2.shape[2:], mode="bilinear", align_corners=False))

        inj2 = self.fim2(s2 + up3, vit_feat)  # (B, 128, H/4, W/4)
        up2 = self.proj_up2(F.interpolate(inj2, size=s1.shape[2:], mode="bilinear", align_corners=False))

        inj1 = self.fim1(s1 + up2, vit_feat)  # (B, 64, H/2, W/2)
        fused = self.ffm(inj1)                # (B, 128, H/2, W/2)

        out = self.head(fused)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out
