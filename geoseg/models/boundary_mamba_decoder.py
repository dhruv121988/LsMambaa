"""
Progressive Boundary-Conditioned Mamba (SS2D) Decoder for GeoSeg
================================================================

A 4-level UNet-style decoder that fuses CNN encoder skip features (res4 -> res1)
using 2D Selective State Space Models (SS2D / Mamba) and progressive boundary gating.

In this swapped architecture:
- Encoder:  CNN Backbone (e.g., ResNet-34 / ResNet-50) capturing local texture
- Decoder:  Mamba SSM Backbone (SS2D) capturing global context across scale fusions

At each decoder level, a boundary attention map derived from the previous level's
boundary prediction gates the incoming CNN skip connection before fusion. The fused
representation is then processed through 2D Selective Scan blocks (VSSBlock) to provide
linear-complexity global context modeling during feature reconstruction.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoseg.models.vmamba_encoder import VSSBlock, LayerNorm2d


# =====================================================================
# Gate type enumeration
# =====================================================================

class GateType(Enum):
    """Boundary gate variants."""
    MULTIPLICATIVE = "multiplicative"
    RESIDUAL = "residual"
    BOUNDED_RESIDUAL = "bounded_residual"


# =====================================================================
# Convolutional Building Blocks
# =====================================================================

class ConvBNReLU(nn.Sequential):
    """Conv2d -> BatchNorm2d -> ReLU."""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, dilation: int = 1,
                 bias: bool = False):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, dilation=dilation, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


# =====================================================================
# Boundary Head: 128 -> 64 -> 1
# =====================================================================

class BoundaryHead(nn.Module):
    """
    Produces a single-channel boundary logit from decoder features.
    Architecture: Conv3x3-BN-ReLU (in_ch -> 64) -> Conv1x1 (64 -> 1)
    """
    def __init__(self, in_channels: int = 128, mid_channels: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            boundary logit: (B, 1, H, W) (raw logit)
        """
        return self.head(x)


# =====================================================================
# Boundary Gate
# =====================================================================

class BoundaryGate(nn.Module):
    """
    Gates a CNN skip connection using the deeper decoder level's raw boundary logit.

    Pipeline:
        1. boundary_logit -> 3x3 conv -> sigmoid -> gate_map in [0, 1]
        2. gate_map -> bilinear upsample to skip spatial resolution
        3. apply gate to skip feature according to gate_type
    """
    def __init__(
        self,
        gate_type: GateType = GateType.MULTIPLICATIVE,
        alpha: float = 1.0,
        beta: float = 0.5,
    ):
        super().__init__()
        self.gate_type = gate_type
        self.alpha = alpha
        self.beta = beta

        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, skip: torch.Tensor, boundary_logit: torch.Tensor) -> torch.Tensor:
        gate_map = torch.sigmoid(self.gate_conv(boundary_logit))

        if gate_map.shape[2:] != skip.shape[2:]:
            gate_map = F.interpolate(
                gate_map, size=skip.shape[2:],
                mode="bilinear", align_corners=False,
            )

        if self.gate_type == GateType.MULTIPLICATIVE:
            return skip * gate_map
        elif self.gate_type == GateType.RESIDUAL:
            return skip + self.alpha * (skip * gate_map)
        elif self.gate_type == GateType.BOUNDED_RESIDUAL:
            return skip * (self.beta + self.alpha * gate_map)
        else:
            raise ValueError(f"Unknown gate type: {self.gate_type}")


# =====================================================================
# Mamba Decoder Level (Powered by 2D Selective State Space SS2D)
# =====================================================================

class _MambaDecoderLevel(nn.Module):
    """
    One level of the progressive boundary-conditioned Mamba decoder.

    Unlike the CNN decoder which uses standard 3x3 convolutional layers,
    this level performs global sequence context modeling via 2D Selective
    Scan blocks (VSSBlock).

    If is_deepest=True (Level 4):
        - Skip (res4) is reduced to decoder_channels
        - Refined through VSSBlock(s)
    Otherwise (Levels 3, 2, 1):
        - Skip is gated and projected to decoder_channels
        - Previous level's features are upsampled and concatenated
        - Fused and refined through VSSBlock(s)
    """

    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        num_mamba_blocks: int = 1,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: type = nn.SiLU,
        drop_path: float = 0.1,
        use_checkpoint: bool = False,
        is_deepest: bool = False,
    ):
        super().__init__()
        self.is_deepest = is_deepest

        # Skip projection
        self.skip_reduce = ConvBNReLU(skip_channels, decoder_channels, kernel_size=1)

        if not is_deepest:
            # 1x1 projection to fuse upsampled previous features + skip
            self.fusion_proj = ConvBNReLU(decoder_channels * 2, decoder_channels, kernel_size=1)

        # Mamba 2D Selective Scan blocks (SS2D) for decoder feature modeling
        mamba_blocks = []
        for _ in range(num_mamba_blocks):
            mamba_blocks.append(
                VSSBlock(
                    hidden_dim=decoder_channels,
                    drop_path=drop_path,
                    norm_layer=LayerNorm2d,
                    channel_first=True,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=3,
                    ssm_conv_bias=True,
                    forward_type="v0",
                    mlp_ratio=0.0,
                    use_checkpoint=use_checkpoint,
                )
            )
        self.mamba_blocks = nn.Sequential(*mamba_blocks)

    def forward(
        self,
        skip: torch.Tensor,
        prev_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            skip: (B, skip_ch, H, W) - gated CNN skip feature
            prev_features: (B, dec_ch, H', W') - features from deeper level
        Returns:
            (B, decoder_channels, H, W)
        """
        skip = self.skip_reduce(skip)

        if self.is_deepest:
            x = skip
        else:
            prev_up = F.interpolate(
                prev_features, size=skip.shape[2:],
                mode="bilinear", align_corners=False,
            )
            x = self.fusion_proj(torch.cat([prev_up, skip], dim=1))

        # Pass through 2D Selective State Space blocks
        x = self.mamba_blocks(x)
        return x


# =====================================================================
# Full Progressive Boundary-Conditioned Mamba Decoder
# =====================================================================

class BoundaryMambaDecoder(nn.Module):
    """
    Progressive Boundary-Conditioned Decoder powered by 2D Selective
    State Space Models (SS2D / Mamba).

    Parameters
    ----------
    encoder_channels : tuple of int
        Channel widths of the 4 CNN encoder skip features [res1, res2, res3, res4].
        E.g. (64, 128, 256, 512) for ResNet-34, or (256, 512, 1024, 2048) for ResNet-50.
    decoder_channels : int
        Internal channel width of decoder stages (default 128).
    num_classes : int
        Number of semantic segmentation classes.
    num_mamba_blocks : int
        Number of VSSBlock layers per decoder stage (default 1).
    gate_type : str or GateType
        'multiplicative', 'residual', or 'bounded_residual'.
    alpha : float
        Scaling factor for residual gates.
    beta : float
        Offset for bounded-residual gate.
    dropout : float
        Dropout before the final segmentation conv.
    use_boundary_heads : bool
        Whether to generate multi-scale boundary predictions.
    gated_levels : tuple of int
        Decoder levels with boundary gating (default (3, 2, 1)).
    """

    def __init__(
        self,
        encoder_channels: Tuple[int, ...] = (64, 128, 256, 512),
        decoder_channels: int = 128,
        num_classes: int = 7,
        num_mamba_blocks: int = 1,
        gate_type: str = "multiplicative",
        alpha: float = 1.0,
        beta: float = 0.5,
        dropout: float = 0.1,
        use_boundary_heads: bool = True,
        gated_levels: Tuple[int, ...] = (3, 2, 1),
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_boundary_heads = bool(use_boundary_heads)
        self.gated_levels = tuple(int(lvl) for lvl in gated_levels) if gated_levels else ()

        if self.gated_levels:
            if isinstance(gate_type, str):
                gate_type = GateType(gate_type.lower())
            self.gate_type_enum = gate_type
        else:
            self.gate_type_enum = None

        c1, c2, c3, c4 = encoder_channels

        # ---- Mamba Decoder Levels (deepest to shallowest) ----
        self.level4 = _MambaDecoderLevel(
            c4, decoder_channels, num_mamba_blocks=num_mamba_blocks,
            use_checkpoint=use_checkpoint, is_deepest=True,
        )
        self.level3 = _MambaDecoderLevel(
            c3, decoder_channels, num_mamba_blocks=num_mamba_blocks,
            use_checkpoint=use_checkpoint, is_deepest=False,
        )
        self.level2 = _MambaDecoderLevel(
            c2, decoder_channels, num_mamba_blocks=num_mamba_blocks,
            use_checkpoint=use_checkpoint, is_deepest=False,
        )
        self.level1 = _MambaDecoderLevel(
            c1, decoder_channels, num_mamba_blocks=num_mamba_blocks,
            use_checkpoint=use_checkpoint, is_deepest=False,
        )

        # ---- Boundary Gates ----
        self.gate3 = (
            BoundaryGate(gate_type, alpha=alpha, beta=beta)
            if 3 in self.gated_levels else None
        )
        self.gate2 = (
            BoundaryGate(gate_type, alpha=alpha, beta=beta)
            if 2 in self.gated_levels else None
        )
        self.gate1 = (
            BoundaryGate(gate_type, alpha=alpha, beta=beta)
            if 1 in self.gated_levels else None
        )

        # ---- Boundary Heads (levels 4, 3, 2 + final) ----
        if self.use_boundary_heads:
            self.boundary_head4 = BoundaryHead(decoder_channels, mid_channels=64)
            self.boundary_head3 = BoundaryHead(decoder_channels, mid_channels=64)
            self.boundary_head2 = BoundaryHead(decoder_channels, mid_channels=64)
            self.final_boundary_head = BoundaryHead(decoder_channels, mid_channels=64)
        else:
            self.boundary_head4 = None
            self.boundary_head3 = None
            self.boundary_head2 = None
            self.final_boundary_head = None

        # ---- Segmentation Heads ----
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3),
            nn.Dropout2d(p=dropout, inplace=False),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

        self.aux_segmentation_head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3),
            nn.Dropout2d(p=dropout, inplace=False),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(
        self,
        res1: torch.Tensor,
        res2: torch.Tensor,
        res3: torch.Tensor,
        res4: torch.Tensor,
        h: int,
        w: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            res1: (B, C1, H/4,  W/4)   CNN skip 1
            res2: (B, C2, H/8,  W/8)   CNN skip 2
            res3: (B, C3, H/16, W/16)  CNN skip 3
            res4: (B, C4, H/32, W/32)  CNN skip 4
            h, w: original input spatial size
        """
        # Level 4: deepest bottleneck
        feat4 = self.level4(res4)
        boundary4 = self.boundary_head4(feat4) if self.boundary_head4 is not None else None

        # Level 3: gated by boundary4
        if 3 in self.gated_levels and self.gate3 is not None and boundary4 is not None:
            gated_res3 = self.gate3(res3, boundary4)
        else:
            gated_res3 = res3
        feat3 = self.level3(gated_res3, prev_features=feat4)
        boundary3 = self.boundary_head3(feat3) if self.boundary_head3 is not None else None

        # Level 2: gated by boundary3
        if 2 in self.gated_levels and self.gate2 is not None and boundary3 is not None:
            gated_res2 = self.gate2(res2, boundary3)
        else:
            gated_res2 = res2
        feat2 = self.level2(gated_res2, prev_features=feat3)
        boundary2 = self.boundary_head2(feat2) if self.boundary_head2 is not None else None

        # Level 1: gated by boundary2
        if 1 in self.gated_levels and self.gate1 is not None and boundary2 is not None:
            gated_res1 = self.gate1(res1, boundary2)
        else:
            gated_res1 = res1
        feat1 = self.level1(gated_res1, prev_features=feat2)

        # Final predictions
        seg_logits = self.segmentation_head(feat1)
        if seg_logits.shape[2:] != (h, w):
            seg_logits = F.interpolate(seg_logits, size=(h, w), mode="bilinear", align_corners=False)

        if self.use_boundary_heads:
            final_boundary = self.final_boundary_head(feat1)
            if final_boundary.shape[2:] != (h, w):
                final_boundary = F.interpolate(final_boundary, size=(h, w), mode="bilinear", align_corners=False)
            seg_aux = self.aux_segmentation_head(feat3)
            if seg_aux.shape[2:] != (h, w):
                seg_aux = F.interpolate(seg_aux, size=(h, w), mode="bilinear", align_corners=False)
        else:
            final_boundary = None
            seg_aux = None

        return {
            "seg_logits": seg_logits,
            "final_boundary": final_boundary,
            "boundary4": boundary4,
            "boundary3": boundary3,
            "boundary2": boundary2,
            "seg_aux": seg_aux,
            "_feat4": feat4,
            "_feat3": feat3,
            "_feat2": feat2,
            "_feat1": feat1,
        }
