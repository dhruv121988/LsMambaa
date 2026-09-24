"""
Progressive Boundary-Conditioned Decoder for GeoSeg
=====================================================

A 4-level UNet-style decoder that fuses encoder skip features (res4→res1)
with progressive boundary gating.  At each decoder level (except the
deepest), a boundary attention map derived from the previous level's
boundary prediction gates the incoming skip connection before fusion.

Architecture overview
---------------------

    Level 4 (deepest):  res4 → reduce → refine → decoder_feat4 + boundary4
    Level 3:            upsample(decoder_feat4) ⊕ Gate(res3, boundary4) → decoder_feat3 + boundary3
    Level 2:            upsample(decoder_feat3) ⊕ Gate(res2, boundary3) → decoder_feat2 + boundary2
    Level 1 (shallowest): upsample(decoder_feat2) ⊕ Gate(res1, boundary2) → decoder_feat1 → seg_logits + final_boundary

Gate types (Section III-E)
--------------------------

    multiplicative:      gated_skip = skip · σ(conv(bnd_logit))
    residual:            gated_skip = skip + α · (skip · σ(conv(bnd_logit)))
    bounded_residual:    gated_skip = skip · (β + α · σ(conv(bnd_logit)))

Returns
-------
    dict with keys:
        seg_logits      (B, num_classes, H, W)   – upsampled to input size
        final_boundary  (B, 1, H, W)             – upsampled to input size
        boundary4       (B, 1, H4, W4)           – at level-4 resolution (16×16)
        boundary3       (B, 1, H3, W3)           – at level-3 resolution (32×32)
        boundary2       (B, 1, H2, W2)           – at level-2 resolution (64×64)

Usage
-----
    from geoseg.models.vmamba_encoder import VMambaEncoder
    from geoseg.models.boundary_gated_decoder import BoundaryGatedDecoder

    encoder = VMambaEncoder()
    decoder = BoundaryGatedDecoder(
        encoder_channels=(96, 192, 384, 768),
        num_classes=6,
        gate_type='multiplicative',
    )

    res1, res2, res3, res4 = encoder(x)          # x: (B, 3, 512, 512)
    outputs = decoder(res1, res2, res3, res4, h=512, w=512)
    seg   = outputs['seg_logits']                  # (B, 6, 512, 512)
    bnd   = outputs['final_boundary']              # (B, 1, 512, 512)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Gate type enumeration (Section III-E)
# =====================================================================

class GateType(Enum):
    """Boundary gate variants."""
    MULTIPLICATIVE = "multiplicative"
    RESIDUAL = "residual"
    BOUNDED_RESIDUAL = "bounded_residual"


# =====================================================================
# Building blocks
# =====================================================================

class ConvBNReLU(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU."""

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


class ConvBN(nn.Sequential):
    """Conv2d → BatchNorm2d (no activation)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, dilation: int = 1,
                 bias: bool = False):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, dilation=dilation, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
        )


# =====================================================================
# Boundary head: 128 → 64 → 1
# =====================================================================

class BoundaryHead(nn.Module):
    """
    Produces a single-channel boundary logit from decoder features.

    Architecture:  Conv3×3-BN-ReLU (in_ch → 64) → Conv1×1 (64 → 1)
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
            boundary logit: (B, 1, H, W)  – raw logit, NOT sigmoid-ed
        """
        return self.head(x)


# =====================================================================
# Boundary gate – swappable gating mechanism
# =====================================================================

class BoundaryGate(nn.Module):
    """
    Gates a skip connection using the previous level's raw boundary logit.

    Pipeline:
        1.  boundary_logit  →  3×3 conv  →  sigmoid  →  gate_map  ∈ [0, 1]
        2.  gate_map  →  bilinear upsample to skip's spatial resolution
        3.  apply gate to skip according to ``gate_type``

    Gate variants:
        multiplicative:     gated = skip · gate_map
        residual:           gated = skip + α · (skip · gate_map)
        bounded_residual:   gated = skip · (β + α · gate_map)
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

        # 3×3 conv on the 1-channel boundary logit → 1-channel gate
        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(
        self,
        skip: torch.Tensor,
        boundary_logit: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            skip:           (B, C, H_skip, W_skip)   – encoder skip feature
            boundary_logit: (B, 1, H_prev, W_prev)   – raw logit from
                            the previous decoder level (may be a different
                            spatial resolution)
        Returns:
            gated_skip:     (B, C, H_skip, W_skip)
        """
        # 1. conv + sigmoid → gate map at boundary_logit's resolution
        gate_map = torch.sigmoid(self.gate_conv(boundary_logit))

        # 2. upsample to match skip's spatial size
        if gate_map.shape[2:] != skip.shape[2:]:
            gate_map = F.interpolate(
                gate_map, size=skip.shape[2:],
                mode="bilinear", align_corners=False,
            )

        # 3. apply gate  (gate_map broadcasts over channel dim)
        if self.gate_type == GateType.MULTIPLICATIVE:
            return skip * gate_map

        elif self.gate_type == GateType.RESIDUAL:
            return skip + self.alpha * (skip * gate_map)

        elif self.gate_type == GateType.BOUNDED_RESIDUAL:
            return skip * (self.beta + self.alpha * gate_map)

        else:
            raise ValueError(f"Unknown gate type: {self.gate_type}")


# =====================================================================
# Decoder level (single stage)
# =====================================================================

class _DecoderLevel(nn.Module):
    """
    One level of the progressive boundary-conditioned decoder.

    If ``is_deepest=True`` (level 4):
        - Only the skip (res4) is processed; no upsampled input, no gate.

    Otherwise (levels 3, 2, 1):
        - The skip is gated by the previous level's boundary map.
        - The skip is channel-reduced to ``decoder_channels``.
        - The previous level's output is bilinearly upsampled to the
          skip's spatial resolution.
        - Both are concatenated and fused through two ConvBNReLU blocks.
    """

    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        is_deepest: bool = False,
    ):
        super().__init__()
        self.is_deepest = is_deepest

        # reduce skip to decoder width
        self.skip_reduce = ConvBNReLU(skip_channels, decoder_channels,
                                      kernel_size=1)

        if is_deepest:
            # level 4: just refine the reduced skip
            self.refine = nn.Sequential(
                ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3),
                ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3),
            )
        else:
            # levels 3, 2, 1: fuse upsampled prev + gated skip
            self.fuse = nn.Sequential(
                ConvBNReLU(decoder_channels * 2, decoder_channels, kernel_size=3),
                ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3),
            )

    def forward(
        self,
        skip: torch.Tensor,
        prev_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            skip:           (B, skip_ch, H, W) – the (already-gated) skip
            prev_features:  (B, dec_ch, H', W') – output from the deeper level
                            (None when ``is_deepest=True``)
        Returns:
            decoder features: (B, decoder_channels, H, W)
        """
        skip = self.skip_reduce(skip)  # (B, dec_ch, H, W)

        if self.is_deepest:
            return self.refine(skip)

        # upsample previous level's features to current spatial size
        prev_up = F.interpolate(
            prev_features, size=skip.shape[2:],
            mode="bilinear", align_corners=False,
        )
        fused = torch.cat([prev_up, skip], dim=1)   # (B, 2·dec_ch, H, W)
        return self.fuse(fused)


# =====================================================================
# Progressive Boundary-Gated Decoder (full module)
# =====================================================================

class BoundaryGatedDecoder(nn.Module):
    """
    Progressive Boundary-Conditioned Decoder.

    Parameters
    ----------
    encoder_channels : tuple of int
        Channel widths of the 4 encoder skip features ``[res1, res2, res3, res4]``.
        Default ``(96, 192, 384, 768)`` matches VMamba-Tiny.
    decoder_channels : int
        Internal channel width of every decoder level.  Default ``128``.
    num_classes : int
        Number of semantic segmentation classes.
    gate_type : str or GateType
        One of ``'multiplicative'``, ``'residual'``, ``'bounded_residual'``.
    alpha : float
        Scaling factor for residual / bounded-residual gates.
    beta : float
        Offset for bounded-residual gate.
    dropout : float
        Dropout before the final segmentation conv.
    """

    def __init__(
        self,
        encoder_channels: Tuple[int, ...] = (96, 192, 384, 768),
        decoder_channels: int = 128,
        num_classes: int = 6,
        gate_type: str = "multiplicative",
        alpha: float = 1.0,
        beta: float = 0.5,
        dropout: float = 0.1,
        use_boundary_heads: bool = True,
        gated_levels: Tuple[int, ...] = (3, 2, 1),
    ):
        super().__init__()

        self.use_boundary_heads = bool(use_boundary_heads)
        if gated_levels is None:
            self.gated_levels = ()
        elif isinstance(gated_levels, (int, str)):
            self.gated_levels = (int(gated_levels),)
        else:
            self.gated_levels = tuple(int(lvl) for lvl in gated_levels)

        # Parse gate type if gating is used
        if self.gated_levels:
            if isinstance(gate_type, str):
                gate_type = GateType(gate_type.lower())
            self.gate_type_enum = gate_type
        else:
            self.gate_type_enum = None

        c1, c2, c3, c4 = encoder_channels

        # ---- Decoder levels (deepest → shallowest) ----
        self.level4 = _DecoderLevel(c4, decoder_channels, is_deepest=True)
        self.level3 = _DecoderLevel(c3, decoder_channels, is_deepest=False)
        self.level2 = _DecoderLevel(c2, decoder_channels, is_deepest=False)
        self.level1 = _DecoderLevel(c1, decoder_channels, is_deepest=False)

        # ---- Boundary gates (selective per level) ----
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

        # ---- Boundary heads (levels 4, 3, 2 + final) ----
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

        # ---- Segmentation head ----
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3),
            nn.Dropout2d(p=dropout, inplace=False),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

        # ---- Auxiliary segmentation head (mid-decoder branching off feat3) ----
        self.aux_segmentation_head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3),
            nn.Dropout2d(p=dropout, inplace=False),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

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
            res1: (B, C1, H/4,  W/4)   encoder skip 1  (shallowest)
            res2: (B, C2, H/8,  W/8)   encoder skip 2
            res3: (B, C3, H/16, W/16)  encoder skip 3
            res4: (B, C4, H/32, W/32)  encoder skip 4  (deepest)
            h, w: original input spatial size (for final upsampling)

        Returns:
            dict with keys:
                seg_logits      (B, num_classes, h, w)
                final_boundary  (B, 1, h, w)
                boundary4       (B, 1, H4, W4)
                boundary3       (B, 1, H3, W3)
                boundary2       (B, 1, H2, W2)
        """
        # ---- Level 4 (deepest): no gate ----
        feat4 = self.level4(res4)                             # (B, D, 16, 16)
        boundary4 = self.boundary_head4(feat4) if self.boundary_head4 is not None else None

        # ---- Level 3: gate res3 with boundary4 if enabled ----
        if 3 in self.gated_levels and self.gate3 is not None and boundary4 is not None:
            gated_res3 = self.gate3(res3, boundary4)          # (B, C3, 32, 32)
        else:
            gated_res3 = res3
        feat3 = self.level3(gated_res3, prev_features=feat4)  # (B, D, 32, 32)
        boundary3 = self.boundary_head3(feat3) if self.boundary_head3 is not None else None

        # ---- Level 2: gate res2 with boundary3 if enabled ----
        if 2 in self.gated_levels and self.gate2 is not None and boundary3 is not None:
            gated_res2 = self.gate2(res2, boundary3)          # (B, C2, 64, 64)
        else:
            gated_res2 = res2
        feat2 = self.level2(gated_res2, prev_features=feat3)  # (B, D, 64, 64)
        boundary2 = self.boundary_head2(feat2) if self.boundary_head2 is not None else None

        # ---- Level 1: gate res1 with boundary2 if enabled ----
        if 1 in self.gated_levels and self.gate1 is not None and boundary2 is not None:
            gated_res1 = self.gate1(res1, boundary2)          # (B, C1, 128,128)
        else:
            gated_res1 = res1
        feat1 = self.level1(gated_res1, prev_features=feat2)  # (B, D, 128,128)

        # ---- Final outputs ----
        seg_logits = self.segmentation_head(feat1)            # (B, K, 128,128)
        seg_logits = F.interpolate(
            seg_logits, size=(h, w), mode="bilinear", align_corners=False,
        )

        # ---- Mid-decoder auxiliary segmentation head (from feat3) ----
        aux_seg_logits = self.aux_segmentation_head(feat3)    # (B, K, 32, 32)
        aux_seg_logits = F.interpolate(
            aux_seg_logits, size=(h, w), mode="bilinear", align_corners=False,
        )

        outputs = {
            "seg_logits": seg_logits,
            "seg_aux": aux_seg_logits,
            "_feat4": feat4,
            "_feat3": feat3,
            "_feat2": feat2,
            "_feat1": feat1,
        }

        if self.use_boundary_heads and self.final_boundary_head is not None:
            final_boundary = self.final_boundary_head(feat1)  # (B, 1, 128,128)
            final_boundary = F.interpolate(
                final_boundary, size=(h, w), mode="bilinear", align_corners=False,
            )
            outputs.update({
                "final_boundary": final_boundary,
                "boundary4": boundary4,
                "boundary3": boundary3,
                "boundary2": boundary2,
            })

        return outputs


# =====================================================================
# Quick self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BoundaryGatedDecoder – shape verification test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate encoder outputs for a 512×512 input
    B = 2
    res1 = torch.randn(B, 96,  128, 128, device=device)
    res2 = torch.randn(B, 192,  64,  64, device=device)
    res3 = torch.randn(B, 384,  32,  32, device=device)
    res4 = torch.randn(B, 768,  16,  16, device=device)

    for gate_name in ["multiplicative", "residual", "bounded_residual"]:
        print(f"\n--- Gate type: {gate_name} ---")
        decoder = BoundaryGatedDecoder(
            encoder_channels=(96, 192, 384, 768),
            decoder_channels=128,
            num_classes=6,
            gate_type=gate_name,
            alpha=1.0,
            beta=0.5,
        ).to(device)

        with torch.no_grad():
            outputs = decoder(res1, res2, res3, res4, h=512, w=512)

        expected = {
            "seg_logits":     (B, 6, 512, 512),
            "final_boundary": (B, 1, 512, 512),
            "boundary4":      (B, 1,  16,  16),
            "boundary3":      (B, 1,  32,  32),
            "boundary2":      (B, 1,  64,  64),
        }

        for key, exp_shape in expected.items():
            actual = tuple(outputs[key].shape)
            status = "✓" if actual == exp_shape else "✗"
            print(f"  {key:20s} : {str(actual):25s}  (expected {exp_shape})  {status}")
            assert actual == exp_shape, \
                f"Shape mismatch for '{key}': {actual} != {exp_shape}"

    # Parameter count
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder parameters: {n_dec:,}")
    print("\nAll assertions passed ✓")
