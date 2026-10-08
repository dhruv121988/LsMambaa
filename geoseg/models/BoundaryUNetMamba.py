"""
BoundaryUNetMamba - Classic U-Net Contracting Encoder + Mamba (SS2D) Decoder
=============================================================================

Architecture Overview:
- Encoder:  Classic U-Net Contracting Path (DoubleConv blocks + MaxPool2d downsampling)
            extracting multi-scale hierarchical spatial feature representations purely
            through convolutional inductive biases.
- Decoder:  BoundaryMambaDecoder (Progressive Boundary-Conditioned 2D Selective State
            Space / SS2D Mamba decoder) providing global sequence modeling and
            boundary-guided feature fusion across scale reconstructions.

Interface Contract:
- ``self.backbone`` -> :class:`ClassicUNetEncoder` so that GeoSeg's layerwise learning
  rate regex ``{"backbone.*": dict(lr=backbone_lr, ...)}`` assigns encoder-specific LR.
- **Training mode** (``self.training == True``):
  Returns ``(seg_logits, aux_dict)`` when ``use_boundary_heads=True``.
- **Eval mode** (``self.training == False``):
  Returns ``seg_logits``.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from geoseg.models.boundary_mamba_decoder import BoundaryMambaDecoder


# =====================================================================
# Classic U-Net Convolutional Building Blocks
# =====================================================================

class DoubleConv(nn.Module):
    """
    Classic Ronneberger U-Net Double Convolution block:
    [Conv2d(3x3) -> BatchNorm2d -> ReLU] x 2
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ClassicUNetEncoder(nn.Module):
    """
    Classic U-Net Contracting Path (Encoder).

    Constructed purely with standard DoubleConv [Conv-BN-ReLU x 2] blocks and
    MaxPool2d downsampling operations to extract 4 hierarchical multi-scale
    feature stages (res1, res2, res3, res4).

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 3 for RGB).
    channels : tuple of int
        Channel dimensions for the 4 skip feature outputs (default (64, 128, 256, 512)).
    strides : tuple of int
        Downsampling stride levels for the 4 skip stages.
        Default is (4, 8, 16, 32), matching standard modern vision decoders and
        optimizing memory footprint on high-resolution inputs.
        Alternative is (2, 4, 8, 16) for original 4-pool Ronneberger contracting paths.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Tuple[int, ...] = (64, 128, 256, 512),
        strides: Tuple[int, ...] = (4, 8, 16, 32),
    ):
        super().__init__()
        self.channels = channels
        self.strides = strides
        c1, c2, c3, c4 = channels

        if strides == (4, 8, 16, 32):
            # Stage 0: Initial DoubleConv (stride 1)
            self.inc = DoubleConv(in_channels, 32)
            # Pool to stride 2
            self.down0 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(32, 64),
            )
            # Stage 1: Pool to stride 4 -> res1
            self.down1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(64, c1),
            )
            # Stage 2: Pool to stride 8 -> res2
            self.down2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(c1, c2),
            )
            # Stage 3: Pool to stride 16 -> res3
            self.down3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(c2, c3),
            )
            # Stage 4: Pool to stride 32 -> res4
            self.down4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(c3, c4),
            )
        elif strides == (2, 4, 8, 16):
            # Classic 4-pool Ronneberger UNet contracting path
            self.inc = DoubleConv(in_channels, 64)
            self.down0 = nn.Identity()
            # Stage 1: Pool to stride 2 -> res1
            self.down1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(64, c1),
            )
            # Stage 2: Pool to stride 4 -> res2
            self.down2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(c1, c2),
            )
            # Stage 3: Pool to stride 8 -> res3
            self.down3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(c2, c3),
            )
            # Stage 4: Pool to stride 16 -> res4
            self.down4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(c3, c4),
            )
        else:
            raise ValueError(f"Unsupported strides: {strides}. Choose (4, 8, 16, 32) or (2, 4, 8, 16).")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract 4 hierarchical feature maps.

        Args:
            x: (B, in_channels, H, W)
        Returns:
            res1, res2, res3, res4
        """
        x = self.inc(x)
        x = self.down0(x)
        res1 = self.down1(x)
        res2 = self.down2(res1)
        res3 = self.down3(res2)
        res4 = self.down4(res3)
        return res1, res2, res3, res4


# =====================================================================
# Full BoundaryUNetMamba Model
# =====================================================================

class BoundaryUNetMamba(nn.Module):
    """
    Classic U-Net Contracting Encoder + Progressive Boundary-Conditioned Mamba Decoder.

    Parameters
    ----------
    num_classes : int
        Number of semantic segmentation classes (e.g., 7 for LoveDA).
    encoder_channels : tuple of int
        Channels for the 4 U-Net encoder stages (default (64, 128, 256, 512)).
    strides : tuple of int
        Downsampling stride levels for the 4 stages (default (4, 8, 16, 32)).
    decoder_channels : int
        Internal channel width of the Mamba decoder stages (default 128).
    num_mamba_blocks : int
        Number of VSSBlock layers per decoder stage (default 1).
    gate_type : str
        Boundary gate variant: 'multiplicative', 'residual', or 'bounded_residual'.
    alpha : float
        Gate scaling factor for residual / bounded-residual gates.
    beta : float
        Gate offset for bounded-residual gate.
    dropout : float
        Dropout rate before the segmentation head.
    use_boundary_heads : bool
        Whether to enable multi-scale boundary auxiliary heads.
    gated_levels : tuple of int
        Decoder levels with boundary gating (default (3, 2, 1)).
    use_checkpoint : bool
        Enable gradient checkpointing in decoder Mamba blocks.
    """

    def __init__(
        self,
        num_classes: int = 7,
        encoder_channels: Tuple[int, ...] = (64, 128, 256, 512),
        strides: Tuple[int, ...] = (4, 8, 16, 32),
        decoder_channels: int = 128,
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
        self.num_classes = num_classes
        self.use_boundary_heads = bool(use_boundary_heads)
        self.gated_levels = gated_levels
        self.encoder_channels = encoder_channels

        # ---- Classic U-Net Contracting Encoder ----
        # Named self.backbone for GeoSeg layerwise learning-rate regex match
        self.backbone = ClassicUNetEncoder(
            in_channels=3,
            channels=encoder_channels,
            strides=strides,
        )

        # ---- Progressive Boundary-Conditioned Mamba Decoder ----
        self.decoder = BoundaryMambaDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            num_mamba_blocks=num_mamba_blocks,
            gate_type=gate_type,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            use_boundary_heads=use_boundary_heads,
            gated_levels=gated_levels,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) input image tensor
        Returns:
            Training mode: (seg_logits, aux_dict) or seg_logits
            Eval mode: seg_logits
        """
        h, w = x.shape[2:]

        # U-Net Contracting Encoder: 4 multi-scale skip features
        res1, res2, res3, res4 = self.backbone(x)

        # Progressive Boundary Mamba Decoder
        outputs = self.decoder(res1, res2, res3, res4, h, w)

        # First-pass sanity check shape logging
        if not getattr(self, "_first_forward_logged", False):
            self._first_forward_logged = True
            print("\n" + "=" * 76)
            print("SANITY CHECK: Intermediate Tensor Shapes (BoundaryUNetMamba)")
            print("=" * 76)
            print(f"  Input image x:          {tuple(x.shape)}")
            print("  --- Classic U-Net Contracting Encoder (DoubleConv + MaxPool) ---")
            print(f"  res1 (skip level 1):    {tuple(res1.shape)}")
            print(f"  res2 (skip level 2):    {tuple(res2.shape)}")
            print(f"  res3 (skip level 3):    {tuple(res3.shape)}")
            print(f"  res4 (skip level 4):    {tuple(res4.shape)}")
            print("  --- Mamba Decoder Features (SS2D VSSBlocks) ---")
            print(f"  decoder_feat4:          {tuple(outputs['_feat4'].shape)}")
            print(f"  decoder_feat3:          {tuple(outputs['_feat3'].shape)}")
            print(f"  decoder_feat2:          {tuple(outputs['_feat2'].shape)}")
            print(f"  decoder_feat1:          {tuple(outputs['_feat1'].shape)}")
            print("  --- Boundary Predictions ---")
            bnd4 = tuple(outputs['boundary4'].shape) if outputs.get('boundary4') is not None else "None"
            bnd3 = tuple(outputs['boundary3'].shape) if outputs.get('boundary3') is not None else "None"
            bnd2 = tuple(outputs['boundary2'].shape) if outputs.get('boundary2') is not None else "None"
            final_bnd = tuple(outputs['final_boundary'].shape) if outputs.get('final_boundary') is not None else "None"
            print(f"  boundary4:              {bnd4}")
            print(f"  boundary3:              {bnd3}")
            print(f"  boundary2:              {bnd2}")
            print(f"  final_boundary:         {final_bnd}")
            print("  --- Segmentation Head ---")
            print(f"  seg_logits:             {tuple(outputs['seg_logits'].shape)}")
            print("=" * 76 + "\n")

        if not self.use_boundary_heads:
            return outputs["seg_logits"]

        if self.training:
            seg_logits = outputs["seg_logits"]
            aux = {
                "final_boundary": outputs.get("final_boundary"),
                "boundary4": outputs.get("boundary4"),
                "boundary3": outputs.get("boundary3"),
                "boundary2": outputs.get("boundary2"),
                "seg_aux": outputs.get("seg_aux"),
            }
            return seg_logits, aux
        else:
            return outputs["seg_logits"]


# Convenience Aliases
UNetMamba = BoundaryUNetMamba
UNetBoundaryMamba = BoundaryUNetMamba


if __name__ == "__main__":
    print("Testing BoundaryUNetMamba...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BoundaryUNetMamba(
        num_classes=7,
        encoder_channels=(64, 128, 256, 512),
        strides=(4, 8, 16, 32),
        decoder_channels=128,
        use_boundary_heads=True,
    ).to(device)

    x = torch.randn(2, 3, 512, 512, device=device)

    # Train mode
    model.train()
    logits, aux = model(x)
    print("Train mode output:")
    print("  seg_logits:", logits.shape)
    print("  final_boundary:", aux["final_boundary"].shape)
    print("  boundary4:", aux["boundary4"].shape)

    # Eval mode
    model.eval()
    with torch.no_grad():
        val_logits = model(x)
    print("Eval mode output:", val_logits.shape)
    print("BoundaryUNetMamba test passed successfully!")
