"""
BoundaryCNNMambaUNet - Swapped Architecture for GeoSeg
======================================================

Swapped counterpart to BoundaryVMambaUNet:
- Encoder:  CNN Backbone (e.g., ResNet-34 / ResNet-50) capturing high-resolution
            local textures and hierarchical spatial features.
- Decoder:  BoundaryMambaDecoder (Progressive Boundary-Gated Mamba / SS2D)
            providing global sequence modeling across scale fusions.

Interface contract
------------------
- ``self.backbone`` -> the CNN encoder backbone (e.g. ResNet), so that the GeoSeg
  config pattern ``{"backbone.*": dict(lr=backbone_lr, ...)}`` assigns a lower LR
  to the pretrained encoder.
- **Training mode** (``self.training == True``):
  Returns a tuple ``(seg_logits, aux_dict)`` when ``use_boundary_heads=True``:
    * ``prediction[0]`` is the seg-logit tensor for metric computation
    * ``prediction[1]`` is a dict of boundary maps for the loss function
- **Eval mode** (``self.training == False``):
  Returns a plain tensor ``seg_logits`` for validation / inference.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from geoseg.models.boundary_mamba_decoder import BoundaryMambaDecoder
from geoseg.models.BoundaryUNetMamba import ClassicUNetEncoder


class BoundaryCNNMambaUNet(nn.Module):
    """
    CNN Encoder (Classic UNet / ResNet) + Progressive Boundary-Conditioned Mamba Decoder.

    Parameters
    ----------
    num_classes : int
        Number of semantic classes (e.g. 7 for LoveDA or SEN-2 LULC).
    backbone_name : str
        Backbone name: 'unet', 'classic_unet', or timm model ('resnet34', 'resnet50', etc.).
    pretrained : bool
        Whether to load pretrained ImageNet weights for timm CNN backbones.
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
        backbone_name: str = "resnet34",
        pretrained: bool = False,
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

        # ---- Encoder Backbone ----
        # Named self.backbone for GeoSeg layerwise learning-rate regex match
        if backbone_name.lower() in ("unet", "classic_unet", "unet_encoder"):
            self.backbone = ClassicUNetEncoder(
                in_channels=3,
                channels=(64, 128, 256, 512),
                strides=(4, 8, 16, 32),
            )
            encoder_channels = (64, 128, 256, 512)
        else:
            self.backbone = timm.create_model(
                backbone_name,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                pretrained=pretrained,
            )
            encoder_channels = tuple(self.backbone.feature_info.channels())

        self.encoder_channels = encoder_channels

        # ---- Mamba Decoder ----
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
        Args:
            x: (B, 3, H, W) - input image
        Returns:
            Training mode: (seg_logits, aux_dict) or seg_logits
            Eval mode: seg_logits
        """
        h, w = x.shape[2:]

        # CNN Encoder extraction -> 4 multi-scale skip features (strides 4, 8, 16, 32)
        res1, res2, res3, res4 = self.backbone(x)

        # Mamba Decoder reconstruction
        outputs = self.decoder(res1, res2, res3, res4, h, w)

        # Sanity check logging on first forward pass
        if not getattr(self, "_first_forward_logged", False):
            self._first_forward_logged = True
            print("\n" + "=" * 76)
            print("SANITY CHECK: Intermediate Tensor Shapes (BoundaryCNNMambaUNet)")
            print("=" * 76)
            print(f"  Input image x:          {tuple(x.shape)}")
            print("  --- CNN Encoder Skip Outputs ---")
            print(f"  res1 (1/4 scale):       {tuple(res1.shape)}")
            print(f"  res2 (1/8 scale):       {tuple(res2.shape)}")
            print(f"  res3 (1/16 scale):      {tuple(res3.shape)}")
            print(f"  res4 (1/32 scale):      {tuple(res4.shape)}")
            print("  --- Mamba Decoder Features (SS2D) ---")
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


# Alias for convenience
CNNBoundaryMambaUNet = BoundaryCNNMambaUNet


if __name__ == "__main__":
    print("Testing BoundaryCNNMambaUNet...")
    model = BoundaryCNNMambaUNet(
        num_classes=7,
        backbone_name="resnet34",
        pretrained=False,
        decoder_channels=128,
        use_boundary_heads=True,
    )
    x = torch.randn(2, 3, 512, 512)
    model.train()
    logits, aux = model(x)
    print("Train mode output:")
    print("  seg_logits:", logits.shape)
    print("  final_boundary:", aux["final_boundary"].shape)
    print("  boundary4:", aux["boundary4"].shape)

    model.eval()
    with torch.no_grad():
        val_logits = model(x)
    print("Eval mode output:", val_logits.shape)
    print("Test passed successfully!")
