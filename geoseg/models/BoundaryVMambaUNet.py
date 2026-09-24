"""
BoundaryVMambaUNet – Full Segmentation Model for GeoSeg
========================================================

Combines :class:`VMambaEncoder` (VMamba-Tiny backbone with 2D-Selective-Scan)
and :class:`BoundaryGatedDecoder` (progressive boundary-conditioned decoder)
into a single ``nn.Module`` that drops directly into GeoSeg's
``train_supervision.py`` training loop.

Interface contract
------------------
- ``self.backbone``   →  the VMamba VSSM encoder, so that the GeoSeg
  config pattern ``{"backbone.*": dict(lr=backbone_lr, ...)}`` assigns a
  lower LR to the pretrained encoder.
- **Training mode** (``self.training == True``):
  Returns a **tuple** ``(seg_logits, aux_dict)`` so that:
    * ``prediction[0]``  is the seg-logit tensor for metric computation
    * ``prediction[1]``  is a dict of boundary maps for the loss function
  Set ``use_aux_loss = True`` in the config.
- **Eval mode** (``self.training == False``):
  Returns a **plain tensor** ``seg_logits`` for validation / inference.

Example config (e.g. ``config/loveda/boundary_vmamba.py``)::

    from geoseg.models.BoundaryVMambaUNet import BoundaryVMambaUNet

    net = BoundaryVMambaUNet(
        num_classes=num_classes,
        gate_type='multiplicative',
        pretrained_backbone_path='pretrain_weights/vssm_tiny_0230_ckpt_epoch_262.pth',
    )

    layerwise_params = {"backbone.*": dict(lr=backbone_lr,
                                            weight_decay=backbone_weight_decay)}
    net_params = process_model_params(net, layerwise_params=layerwise_params)
    ...
    use_aux_loss = True
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoseg.models.vmamba_encoder import VSSM
from geoseg.models.boundary_gated_decoder import BoundaryGatedDecoder


class BoundaryVMambaUNet(nn.Module):
    """
    VMamba-Tiny encoder + Progressive Boundary-Conditioned Decoder.

    Parameters
    ----------
    num_classes : int
        Number of semantic classes (default 7, e.g. LoveDA).
    decoder_channels : int
        Internal channel width of every decoder level (default 128).
    gate_type : str
        Boundary gate variant: ``'multiplicative'``, ``'residual'``,
        or ``'bounded_residual'``.
    alpha : float
        Gate scaling factor (for residual / bounded-residual gates).
    beta : float
        Gate offset (for bounded-residual gate).
    dropout : float
        Dropout rate before the segmentation head.
    pretrained_backbone_path : str or None
        Path to a VMamba-Tiny ImageNet checkpoint.
    use_checkpoint : bool
        Enable gradient checkpointing in the backbone.
    """

    def __init__(
        self,
        num_classes: int = 7,
        decoder_channels: int = 128,
        gate_type: str = "multiplicative",
        alpha: float = 1.0,
        beta: float = 0.5,
        dropout: float = 0.1,
        pretrained_backbone_path: str = None,
        use_checkpoint: bool = True,
        use_boundary_heads: bool = True,
        gated_levels: Tuple[int, ...] = (3, 2, 1),
    ):
        super().__init__()
        self.use_boundary_heads = bool(use_boundary_heads)
        self.gated_levels = gated_levels

        # ---- Encoder (VMamba-Tiny) ----
        # Named ``self.backbone`` so that the GeoSeg config pattern
        #   {"backbone.*": dict(lr=backbone_lr, ...)}
        # automatically assigns a separate learning rate.
        self.backbone = VSSM(
            patch_size=4,
            in_chans=3,
            depths=[2, 2, 9, 2],
            dims=96,
            drop_path_rate=0.2,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v0",
            mlp_ratio=0.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            patch_norm=True,
            norm_layer="ln",
            downsample_version="v1",
            patchembed_version="v1",
            use_checkpoint=use_checkpoint,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained_backbone_path,
        )

        encoder_channels = tuple(self.backbone.dims)  # (96, 192, 384, 768)

        # Pretrained status tracking & logging
        self.pretrained_loaded = bool(getattr(self.backbone, "_pretrained_loaded", False))
        if self.pretrained_loaded:
            print("\n" + "=" * 65)
            print("  Backbone: PRETRAINED")
            print(f"  Source: {pretrained_backbone_path}")
            print("=" * 65 + "\n")
        else:
            print("\n" + "=" * 65)
            print("  Backbone: RANDOM INIT — WARNING")
            if pretrained_backbone_path:
                print(f"  (Failed to load from: {pretrained_backbone_path})")
            else:
                print("  (No pretrained_backbone_path specified)")
            print("=" * 65 + "\n")

        # ---- Decoder (randomly initialized) ----
        self.decoder = BoundaryGatedDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            gate_type=gate_type,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            use_boundary_heads=use_boundary_heads,
            gated_levels=gated_levels,
        )

    def load_pretrained_backbone(self, ckpt_path: str) -> bool:
        """
        Load pretrained VMamba weights into encoder backbone.
        Decoder and boundary heads remain untouched (randomly initialized).
        """
        success = self.backbone.load_pretrained(ckpt_path)
        self.pretrained_loaded = bool(success)
        if self.pretrained_loaded:
            print("\n" + "=" * 65)
            print("  Backbone: PRETRAINED")
            print(f"  Source: {ckpt_path}")
            print("=" * 65 + "\n")
        else:
            print("\n" + "=" * 65)
            print("  Backbone: RANDOM INIT — WARNING")
            print(f"  (Failed to load from: {ckpt_path})")
            print("=" * 65 + "\n")
        return success

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) – input image (e.g. 512×512)

        Returns:
            Training mode:
                tuple ``(seg_logits, aux_dict)`` if use_boundary_heads is True,
                or plain tensor ``seg_logits`` if use_boundary_heads is False.
                - seg_logits:   (B, num_classes, H, W)
                - aux_dict:     dict with keys
                    ``final_boundary``, ``boundary4``, ``boundary3``, ``boundary2``
            Eval mode:
                seg_logits:  (B, num_classes, H, W)
        """
        h, w = x.shape[2:]

        # Encoder → 4 multi-scale skip features
        res1, res2, res3, res4 = self.backbone(x)

        # Decoder → seg logits + boundary maps
        outputs = self.decoder(res1, res2, res3, res4, h, w)

        # First forward pass sanity check logging
        if not getattr(self, "_first_forward_logged", False):
            self._first_forward_logged = True
            print("\n" + "=" * 76)
            print("SANITY CHECK: Intermediate Tensor Shapes (First Forward Pass)")
            print("=" * 76)
            print(f"  Input image x:          {tuple(x.shape)}")
            print("  --- Encoder Skip Outputs (VMamba-Tiny) ---")
            print(f"  res1:                   {tuple(res1.shape)}   (paper spec: [B, 96, 128, 128])")
            print(f"  res2:                   {tuple(res2.shape)}   (paper spec: [B, 192, 64, 64])")
            print(f"  res3:                   {tuple(res3.shape)}   (paper spec: [B, 384, 32, 32])")
            print(f"  res4:                   {tuple(res4.shape)}   (paper spec: [B, 768, 16, 16])")
            print("  --- Decoder Intermediate Features ---")
            print(f"  decoder_feat4:          {tuple(outputs['_feat4'].shape)}   (paper spec: [B, 128, 16, 16])")
            print(f"  decoder_feat3:          {tuple(outputs['_feat3'].shape)}   (paper spec: [B, 128, 32, 32])")
            print(f"  decoder_feat2:          {tuple(outputs['_feat2'].shape)}   (paper spec: [B, 128, 64, 64])")
            print(f"  decoder_feat1:          {tuple(outputs['_feat1'].shape)}  (paper spec: [B, 128, 128, 128])")
            print("  --- Boundary Maps ---")
            bnd4_shape = tuple(outputs['boundary4'].shape) if outputs.get('boundary4') is not None else "None (disabled in plain baseline; full model: [B, 1, 16, 16])"
            bnd3_shape = tuple(outputs['boundary3'].shape) if outputs.get('boundary3') is not None else "None (disabled in plain baseline; full model: [B, 1, 32, 32])"
            bnd2_shape = tuple(outputs['boundary2'].shape) if outputs.get('boundary2') is not None else "None (disabled in plain baseline; full model: [B, 1, 64, 64])"
            final_bnd_shape = tuple(outputs['final_boundary'].shape) if outputs.get('final_boundary') is not None else "None (disabled in plain baseline; full model: [B, 1, 512, 512])"
            print(f"  boundary4:              {bnd4_shape}")
            print(f"  boundary3:              {bnd3_shape}")
            print(f"  boundary2:              {bnd2_shape}")
            print(f"  final_boundary:         {final_bnd_shape}")
            print("  --- Segmentation Head ---")
            print(f"  seg_logits:             {tuple(outputs['seg_logits'].shape)}   (paper spec: [B, 7, 512, 512])")
            print("=" * 76 + "\n")

        if not self.use_boundary_heads:
            # Baseline plain VMamba-UNet: no boundary heads or auxiliary dict
            return outputs["seg_logits"]

        if self.training:
            # Return tuple: (seg_logits, aux_dict)
            # train_supervision.py uses prediction[0] for metrics
            # and passes the full tuple to self.loss(prediction, mask)
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
            # Eval mode: plain tensor for validation_step compatibility
            return outputs["seg_logits"]


# =====================================================================
# Quick self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BoundaryVMambaUNet – interface verification")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BoundaryVMambaUNet(
        num_classes=7,
        gate_type="multiplicative",
        pretrained_backbone_path=None,
    ).to(device)

    x = torch.randn(2, 3, 512, 512, device=device)

    # ---- Training mode ----
    model.train()
    prediction = model(x)
    assert isinstance(prediction, tuple), "Training output should be a tuple"
    seg_logits, aux = prediction
    print(f"\n[Training mode]")
    print(f"  prediction[0] (seg_logits) : {tuple(seg_logits.shape)}")
    assert tuple(seg_logits.shape) == (2, 7, 512, 512)
    for k, v in aux.items():
        print(f"  aux['{k}']  : {tuple(v.shape)}")

    # Verify prediction[0] works for metrics (as train_supervision.py expects)
    pre_mask = torch.softmax(prediction[0], dim=1).argmax(dim=1)
    print(f"  argmax mask shape          : {tuple(pre_mask.shape)}")
    assert tuple(pre_mask.shape) == (2, 512, 512)

    # ---- Eval mode ----
    model.eval()
    with torch.no_grad():
        prediction = model(x)
    assert isinstance(prediction, torch.Tensor), "Eval output should be a tensor"
    print(f"\n[Eval mode]")
    print(f"  prediction (seg_logits)    : {tuple(prediction.shape)}")
    assert tuple(prediction.shape) == (2, 7, 512, 512)

    # ---- Verify backbone.* param naming for layerwise LR ----
    backbone_params = [n for n, _ in model.named_parameters()
                       if n.startswith("backbone.")]
    decoder_params = [n for n, _ in model.named_parameters()
                      if n.startswith("decoder.")]
    total_params = list(model.named_parameters())
    print(f"\n[Parameter groups]")
    print(f"  backbone.* params: {len(backbone_params)}")
    print(f"  decoder.*  params: {len(decoder_params)}")
    print(f"  total params     : {len(total_params)}")
    assert len(backbone_params) > 0, "No params under backbone.*"
    assert len(decoder_params) > 0, "No params under decoder.*"
    assert len(backbone_params) + len(decoder_params) == len(total_params), \
        "All params should be under backbone.* or decoder.*"

    n_total = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for n, p in model.named_parameters()
                     if n.startswith("backbone."))
    n_decoder = sum(p.numel() for n, p in model.named_parameters()
                    if n.startswith("decoder."))
    print(f"\n  Backbone : {n_backbone:>12,} params")
    print(f"  Decoder  : {n_decoder:>12,} params")
    print(f"  Total    : {n_total:>12,} params")

    print(f"\n{'=' * 60}")
    print("All assertions passed ✓")
    print(f"{'=' * 60}")
