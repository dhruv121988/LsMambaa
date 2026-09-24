"""
BoundaryVMambaLoss – Multi-task loss for BoundaryVMambaUNet
============================================================

Implements the exact multi-task loss formula:

    Total = SegLoss + λ_aux * AuxLoss + λ_bnd * BoundaryLoss

where:
    SegLoss      = CE(label_smoothing=0.1) + DiceLoss on seg_logits vs GT mask
    AuxLoss      = CE(label_smoothing=0.1) on genuine mid-decoder auxiliary
                   segmentation logits (branching off decoder_feat3) vs GT mask
    BoundaryLoss = mean(BCE(boundary4, gt_boundary),
                        BCE(boundary3, gt_boundary),
                        BCE(boundary2, gt_boundary),
                        BCE(final_boundary, gt_boundary))
                   All four boundary BCE terms are averaged together under the
                   single λ_bnd weight.

Default hyperparameters:
    λ_aux = 0.4
    λ_bnd = 0.3
Both overridable via config and CLI.
"""

from typing import Dict, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .soft_ce import SoftCrossEntropyLoss
from .dice import DiceLoss

__all__ = ["BoundaryVMambaLoss"]


# =====================================================================
# Boundary GT extraction
# =====================================================================

def _extract_boundary(mask: Tensor) -> Tensor:
    """
    Derive a binary boundary map from a segmentation mask using a
    Laplacian edge-detection kernel.

    Args:
        mask: (B, H, W) integer segmentation labels

    Returns:
        boundary: (B, 1, H, W) float tensor, 1.0 at edges, 0.0 elsewhere
    """
    # Laplacian 3×3 kernel
    kernel = torch.tensor(
        [-1, -1, -1,
         -1,  8, -1,
         -1, -1, -1],
        dtype=torch.float32, device=mask.device,
    ).reshape(1, 1, 3, 3)

    x = mask.unsqueeze(1).float()                      # (B, 1, H, W)
    edges = F.conv2d(x, kernel, padding=1)             # (B, 1, H, W)
    edges = edges.clamp(min=0)
    boundary = (edges >= 0.1).float()
    return boundary


# =====================================================================
# Boundary loss variants
# =====================================================================

def _bce_boundary_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Optional[Tensor] = None,
    pos_weight: Optional[float] = None,
) -> Tensor:
    """
    Standard binary cross-entropy with logits for boundary detection.

    Args:
        pred:       (B, 1, H, W) raw logits
        target:     (B, 1, H, W) binary boundary GT
        valid_mask: (B, 1, H, W) optional mask (1 = valid, 0 = ignore)
        pos_weight: scalar weight for positive (boundary) class
    """
    pw = None
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=pred.device, dtype=pred.dtype)

    loss = F.binary_cross_entropy_with_logits(
        pred, target, pos_weight=pw, reduction="none",
    )

    if valid_mask is not None:
        loss = loss * valid_mask
        return loss.sum() / valid_mask.sum().clamp(min=1)

    return loss.mean()


def _focal_boundary_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Optional[Tensor] = None,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> Tensor:
    """
    Focal binary cross-entropy for boundary detection.
    """
    p = torch.sigmoid(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t).pow(gamma)

    loss = focal_weight * bce

    if valid_mask is not None:
        loss = loss * valid_mask
        return loss.sum() / valid_mask.sum().clamp(min=1)

    return loss.mean()


# =====================================================================
# Main loss module
# =====================================================================

class BoundaryVMambaLoss(nn.Module):
    """
    Multi-task loss for :class:`BoundaryVMambaUNet`.

    Total = SegLoss + λ_aux * AuxLoss + λ_bnd * BoundaryLoss

    where:
        SegLoss      = CE(label_smoothing=0.1) + DiceLoss on seg_logits vs GT mask
        AuxLoss      = CE(label_smoothing=0.1) on mid-decoder auxiliary segmentation logits vs GT mask
        BoundaryLoss = mean(BCE(boundary4, gt_boundary),
                            BCE(boundary3, gt_boundary),
                            BCE(boundary2, gt_boundary),
                            BCE(final_boundary, gt_boundary))
                       All four boundary BCE terms are averaged under the single λ_bnd weight.

    Parameters
    ----------
    ignore_index : int
        Label value to ignore (default 255).
    label_smoothing : float
        Label smoothing factor for segmentation CE loss (default 0.1).
    lambda_aux : float
        Weight for mid-decoder auxiliary segmentation loss (default 0.4).
    lambda_bnd : float
        Weight for averaged boundary loss (default 0.3).
    boundary_loss_type : str
        'bce' for binary cross-entropy, 'focal' for focal BCE (default 'bce').
    bce_pos_weight : float or None
        Positive class weight for boundary BCE (default None).
    focal_alpha : float
        Focal loss alpha parameter (default 0.75).
    focal_gamma : float
        Focal loss gamma focusing parameter (default 2.0).
    """

    def __init__(
        self,
        ignore_index: int = 255,
        label_smoothing: float = 0.1,
        lambda_aux: float = 0.4,
        lambda_bnd: float = 0.3,
        boundary_loss_type: str = "bce",
        bce_pos_weight: Optional[float] = None,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        assert boundary_loss_type in ("bce", "focal"), \
            f"boundary_loss_type must be 'bce' or 'focal', got '{boundary_loss_type}'"

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.lambda_aux = lambda_aux
        self.lambda_bnd = lambda_bnd
        self.boundary_loss_type = boundary_loss_type
        self.bce_pos_weight = bce_pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # ---- SegLoss: CE(label_smoothing=0.1) + DiceLoss ----
        self.ce_loss = SoftCrossEntropyLoss(
            smooth_factor=label_smoothing,
            ignore_index=ignore_index,
        )
        self.dice_loss = DiceLoss(
            smooth=0.05,
            ignore_index=ignore_index,
        )

        # ---- AuxLoss: CE on genuine mid-decoder auxiliary segmentation output ----
        self.aux_ce_loss = SoftCrossEntropyLoss(
            smooth_factor=label_smoothing,
            ignore_index=ignore_index,
        )

        self.last_losses: Dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_boundary_gt(self, mask: Tensor) -> Tensor:
        """Derive boundary GT from segmentation mask."""
        return _extract_boundary(mask)

    def _get_valid_mask(self, mask: Tensor) -> Tensor:
        """
        Create a spatial validity mask that excludes ignore-index pixels
        from boundary loss computation.
        """
        return (mask != self.ignore_index).unsqueeze(1).float()

    def _compute_single_boundary_loss(
        self,
        pred: Tensor,
        boundary_gt: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Compute boundary loss for a single boundary head."""
        if pred.shape[2:] != boundary_gt.shape[2:]:
            pred = F.interpolate(
                pred, size=boundary_gt.shape[2:],
                mode="bilinear", align_corners=False,
            )

        if self.boundary_loss_type == "bce":
            return _bce_boundary_loss(
                pred, boundary_gt, valid_mask,
                pos_weight=self.bce_pos_weight,
            )
        else:  # focal
            return _focal_boundary_loss(
                pred, boundary_gt, valid_mask,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        prediction: Union[Tensor, tuple],
        mask: Tensor,
        gt_boundary: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute: Total = SegLoss + λ_aux * AuxLoss + λ_bnd * BoundaryLoss

        Args:
            prediction:
                Training mode: tuple ``(seg_logits, aux_dict)`` where
                    - seg_logits: (B, num_classes, H, W)
                    - aux_dict: dict containing:
                        ``'seg_aux'``: (B, num_classes, H, W) mid-decoder aux logits
                        ``'boundary4'``: (B, 1, H/32, W/32)
                        ``'boundary3'``: (B, 1, H/16, W/16)
                        ``'boundary2'``: (B, 1, H/8, W/8)
                        ``'final_boundary'``: (B, 1, H, W)
                Eval mode: plain ``seg_logits`` tensor.
            mask: (B, H, W) ground-truth integer segmentation labels.
            gt_boundary: Optional (B, 1, H, W) binary boundary ground truth.
                         If None, derived on-the-fly from mask.

        Returns:
            Scalar total loss tensor.
        """
        # Unpack prediction
        if isinstance(prediction, (tuple, list)):
            seg_logits, aux = prediction
        else:
            # Eval mode (plain tensor): compute SegLoss only
            loss_ce = self.ce_loss(prediction, mask)
            loss_dice = self.dice_loss(prediction, mask)
            return loss_ce + loss_dice

        # --------------------------------------------------------------
        # 1. SegLoss = CE(label_smoothing=0.1) + DiceLoss on seg_logits vs GT mask
        # --------------------------------------------------------------
        loss_ce = self.ce_loss(seg_logits, mask)
        loss_dice = self.dice_loss(seg_logits, mask)
        seg_loss = loss_ce + loss_dice

        # --------------------------------------------------------------
        # 2. AuxLoss = CE on genuine mid-decoder auxiliary segmentation output
        # --------------------------------------------------------------
        loss_aux = torch.tensor(0.0, device=seg_logits.device, dtype=seg_logits.dtype)
        if isinstance(aux, dict) and "seg_aux" in aux and aux["seg_aux"] is not None:
            loss_aux = self.aux_ce_loss(aux["seg_aux"], mask)

        # --------------------------------------------------------------
        # 3. BoundaryLoss = mean(BCE(boundary4, gt_boundary),
        #                        BCE(boundary3, gt_boundary),
        #                        BCE(boundary2, gt_boundary),
        #                        BCE(final_boundary, gt_boundary))
        # All four terms averaged under the single λ_bnd weight
        # --------------------------------------------------------------
        if gt_boundary is None:
            boundary_gt = self._get_boundary_gt(mask)
        else:
            boundary_gt = gt_boundary
            if boundary_gt.dim() == 3:
                boundary_gt = boundary_gt.unsqueeze(1).float()
            else:
                boundary_gt = boundary_gt.float()

        valid_mask = self._get_valid_mask(mask)

        boundary_keys = ["boundary4", "boundary3", "boundary2", "final_boundary"]
        bnd_losses: List[Tensor] = []
        for key in boundary_keys:
            if isinstance(aux, dict) and key in aux and aux[key] is not None:
                bce_term = self._compute_single_boundary_loss(
                    aux[key], boundary_gt, valid_mask,
                )
                bnd_losses.append(bce_term)

        if bnd_losses:
            boundary_loss = sum(bnd_losses) / len(bnd_losses)
        else:
            boundary_loss = torch.tensor(0.0, device=seg_logits.device, dtype=seg_logits.dtype)

        # --------------------------------------------------------------
        # Total = SegLoss + λ_aux * AuxLoss + λ_bnd * BoundaryLoss
        # --------------------------------------------------------------
        total = seg_loss + self.lambda_aux * loss_aux + self.lambda_bnd * boundary_loss

        # Store loss components for logging and unit-test inspection
        self.last_losses = {
            "seg_loss": seg_loss.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
            "aux_loss": loss_aux.detach(),
            "boundary_loss": boundary_loss.detach(),
            "num_boundary_terms": torch.tensor(len(bnd_losses), device=seg_logits.device),
            "total_loss": total.detach(),
        }

        return total
