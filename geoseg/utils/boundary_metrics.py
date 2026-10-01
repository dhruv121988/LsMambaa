"""
Boundary Evaluation Metrics
===========================

Implements specialized boundary-aware segmentation metrics:
1. Boundary IoU (Cheng et al., CVPR 2021)
2. Boundary F1, Precision, and Recall (BF-score with configurable tolerance)
3. Boundary-Distance Accuracy (Accuracy vs Distance from Nearest Boundary)
4. State-accumulating BoundaryEvaluator matching GeoSeg Evaluator interface
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


# =====================================================================
# Boundary Extraction Utilities
# =====================================================================

def mask_to_boundary(
    mask: np.ndarray,
    connectivity: int = 8,
    ignore_index: Optional[int] = None,
) -> np.ndarray:
    """
    Extract binary boundary map from a multi-class integer segmentation mask.
    A pixel is considered a boundary pixel if at least one neighbor in its
    neighborhood has a different semantic class label.

    Transitions to/from ``ignore_index`` are excluded from being counted as boundaries.

    Args:
        mask: (H, W) integer numpy array
        connectivity: 4 or 8 neighborhood
        ignore_index: label to exclude from boundary detection

    Returns:
        boundary: (H, W) boolean numpy array
    """
    h, w = mask.shape
    if ignore_index is not None:
        valid_mask = (mask != ignore_index)
    else:
        valid_mask = np.ones((h, w), dtype=bool)

    boundary = np.zeros((h, w), dtype=bool)

    # 4-connectivity neighbor shifts: (dy, dx)
    shifts_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 8-connectivity adds diagonals
    shifts_8 = shifts_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    shifts = shifts_8 if connectivity == 8 else shifts_4

    for dy, dx in shifts:
        # Source slices
        src_y1 = max(0, -dy)
        src_y2 = h - max(0, dy)
        src_x1 = max(0, -dx)
        src_x2 = w - max(0, dx)

        # Target neighbor slices
        dst_y1 = max(0, dy)
        dst_y2 = h - max(0, -dy)
        dst_x1 = max(0, dx)
        dst_x2 = w - max(0, -dx)

        curr = mask[src_y1:src_y2, src_x1:src_x2]
        neighbor = mask[dst_y1:dst_y2, dst_x1:dst_x2]

        curr_valid = valid_mask[src_y1:src_y2, src_x1:src_x2]
        neighbor_valid = valid_mask[dst_y1:dst_y2, dst_x1:dst_x2]

        diff = (curr != neighbor) & curr_valid & neighbor_valid
        boundary[src_y1:src_y2, src_x1:src_x2] |= diff

    return boundary


def class_mask_to_boundary(
    binary_mask: np.ndarray,
    dilation_width: int = 1,
) -> np.ndarray:
    """
    Extract boundary region of a single binary class mask.

    Args:
        binary_mask: (H, W) boolean or 0/1 array
        dilation_width: pixel thickness d for boundary region

    Returns:
        boundary: (H, W) boolean array
    """
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)

    mask_u8 = binary_mask.astype(np.uint8)
    kernel_size = 2 * dilation_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    eroded = cv2.erode(mask_u8, kernel)
    boundary = (mask_u8 - eroded) > 0
    return boundary


def class_mask_to_boundary_official(
    binary_mask: np.ndarray,
    dilation_ratio: float = 0.02,
    dilation_px: Optional[int] = None,
) -> np.ndarray:
    """
    Extract boundary region of a binary mask per Cheng et al. CVPR 2021 (boundary-iou-api).

    If dilation_px is provided, uses that fixed pixel distance.
    Otherwise, computes dilation = int(round(dilation_ratio * img_diag)).
    Applies 1px zero-padding to capture boundaries at image edges correctly.

    Args:
        binary_mask: (H, W) boolean or 0/1 array
        dilation_ratio: ratio of image diagonal (default 0.02, per CVPR 2021)
        dilation_px: optional fixed pixel dilation (overrides dilation_ratio if given)

    Returns:
        boundary: (H, W) boolean array
    """
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)

    h, w = binary_mask.shape
    if dilation_px is not None:
        dilation = max(1, int(dilation_px))
    else:
        img_diag = np.sqrt(h ** 2 + w ** 2)
        dilation = max(1, int(round(dilation_ratio * img_diag)))

    mask_u8 = binary_mask.astype(np.uint8)
    new_mask = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    boundary = (mask_u8 - mask_erode) > 0
    return boundary


# =====================================================================
# Metric Functions (Functional API)
# =====================================================================

def compute_boundary_iou(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    num_classes: int,
    dilation_width: Optional[int] = 2,
    dilation_ratio: Optional[float] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Compute Boundary IoU (Cheng et al., CVPR 2021) for multi-class segmentation.

    For each class c:
        G_d = boundary region of GT mask of class c (pixels within d of boundary)
        P_d = boundary region of Pred mask of class c (pixels within d of boundary)
        BIoU = |G_d ∩ P_d| / |(G_d ∩ P) ∪ (P_d ∩ G)|

    Args:
        gt_mask: (H, W) ground truth integer mask
        pred_mask: (H, W) predicted integer mask
        num_classes: number of semantic classes
        dilation_width: pixel boundary width d (e.g. 2 px). If None and dilation_ratio given, uses ratio.
        dilation_ratio: ratio of image diagonal (e.g. 0.02).
        ignore_index: class index to ignore

    Returns:
        biou_per_class: (num_classes,) float array of per-class Boundary IoUs
        mean_biou: scalar mean Boundary IoU across classes present in GT or pred
    """
    biou_per_class = np.full(num_classes, np.nan, dtype=np.float64)

    valid_mask = np.ones_like(gt_mask, dtype=bool)
    if ignore_index is not None:
        valid_mask = (gt_mask != ignore_index)

    for c in range(num_classes):
        if c == ignore_index:
            continue

        gt_c = (gt_mask == c) & valid_mask
        pred_c = (pred_mask == c) & valid_mask

        if not np.any(gt_c) and not np.any(pred_c):
            continue

        # Extract boundary pixels within distance d of class boundary
        if dilation_ratio is not None and dilation_width is None:
            gt_bnd_region = class_mask_to_boundary_official(gt_c, dilation_ratio=dilation_ratio)
            pred_bnd_region = class_mask_to_boundary_official(pred_c, dilation_ratio=dilation_ratio)
        else:
            d_w = dilation_width if dilation_width is not None else 2
            gt_bnd_region = class_mask_to_boundary(gt_c, dilation_width=d_w)
            pred_bnd_region = class_mask_to_boundary(pred_c, dilation_width=d_w)

        # Boundary intersection and union per Cheng et al. CVPR 2021 Eq. (1)
        intersection = np.logical_and(gt_bnd_region, pred_bnd_region).sum()
        union = np.logical_or(gt_bnd_region, pred_bnd_region).sum()

        if union > 0:
            biou_per_class[c] = intersection / union
        else:
            biou_per_class[c] = 0.0

    valid_ious = biou_per_class[~np.isnan(biou_per_class)]
    mean_biou = float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0
    return biou_per_class, mean_biou


def compute_boundary_f1_precision_recall(
    gt_boundary: np.ndarray,
    pred_boundary: np.ndarray,
    tolerance: int = 2,
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    """
    Compute Boundary Precision, Recall, and Boundary F1 score (BF-score)
    with a spatial slack/tolerance width.

    A predicted boundary pixel is a True Positive if it is within ``tolerance``
    pixels of any ground-truth boundary pixel.
    A ground-truth boundary pixel is recalled if it is within ``tolerance``
    pixels of any predicted boundary pixel.

    Args:
        gt_boundary: (H, W) boolean or 0/1 array of ground truth boundaries
        pred_boundary: (H, W) boolean or 0/1 array of predicted boundaries
        tolerance: spatial tolerance in pixels (default: 2)
        eps: numerical stability epsilon

    Returns:
        precision: float
        recall: float
        f1: float
    """
    gt_bnd = (gt_boundary > 0).astype(np.uint8)
    pred_bnd = (pred_boundary > 0).astype(np.uint8)

    n_gt = np.sum(gt_bnd)
    n_pred = np.sum(pred_bnd)

    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0

    kernel_size = 2 * tolerance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    gt_dilated = cv2.dilate(gt_bnd, kernel)
    pred_dilated = cv2.dilate(pred_bnd, kernel)

    # True positives:
    # TP_pred: predicted edge pixels that lie within tolerance of a true edge
    tp_pred = np.sum((pred_bnd > 0) & (gt_dilated > 0))
    # TP_gt: true edge pixels that lie within tolerance of a predicted edge
    tp_gt = np.sum((gt_bnd > 0) & (pred_dilated > 0))

    precision = float(tp_pred / (n_pred + eps))
    recall = float(tp_gt / (n_gt + eps))

    if precision + recall > 0:
        f1 = float(2.0 * precision * recall / (precision + recall + eps))
    else:
        f1 = 0.0

    return precision, recall, f1


DEFAULT_DISTANCE_BUCKETS = [
    (0, 1, "0-1 px"),
    (2, 4, "2-4 px"),
    (5, 8, "5-8 px"),
    (9, 16, "9-16 px"),
    (17, 32, "17-32 px"),
    (33, float("inf"), "33+ px"),
]


def compute_boundary_distance_accuracy(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    gt_boundary: Optional[np.ndarray] = None,
    distance_buckets: Optional[List[Tuple[float, float, str]]] = None,
    ignore_index: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Computes segmentation accuracy as a function of Euclidean pixel distance
    from the nearest ground-truth boundary.

    Buckets pixels into distance bands (e.g. 0-1px, 2-4px, 5-8px, etc.) and
    reports classification accuracy and pixel proportions per bucket.

    Args:
        gt_mask: (H, W) ground truth semantic mask
        pred_mask: (H, W) predicted semantic mask
        gt_boundary: optional precomputed binary boundary mask. If None, derived on-the-fly.
        distance_buckets: list of tuples (min_dist, max_dist, label)
        ignore_index: class index to exclude

    Returns:
        results: Dict mapping bucket label to dict with 'correct', 'total', 'accuracy', 'pixel_pct'
    """
    if distance_buckets is None:
        distance_buckets = DEFAULT_DISTANCE_BUCKETS

    valid_mask = (gt_mask != ignore_index) if ignore_index is not None else np.ones_like(gt_mask, dtype=bool)

    if gt_boundary is None:
        gt_boundary = mask_to_boundary(gt_mask, connectivity=8, ignore_index=ignore_index)

    gt_bnd_u8 = gt_boundary.astype(np.uint8)

    # Compute Euclidean distance to nearest boundary pixel (where gt_bnd == 1)
    if np.any(gt_bnd_u8):
        dist_map = cv2.distanceTransform(1 - gt_bnd_u8, cv2.DIST_L2, 5)
    else:
        dist_map = np.full_like(gt_mask, 999.0, dtype=np.float32)

    correct_map = (pred_mask == gt_mask) & valid_mask

    results = {}
    total_valid_pixels = np.sum(valid_mask)

    for min_d, max_d, label in distance_buckets:
        if max_d == float("inf"):
            in_bucket = (dist_map >= min_d) & valid_mask
        else:
            in_bucket = (dist_map >= min_d) & (dist_map <= max_d) & valid_mask

        n_bucket = int(np.sum(in_bucket))
        n_correct = int(np.sum(correct_map & in_bucket))
        acc = float(n_correct / n_bucket) if n_bucket > 0 else np.nan
        pct = float((n_bucket / total_valid_pixels) * 100.0) if total_valid_pixels > 0 else 0.0

        results[label] = {
            "correct": n_correct,
            "total": n_bucket,
            "accuracy": acc,
            "pixel_pct": pct,
        }

    return results


# =====================================================================
# Stateful Evaluator Class (Accumulates across dataset batches)
# =====================================================================

class BoundaryEvaluator:
    """
    Stateful boundary evaluator that accumulates boundary statistics across
    all batches/samples during validation or testing.

    Matches the usage pattern of GeoSeg's ``Evaluator`` in ``tools/metric.py``.

    Parameters
    ----------
    num_classes : int
        Number of semantic classes.
    tolerance : int
        Distance tolerance in pixels for boundary F1/Precision/Recall (default: 2).
    dilation_width : int
        Dilation width d in pixels for Boundary IoU (default: 2).
    ignore_index : int or None
        Class index to ignore (e.g. 7 for LoveDA).
    distance_buckets : list of tuples, optional
        Distance ranges for distance-accuracy analysis.
    """

    def __init__(
        self,
        num_classes: int,
        tolerance: int = 2,
        dilation_width: int = 2,
        ignore_index: Optional[int] = None,
        distance_buckets: Optional[List[Tuple[float, float, str]]] = None,
    ):
        self.num_classes = num_classes
        self.tolerance = tolerance
        self.dilation_width = dilation_width
        self.dilation_ratio = 0.02
        self.ignore_index = ignore_index
        self.distance_buckets = (
            distance_buckets if distance_buckets is not None else DEFAULT_DISTANCE_BUCKETS
        )
        self.eps = 1e-8
        self.reset()

    def reset(self):
        """Reset all metric accumulators."""
        # Boundary IoU accumulators (per class) - 2px fixed
        self.biou_intersection = np.zeros(self.num_classes, dtype=np.float64)
        self.biou_union = np.zeros(self.num_classes, dtype=np.float64)
        self.biou_class_present = np.zeros(self.num_classes, dtype=bool)

        # Boundary IoU accumulators (per class) - Official CVPR 2021 ratio
        self.biou_intersection_ratio = np.zeros(self.num_classes, dtype=np.float64)
        self.biou_union_ratio = np.zeros(self.num_classes, dtype=np.float64)

        # Overall boundary F1/P/R accumulators
        self.overall_tp_pred = 0.0
        self.overall_tp_gt = 0.0
        self.overall_n_pred = 0.0
        self.overall_n_gt = 0.0

        # Per-class boundary F1/P/R accumulators
        self.class_tp_pred = np.zeros(self.num_classes, dtype=np.float64)
        self.class_tp_gt = np.zeros(self.num_classes, dtype=np.float64)
        self.class_n_pred = np.zeros(self.num_classes, dtype=np.float64)
        self.class_n_gt = np.zeros(self.num_classes, dtype=np.float64)
        self.class_bnd_present = np.zeros(self.num_classes, dtype=bool)

        # Distance-accuracy accumulators: {label: [correct, total]}
        self.dist_acc_counts = {
            label: [0, 0] for _, _, label in self.distance_buckets
        }

    def add_batch(
        self,
        gt_image: np.ndarray,
        pre_image: np.ndarray,
        pred_boundary_map: Optional[np.ndarray] = None,
    ):
        """
        Add a single sample or batch of predictions and ground-truth masks.

        Args:
            gt_image: (H, W) or (B, H, W) ground-truth integer mask
            pre_image: (H, W) or (B, H, W) predicted integer mask
            pred_boundary_map: optional (H, W) or (B, H, W) explicit boundary logits/probabilities
        """
        if gt_image.ndim == 2:
            self._add_single_image(gt_image, pre_image, pred_boundary_map)
        elif gt_image.ndim == 3:
            b = gt_image.shape[0]
            for i in range(b):
                bnd_i = pred_boundary_map[i] if pred_boundary_map is not None else None
                self._add_single_image(gt_image[i], pre_image[i], bnd_i)
        else:
            raise ValueError(f"Expected 2D or 3D arrays, got shape {gt_image.shape}")

    def _add_single_image(
        self,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        pred_boundary_prob: Optional[np.ndarray] = None,
    ):
        assert gt_mask.shape == pred_mask.shape, (
            f"Shape mismatch: GT {gt_mask.shape} vs Pred {pred_mask.shape}"
        )
        valid = (gt_mask != self.ignore_index) if self.ignore_index is not None else np.ones_like(gt_mask, dtype=bool)

        # 1. Boundary IoU Accumulation
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue

            gt_c = (gt_mask == c) & valid
            pred_c = (pred_mask == c) & valid

            if not np.any(gt_c) and not np.any(pred_c):
                continue

            self.biou_class_present[c] = True

            # 1a. Fixed pixel boundary (2px tolerance mode)
            gt_bnd_reg = class_mask_to_boundary(gt_c, dilation_width=self.dilation_width)
            pred_bnd_reg = class_mask_to_boundary(pred_c, dilation_width=self.dilation_width)
            inter = np.logical_and(gt_bnd_reg, pred_bnd_reg).sum()
            union = np.logical_or(gt_bnd_reg, pred_bnd_reg).sum()
            self.biou_intersection[c] += inter
            self.biou_union[c] += union

            # 1b. Official ratio-based boundary (Cheng et al. CVPR 2021)
            gt_bnd_ratio = class_mask_to_boundary_official(gt_c, dilation_ratio=self.dilation_ratio)
            pred_bnd_ratio = class_mask_to_boundary_official(pred_c, dilation_ratio=self.dilation_ratio)
            inter_ratio = np.logical_and(gt_bnd_ratio, pred_bnd_ratio).sum()
            union_ratio = np.logical_or(gt_bnd_ratio, pred_bnd_ratio).sum()
            self.biou_intersection_ratio[c] += inter_ratio
            self.biou_union_ratio[c] += union_ratio

        # 2. Extract multi-class transition boundaries for overall Boundary F1
        gt_bnd_full = mask_to_boundary(gt_mask, connectivity=8, ignore_index=self.ignore_index)

        if pred_boundary_prob is not None:
            # If explicit boundary probabilities from a boundary head were passed
            pred_bnd_full = (pred_boundary_prob > 0.5)
        else:
            pred_bnd_full = mask_to_boundary(pred_mask, connectivity=8, ignore_index=self.ignore_index)

        gt_u8 = gt_bnd_full.astype(np.uint8)
        pred_u8 = pred_bnd_full.astype(np.uint8)

        n_gt = np.sum(gt_u8)
        n_pred = np.sum(pred_u8)

        self.overall_n_gt += n_gt
        self.overall_n_pred += n_pred

        if n_gt > 0 and n_pred > 0:
            ksize = 2 * self.tolerance + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            gt_dil = cv2.dilate(gt_u8, kernel)
            pred_dil = cv2.dilate(pred_u8, kernel)

            self.overall_tp_pred += np.sum((pred_u8 > 0) & (gt_dil > 0))
            self.overall_tp_gt += np.sum((gt_u8 > 0) & (pred_dil > 0))

        # Per-class boundary F1
        ksize = 2 * self.tolerance + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue

            gt_c_bnd = class_mask_to_boundary((gt_mask == c) & valid, dilation_width=1).astype(np.uint8)
            pred_c_bnd = class_mask_to_boundary((pred_mask == c) & valid, dilation_width=1).astype(np.uint8)

            c_n_gt = np.sum(gt_c_bnd)
            c_n_pred = np.sum(pred_c_bnd)

            if c_n_gt > 0 or c_n_pred > 0:
                self.class_bnd_present[c] = True

            self.class_n_gt[c] += c_n_gt
            self.class_n_pred[c] += c_n_pred

            if c_n_gt > 0 and c_n_pred > 0:
                c_gt_dil = cv2.dilate(gt_c_bnd, kernel)
                c_pred_dil = cv2.dilate(pred_c_bnd, kernel)

                self.class_tp_pred[c] += np.sum((pred_c_bnd > 0) & (c_gt_dil > 0))
                self.class_tp_gt[c] += np.sum((gt_c_bnd > 0) & (c_pred_dil > 0))

        # 3. Distance-Accuracy Accumulation
        if np.any(gt_u8):
            dist_map = cv2.distanceTransform(1 - gt_u8, cv2.DIST_L2, 5)
        else:
            dist_map = np.full_like(gt_mask, 999.0, dtype=np.float32)

        correct_map = (pred_mask == gt_mask) & valid

        for min_d, max_d, label in self.distance_buckets:
            if max_d == float("inf"):
                in_bucket = (dist_map >= min_d) & valid
            else:
                in_bucket = (dist_map >= min_d) & (dist_map <= max_d) & valid

            self.dist_acc_counts[label][0] += int(np.sum(correct_map & in_bucket))
            self.dist_acc_counts[label][1] += int(np.sum(in_bucket))

    # ------------------------------------------------------------------
    # Metric Queries
    # ------------------------------------------------------------------

    def boundary_iou(self) -> np.ndarray:
        """Return per-class Boundary IoU array of shape (num_classes,)."""
        biou = np.full(self.num_classes, np.nan, dtype=np.float64)
        for c in range(self.num_classes):
            if self.biou_class_present[c]:
                if self.biou_union[c] > 0:
                    biou[c] = self.biou_intersection[c] / (self.biou_union[c] + self.eps)
                else:
                    biou[c] = 0.0
        return biou

    def mean_boundary_iou(self) -> float:
        """Return mean Boundary IoU across classes present in the dataset."""
        bious = self.boundary_iou()
        valid = bious[~np.isnan(bious)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    def boundary_iou_ratio(self) -> np.ndarray:
        """Return per-class Boundary IoU array of shape (num_classes,) [Official CVPR 2021 ratio mode]."""
        biou = np.full(self.num_classes, np.nan, dtype=np.float64)
        for c in range(self.num_classes):
            if self.biou_class_present[c]:
                if self.biou_union_ratio[c] > 0:
                    biou[c] = self.biou_intersection_ratio[c] / (self.biou_union_ratio[c] + self.eps)
                else:
                    biou[c] = 0.0
        return biou

    def mean_boundary_iou_ratio(self) -> float:
        """Return mean Boundary IoU across classes [Official CVPR 2021 ratio mode]."""
        bious = self.boundary_iou_ratio()
        valid = bious[~np.isnan(bious)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    def overall_boundary_metrics(self) -> Tuple[float, float, float]:
        """Return (overall_precision, overall_recall, overall_f1)."""
        prec = float(self.overall_tp_pred / (self.overall_n_pred + self.eps))
        rec = float(self.overall_tp_gt / (self.overall_n_gt + self.eps))
        f1 = float(2.0 * prec * rec / (prec + rec + self.eps)) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    def class_boundary_f1(self) -> np.ndarray:
        """Return per-class Boundary F1 array of shape (num_classes,)."""
        bf1 = np.full(self.num_classes, np.nan, dtype=np.float64)
        for c in range(self.num_classes):
            if self.class_bnd_present[c]:
                prec = self.class_tp_pred[c] / (self.class_n_pred[c] + self.eps)
                rec = self.class_tp_gt[c] / (self.class_n_gt[c] + self.eps)
                if prec + rec > 0:
                    bf1[c] = 2.0 * prec * rec / (prec + rec + self.eps)
                else:
                    bf1[c] = 0.0
        return bf1

    def class_boundary_precision(self) -> np.ndarray:
        """Return per-class Boundary Precision array of shape (num_classes,)."""
        prec = np.full(self.num_classes, np.nan, dtype=np.float64)
        for c in range(self.num_classes):
            if self.class_bnd_present[c]:
                prec[c] = self.class_tp_pred[c] / (self.class_n_pred[c] + self.eps)
        return prec

    def class_boundary_recall(self) -> np.ndarray:
        """Return per-class Boundary Recall array of shape (num_classes,)."""
        rec = np.full(self.num_classes, np.nan, dtype=np.float64)
        for c in range(self.num_classes):
            if self.class_bnd_present[c]:
                rec[c] = self.class_tp_gt[c] / (self.class_n_gt[c] + self.eps)
        return rec

    def distance_accuracy(self) -> Dict[str, Dict[str, float]]:
        """Return distance-bucket accuracy statistics."""
        total_pixels = sum(counts[1] for counts in self.dist_acc_counts.values())
        results = {}
        for label, (n_corr, n_tot) in self.dist_acc_counts.items():
            acc = float(n_corr / n_tot) if n_tot > 0 else np.nan
            pct = float((n_tot / total_pixels) * 100.0) if total_pixels > 0 else 0.0
            results[label] = {
                "correct": n_corr,
                "total": n_tot,
                "accuracy": acc,
                "pixel_pct": pct,
            }
        return results

    def summary(self, class_names: Optional[List[str]] = None) -> Dict[str, Union[float, np.ndarray, dict]]:
        """Return a structured summary of all boundary metrics."""
        prec, rec, f1 = self.overall_boundary_metrics()
        biou_per_c_2px = self.boundary_iou()
        m_biou_2px = self.mean_boundary_iou()
        biou_per_c_ratio = self.boundary_iou_ratio()
        m_biou_ratio = self.mean_boundary_iou_ratio()

        bf1_per_c = self.class_boundary_f1()
        bprec_per_c = self.class_boundary_precision()
        brec_per_c = self.class_boundary_recall()
        valid_bf1 = bf1_per_c[~np.isnan(bf1_per_c)]
        m_bf1 = float(np.mean(valid_bf1)) if len(valid_bf1) > 0 else 0.0
        dist_acc = self.distance_accuracy()

        return {
            "overall_precision": prec,
            "overall_recall": rec,
            "overall_f1": f1,
            "mean_boundary_iou": m_biou_2px,
            "mean_boundary_iou_2px": m_biou_2px,
            "mean_boundary_iou_ratio": m_biou_ratio,
            "boundary_iou_per_class": biou_per_c_2px,
            "boundary_iou_per_class_2px": biou_per_c_2px,
            "boundary_iou_per_class_ratio": biou_per_c_ratio,
            "mean_boundary_f1": m_bf1,
            "boundary_f1_per_class": bf1_per_c,
            "boundary_precision_per_class": bprec_per_c,
            "boundary_recall_per_class": brec_per_c,
            "distance_accuracy": dist_acc,
        }
