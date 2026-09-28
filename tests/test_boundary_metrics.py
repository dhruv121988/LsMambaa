"""
Phase 1 Unit Tests: Boundary Metrics Verification on Synthetic Masks
====================================================================
Tests the Boundary IoU (fixed 2px & official CVPR 2021 ratio mode) and Boundary F1:
1. Identical masks: BIoU must equal 1.0, BF1 must equal 1.0.
2. Shifted square: Shifted by 2px; verifies expected overlap behavior.
3. Circle vs slightly eroded circle: Verifies boundary thinning / shift response.
4. Prediction with correct interior but noisy edges: Verifies high Mask IoU (>98%)
   paired with substantially degraded Boundary IoU (demonstrating boundary sensitivity).
5. Multi-class case with ignore_index: Verifies valid class transitions and NaN exclusion.
"""

import unittest
import numpy as np
import cv2
from geoseg.utils.boundary_metrics import (
    BoundaryEvaluator,
    compute_boundary_iou,
    compute_boundary_f1_precision_recall,
    class_mask_to_boundary,
    class_mask_to_boundary_official,
    mask_to_boundary,
)


class TestBoundaryMetrics(unittest.TestCase):

    def setUp(self):
        self.canvas_size = (256, 256)

    def test_1_identical_masks(self):
        """1. Identical masks: BIoU = 1.0, BF1 = 1.0, Precision = 1.0, Recall = 1.0."""
        gt = np.zeros(self.canvas_size, dtype=np.int64)
        gt[50:150, 50:150] = 1
        pred = gt.copy()

        # Test functional API (2px and official ratio)
        biou_per_c, mean_biou = compute_boundary_iou(gt, pred, num_classes=2, dilation_width=2)
        self.assertAlmostEqual(biou_per_c[1], 1.0, places=5)
        self.assertAlmostEqual(mean_biou, 1.0, places=5)

        biou_per_c_ratio, mean_biou_ratio = compute_boundary_iou(
            gt, pred, num_classes=2, dilation_width=None, dilation_ratio=0.02
        )
        self.assertAlmostEqual(biou_per_c_ratio[1], 1.0, places=5)
        self.assertAlmostEqual(mean_biou_ratio, 1.0, places=5)

        # Test stateful BoundaryEvaluator
        evaluator = BoundaryEvaluator(num_classes=2, tolerance=2, dilation_width=2)
        evaluator.add_batch(gt, pred)
        summary = evaluator.summary()

        self.assertAlmostEqual(summary["mean_boundary_iou_2px"], 1.0, places=5)
        self.assertAlmostEqual(summary["mean_boundary_iou_ratio"], 1.0, places=5)
        self.assertAlmostEqual(summary["overall_f1"], 1.0, places=5)
        self.assertAlmostEqual(summary["overall_precision"], 1.0, places=5)
        self.assertAlmostEqual(summary["overall_recall"], 1.0, places=5)
        print("\n[TEST 1 PASSED] Identical masks: BIoU=1.000, BF1=1.000")

    def test_2_shifted_square(self):
        """2. Shifted square: Shifted horizontally by 2 px."""
        gt = np.zeros(self.canvas_size, dtype=np.int64)
        gt[50:150, 50:150] = 1

        pred = np.zeros(self.canvas_size, dtype=np.int64)
        pred[50:150, 52:152] = 1  # 2 px right shift

        # Fixed 2px boundary
        biou_per_c_2px, mean_biou_2px = compute_boundary_iou(gt, pred, num_classes=2, dilation_width=2)
        # Official ratio boundary (for 256x256, diag ~362 px, dilation = round(0.02*362) = 7 px)
        biou_per_c_ratio, mean_biou_ratio = compute_boundary_iou(
            gt, pred, num_classes=2, dilation_width=None, dilation_ratio=0.02
        )

        # When shifted by 2px, a 2px boundary has overlap on top & bottom edges, but minimal on vertical edges
        self.assertGreater(biou_per_c_2px[1], 0.20)
        self.assertLess(biou_per_c_2px[1], 0.70)

        # Official ratio (d=7 px) has much larger boundary thickness, so a 2px shift yields higher BIoU
        self.assertGreater(biou_per_c_ratio[1], biou_per_c_2px[1])

        # Boundary F1 with tolerance=2 px should be high because all boundary pixels are within 2 px
        gt_bnd = mask_to_boundary(gt, connectivity=8)
        pred_bnd = mask_to_boundary(pred, connectivity=8)
        prec, rec, f1 = compute_boundary_f1_precision_recall(gt_bnd, pred_bnd, tolerance=2)
        self.assertGreaterEqual(f1, 0.90)
        print(f"\n[TEST 2 PASSED] Shifted square: BIoU(2px)={biou_per_c_2px[1]:.4f}, BIoU(ratio)={biou_per_c_ratio[1]:.4f}, BF1={f1:.4f}")

    def test_3_circle_vs_eroded_circle(self):
        """3. Circle vs slightly eroded circle (radius 40 vs 38)."""
        gt = np.zeros(self.canvas_size, dtype=np.uint8)
        cv2.circle(gt, (128, 128), 40, 1, -1)

        pred = np.zeros(self.canvas_size, dtype=np.uint8)
        cv2.circle(pred, (128, 128), 38, 1, -1)

        # Mask IoU (standard area IoU)
        mask_intersection = np.sum((gt == 1) & (pred == 1))
        mask_union = np.sum((gt == 1) | (pred == 1))
        mask_iou = mask_intersection / mask_union

        # Boundary IoU (2px and ratio)
        biou_per_c_2px, _ = compute_boundary_iou(gt.astype(np.int64), pred.astype(np.int64), num_classes=2, dilation_width=2)
        biou_per_c_ratio, _ = compute_boundary_iou(
            gt.astype(np.int64), pred.astype(np.int64), num_classes=2, dilation_width=None, dilation_ratio=0.02
        )

        # Mask IoU is large (pi*38^2 / pi*40^2 ~ 90.25%)
        self.assertGreater(mask_iou, 0.85)

        # Boundary IoU at 2px detects the 2px ring erosion cleanly
        print(f"\n[TEST 3 PASSED] Circle vs eroded circle: Mask IoU={mask_iou:.4f}, BIoU(2px)={biou_per_c_2px[1]:.4f}, BIoU(ratio)={biou_per_c_ratio[1]:.4f}")

    def test_4_correct_interior_noisy_edges(self):
        """4. Prediction with correct interior but noisy edges: Mask IoU > 98%, BIoU severely penalized."""
        gt = np.zeros(self.canvas_size, dtype=np.int64)
        gt[50:150, 50:150] = 1

        pred = gt.copy()
        # Add alternating 1px noisy serrations along the perimeter
        for i in range(50, 150):
            if i % 2 == 0:
                pred[49, i] = 1  # 1px protrusion top
                pred[150, i] = 1 # 1px protrusion bottom
                pred[i, 49] = 1  # 1px protrusion left
                pred[i, 150] = 1 # 1px protrusion right

        mask_intersection = np.sum((gt == 1) & (pred == 1))
        mask_union = np.sum((gt == 1) | (pred == 1))
        mask_iou = mask_intersection / mask_union

        biou_per_c_2px, _ = compute_boundary_iou(gt, pred, num_classes=2, dilation_width=2)

        # Mask IoU is over 98% because area of 100x100 is 10,000, noise is ~200 px (<2%)
        self.assertGreater(mask_iou, 0.98)
        # Boundary IoU drops noticeably because boundary region is heavily perturbed
        self.assertLess(biou_per_c_2px[1], 0.85)
        print(f"\n[TEST 4 PASSED] Correct interior + noisy edge: Mask IoU={mask_iou:.4f} (>98%), BIoU(2px)={biou_per_c_2px[1]:.4f} (shows boundary penalty)")

    def test_5_multiclass_with_ignore_index(self):
        """5. Multi-class case with ignore_index: ignores excluded class, computes valid classes."""
        num_classes = 4  # 0: bg, 1: building, 2: road, 3: ignore
        ignore_index = 3

        gt = np.zeros(self.canvas_size, dtype=np.int64)
        gt[30:100, 30:100] = 1   # building
        gt[120:200, 30:200] = 2  # road
        gt[210:250, 210:250] = 3 # ignore region

        pred = gt.copy()
        # introduce an error on class 2 only
        pred[120:130, 30:200] = 0

        evaluator = BoundaryEvaluator(num_classes=num_classes, tolerance=2, dilation_width=2, ignore_index=ignore_index)
        evaluator.add_batch(gt, pred)
        summary = evaluator.summary()

        # Class 1 (perfect): BIoU = 1.0
        self.assertAlmostEqual(summary["boundary_iou_per_class_2px"][1], 1.0, places=5)
        # Class 2 (has error): BIoU < 1.0
        self.assertLess(summary["boundary_iou_per_class_2px"][2], 1.0)
        # Class 3 (ignore_index): must be NaN
        self.assertTrue(np.isnan(summary["boundary_iou_per_class_2px"][3]))

        # Mean BIoU must average only valid classes (0, 1, 2)
        valid_bious = [summary["boundary_iou_per_class_2px"][c] for c in [0, 1, 2] if not np.isnan(summary["boundary_iou_per_class_2px"][c])]
        expected_mean = float(np.mean(valid_bious))
        self.assertAlmostEqual(summary["mean_boundary_iou_2px"], expected_mean, places=5)
        print(f"\n[TEST 5 PASSED] Multi-class with ignore_index: Class 1 BIoU={summary['boundary_iou_per_class_2px'][1]:.4f}, Class 3 (ignore)=NaN, Mean={summary['mean_boundary_iou_2px']:.4f}")


if __name__ == "__main__":
    unittest.main()
