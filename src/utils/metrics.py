"""Clinical Evaluation Metrics for 3D BraTS Segmentation.

Computes:
- Dice Similarity Coefficient (DSC) for Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET)
- 95th Percentile Hausdorff Distance (HD95) for boundary precision
"""

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric


class BraTSMetrics:
    """Manages computation and accumulation of Dice and HD95 metrics over validation/test batches."""

    def __init__(self, include_background: bool = True):
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="mean_batch",
            get_not_nans=False,
        )
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=include_background,
            percentile=95,
            reduction="mean_batch",
            get_not_nans=False,
        )

    def reset(self):
        self.dice_metric.reset()
        self.hd95_metric.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor, compute_hd95: bool = False):
        """Update metrics with batch predictions and ground truth.

        Args:
            y_pred: Binarized predictions [B, 3, H, W, D]
            y_true: Ground truth binary masks [B, 3, H, W, D]
            compute_hd95: Whether to compute HD95 (slower, recommended for final eval)
        """
        self.dice_metric(y_pred=y_pred, y=y_true)
        if compute_hd95:
            self.hd95_metric(y_pred=y_pred, y=y_true)

    def aggregate(self, compute_hd95: bool = False):
        """Computes mean metrics across accumulated samples."""
        dice_scores = self.dice_metric.aggregate()
        results = {
            "dice_wt": float(dice_scores[0].item()),
            "dice_tc": float(dice_scores[1].item()),
            "dice_et": float(dice_scores[2].item()),
            "dice_mean": float(dice_scores.mean().item()),
        }

        if compute_hd95:
            try:
                hd95_scores = self.hd95_metric.aggregate()
                results.update(
                    {
                        "hd95_wt": float(hd95_scores[0].item()),
                        "hd95_tc": float(hd95_scores[1].item()),
                        "hd95_et": float(hd95_scores[2].item()),
                        "hd95_mean": float(hd95_scores.mean().item()),
                    }
                )
            except Exception as e:
                print(f"[Metrics] Warning: Could not compute HD95: {e}")

        return results
