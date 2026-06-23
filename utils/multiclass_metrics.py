# -*- coding: utf-8 -*-
"""Multi-class segmentation metrics (pixel-wise) + some boundary/surface metrics.

We report classic metrics (Dice/mIoU/F1/F2) and more publication-friendly ones:
  - Cohen's kappa (agreement)
  - Multi-class MCC (Gorodkin)
  - Boundary F1 (BFScore, tolerance via dilation)
  - NSD (Normalized Surface Dice) and HD95 (95% Hausdorff)

All metrics support an ignore_index mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def confusion_matrix(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> np.ndarray:
    """Compute confusion matrix where rows=GT and cols=Pred.

    This matches the convention used by most segmentation papers and by the
    common segmentation baseline code (fast_hist_ignore).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have same shape, got {pred.shape} vs {gt.shape}")

    # NOTE: For fair comparison we follow the common fast-hist implementation
    # (fast_hist_ignore) semantics:
    #   - ignore pixels where GT == ignore_index
    #   - ignore pixels where GT is out of [0, num_classes)
    #   - rows = GT, cols = Pred
    gt = gt.astype(np.int64, copy=False)
    pred = pred.astype(np.int64, copy=False)

    if ignore_index is None:
        k = (gt >= 0) & (gt < int(num_classes))
    else:
        k = (gt != int(ignore_index)) & (gt >= 0) & (gt < int(num_classes))

    gt_k = gt[k].astype(np.int64, copy=False)
    pred_k = pred[k].astype(np.int64, copy=False)

    # Pred should already be in-range (argmax), but clip for safety.
    pred_k = np.clip(pred_k, 0, int(num_classes) - 1)

    cm = np.bincount(
        int(num_classes) * gt_k + pred_k,
        minlength=int(num_classes) ** 2,
    ).reshape((int(num_classes), int(num_classes)))
    return cm


def boundary_f1_per_class(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None,
    radius: int = 2,
) -> np.ndarray:
    """Boundary F1 (BFScore) for each class (one-vs-rest).

    Args:
        gt/pred: (H,W) int arrays.
        num_classes: total classes.
        ignore_index: ignored pixels (e.g., 255).
        radius: boundary matching tolerance (pixels).

    Returns:
        (num_classes,) float64, NaN when cv2 is unavailable.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have same shape, got {pred.shape} vs {gt.shape}")

    valid = np.ones_like(gt, dtype=bool)
    if ignore_index is not None:
        valid = (gt != int(ignore_index))

    out = np.full((num_classes,), np.nan, dtype=np.float64)
    for c in range(int(num_classes)):
        gt_c = ((gt == c) & valid).astype(np.uint8)
        pred_c = ((pred == c) & valid).astype(np.uint8)
        out[c] = boundary_f1_score(pred_c, gt_c, radius=int(radius))
    return out


@dataclass
class PerClassMetrics:
    dice: np.ndarray
    iou: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    specificity: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    support: np.ndarray


def per_class_metrics_from_cm(cm: np.ndarray, eps: float = 1e-6) -> PerClassMetrics:
    """Compute one-vs-rest metrics for each class from confusion matrix."""
    num_classes = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    tn = cm.sum().astype(np.float64) - (tp + fp + fn)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    f1 = (2 * precision * recall) / (precision + recall + eps)
    beta2 = 2.0
    f2 = (1 + beta2 ** 2) * precision * recall / ((beta2 ** 2) * precision + recall + eps)

    support = cm.sum(axis=1).astype(np.float64)
    return PerClassMetrics(
        dice=dice,
        iou=iou,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        f2=f2,
        support=support,
    )


def overall_metrics_from_cm(cm: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    """Compute overall (multi-class) metrics from confusion matrix."""
    total = float(cm.sum())
    if total <= 0:
        return {
            "acc": float("nan"),
            "kappa": float("nan"),
            "mcc": float("nan"),
        }

    diag = np.diag(cm).astype(np.float64)
    acc = float(diag.sum() / (total + eps))

    # Cohen's kappa
    row = cm.sum(axis=1).astype(np.float64)
    col = cm.sum(axis=0).astype(np.float64)
    pe = float((row * col).sum() / ((total + eps) ** 2))
    kappa = float((acc - pe) / (1.0 - pe + eps))

    # Multi-class MCC (Gorodkin 2004)
    c = float(diag.sum())
    s = total
    sum_pk_tk = float((row * col).sum())
    denom = np.sqrt((s ** 2 - float((col ** 2).sum())) * (s ** 2 - float((row ** 2).sum())) + eps)
    mcc = float((c * s - sum_pk_tk) / (denom + eps))

    return {"acc": acc, "kappa": kappa, "mcc": mcc}


def freq_weighted_iou(per_class_iou: np.ndarray, support: np.ndarray, eps: float = 1e-6) -> float:
    s = float(support.sum())
    if s <= 0:
        return float("nan")
    return float((per_class_iou * support).sum() / (s + eps))


def mean_excluding_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanmean(x))


def boundary_f1_score(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    radius: int = 2,
    eps: float = 1e-6,
) -> float:
    """Boundary F1 (BFScore) with tolerance using dilation.

    Implementation:
      - boundary = mask XOR erode(mask)
      - match boundary pixels if within `radius` pixels of the other boundary
    """
    pred = (pred_bin > 0).astype(np.uint8)
    gt = (gt_bin > 0).astype(np.uint8)
    if pred.shape != gt.shape:
        raise ValueError("pred_bin and gt_bin must have same shape")

    try:
        import cv2
    except Exception:
        # If cv2 is not available, fallback to NaN
        return float("nan")

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 and gt.sum() > 0:
        return 0.0
    if pred.sum() > 0 and gt.sum() == 0:
        return 0.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    pred_er = cv2.erode(pred, np.ones((3, 3), np.uint8), iterations=1)
    gt_er = cv2.erode(gt, np.ones((3, 3), np.uint8), iterations=1)
    pred_b = (pred ^ pred_er).astype(np.uint8)
    gt_b = (gt ^ gt_er).astype(np.uint8)

    pred_b_d = cv2.dilate(pred_b, k, iterations=1)
    gt_b_d = cv2.dilate(gt_b, k, iterations=1)

    # precision: how many predicted boundary pixels fall near GT boundary
    prec = float((pred_b * gt_b_d).sum() / (pred_b.sum() + eps))
    rec = float((gt_b * pred_b_d).sum() / (gt_b.sum() + eps))
    f1 = (2 * prec * rec) / (prec + rec + eps)
    return float(f1)


def surface_metrics_binary(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    nsd_tolerance: int = 2,
) -> Tuple[float, float]:
    """Compute (NSD, HD95) for a single binary mask pair.

    Requires MONAI (already used in this repo).
    Returns NaN when undefined.
    """
    try:
        import torch
        from monai.metrics import SurfaceDiceMetric
        from utils.HausdorffDistance import HausdorffDistanceMetric
    except Exception:
        return float("nan"), float("nan")

    pred = torch.from_numpy((pred_bin > 0).astype(np.uint8))[None, None].float()
    gt = torch.from_numpy((gt_bin > 0).astype(np.uint8))[None, None].float()

    # NSD
    try:
        nsd_metric = SurfaceDiceMetric(class_thresholds=[int(nsd_tolerance)])
        nsd = float(nsd_metric(pred, gt).squeeze().cpu().numpy())
    except Exception:
        nsd = float("nan")

    # HD95
    try:
        hd_metric = HausdorffDistanceMetric(percentile=95.0)
        hd = float(hd_metric(pred, gt).squeeze().cpu().numpy())
    except Exception:
        hd = float("nan")

    return nsd, hd
