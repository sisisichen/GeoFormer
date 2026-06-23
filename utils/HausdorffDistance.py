# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from monai.metrics.utils import (
    do_metric_reduction,
    get_mask_edges,
    get_surface_distance,
    ignore_background,
    prepare_spacing,
)
from monai.utils import MetricReduction, convert_data_type
from monai.metrics.metric import CumulativeIterationMetric


class HausdorffDistanceMetric(CumulativeIterationMetric):
    """
    Compute Hausdorff Distance between two tensors.

    Notes for robustness in sparse-defect segmentation:
    - When GT has no foreground, HD is undefined -> return NaN.
    - When Pred has no foreground but GT has foreground, surface distance is undefined in classic formulation.
      We also return NaN to avoid arbitrary large constants and to prevent crashes.
      (You can change this to a large finite fallback if you prefer.)
    """

    def __init__(
        self,
        include_background: bool = False,
        distance_metric: str = "euclidean",
        percentile: float | None = None,
        directed: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.percentile = percentile
        self.directed = directed
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")

        return compute_hausdorff_distance(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            percentile=self.percentile,
            directed=self.directed,
            spacing=kwargs.get("spacing"),
        )

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_hausdorff_distance(
    y_pred: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    include_background: bool = False,
    distance_metric: str = "euclidean",
    percentile: float | None = None,
    directed: bool = False,
    spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[int | float]] | None = None,
) -> torch.Tensor:
    """
    Compute Hausdorff distance (percentile HD if percentile is not None).

    Robustness rules:
    - If GT has no foreground -> NaN (undefined).
    - If Pred has no foreground -> NaN (undefined) (even if GT has foreground).
    - Otherwise compute standard surface-distance-based Hausdorff.
    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
    y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    batch_size, n_class = y_pred.shape[:2]
    hd = np.empty((batch_size, n_class), dtype=np.float32)

    img_dim = y_pred.ndim - 2
    spacing_list = prepare_spacing(spacing=spacing, batch_size=batch_size, img_dim=img_dim)

    for b, c in np.ndindex(batch_size, n_class):
        edges_pred, edges_gt = get_mask_edges(y_pred[b, c], y[b, c])

        if not np.any(edges_gt):
            # GT empty => HD undefined
            warnings.warn(f"the ground truth of class {c} is all 0, HD is undefined -> NaN.")
            hd[b, c] = np.nan
            continue

        if not np.any(edges_pred):
            # Pred empty => HD undefined (avoid arbitrary large fallback)
            hd[b, c] = np.nan
            continue

        # Normal case
        distance_1 = compute_percent_hausdorff_distance(
            edges_pred, edges_gt, distance_metric, percentile, spacing_list[b]
        )
        if directed:
            hd[b, c] = distance_1
        else:
            distance_2 = compute_percent_hausdorff_distance(
                edges_gt, edges_pred, distance_metric, percentile, spacing_list[b]
            )
            # if any side is nan, max(nan, x) would be nan in numpy; handle safely:
            if np.isnan(distance_1) and np.isnan(distance_2):
                hd[b, c] = np.nan
            elif np.isnan(distance_1):
                hd[b, c] = distance_2
            elif np.isnan(distance_2):
                hd[b, c] = distance_1
            else:
                hd[b, c] = max(distance_1, distance_2)

    return convert_data_type(hd, output_type=torch.Tensor, device=y_pred.device, dtype=torch.float)[0]


def compute_percent_hausdorff_distance(
    edges_pred: np.ndarray,
    edges_gt: np.ndarray,
    distance_metric: str = "euclidean",
    percentile: float | None = None,
    spacing: int | float | np.ndarray | Sequence[int | float] | None = None,
) -> float:
    """
    Compute directed Hausdorff distance based on surface distances.
    If percentile is set, compute HD@percentile (e.g., HD95).
    """
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric=distance_metric, spacing=spacing)

    # if empty, undefined
    if surface_distance.shape == (0,):
        return np.nan

    if percentile is None:
        return float(surface_distance.max())

    if 0 <= percentile <= 100:
        return float(np.percentile(surface_distance, percentile))

    raise ValueError(f"percentile should be a value between 0 and 100, get {percentile}.")
