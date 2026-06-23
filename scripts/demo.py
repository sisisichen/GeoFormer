#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a fast, checkpoint-free demo on the bundled RGB-D sample data.

This is not a replacement for GeoFormerX evaluation with a trained checkpoint.
It is a repository smoke test that verifies the data layout, palette handling,
metrics, prediction writing, and qualitative output paths without requiring
SAM weights or a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pavement_config import CLASS_NAMES, CLASS_RGB_VALUES, IGNORE_INDEX  # noqa: E402


PALETTE = np.asarray(CLASS_RGB_VALUES, dtype=np.uint8)
NUM_CLASSES = len(CLASS_RGB_VALUES)


def _find_depth_path(data_root: Path, split: str, image_name: str) -> Path:
    depth_name = image_name.replace("DL2", "DL3", 1)
    candidates = [
        data_root / "3Ddate" / split / "image" / depth_name,
        data_root / "3Ddate" / split / depth_name,
        data_root / "3Ddate" / depth_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Depth image for {image_name} was not found under {data_root / '3Ddate'}")


def _rgb_mask_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    flat = mask_rgb.reshape(-1, 3).astype(np.int32)
    palette = PALETTE.astype(np.int32)
    dist = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    label = dist.argmin(axis=1).astype(np.uint8).reshape(mask_rgb.shape[:2])
    return label


def _predict_demo(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Deterministic RGB-D heuristic used only for the bundled synthetic demo."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    d = depth.astype(np.int16)
    gray = ((r + g + b) / 3.0).astype(np.float32)

    pred = np.zeros(rgb.shape[:2], dtype=np.uint8)

    pred[(b > 155) & (r < 120) & (g < 135)] = 5  # marking
    pred[(g > 135) & (b > 135) & (r < 120)] = 4  # patch
    pred[(r > 165) & (g > 70) & (g < 150) & (b < 100)] = 7  # manhole
    pred[(r > 105) & (b > 115) & (g < 105)] = 3  # seal
    pred[(r > 85) & (r < 155) & (g > 40) & (g < 105) & (b < 80)] = 6  # joint
    pred[(gray < 92) & (d < 95)] = 2  # pothole
    pred[(gray < 48) & (d < 95)] = 1  # crack
    return pred


def _confusion_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    valid = (gt != IGNORE_INDEX) & (gt >= 0) & (gt < NUM_CLASSES)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (gt[valid].astype(np.int64), pred[valid].astype(np.int64)), 1)
    return cm


def _metrics_from_cm(cm: np.ndarray) -> Tuple[Dict[str, float], Iterable[Dict[str, float]]]:
    rows = []
    dice_values = []
    iou_values = []
    for class_id, name in enumerate(CLASS_NAMES):
        tp = float(cm[class_id, class_id])
        fp = float(cm[:, class_id].sum() - tp)
        fn = float(cm[class_id, :].sum() - tp)
        denom_iou = tp + fp + fn
        denom_dice = 2.0 * tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else float("nan")
        dice = 2.0 * tp / denom_dice if denom_dice > 0 else float("nan")
        support = int(cm[class_id, :].sum())
        rows.append({"class_id": class_id, "class_name": name, "support": support, "iou": iou, "dice": dice})
        if class_id != 0 and support > 0:
            iou_values.append(iou)
            dice_values.append(dice)

    overall = {
        "pixel_accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "mean_foreground_iou": float(np.nanmean(iou_values)) if iou_values else float("nan"),
        "mean_foreground_dice": float(np.nanmean(dice_values)) if dice_values else float("nan"),
    }
    return overall, rows


def _save_overlay(rgb: np.ndarray, label: np.ndarray, out_path: Path, alpha: float = 0.38) -> None:
    color = PALETTE[label]
    overlay = np.clip((1.0 - alpha) * rgb.astype(np.float32) + alpha * color.astype(np.float32), 0, 255)
    Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(out_path)


def run_demo(data_root: Path, split: str, out_dir: Path) -> None:
    image_dir = data_root / split / "image"
    label_dir = data_root / split / "label"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Example data not found at {data_root}. Run: python scripts/create_example_dataset.py --overwrite"
        )

    pred_dir = out_dir / "predictions"
    overlay_dir = out_dir / "overlays"
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    processed = []
    for image_path in sorted(image_dir.glob("*.png")):
        depth_path = _find_depth_path(data_root, split, image_path.name)
        label_path = label_dir / f"{image_path.stem}.bmp"
        if not label_path.exists():
            raise FileNotFoundError(f"Label missing: {label_path}")

        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        depth = np.asarray(Image.open(depth_path).convert("L"), dtype=np.uint8)
        gt = _rgb_mask_to_label(np.asarray(Image.open(label_path).convert("RGB"), dtype=np.uint8))
        pred = _predict_demo(rgb, depth)

        cm_total += _confusion_matrix(gt, pred)
        Image.fromarray(PALETTE[pred], mode="RGB").save(pred_dir / f"{image_path.stem}_pred.png")
        _save_overlay(rgb, pred, overlay_dir / f"{image_path.stem}_overlay.png")
        processed.append(image_path.name)

    overall, rows = _metrics_from_cm(cm_total)
    summary = {"data_root": str(data_root), "split": split, "num_images": len(processed), "images": processed, **overall}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "support", "iou", "dice"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"Predictions: {pred_dir}")
    print(f"Overlays: {overlay_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GeoFormerX bundled smoke demo.")
    parser.add_argument("--data", type=Path, default=ROOT / "examples" / "pavement_rgbd_small")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_demo(args.data, args.split, args.out)


if __name__ == "__main__":
    main()
