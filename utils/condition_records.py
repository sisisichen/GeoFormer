"""Machine-readable pavement condition records derived from semantic maps."""

from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence

import numpy as np

from pavement_config import CLASS_NAMES, FG_CLASS_IDS, IGNORE_INDEX


CONDITION_RECORD_SCHEMA = "GeoFormerX.condition_record.v1"


def build_condition_record(
    prediction: np.ndarray,
    image_id: str,
    model_id: str,
    *,
    valid_mask: Optional[np.ndarray] = None,
    semantic_map: Optional[str] = None,
    checkpoint: Optional[str] = None,
    class_names: Sequence[str] = CLASS_NAMES,
    foreground_class_ids: Sequence[int] = FG_CLASS_IDS,
    ignore_index: int = IGNORE_INDEX,
) -> dict:
    """Summarize one semantic map using the manuscript's class-indexed fields.

    Relative coverage is the fraction of valid image pixels assigned to a
    foreground class.  If ``valid_mask`` is omitted, prediction pixels equal to
    ``ignore_index`` are excluded.  A supplied mask is additionally intersected
    with that rule, which lets evaluation exclude audited ignore-label pixels.
    """
    pred = np.asarray(prediction)
    if pred.ndim != 2:
        raise ValueError(f"Expected a two-dimensional semantic map, got {pred.shape}")

    if valid_mask is None:
        valid = pred != int(ignore_index)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != pred.shape:
            raise ValueError(
                f"valid_mask shape {valid.shape} does not match prediction shape {pred.shape}"
            )
        valid = valid & (pred != int(ignore_index))

    valid_values = pred[valid]
    if valid_values.size:
        invalid_values = valid_values[(valid_values < 0) | (valid_values >= len(class_names))]
        if invalid_values.size:
            values = sorted({int(value) for value in invalid_values.tolist()})
            raise ValueError(f"Unexpected semantic class IDs on valid pixels: {values}")

    valid_pixel_count = int(valid.sum())
    class_rows = []
    pixel_counts = []
    presence = []
    relative_coverage = []
    for class_id in (int(value) for value in foreground_class_ids):
        count = int(np.count_nonzero(valid & (pred == class_id)))
        present = bool(count > 0)
        coverage = float(count / valid_pixel_count) if valid_pixel_count else 0.0
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": str(class_names[class_id]),
                "pixel_count": count,
                "presence": present,
                "relative_coverage": coverage,
            }
        )
        pixel_counts.append(count)
        presence.append(present)
        relative_coverage.append(coverage)

    record = {
        "schema": CONDITION_RECORD_SCHEMA,
        "image_id": str(image_id),
        "model_id": str(model_id),
        "valid_pixel_count": valid_pixel_count,
        "ignored_pixel_count": int(pred.size - valid_pixel_count),
        "foreground_class_ids": [int(value) for value in foreground_class_ids],
        "class_names": [str(class_names[int(value)]) for value in foreground_class_ids],
        "pixel_counts": pixel_counts,
        "presence": presence,
        "relative_coverage": relative_coverage,
        "classes": class_rows,
    }
    if semantic_map is not None:
        record["semantic_map"] = str(semantic_map)
    if checkpoint is not None:
        record["checkpoint"] = str(checkpoint)
    return record


def condition_record_csv_row(record: Mapping) -> dict:
    """Flatten a condition record into one fixed-width CSV row."""
    row = {
        "image_id": record["image_id"],
        "model_id": record["model_id"],
        "checkpoint": record.get("checkpoint", ""),
        "semantic_map": record.get("semantic_map", ""),
        "valid_pixel_count": record["valid_pixel_count"],
        "ignored_pixel_count": record["ignored_pixel_count"],
    }
    for item in record["classes"]:
        slug = re.sub(r"[^a-z0-9]+", "_", str(item["class_name"]).lower()).strip("_")
        prefix = f"class_{int(item['class_id'])}_{slug}"
        row[f"{prefix}_pixel_count"] = int(item["pixel_count"])
        row[f"{prefix}_presence"] = bool(item["presence"])
        row[f"{prefix}_relative_coverage"] = float(item["relative_coverage"])
    return row
