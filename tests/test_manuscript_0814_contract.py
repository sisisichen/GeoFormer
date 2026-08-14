from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data.dataset import load_grayscale_intensity
from utils.condition_records import (
    CONDITION_RECORD_SCHEMA,
    build_condition_record,
    condition_record_csv_row,
)


def test_intensity_loader_returns_one_information_channel(tmp_path: Path) -> None:
    gray = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    single_path = tmp_path / "single.png"
    triplicate_path = tmp_path / "triplicate.png"
    Image.fromarray(gray, mode="L").save(single_path)
    Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), mode="RGB").save(triplicate_path)

    loaded_single = load_grayscale_intensity(str(single_path))
    loaded_triplicate = load_grayscale_intensity(str(triplicate_path))

    assert loaded_single.shape == (2, 2, 1)
    assert loaded_single.dtype == np.uint8
    np.testing.assert_array_equal(loaded_single, loaded_triplicate)
    np.testing.assert_array_equal(loaded_single[:, :, 0], gray)


def test_intensity_loader_rejects_real_color_content(tmp_path: Path) -> None:
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    color[0, 0] = [255, 0, 0]
    path = tmp_path / "not_grayscale.png"
    Image.fromarray(color, mode="RGB").save(path)

    with pytest.raises(ValueError, match="R=G=B"):
        load_grayscale_intensity(str(path))


def test_condition_record_uses_valid_pixels_and_foreground_order() -> None:
    prediction = np.array(
        [
            [0, 1, 2, 255],
            [5, 7, 4, 6],
        ],
        dtype=np.uint8,
    )
    valid_mask = np.array(
        [
            [True, True, False, True],
            [True, True, True, True],
        ]
    )

    record = build_condition_record(
        prediction,
        image_id="sample",
        model_id="GeoFormerX-G8-D0-S0",
        valid_mask=valid_mask,
        semantic_map="pred_label/sample.png",
        checkpoint="model_final.pth",
    )

    assert record["schema"] == CONDITION_RECORD_SCHEMA
    assert record["valid_pixel_count"] == 6
    assert record["ignored_pixel_count"] == 2
    assert record["foreground_class_ids"] == [1, 2, 3, 4, 5, 6, 7]
    assert record["pixel_counts"] == [1, 0, 0, 1, 1, 1, 1]
    assert record["presence"] == [True, False, False, True, True, True, True]
    np.testing.assert_allclose(
        record["relative_coverage"],
        [1 / 6, 0.0, 0.0, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
    )

    row = condition_record_csv_row(record)
    assert row["class_1_crack_pixel_count"] == 1
    assert row["class_2_pothole_presence"] is False
    assert row["class_7_manhole_cover_relative_coverage"] == pytest.approx(1 / 6)


def test_condition_record_all_ignored_is_finite_zero() -> None:
    prediction = np.full((2, 3), 255, dtype=np.uint8)
    record = build_condition_record(prediction, image_id="ignored", model_id="model")

    assert record["valid_pixel_count"] == 0
    assert record["ignored_pixel_count"] == 6
    assert record["pixel_counts"] == [0] * 7
    assert record["presence"] == [False] * 7
    assert record["relative_coverage"] == [0.0] * 7
