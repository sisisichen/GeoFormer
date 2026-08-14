from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import evaluate  # noqa: E402
from geoformerx_recipe import evaluation_args  # noqa: E402


EVALUATOR_SOURCE = (REPO_ROOT / "_evaluate_core.py").read_text(encoding="utf-8")


def _value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def _production_hann_window(tile: int, stride: int) -> np.ndarray:
    """Load only the production NumPy helper, without importing torch/model code."""
    tree = ast.parse(EVALUATOR_SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_blend_window"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "_evaluate_core.py", "exec"), namespace)
    return namespace["make_blend_window"](tile, stride, "hann")


def _reconstruct(constants: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tile, stride = 4, 2
    window = _production_hann_window(tile, stride)
    accumulator = np.zeros((tile, tile + stride), dtype=np.float32)
    weights = np.zeros_like(accumulator)
    for x0, value in zip((0, stride), constants):
        accumulator[:, x0 : x0 + tile] += value * window
        weights[:, x0 : x0 + tile] += window
    reconstruction = accumulator / np.maximum(weights, 1e-6)
    return reconstruction, weights, window


def test_canonical_evaluation_command_contains_final_protocol() -> None:
    command = evaluation_args(
        data_path="data",
        checkpoint="sam",
        ckpt="model.pth",
        out_dir="eval",
        device="cpu",
        device_ids=(0,),
    )
    assert _value(command, "--tta_hflip") == "1"
    assert _value(command, "--blend") == "hann"
    assert _value(command, "--stitch_mode") == "logits"
    assert _value(command, "--prompt_mode") == "full"
    assert _value(command, "--collect_debug") == "0"
    assert _value(command, "--tile_size") == "256"
    assert _value(command, "--tile_stride") == "128"


def test_hflip_logits_are_unflipped_before_equal_merge() -> None:
    original = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4)
    reflected_native = np.flip(original + 2.0, axis=-1)

    restored = np.flip(reflected_native, axis=-1)
    merged = 0.5 * (original + restored)
    wrong_merge_without_unflip = 0.5 * (original + reflected_native)

    np.testing.assert_allclose(merged, original + 1.0)
    assert not np.allclose(merged, wrong_merge_without_unflip)
    assert "torch.flip(data['img'], dims=[3])" in EVALUATOR_SOURCE
    assert "logits_fg_f = torch.flip(logits_fg_f, dims=[3])" in EVALUATOR_SOURCE
    assert "logits_fg = 0.5 * (logits_fg + logits_fg_f)" in EVALUATOR_SOURCE


def test_hann_reconstruction_accumulates_and_normalizes_weights() -> None:
    reconstruction, weights, window = _reconstruct([2.0, 6.0])
    expected_overlap = (2.0 * window[:, 2:] + 6.0 * window[:, :2]) / (
        window[:, 2:] + window[:, :2]
    )

    np.testing.assert_allclose(reconstruction[:, 2:4], expected_overlap, rtol=1e-6)
    assert np.all(weights[:, 2:4] > 0)
    assert not np.allclose(reconstruction[:, 2:4], 2.0 + 6.0)

    constant_reconstruction, _, _ = _reconstruct([3.0, 3.0])
    np.testing.assert_allclose(constant_reconstruction, 3.0, rtol=1e-6)


def test_production_reconstructs_logits_before_complete_image_argmax() -> None:
    accumulate = "logits_acc[:, y0:y0 + tile, x0:x0 + tile] += logits_all_np[b] * blend_w[None, :, :]"
    normalize = "logits_acc = logits_acc / cnt_acc[None, :, :]"
    crop = "logits_acc = logits_acc[:, :int(orig_h), :int(orig_w)]"
    argmax = "pred = logits_acc.argmax(axis=0)"
    probability = "prob_all = e /"

    indices = [EVALUATOR_SOURCE.index(text) for text in (accumulate, normalize, crop, argmax, probability)]
    assert indices == sorted(indices)
    assert "cnt_acc = np.maximum(cnt_acc, 1e-6)" in EVALUATOR_SOURCE
    assert "default=-1.0" in EVALUATOR_SOURCE


def test_canonical_timing_includes_input_loading_and_cuda_sync() -> None:
    start = "t0 = time.perf_counter()"
    intensity_load = "intensity = load_grayscale_intensity(str(img_path))"
    inference = "pred, reconstructed_logits, debug = infer_prob_maps_fg("
    synchronization = "torch.cuda.synchronize(device)"
    stop = "infer_times.append(time.perf_counter() - t0)"
    probability = "prob_fg = foreground_probabilities_from_logits(reconstructed_logits)"
    reference_load = "if gt is None:\n            gt = load_reference_label(lbl_dir, name)"

    start_index = EVALUATOR_SOURCE.index(start)
    assert start_index < EVALUATOR_SOURCE.index(intensity_load, start_index)
    assert start_index < EVALUATOR_SOURCE.index(inference, start_index)
    assert EVALUATOR_SOURCE.rindex(synchronization) < EVALUATOR_SOURCE.index(stop, start_index)
    assert EVALUATOR_SOURCE.index(stop, start_index) < EVALUATOR_SOURCE.index(probability)
    assert EVALUATOR_SOURCE.index(stop, start_index) < EVALUATOR_SOURCE.index(reference_load)


def test_protocol_manifest_is_complete_and_canonical(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint hash fixture")
    out_dir = tmp_path / "eval"
    args = argparse.Namespace(
        ckpt=str(checkpoint),
        out_dir=str(out_dir),
        ablation="full",
    )
    command = evaluation_args(
        data_path="data",
        checkpoint="sam",
        ckpt=str(checkpoint),
        out_dir=str(out_dir),
        device="cpu",
        device_ids=(0,),
    )

    manifest_path = evaluate.write_protocol_manifest(
        args, command, launcher_argv=["evaluate.py", "--dry_run"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == "resolved_evaluation_protocol.json"
    assert manifest["protocol_name"] == "GeoFormerX_final_paper_evaluation_v1"
    assert manifest["tta_horizontal_flip"] is True
    assert manifest["tta_merge"] == "0.5_original_plus_0.5_unflipped_hflip"
    assert manifest["blend"] == "hann"
    assert manifest["stitch_mode"] == "logits"
    assert manifest["tile_size"] == 256
    assert manifest["tile_stride"] == 128
    assert manifest["tiles_per_512x256_image"] == 3
    assert manifest["input_channels"] == "1_grayscale_intensity_plus_1_range_coded"
    assert manifest["decision"] == "argmax_after_weight_normalized_reconstruction"
    assert manifest["condition_record_schema"] == "GeoFormerX.condition_record.v1"
    assert manifest["diagnostic_collection"] is False
    assert "cuda_synchronization" in manifest["timing_includes"]
    assert "reference_mask_loading" in manifest["timing_excludes"]
    assert manifest["canonical"] is True
    assert manifest["evaluator_file_sha256"]
    assert manifest["config_sha256"]
    assert manifest["checkpoint_sha256"]
    assert manifest["command_line"] == ["evaluate.py", "--dry_run"]
    assert manifest["date_time_utc"]
    assert "pytorch_version" in manifest


def test_noncanonical_override_prints_explicit_warning(capsys) -> None:
    command = evaluation_args("data", "sam", "model.pth", "eval", device="cpu")
    command[command.index("--stitch_mode") + 1] = "hard"

    canonical = evaluate.is_canonical_evaluation(command, "full")
    evaluate.print_protocol_status(canonical)
    output = capsys.readouterr().out

    assert canonical is False
    assert "NONCANONICAL_EVALUATION_PROTOCOL" in output
    assert "must not be labeled" in output
