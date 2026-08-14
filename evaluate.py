# -*- coding: utf-8 -*-
"""One-command GeoFormerX evaluation entry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from geoformerx_recipe import apply_ablation_variant, evaluation_args


PROTOCOL_NAME = "GeoFormerX_final_paper_evaluation_v1"
CONDITION_RECORD_SCHEMA = "GeoFormerX.condition_record.v1"
PROTOCOL_FILENAME = "resolved_evaluation_protocol.json"
REPO_ROOT = Path(__file__).resolve().parent
EVALUATOR_PATH = REPO_ROOT / "_evaluate_core.py"
CONFIG_PATH = REPO_ROOT / "configs" / "GeoFormerX_G8_D0_S0.yaml"


def _command_value(command: Sequence[str], flag: str) -> Optional[str]:
    command_list = list(command)
    if flag not in command_list:
        return None
    index = command_list.index(flag)
    return command_list[index + 1] if index + 1 < len(command_list) else None


def _sha256_if_available(path: Path) -> Optional[str]:
    try:
        is_file = path.is_file()
    except OSError:
        return None
    if not is_file:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _pytorch_version() -> Optional[str]:
    """Read package metadata without importing torch or initializing CUDA."""
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def is_canonical_evaluation(command: Sequence[str], ablation: str = "full") -> bool:
    return (
        _command_value(command, "--tta_hflip") == "1"
        and _command_value(command, "--blend") == "hann"
        and _command_value(command, "--stitch_mode") == "logits"
        and _command_value(command, "--prompt_mode") == "full"
        and _command_value(command, "--collect_debug") == "0"
        and str(ablation).lower().strip() == "full"
    )


def build_protocol_manifest(
    args: argparse.Namespace,
    command: Sequence[str],
    launcher_argv: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    checkpoint = Path(args.ckpt).expanduser()
    canonical = is_canonical_evaluation(command, args.ablation)
    return {
        "protocol_name": PROTOCOL_NAME,
        "tta_horizontal_flip": _command_value(command, "--tta_hflip") == "1",
        "tta_merge": "0.5_original_plus_0.5_unflipped_hflip",
        "blend": _command_value(command, "--blend"),
        "stitch_mode": _command_value(command, "--stitch_mode"),
        "tile_size": int(_command_value(command, "--tile_size") or 0),
        "tile_stride": int(_command_value(command, "--tile_stride") or 0),
        "tiles_per_512x256_image": 3,
        "input_channels": "1_grayscale_intensity_plus_1_range_coded",
        "decision": "argmax_after_weight_normalized_reconstruction",
        "condition_record_schema": CONDITION_RECORD_SCHEMA,
        "condition_record_outputs": ["condition_records.json", "condition_records.csv"],
        "diagnostic_collection": _command_value(command, "--collect_debug") == "1",
        "timing_includes": [
            "intensity_range_loading_and_preprocessing",
            "three_original_and_three_horizontally_reflected_tile_forwards",
            "inverse_flipping_and_equal_logit_merge",
            "hann_weighted_complete_image_reconstruction",
            "final_argmax",
            "cuda_synchronization",
        ],
        "timing_excludes": [
            "reference_mask_loading",
            "metric_computation",
            "prediction_saving",
            "foreground_probability_export",
            "optional_diagnostic_collection",
        ],
        "canonical": canonical,
        "paper_reproduction_eligible": canonical,
        "ablation": args.ablation,
        "evaluator_file": EVALUATOR_PATH.name,
        "evaluator_file_sha256": _sha256_if_available(EVALUATOR_PATH),
        "config_file": CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": _sha256_if_available(CONFIG_PATH),
        "checkpoint_path": args.ckpt,
        "checkpoint_sha256": _sha256_if_available(checkpoint),
        "command_line": list(launcher_argv if launcher_argv is not None else sys.argv),
        "resolved_evaluator_command": list(command),
        "date_time_utc": datetime.now(timezone.utc).isoformat(),
        "pytorch_version": _pytorch_version(),
    }


def write_protocol_manifest(
    args: argparse.Namespace,
    command: Sequence[str],
    launcher_argv: Optional[Sequence[str]] = None,
) -> Path:
    output_path = Path(args.out_dir) / PROTOCOL_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_protocol_manifest(args, command, launcher_argv=launcher_argv)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_path


def print_protocol_status(canonical: bool) -> None:
    if not canonical:
        print("NONCANONICAL_EVALUATION_PROTOCOL")
        print("This run must not be labeled as a canonical paper-reproduction result.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GeoFormerX with the final tiled-inference recipe.")
    p.add_argument("--data_path", default="./data/pavement_rgbd", help="Dataset root containing the split to evaluate.")
    p.add_argument("--checkpoint", default="./checkpoints/sam", help="Directory containing the SAM checkpoint.")
    p.add_argument("--ckpt", default="./runs/geoformerx/model_final.pth", help="Trained GeoFormerX checkpoint.")
    p.add_argument("--split", default="test", help="Split name under data_path, usually test/val/train.")
    p.add_argument("--out_dir", default="./runs/geoformerx/eval_test", help="Directory for reports and predictions.")
    p.add_argument("--model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device_ids", nargs="+", type=int, default=[0])
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--ablation", default="full", choices=["full", "no_range", "no_geometry_gate", "moe_adapter", "decoder_only"], help="Evaluation architecture variant; must match the checkpoint.")
    p.add_argument("--stitch_mode", default="logits", choices=["logits", "hard"], help="Use hard for the patch-mosaic stitching ablation.")
    p.add_argument("--save_paper_outputs", type=int, default=1, choices=[0, 1])
    p.add_argument("--max_visuals", type=int, default=24)
    p.add_argument("--collect_debug", type=int, default=0, choices=[0, 1])
    p.add_argument("--dry_run", action="store_true", help="Print the resolved command without executing it.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cmd = apply_ablation_variant(evaluation_args(
        data_path=args.data_path,
        checkpoint=args.checkpoint,
        ckpt=args.ckpt,
        out_dir=args.out_dir,
        split=args.split,
        model_type=args.model_type,
        device=args.device,
        device_ids=args.device_ids,
        batch_size=args.batch_size,
    ), args.ablation)
    cmd += [
        "--save_paper_outputs", str(int(args.save_paper_outputs)),
        "--max_visuals", str(int(args.max_visuals)),
    ]
    collect_debug_idx = cmd.index("--collect_debug")
    cmd[collect_debug_idx + 1] = str(int(args.collect_debug))
    stitch_idx = cmd.index("--stitch_mode")
    cmd[stitch_idx + 1] = str(args.stitch_mode)
    manifest_path = write_protocol_manifest(args, cmd)
    canonical = is_canonical_evaluation(cmd, args.ablation)
    print_protocol_status(canonical)
    print(f"Resolved evaluation protocol: {manifest_path}")
    print(" ".join(str(x) for x in cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
        print(f"\nEvaluation finished. Summary: {Path(args.out_dir) / 'summary.txt'}")


if __name__ == "__main__":
    main()
