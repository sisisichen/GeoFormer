# -*- coding: utf-8 -*-
"""One-command GeoFormerX training entry.

The full recipe is intentionally hidden behind a small CLI.  It trains the base
model, trains the line-specialist refinement stage, then merges all class-best
states into one final checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from geoformerx_recipe import base_stage_args, specialist_stage_args, merge_args, apply_ablation_variant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GeoFormerX with the final reproducible recipe.")
    p.add_argument("--data_path", default="./data/pavement_rgbd", help="Dataset root containing train/val/test and 3Ddate.")
    p.add_argument("--checkpoint", default="./checkpoints/sam", help="Directory containing the SAM checkpoint.")
    p.add_argument("--work_dir", default="./runs/geoformerx", help="Output directory for all training artifacts.")
    p.add_argument("--model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device_ids", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=50, help="Epochs for each training stage. Default: 50.")
    p.add_argument("--base_batch_size", type=int, default=28)
    p.add_argument("--specialist_batch_size", type=int, default=24)
    p.add_argument("--overwrite", action="store_true", help="Delete work_dir before training.")
    p.add_argument("--skip_base", action="store_true", help="Reuse work_dir/base/model_best.pth and train only the specialist stage.")
    p.add_argument("--ablation", default="full", choices=["full", "no_depth", "no_geometry_gate", "no_style_routing"], help="Train a paper ablation variant.")
    p.add_argument("--dry_run", action="store_true", help="Print resolved commands without executing them.")
    return p.parse_args()


def run(cmd: list[str], title: str, dry_run: bool = False) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(" ".join(str(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    if args.overwrite and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = work_dir / "base" / "model_best.pth"
    final_ckpt = work_dir / "model_final.pth"

    if not args.skip_base:
        run(
            apply_ablation_variant(base_stage_args(
                data_path=args.data_path,
                checkpoint=args.checkpoint,
                work_dir=str(work_dir),
                model_type=args.model_type,
                device=args.device,
                device_ids=args.device_ids,
                num_epochs=args.epochs,
                batch_size=args.base_batch_size,
            ), args.ablation),
            f"Stage 1/3: base GeoFormerX training [{args.ablation}]",
            dry_run=args.dry_run,
        )
    elif not base_ckpt.exists():
        raise FileNotFoundError(f"--skip_base was set, but base checkpoint is missing: {base_ckpt}")

    run(
        apply_ablation_variant(specialist_stage_args(
            data_path=args.data_path,
            checkpoint=args.checkpoint,
            work_dir=str(work_dir),
            init_from=str(base_ckpt),
            model_type=args.model_type,
            device=args.device,
            device_ids=args.device_ids,
            num_epochs=args.epochs,
            batch_size=args.specialist_batch_size,
        ), args.ablation),
        f"Stage 2/3: line-specialist refinement training [{args.ablation}]",
        dry_run=args.dry_run,
    )

    run(merge_args(str(work_dir)), "Stage 3/3: merge one final checkpoint", dry_run=args.dry_run)

    manifest = {
        "final_checkpoint": str(final_ckpt),
        "base_checkpoint": str(base_ckpt),
        "data_path": args.data_path,
        "sam_checkpoint_dir": args.checkpoint,
        "epochs_per_stage": args.epochs,
        "ablation": args.ablation,
    }
    if not args.dry_run:
        (work_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nTraining pipeline finished.")
    print(f"Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
