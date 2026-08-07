# -*- coding: utf-8 -*-
"""One-command GeoFormerX-G8-D0-S0 training entry."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from geoformerx_recipe import base_stage_args, apply_ablation_variant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GeoFormerX with the final reproducible recipe.")
    p.add_argument("--data_path", default="./data/pavement_rgbd", help="Dataset root containing train/val/test and 3Ddate.")
    p.add_argument("--checkpoint", default="./checkpoints/sam", help="Directory containing the SAM checkpoint.")
    p.add_argument("--work_dir", default="./runs/geoformerx", help="Output directory for all training artifacts.")
    p.add_argument("--model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device_ids", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=50, help="Number of epochs. Default: 50.")
    p.add_argument("--batch_size", type=int, default=28)
    p.add_argument("--seed", type=int, default=2028, help="Formal-run seed; the paper reports 2026, 2027, and 2028.")
    p.add_argument("--overwrite", action="store_true", help="Delete work_dir before training.")
    p.add_argument("--ablation", default="full", choices=["full", "no_range", "no_geometry_gate", "moe_adapter", "decoder_only"], help="Optional architecture-screening variant.")
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

    run(
        apply_ablation_variant(base_stage_args(
            data_path=args.data_path,
            checkpoint=args.checkpoint,
            work_dir=str(work_dir),
            model_type=args.model_type,
            device=args.device,
            device_ids=args.device_ids,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        ), args.ablation),
        f"GeoFormerX-G8-D0-S0 training [{args.ablation}]",
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        if not base_ckpt.exists():
            raise FileNotFoundError(f"Training completed without a selected checkpoint: {base_ckpt}")
        shutil.copy2(base_ckpt, final_ckpt)

    manifest = {
        "final_checkpoint": str(final_ckpt),
        "base_checkpoint": str(base_ckpt),
        "data_path": args.data_path,
        "sam_checkpoint_dir": args.checkpoint,
        "epochs": args.epochs,
        "seed": args.seed,
        "canonical_model": "GeoFormerX-G8-D0-S0",
        "ablation": args.ablation,
    }
    if not args.dry_run:
        (work_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run:
        print("\nDry run finished; no training was started.")
    else:
        print("\nTraining finished.")
        print(f"Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
