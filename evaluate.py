# -*- coding: utf-8 -*-
"""One-command GeoFormerX evaluation entry."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from geoformerx_recipe import evaluation_args, apply_ablation_variant


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
    p.add_argument("--ablation", default="full", choices=["full", "no_depth", "no_geometry_gate", "no_style_routing"], help="Evaluation architecture variant; must match the checkpoint.")
    p.add_argument("--stitch_mode", default="logits", choices=["logits", "hard"], help="Use hard for the patch-mosaic stitching ablation.")
    p.add_argument("--save_paper_outputs", type=int, default=1, choices=[0, 1])
    p.add_argument("--max_visuals", type=int, default=24)
    p.add_argument("--collect_debug", type=int, default=1, choices=[0, 1])
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
        "--stitch_mode", str(args.stitch_mode),
        "--save_paper_outputs", str(int(args.save_paper_outputs)),
        "--max_visuals", str(int(args.max_visuals)),
        "--collect_debug", str(int(args.collect_debug)),
    ]
    print(" ".join(str(x) for x in cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
        print(f"\nEvaluation finished. Summary: {Path(args.out_dir) / 'summary.txt'}")


if __name__ == "__main__":
    main()
