#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a tiny synthetic RGB-D pavement dataset for the GeoFormerX demo.

The generated files follow the same layout expected by the training and
evaluation code:

  examples/pavement_rgbd_small/
    train|val|test/image/DL2xxxx.png
    train|val|test/label/DL2xxxx.bmp
    3Ddate/train|val|test/image/DL3xxxx.png
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pavement_config import CLASS_RGB_VALUES  # noqa: E402


SPLIT_COUNTS = {"train": 3, "val": 2, "test": 2}


def _road_texture(rng: np.random.Generator, size: int) -> np.ndarray:
    base = np.full((size, size, 3), [118, 118, 112], dtype=np.int16)
    noise = rng.normal(0, 9, size=(size, size, 1)).astype(np.int16)
    texture = base + noise
    # Subtle longitudinal surface variation.
    x = np.linspace(-10, 10, size, dtype=np.float32)[None, :, None]
    texture = texture + x.astype(np.int16)
    return np.clip(texture, 0, 255).astype(np.uint8)


def _depth_texture(rng: np.random.Generator, size: int) -> np.ndarray:
    base = np.full((size, size), 150, dtype=np.int16)
    noise = rng.normal(0, 8, size=(size, size)).astype(np.int16)
    y = np.linspace(-12, 12, size, dtype=np.float32)[:, None]
    depth = base + noise + y.astype(np.int16)
    return np.clip(depth, 0, 255).astype(np.uint8)


def _draw_sample(split: str, index: int, out_dir: Path, seed: int, size: int = 256) -> None:
    rng = np.random.default_rng(seed)
    rgb = Image.fromarray(_road_texture(rng, size), mode="RGB")
    label = Image.new("RGB", (size, size), tuple(CLASS_RGB_VALUES[0]))
    depth = Image.fromarray(_depth_texture(rng, size), mode="L")

    draw_rgb = ImageDraw.Draw(rgb)
    draw_label = ImageDraw.Draw(label)
    draw_depth = ImageDraw.Draw(depth)

    # Crack: jagged dark line.
    y0 = 34 + 17 * ((index + seed) % 5)
    crack = [(18, y0), (58, y0 + 7), (96, y0 - 3), (139, y0 + 10), (214, y0 + 2)]
    draw_rgb.line(crack, fill=(20, 20, 18), width=3)
    draw_label.line(crack, fill=tuple(CLASS_RGB_VALUES[1]), width=3)
    draw_depth.line(crack, fill=55, width=4)

    # Pothole: dark depression.
    hole = (34, 142, 100, 207) if index % 2 == 0 else (146, 128, 214, 198)
    draw_rgb.ellipse(hole, fill=(58, 70, 54), outline=(37, 43, 35), width=3)
    draw_label.ellipse(hole, fill=tuple(CLASS_RGB_VALUES[2]))
    draw_depth.ellipse(hole, fill=42)

    # Seal and joint: two elongated structural defects.
    seal = [(126, 24), (130, 78), (134, 132), (142, 184), (146, 235)]
    draw_rgb.line(seal, fill=(135, 66, 166), width=5)
    draw_label.line(seal, fill=tuple(CLASS_RGB_VALUES[3]), width=5)
    draw_depth.line(seal, fill=136, width=5)

    joint_y = 218 - 11 * (index % 3)
    joint = [(12, joint_y), (244, joint_y - 5)]
    draw_rgb.line(joint, fill=(118, 69, 25), width=6)
    draw_label.line(joint, fill=tuple(CLASS_RGB_VALUES[6]), width=6)
    draw_depth.line(joint, fill=92, width=6)

    # Patch and road marking.
    patch = (158, 34, 228, 91)
    draw_rgb.rounded_rectangle(patch, radius=5, fill=(68, 178, 198), outline=(47, 135, 151), width=2)
    draw_label.rounded_rectangle(patch, radius=5, fill=tuple(CLASS_RGB_VALUES[4]))
    draw_depth.rounded_rectangle(patch, radius=5, fill=112)

    mark = [(24, 94), (95, 112), (168, 119), (235, 139)]
    draw_rgb.line(mark, fill=(55, 88, 245), width=6)
    draw_label.line(mark, fill=tuple(CLASS_RGB_VALUES[5]), width=6)
    draw_depth.line(mark, fill=164, width=6)

    # Manhole cover.
    cover = (173, 169, 225, 221)
    draw_rgb.ellipse(cover, fill=(221, 112, 45), outline=(151, 65, 24), width=3)
    draw_label.ellipse(cover, fill=tuple(CLASS_RGB_VALUES[7]))
    draw_depth.ellipse(cover, fill=82)

    split_root = out_dir / split
    image_dir = split_root / "image"
    label_dir = split_root / "label"
    depth_dir = out_dir / "3Ddate" / split / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    sample_id = index + 1
    stem2d = f"DL2{sample_id:04d}"
    stem3d = f"DL3{sample_id:04d}"
    rgb.save(image_dir / f"{stem2d}.png")
    label.save(label_dir / f"{stem2d}.bmp")
    depth.save(depth_dir / f"{stem3d}.png")


def build_dataset(out_dir: Path, overwrite: bool = False) -> None:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, count in SPLIT_COUNTS.items():
        for i in range(count):
            _draw_sample(split, i, out_dir, seed=1729 + i * 37 + len(split) * 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the bundled GeoFormerX example dataset.")
    parser.add_argument("--out", type=Path, default=ROOT / "examples" / "pavement_rgbd_small")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args.out, overwrite=args.overwrite)
    print(f"Example dataset written to: {args.out}")


if __name__ == "__main__":
    main()
