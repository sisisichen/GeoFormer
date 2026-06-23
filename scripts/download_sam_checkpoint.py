#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download official Segment Anything checkpoints used by GeoFormerX."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


CHECKPOINTS = {
    "vit_b": (
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    "vit_l": (
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    "vit_h": (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a SAM checkpoint.")
    parser.add_argument("--model_type", default="vit_b", choices=sorted(CHECKPOINTS))
    parser.add_argument("--out_dir", type=Path, default=Path("checkpoints") / "sam")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filename, url = CHECKPOINTS[args.model_type]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / filename
    if out_path.exists() and not args.overwrite:
        print(f"Checkpoint already exists: {out_path}")
        return
    print(f"Downloading {filename} from {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
