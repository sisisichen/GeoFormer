# -*- coding: utf-8 -*-
"""Configuration for GeoFormerX 2D+3D pavement multi-class segmentation.

This project was originally designed around per-task binary masks stored as .npy.
Your dataset is:

  data/
    train/image   (2D RGB, filenames like DL2xxxx.png)
    train/label   (RGB mask .bmp)
    val/image
    val/label
    test/image
    test/label
    3Ddate/       (3D png mapped from .bin, filenames like DL3xxxx.png)

We modify SAM's mask decoder to output 7 foreground-class logits in **one forward**
(for Crack..Manhole). Background is produced by thresholding / argmax fusion.

At evaluation time we tile, run inference once per tile, stitch back to the original size,
then render with the same palette.
"""

from __future__ import annotations

from typing import Dict, List


# ----------------------
# Image / mask formats
# ----------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MASK_EXT = ".bmp"


# ----------------------
# Multi-class palette
# ----------------------
NUM_CLASSES = 8

CLASS_RGB_VALUES: List[List[int]] = [
    [255, 255, 255],  # 0 Background
    [255,   0,   0],  # 1 Crack
    [  0, 255,   0],  # 2 Pothole
    [140,  40, 225],  # 3 Seal
    [  0, 190, 255],  # 4 Patch
    [  0,   0, 255],  # 5 Marking
    [140,  70,   0],  # 6 Joint
    [255, 100,  50],  # 7 Manhole
]

# Backward-compat alias (some scripts used CLASS_COLORS)
CLASS_COLORS = CLASS_RGB_VALUES

CLASS_NAMES: List[str] = [
    "Background", "Crack", "Pothole", "Seal",
    "Patch", "Marking", "Joint", "Manhole",
]


# Foreground classes (exclude background=0)
FG_CLASS_IDS: List[int] = [1, 2, 3, 4, 5, 6, 7]
NUM_FG_CLASSES: int = len(FG_CLASS_IDS)
CHANNEL_TO_CLASSID: List[int] = FG_CLASS_IDS[:]
CLASSID_TO_CHANNEL = {cid: i for i, cid in enumerate(FG_CLASS_IDS)}


# ----------------------
# Mask color noise handling
# ----------------------
IGNORE_INDEX = 255
DIST_THRESHOLD = 25.0


# ----------------------
# Tiling settings
# ----------------------
TILE_SIZE = 256
TILE_STRIDE = 256


# ----------------------
# Class -> GeoFormerX task mapping
# (align with data/datainfo.py TASK_FOLDER_META)
# ----------------------
CLASSID_TO_TASK: Dict[int, str] = {
    1: "PavementCrack",
    2: "PavementPothole",
    3: "JointSealing",
    4: "PavementPatch",
    5: "RoadMarking",
    6: "ExpansionJoint",
    7: "ManholeCover",
}


def classid_to_task_folder(class_id: int, modal: str = "VehicleProfiler") -> str:
    """Convert class_id (1..7) to TaskFolder name like 'VehicleProfiler_PavementCrack'."""
    if class_id not in CLASSID_TO_TASK:
        raise ValueError(f"class_id must be in {sorted(CLASSID_TO_TASK.keys())}, got {class_id}")
    return f"{modal}_{CLASSID_TO_TASK[class_id]}"


def default_force_expert_map(expert_num: int = 8) -> Dict[int, int]:
    """Default mapping: each foreground class uses a dedicated expert.

    If expert_num >= 7:
      Crack..Manhole -> expert 0..6
      (remaining experts unused by default)
    If expert_num < 7:
      map by modulo.
    """
    m: Dict[int, int] = {}
    ordered = sorted(CLASSID_TO_TASK.keys())
    for k, class_id in enumerate(ordered):
        if int(expert_num) >= 7:
            m[class_id] = k
        else:
            m[class_id] = k % max(1, int(expert_num))
    return m
