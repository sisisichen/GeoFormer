# -*- coding: utf-8 -*-
"""Evaluate pavement multi-class segmentation (2D+3D) with **one forward per tile**.

Key design
----------
- Input images are 512x256 (W x H). We tile them online into 256x256.
- The model outputs 7 foreground-class logits in one forward (classes 1..7).
- We stitch tile predictions back to the original resolution, then fuse to an 8-class
  label map (0 background + 7 defects) using softmax over [bg_logit(=0), fg_logits].

This script:
  1) loads the trained checkpoint (MoE GeoFormerX + 2D/3D fusion)
  2) runs tiled inference on the test split
  3) saves color predictions (same palette as labels)
  4) reports Dice / IoU / F1 / F2 and extra metrics for publication

Recommended fair setting
------------------------
Use --prompt_mode full (default) to avoid any GT leakage (no gt_box).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from segment_anything import sam_model_registry
from model.geoformerx import GeoFormerX

# Same checkpoint naming as training scripts
sam_model_checkpoint = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


from data.dataset import find_3d_path, rgb_to_label_nearest
from data.dataset import PAV_DIST_THRESHOLD, PAV_IGNORE_INDEX

from data.dataset import resolve_task_folder_meta
from pavement_config import (
    CLASS_COLORS,
    CLASS_NAMES,
    CLASSID_TO_TASK,
    classid_to_task_folder,
    FG_CLASS_IDS,
    NUM_FG_CLASSES,
)

from utils.multiclass_metrics import (
    confusion_matrix,
    per_class_metrics_from_cm,
    boundary_f1_per_class,
    overall_metrics_from_cm,
    freq_weighted_iou,
)

from utils.multiclass_loss import logits_with_bg
from utils.publication_outputs import (
    count_parameters, error_overlay, foreground_mdice_from_cm, nanmean,
    plot_alpha_hist, plot_expert_heatmap, save_json, save_qual_panel,
    seam_band_mask, write_csv,
)


def parse_class_weight_map(s: str):
    out = {}
    s = str(s or '').strip()
    if not s:
        return out
    for part in s.split(','):
        part = part.strip()
        if not part or ':' not in part:
            continue
        k, v = part.split(':', 1)
        out[int(k.strip())] = float(v.strip())
    return out


def parse_class_path_map(s: str):
    """Parse class-to-checkpoint map, e.g. '1:./c1.pth,5:./c5.pth'."""
    out = {}
    s = str(s or '').strip()
    if not s:
        return out
    for part in s.split(','):
        part = part.strip()
        if not part or ':' not in part:
            continue
        k, v = part.split(':', 1)
        out[int(k.strip())] = v.strip().strip('"').strip("'")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, required=True, help='Dataset root. Expect {split}/image, {split}/label, and 3Ddate/..')
    p.add_argument('--checkpoint', type=str, required=True, help='SAM checkpoint dir (contains sam_vit_*.pth)')
    p.add_argument('--model_type', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    p.add_argument('--sam_image_size', type=int, default=0,
                   help='SAM image size. 0 means auto: read from checkpoint model_cfg if present, else use 256.')
    p.add_argument('--model', type=str, default='geoformerx', choices=['geoformerx'])
    p.add_argument('--expert_num', type=int, default=8)
    p.add_argument('--num_fg_classes', type=int, default=7, help='Foreground classes (exclude background).')

    # --- GeoFormerX / MoE / Fusion hyperparams (keep aligned with training for reproducibility) ---
    p.add_argument('--bottleneck_dim', type=int, default=16)
    p.add_argument('--embedding_dim', type=int, default=16)
    p.add_argument('--moe_topk', type=int, default=2)
    p.add_argument('--moe_temp', type=float, default=1.0)
    p.add_argument('--moe_noise_std', type=float, default=0.0)
    p.add_argument('--moe_style_scale', type=float, default=0.25)
    p.add_argument('--fusion_hidden', type=int, default=16)
    p.add_argument('--fusion_mode', type=str, default='auto', choices=['auto', 'global', 'hybrid', 'fixed'])
    p.add_argument('--use_logit_refiner', type=int, default=-1, help='-1:auto detect from checkpoint, 0:disable, 1:enable')
    p.add_argument('--refiner_hidden', type=int, default=32)
    p.add_argument('--use_specialist_refiner', type=int, default=-1, help='-1:auto detect from checkpoint, 0:disable, 1:enable')
    p.add_argument('--line_refiner_hidden', type=int, default=48)
    p.add_argument('--surface_refiner_hidden', type=int, default=32)

    p.add_argument('--tile_size', type=int, default=256)
    p.add_argument('--tile_stride', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)

    p.add_argument('--prompt_mode', type=str, default='full', choices=['full', 'gt_box'],
                   help='full: use full tile box; gt_box: use oracle gt bbox (NOT fair, debug only).')

    p.add_argument('--use_fusion_2d3d', type=int, default=1, choices=[0, 1])

    p.add_argument(
        '--fg_thresh',
        type=float,
        default=-1.0,
        help='Optional FG confidence threshold. If >0, pixels with max_fg_prob<thresh are forced to background. '
             'Set -1 to disable (default, fair & consistent with argmax).',
    )

    # Inference-time boosters (optional)
    p.add_argument('--tta_hflip', type=int, default=0, choices=[0, 1],
                   help='1: horizontal flip test-time augmentation (average logits).')
    p.add_argument('--blend', type=str, default='none', choices=['none', 'hann'],
                   help='Logit blending window for overlapped tiling (use with stride < tile).')
    p.add_argument('--class_logit_bias_map', type=str, default='',
                   help='Optional per-class logit bias added before argmax, e.g. 1:0.06 to slightly favor Crack.')

    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--device_ids', type=int, nargs='+', default=[0])

    p.add_argument('--ckpt', type=str, required=True, help='Trained model checkpoint (.pth)')
    p.add_argument('--prefer_ema', type=int, default=0, choices=[0, 1], help='1: prefer ema_model in checkpoints when available.')
    p.add_argument('--ckpt_secondary', type=str, default='', help='Optional secondary checkpoint for class-wise logit fusion.')
    p.add_argument('--secondary_class_weight_map', type=str, default='',
                   help='Per-class weight for secondary checkpoint fusion, e.g. 1:0.8,5:0.4,6:0.4')
    p.add_argument('--secondary_fuse_mode', type=str, default='delta', choices=['delta', 'logits'],
                   help='delta: add selected residual logits (secondary-primary); logits: add raw secondary logits. delta is safer for base+specialist fusion.')
    p.add_argument('--class_ckpt_map', type=str, default='',
                   help="Optional multi-checkpoint class fusion, e.g. '1:./model_best_c1.pth,5:./model_best_c5.pth,6:./model_best_c6.pth'.")
    p.add_argument('--class_ckpt_weight_map', type=str, default='',
                   help="Weights for class_ckpt_map. Missing classes default to 1.0, e.g. '1:0.7,5:0.4,6:0.5'.")
    p.add_argument('--class_ckpt_fuse_mode', type=str, default='delta', choices=['delta', 'logits', 'replace'],
                   help='delta: add weight*(class_ckpt-base); logits: add weight*class_ckpt; replace: replace selected class logit.')
    p.add_argument('--use_bundle_fusion', type=int, default=0, choices=[0, 1],
                   help='1: if --ckpt is a merged bundle made by _merge_bundle.py, use its internal base_model + class_bank for exact one-file fusion.')
    p.add_argument('--out_dir', type=str, default='./eval_out')
    p.add_argument('--save_prob', type=int, default=0, choices=[0, 1])
    p.add_argument('--save_paper_outputs', type=int, default=1, choices=[0, 1],
                   help='1: save publication-oriented CSV/JSON metrics, overlays, panels, alpha histograms and expert heatmaps.')
    p.add_argument('--collect_debug', type=int, default=1, choices=[0, 1],
                   help='1: collect fusion alpha and MoE routing probabilities during evaluation.')
    p.add_argument('--max_visuals', type=int, default=24,
                   help='Maximum number of qualitative panels saved for paper figures. 0 disables panels.')
    p.add_argument('--visual_names', type=str, default='',
                   help='Optional comma-separated image stems to always save as qualitative panels.')
    p.add_argument('--seam_band', type=int, default=8,
                   help='Half-width in pixels for seam-band Dice around internal tile borders.')
    p.add_argument('--stitch_mode', type=str, default='logits', choices=['logits', 'hard'],
                   help='logits: average logits before argmax; hard: argmax each tile then vote/average one-hot labels.')
    p.add_argument('--print_report', type=str, default='full', choices=['summary', 'report', 'full'],
                   help='What to print to the console at the end of evaluation.')

    p.add_argument(
        '--split',
        type=str,
        default='test',
        help='Which split to evaluate. Default: test. Generated OOD roots can still use split=test.',
    )

    return p.parse_args()


def safe_tile_coords(h: int, w: int, tile: int, stride: int) -> List[Tuple[int, int]]:
    """Safe tiling coords that never go out of bounds and always cover borders."""
    if h < tile or w < tile:
        raise ValueError(f'h={h}, w={w} must be >= tile={tile}')
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)
    return [(x, y) for y in ys for x in xs]


def pad_to_multiple(img4: np.ndarray, label: np.ndarray, tile: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pad H,W to multiples of tile."""
    h, w = img4.shape[:2]
    pad_h = (tile - (h % tile)) % tile
    pad_w = (tile - (w % tile)) % tile
    if pad_h == 0 and pad_w == 0:
        return img4, label

    img4_p = np.pad(img4, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    lbl_p = np.pad(label, ((0, pad_h), (0, pad_w)), mode='edge')
    return img4_p, lbl_p


def full_box(tile: int) -> np.ndarray:
    return np.array([0, 0, tile, tile], dtype=np.float32)


def hflip_boxes(boxes: torch.Tensor, tile: int) -> torch.Tensor:
    """Flip boxes horizontally within a tile.

    Args:
        boxes: (B,4) in (x1,y1,x2,y2)
    """
    if boxes is None:
        return boxes
    if boxes.numel() == 0:
        return boxes
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    out = boxes.clone()
    out[:, 0] = float(tile) - x2
    out[:, 2] = float(tile) - x1
    return out


def make_blend_window(tile: int, stride: int, mode: str = 'none') -> np.ndarray:
    """Return (tile,tile) float32 blending weights.

    - If stride >= tile, blending is meaningless; return all-ones.
    - For overlapped tiling (stride < tile), Hann window reduces border artifacts.
    """
    mode = str(mode).lower()
    if int(stride) >= int(tile):
        return np.ones((tile, tile), dtype=np.float32)
    if mode == 'hann':
        w = np.hanning(tile).astype(np.float32)
        w = np.maximum(w, 1e-3)
        win = w[:, None] * w[None, :]
        return win.astype(np.float32)
    return np.ones((tile, tile), dtype=np.float32)


def bbox_from_fg(label_tile: np.ndarray, jitter: int = 0) -> np.ndarray:
    """Oracle bbox (union of all fg). For debug only."""
    fg = (label_tile > 0) & (label_tile != int(PAV_IGNORE_INDEX))
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return full_box(label_tile.shape[0])
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    if jitter > 0:
        x1 = max(0, x1 - np.random.randint(0, jitter + 1))
        y1 = max(0, y1 - np.random.randint(0, jitter + 1))
        x2 = min(label_tile.shape[1], x2 + np.random.randint(0, jitter + 1))
        y2 = min(label_tile.shape[0], y2 + np.random.randint(0, jitter + 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def label_to_color(label: np.ndarray) -> np.ndarray:
    h, w = label.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in enumerate(CLASS_COLORS):
        out[label == cid] = np.array(rgb, dtype=np.uint8)
    return out


def build_model(
    args: argparse.Namespace,
    ckpt_path: str | None = None,
    prefer_ema: int | None = None,
    state_override: dict | None = None,
    cfg_override: dict | None = None,
) -> torch.nn.Module:
    sam_ckpt = os.path.join(args.checkpoint, sam_model_checkpoint[args.model_type])
    if not os.path.exists(sam_ckpt):
        cand = list(Path(args.checkpoint).glob('*.pth'))
        if len(cand) == 0:
            raise FileNotFoundError(f'No SAM checkpoint found under {args.checkpoint}')
        sam_ckpt = str(cand[0])

    prefer_ema = int(getattr(args, 'prefer_ema', 0) if prefer_ema is None else prefer_ema)
    if state_override is not None:
        ckpt = {'model': state_override, 'model_cfg': dict(cfg_override or {})}
        model_state = state_override
        model_cfg = dict(cfg_override or {})
    else:
        ckpt_path = str(args.ckpt if ckpt_path is None else ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict) and prefer_ema == 1 and ('ema_model' in ckpt):
            model_state = ckpt['ema_model']
        else:
            model_state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model_cfg = ckpt.get('model_cfg', {}) if isinstance(ckpt, dict) else {}

    fusion_mode = str(getattr(args, 'fusion_mode', 'auto')).lower()
    if fusion_mode == 'auto':
        fusion_mode = str(model_cfg.get('fusion_mode', 'auto')).lower() if isinstance(model_cfg, dict) and str(model_cfg.get('fusion_mode', '')) else fusion_mode
        if fusion_mode == 'auto':
            fusion_mode = 'hybrid' if any(str(k).startswith('fusion_2d3d.spatial_gate') for k in model_state.keys()) else 'global'
    use_logit_refiner = int(getattr(args, 'use_logit_refiner', -1))
    if use_logit_refiner < 0:
        if isinstance(model_cfg, dict) and 'use_logit_refiner' in model_cfg:
            use_logit_refiner = int(model_cfg.get('use_logit_refiner', 0))
        else:
            use_logit_refiner = 1 if any(str(k).startswith('logit_refiner.') for k in model_state.keys()) else 0
    refiner_hidden = int(getattr(args, 'refiner_hidden', 32))
    if isinstance(model_cfg, dict) and 'refiner_hidden' in model_cfg:
        refiner_hidden = int(model_cfg.get('refiner_hidden', refiner_hidden))
    for k, v in model_state.items():
        if str(k).startswith('logit_refiner.net.0.weight') and hasattr(v, 'shape'):
            refiner_hidden = int(v.shape[0])
            break

    use_specialist_refiner = int(getattr(args, 'use_specialist_refiner', -1))
    if use_specialist_refiner < 0:
        if isinstance(model_cfg, dict) and 'use_specialist_refiner' in model_cfg:
            use_specialist_refiner = int(model_cfg.get('use_specialist_refiner', 0))
        else:
            use_specialist_refiner = 1 if any(str(k).startswith('specialist_refiner.') for k in model_state.keys()) else 0
    line_refiner_hidden = int(getattr(args, 'line_refiner_hidden', 48))
    surface_refiner_hidden = int(getattr(args, 'surface_refiner_hidden', 32))
    if isinstance(model_cfg, dict):
        line_refiner_hidden = int(model_cfg.get('line_refiner_hidden', line_refiner_hidden))
        surface_refiner_hidden = int(model_cfg.get('surface_refiner_hidden', surface_refiner_hidden))
    for k, v in model_state.items():
        if str(k).startswith('specialist_refiner.line_head.stem.net.0.weight') and hasattr(v, 'shape'):
            line_refiner_hidden = int(v.shape[0])
        if str(k).startswith('specialist_refiner.surface_head.stem.net.0.weight') and hasattr(v, 'shape'):
            surface_refiner_hidden = int(v.shape[0])

    sam_image_size = int(getattr(args, 'sam_image_size', 0))
    if sam_image_size <= 0:
        sam_image_size = int(model_cfg.get('sam_image_size', 256))

    sam_model = sam_model_registry[args.model_type](
        image_size=int(sam_image_size),
        keep_resolution=True,
        checkpoint=sam_ckpt,
        num_multimask_outputs=int(args.num_fg_classes),
    )

    model = GeoFormerX(
        sam=sam_model,
        bottleneck_dim=int(args.bottleneck_dim),
        embedding_dim=int(args.embedding_dim),
        expert_num=int(args.expert_num),
        gate_topk=int(args.moe_topk),
        gate_temperature=float(args.moe_temp),
        gate_noise_std=float(args.moe_noise_std),
        style_scale=float(args.moe_style_scale),
        use_fusion_2d3d=bool(int(args.use_fusion_2d3d) == 1),
        fusion_hidden=int(args.fusion_hidden),
        fusion_mode=fusion_mode,
        use_logit_refiner=bool(use_logit_refiner == 1),
        refiner_hidden=refiner_hidden,
        use_specialist_refiner=bool(use_specialist_refiner == 1),
        line_refiner_hidden=line_refiner_hidden,
        surface_refiner_hidden=surface_refiner_hidden,
    )

    if state_override is not None:
        model.load_parameters(state_override)
    elif isinstance(ckpt, dict):
        if prefer_ema == 1 and ('ema_model' in ckpt):
            model.load_parameters(ckpt['ema_model'])
        elif 'model' in ckpt:
            model.load_parameters(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    device = torch.device(args.device)
    model.to(device)
    if torch.cuda.device_count() > 1 and len(args.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    model.eval()
    return model


def infer_prob_maps_fg(
    model: torch.nn.Module,
    img4: np.ndarray,
    label: np.ndarray,
    meta: Tuple[int, int, int, int, int],
    tile: int,
    stride: int,
    batch_size: int,
    prompt_mode: str,
    fg_thresh: float,
    tta_hflip: bool,
    blend: str,
    device: torch.device,
    class_logit_bias_map: Dict[int, float] | None = None,
    secondary_model: torch.nn.Module | None = None,
    secondary_class_weight_map: Dict[int, float] | None = None,
    secondary_fuse_mode: str = 'delta',
    class_ckpt_models: Dict[int, torch.nn.Module] | None = None,
    class_ckpt_weight_map: Dict[int, float] | None = None,
    class_ckpt_fuse_mode: str = 'delta',
    collect_debug: bool = False,
    stitch_mode: str = 'logits',
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return (pred_label, prob_fg, debug) for one image.

    - pred_label: (H,W) uint8 in {0..7} produced by argmax over 8-class logits
                 (bg logit is fixed to 0, fg logits are predicted).
    - prob_fg   : (7,H,W) float32 softmax probability maps for foreground classes.

    NOTE: We stitch by averaging **logits**, then apply argmax, to keep the
    evaluation prediction rule consistent with validation (argmax on logits).
    """
    orig_h, orig_w = img4.shape[:2]
    img4_p, label_p = pad_to_multiple(img4, label, tile=tile)
    Hp, Wp = img4_p.shape[:2]

    coords = safe_tile_coords(Hp, Wp, tile=tile, stride=stride)

    # accumulate 8-class logits (bg + 7 fg) to stay consistent with validation
    logits_acc = np.zeros((1 + NUM_FG_CLASSES, Hp, Wp), dtype=np.float32)
    cnt_acc = np.zeros((Hp, Wp), dtype=np.float32)

    blend_w = make_blend_window(tile=tile, stride=stride, mode=str(blend))
    resize_warned = False
    stitch_mode = str(stitch_mode).lower().strip()
    if stitch_mode not in {'logits', 'hard'}:
        stitch_mode = 'logits'
    debug = {'alpha_rows': [], 'gate_sum': [], 'gate_top1': [], 'gate_count': [], 'gate_entropy_sum': []}

    modal_idx, l1_idx, l2_idx, l3_idx, l4_idx = meta

    # batch over tiles
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        imgs = []
        boxes = []
        names = []

        for (x0, y0) in batch_coords:
            tile_img = img4_p[y0:y0 + tile, x0:x0 + tile, :]
            tile_lbl = label_p[y0:y0 + tile, x0:x0 + tile]
            imgs.append(tile_img.transpose(2, 0, 1))

            if prompt_mode == 'gt_box':
                boxes.append(bbox_from_fg(tile_lbl))
            else:
                boxes.append(full_box(tile))

            names.append(f"x{x0}y{y0}")

        img_t = torch.from_numpy(np.stack(imgs, axis=0)).float().to(device)
        box_t = torch.from_numpy(np.stack(boxes, axis=0)).float().to(device)

        data: Dict[str, object] = {
            'img': img_t,
            'box': box_t,
            'modal': torch.full((img_t.shape[0],), int(modal_idx), dtype=torch.long, device=device),
            'route': (
                torch.full((img_t.shape[0],), int(l1_idx), dtype=torch.long, device=device),
                torch.full((img_t.shape[0],), int(l2_idx), dtype=torch.long, device=device),
                torch.full((img_t.shape[0],), int(l3_idx), dtype=torch.long, device=device),
                torch.full((img_t.shape[0],), int(l4_idx), dtype=torch.long, device=device),
                torch.zeros((img_t.shape[0],), dtype=torch.long, device=device),
            ),
            'name': names,
        }

        with torch.no_grad():
            # base prediction
            if bool(collect_debug):
                base_out = model(data, return_gates=True, return_alpha=True)
            else:
                base_out = model(data)
            gates_dbg = []
            alpha_dbg = None
            if isinstance(base_out, dict):
                logits_fg = base_out['masks']
                gates_dbg = base_out.get('gates', []) or []
                alpha_dbg = base_out.get('fusion_alpha', None)
            else:
                logits_fg = base_out

            if bool(collect_debug):
                if alpha_dbg is not None:
                    a = alpha_dbg.detach().float()
                    if a.ndim >= 3:
                        a_vals = a.flatten(1).mean(dim=1).detach().cpu().numpy().astype(float)
                    else:
                        a_vals = a.view(-1).detach().cpu().numpy().astype(float)
                    for j, (x0d, y0d) in enumerate(batch_coords):
                        if j < len(a_vals):
                            debug['alpha_rows'].append({'tile_id': int(i + j), 'x0': int(x0d), 'y0': int(y0d), 'alpha': float(a_vals[j])})
                if gates_dbg:
                    while len(debug['gate_sum']) < len(gates_dbg):
                        E0 = int(gates_dbg[len(debug['gate_sum'])].shape[1])
                        debug['gate_sum'].append(np.zeros((E0,), dtype=np.float64))
                        debug['gate_top1'].append(np.zeros((E0,), dtype=np.float64))
                        debug['gate_count'].append(0)
                        debug['gate_entropy_sum'].append(0.0)
                    for li, g in enumerate(gates_dbg):
                        gd = g.detach().float().cpu().numpy().astype(np.float64)
                        debug['gate_sum'][li] += gd.sum(axis=0)
                        debug['gate_top1'][li] += np.bincount(np.argmax(gd, axis=1), minlength=gd.shape[1]).astype(np.float64)
                        debug['gate_entropy_sum'][li] += float((-(gd * np.log(np.clip(gd, 1e-8, 1.0))).sum(axis=1)).sum())
                        debug['gate_count'][li] += int(gd.shape[0])

            if bool(tta_hflip):
                data_f = dict(data)
                data_f['img'] = torch.flip(data['img'], dims=[3])
                data_f['box'] = hflip_boxes(data['box'], tile=tile)
                logits_fg_f = model(data_f)
                if isinstance(logits_fg_f, dict):
                    logits_fg_f = logits_fg_f['masks']
                logits_fg_f = torch.flip(logits_fg_f, dims=[3])
                logits_fg = 0.5 * (logits_fg + logits_fg_f)

            # When training used sam_image_size > tile_size, GeoFormerX may emit
            # logits at SAM resolution (e.g. 320x320) while stitching expects tile
            # resolution (e.g. 256x256). Match training-time loss behavior by
            # resizing logits back to the current tile size before stitching.
            if tuple(logits_fg.shape[-2:]) != (int(tile), int(tile)):
                if not resize_warned:
                    print(f'[Eval] resize logits from {tuple(logits_fg.shape[-2:])} to ({int(tile)}, {int(tile)}) for stitching.', flush=True)
                    resize_warned = True
                logits_fg = F.interpolate(logits_fg.float(), size=(int(tile), int(tile)), mode='bilinear', align_corners=False)

            logits_all = logits_with_bg(logits_fg.float())  # (B,8,...) bg logit=0

            if secondary_model is not None and secondary_class_weight_map:
                logits_fg_sec = secondary_model(data)
                if isinstance(logits_fg_sec, dict):
                    logits_fg_sec = logits_fg_sec['masks']
                if bool(tta_hflip):
                    data_f2 = dict(data)
                    data_f2['img'] = torch.flip(data['img'], dims=[3])
                    data_f2['box'] = hflip_boxes(data['box'], tile=tile)
                    logits_fg_sec_f = secondary_model(data_f2)
                    if isinstance(logits_fg_sec_f, dict):
                        logits_fg_sec_f = logits_fg_sec_f['masks']
                    logits_fg_sec_f = torch.flip(logits_fg_sec_f, dims=[3])
                    logits_fg_sec = 0.5 * (logits_fg_sec + logits_fg_sec_f)
                if tuple(logits_fg_sec.shape[-2:]) != (int(tile), int(tile)):
                    logits_fg_sec = F.interpolate(logits_fg_sec.float(), size=(int(tile), int(tile)), mode='bilinear', align_corners=False)
                fuse_mode = str(secondary_fuse_mode).lower().strip()
                for cid, class_weight in dict(secondary_class_weight_map).items():
                    cid_i = int(cid)
                    class_weight_f = float(class_weight)
                    if 1 <= cid_i < logits_all.shape[1] and class_weight_f != 0.0:
                        sec_logits = logits_fg_sec[:, cid_i - 1].float()
                        if fuse_mode == 'delta':
                            # Safer for base+specialist fusion: only inject the secondary residual.
                            # This avoids double-counting the whole base class logit.
                            base_logits = logits_fg[:, cid_i - 1].float()
                            sec_logits = sec_logits - base_logits
                        logits_all[:, cid_i] = logits_all[:, cid_i] + class_weight_f * sec_logits

            if class_ckpt_models:
                fuse_mode_multi = str(class_ckpt_fuse_mode).lower().strip()
                for cid, class_model in dict(class_ckpt_models).items():
                    cid_i = int(cid)
                    if not (1 <= cid_i < logits_all.shape[1]):
                        continue
                    class_weight_f = float((class_ckpt_weight_map or {}).get(cid_i, 1.0))
                    if class_weight_f == 0.0:
                        continue
                    logits_fg_cls = class_model(data)
                    if isinstance(logits_fg_cls, dict):
                        logits_fg_cls = logits_fg_cls['masks']
                    if bool(tta_hflip):
                        data_f3 = dict(data)
                        data_f3['img'] = torch.flip(data['img'], dims=[3])
                        data_f3['box'] = hflip_boxes(data['box'], tile=tile)
                        logits_fg_cls_f = class_model(data_f3)
                        if isinstance(logits_fg_cls_f, dict):
                            logits_fg_cls_f = logits_fg_cls_f['masks']
                        logits_fg_cls_f = torch.flip(logits_fg_cls_f, dims=[3])
                        logits_fg_cls = 0.5 * (logits_fg_cls + logits_fg_cls_f)
                    if tuple(logits_fg_cls.shape[-2:]) != (int(tile), int(tile)):
                        logits_fg_cls = F.interpolate(logits_fg_cls.float(), size=(int(tile), int(tile)), mode='bilinear', align_corners=False)
                    cls_logits = logits_fg_cls[:, cid_i - 1].float()
                    if fuse_mode_multi == 'replace':
                        logits_all[:, cid_i] = cls_logits
                    elif fuse_mode_multi == 'delta':
                        base_logits = logits_fg[:, cid_i - 1].float()
                        logits_all[:, cid_i] = logits_all[:, cid_i] + class_weight_f * (cls_logits - base_logits)
                    else:
                        logits_all[:, cid_i] = logits_all[:, cid_i] + class_weight_f * cls_logits

            if class_logit_bias_map:
                for cid, class_bias in dict(class_logit_bias_map).items():
                    cid_i = int(cid)
                    if 0 <= cid_i < logits_all.shape[1] and float(class_bias) != 0.0:
                        logits_all[:, cid_i] = logits_all[:, cid_i] + float(class_bias)

        logits_all_np = logits_all.detach().cpu().numpy().astype(np.float32)

        for b, (x0, y0) in enumerate(batch_coords):
            if stitch_mode == 'hard':
                tile_pred = logits_all_np[b].argmax(axis=0)
                tile_onehot = np.eye(1 + NUM_FG_CLASSES, dtype=np.float32)[tile_pred].transpose(2, 0, 1)
                logits_acc[:, y0:y0 + tile, x0:x0 + tile] += tile_onehot * blend_w[None, :, :]
            else:
                logits_acc[:, y0:y0 + tile, x0:x0 + tile] += logits_all_np[b] * blend_w[None, :, :]
            cnt_acc[y0:y0 + tile, x0:x0 + tile] += blend_w

    cnt_acc = np.maximum(cnt_acc, 1e-6)
    logits_acc = logits_acc / cnt_acc[None, :, :]
    logits_acc = logits_acc[:, :int(orig_h), :int(orig_w)]

    # --- prediction by argmax over 8-class logits ---
    pred = logits_acc.argmax(axis=0).astype(np.uint8)  # 0..7

    # --- probabilities (for optional thresholding + saving) ---
    m = logits_acc.max(axis=0, keepdims=True)
    e = np.exp(logits_acc - m)
    prob_all = e / (e.sum(axis=0, keepdims=True) + 1e-12)
    prob_fg = prob_all[1:1 + NUM_FG_CLASSES]

    if float(fg_thresh) > 0:
        max_fg = prob_fg.max(axis=0)
        pred = pred.copy()
        pred[(pred != 0) & (max_fg < float(fg_thresh))] = 0

    # For visualization, force ignore pixels to background (metrics will ignore anyway)
    if label is not None:
        pred[label == int(PAV_IGNORE_INDEX)] = 0

    return pred, prob_fg, debug


def _mean(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64)))


def main() -> None:
    args = parse_args()
    class_logit_bias_map = parse_class_weight_map(getattr(args, 'class_logit_bias_map', ''))
    secondary_class_weight_map = parse_class_weight_map(getattr(args, 'secondary_class_weight_map', ''))

    if int(args.num_fg_classes) != int(NUM_FG_CLASSES):
        raise ValueError(f"num_fg_classes={args.num_fg_classes} but pavement_config.NUM_FG_CLASSES={NUM_FG_CLASSES}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'pred_color').mkdir(exist_ok=True)
    (out_dir / 'pred_label').mkdir(exist_ok=True)
    (out_dir / 'gt_color').mkdir(exist_ok=True)
    if int(args.save_prob) == 1:
        (out_dir / 'prob_fg').mkdir(exist_ok=True)
    if int(getattr(args, 'save_paper_outputs', 1)) == 1:
        (out_dir / 'error_overlay').mkdir(exist_ok=True)
        (out_dir / 'paper_panels').mkdir(exist_ok=True)
        (out_dir / 'paper_figures').mkdir(exist_ok=True)
        (out_dir / 'debug').mkdir(exist_ok=True)

    device = torch.device(args.device)

    # Optional one-file bundle mode.  A bundle created by _merge_bundle.py
    # contains: model (single merged model for normal use/few-shot), plus optional
    # base_model + class_bank for exact in-memory class-delta fusion.
    bundle_ckpt = None
    if int(getattr(args, 'use_bundle_fusion', 0)) == 1:
        try:
            bundle_ckpt = torch.load(str(args.ckpt), map_location='cpu')
        except Exception as e:
            raise RuntimeError(f'Failed to load bundle checkpoint {args.ckpt}: {e}')

    if isinstance(bundle_ckpt, dict) and ('base_model' in bundle_ckpt):
        model = build_model(args, state_override=bundle_ckpt['base_model'], cfg_override=bundle_ckpt.get('base_model_cfg', {}))
        print('[Eval] use_bundle_fusion=1: primary logits loaded from bundle base_model.', flush=True)
    else:
        model = build_model(args)

    secondary_model = None
    if str(getattr(args, 'ckpt_secondary', '')).strip() != '':
        secondary_model = build_model(args, ckpt_path=str(getattr(args, 'ckpt_secondary')), prefer_ema=int(getattr(args, 'prefer_ema', 0)))

    class_ckpt_path_map = parse_class_path_map(getattr(args, 'class_ckpt_map', ''))
    class_ckpt_weight_map = parse_class_weight_map(getattr(args, 'class_ckpt_weight_map', ''))
    class_ckpt_models = {}
    for cid, ckpt_path in sorted(class_ckpt_path_map.items()):
        class_ckpt_models[int(cid)] = build_model(args, ckpt_path=str(ckpt_path), prefer_ema=int(getattr(args, 'prefer_ema', 0)))

    if isinstance(bundle_ckpt, dict) and int(getattr(args, 'use_bundle_fusion', 0)) == 1 and isinstance(bundle_ckpt.get('class_bank', None), dict):
        bank = bundle_ckpt.get('class_bank', {})
        bank_cfg = bundle_ckpt.get('class_bank_cfg', bundle_ckpt.get('model_cfg', {}))
        for cid_raw, state in sorted(bank.items(), key=lambda kv: int(kv[0])):
            cid = int(cid_raw)
            if cid in class_ckpt_models:
                continue
            class_ckpt_models[cid] = build_model(args, state_override=state, cfg_override=bank_cfg)
        if len(class_ckpt_weight_map) == 0:
            recipe = bundle_ckpt.get('fusion_recipe', {}) if isinstance(bundle_ckpt.get('fusion_recipe', {}), dict) else {}
            class_ckpt_weight_map = {int(k): float(v) for k, v in dict(recipe.get('class_ckpt_weight_map', {})).items()}
        print(f'[Eval] use_bundle_fusion=1: loaded class_bank classes={sorted([int(k) for k in bank.keys()])}', flush=True)

    if len(class_logit_bias_map) > 0:
        print(f'[Eval] class_logit_bias_map={class_logit_bias_map}', flush=True)
    if secondary_model is not None and len(secondary_class_weight_map) > 0:
        print(f'[Eval] secondary_class_weight_map={secondary_class_weight_map}, secondary_fuse_mode={str(getattr(args, "secondary_fuse_mode", "delta"))}', flush=True)
    if len(class_ckpt_models) > 0:
        print(f'[Eval] class_ckpt_map={class_ckpt_path_map}, class_ckpt_weight_map={class_ckpt_weight_map}, class_ckpt_fuse_mode={str(getattr(args, "class_ckpt_fuse_mode", "delta"))}', flush=True)

    param_info = count_parameters(model)
    save_json(out_dir / 'trainable_params.json', param_info)

    # meta (use class 1 folder only to generate hierarchy indices)
    task_folder = classid_to_task_folder(1, modal='VehicleProfiler')
    meta = resolve_task_folder_meta(task_folder)

    # Paths
    data_root = Path(args.data_path)
    split = str(args.split)
    img_dir = data_root / split / 'image'
    lbl_dir = data_root / split / 'label'
    threed_root = data_root / '3Ddate'

    if not img_dir.is_dir():
        raise FileNotFoundError(f'Image dir not found: {img_dir}')
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f'Label dir not found: {lbl_dir}')

    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
    if len(img_files) == 0:
        raise RuntimeError(f'No images in {img_dir}')

    cm_total = np.zeros((8, 8), dtype=np.int64)
    cm_seam_total = np.zeros((8, 8), dtype=np.int64)
    bnd_sum = np.zeros((8,), dtype=np.float64)
    bnd_cnt = np.zeros((8,), dtype=np.float64)
    alpha_rows_global: List[Dict[str, object]] = []
    gate_sum_global: List[np.ndarray] = []
    gate_top1_global: List[np.ndarray] = []
    gate_count_global: List[int] = []
    gate_entropy_global: List[float] = []

    # Speed stats (rough, end-to-end per image)
    import time
    t_all0 = time.time()
    infer_times: List[float] = []

    # Per-image logging
    rows: List[Dict[str, object]] = []
    visual_names = {x.strip() for x in str(getattr(args, 'visual_names', '')).replace(';', ',').split(',') if x.strip()}
    max_visuals = int(getattr(args, 'max_visuals', 24))
    saved_visuals = 0

    for img_path in tqdm(img_files, desc='Infer+Eval'):
        name = img_path.stem
        # 2D
        img2d = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        h, w = img2d.shape[:2]

        # 3D path via mapping
        img3d_path = find_3d_path(str(threed_root), split, img_path.name)
        img3d = np.array(Image.open(img3d_path).convert('L'), dtype=np.uint8)[:, :, None]
        if img3d.shape[:2] != img2d.shape[:2]:
            img3d = np.array(Image.fromarray(img3d.squeeze(-1)).resize((w, h), resample=Image.NEAREST), dtype=np.uint8)[:, :, None]

        img4 = np.concatenate([img2d, img3d], axis=-1).astype(np.float32) / 255.0

        # GT label
        gt_path = lbl_dir / f"{name}.bmp"
        if not gt_path.exists():
            # try png
            gt_candidates = list(lbl_dir.glob(f"{name}.*"))
            if len(gt_candidates) == 0:
                raise FileNotFoundError(f'GT label not found for {name}')
            gt_path = gt_candidates[0]

        gt_rgb = np.array(Image.open(gt_path).convert('RGB'), dtype=np.uint8)
        gt = rgb_to_label_nearest(gt_rgb, dist_threshold=PAV_DIST_THRESHOLD, ignore_index=PAV_IGNORE_INDEX)

        # Inference
        t0 = time.time()
        pred, prob_fg, debug = infer_prob_maps_fg(
            model=model,
            img4=img4,
            label=gt,
            meta=meta,
            tile=int(args.tile_size),
            stride=int(args.tile_stride),
            batch_size=int(args.batch_size),
            prompt_mode=str(args.prompt_mode),
            fg_thresh=float(args.fg_thresh),
            tta_hflip=bool(int(getattr(args, 'tta_hflip', 0)) == 1),
            blend=str(getattr(args, 'blend', 'none')),
            device=device,
            class_logit_bias_map=class_logit_bias_map if len(class_logit_bias_map) > 0 else None,
            secondary_model=secondary_model,
            secondary_class_weight_map=secondary_class_weight_map if len(secondary_class_weight_map) > 0 else None,
            secondary_fuse_mode=str(getattr(args, 'secondary_fuse_mode', 'delta')),
            class_ckpt_models=class_ckpt_models if len(class_ckpt_models) > 0 else None,
            class_ckpt_weight_map=class_ckpt_weight_map if len(class_ckpt_weight_map) > 0 else None,
            class_ckpt_fuse_mode=str(getattr(args, 'class_ckpt_fuse_mode', 'delta')),
            collect_debug=bool(int(getattr(args, 'collect_debug', 1)) == 1),
            stitch_mode=str(getattr(args, 'stitch_mode', 'logits')),
        )
        infer_times.append(time.time() - t0)

        # Save (use .bmp for color masks to match your original labels)
        Image.fromarray(label_to_color(pred)).save(out_dir / 'pred_color' / f"{name}.bmp")
        Image.fromarray(pred.astype(np.uint8)).save(out_dir / 'pred_label' / f"{name}.png")
        Image.fromarray(label_to_color(gt.astype(np.uint8))).save(out_dir / 'gt_color' / f"{name}.bmp")

        alpha_mean_img = float('nan')
        gate_mean_img = None
        if isinstance(debug, dict):
            ar = debug.get('alpha_rows', []) or []
            if ar:
                for rr in ar:
                    rr = dict(rr); rr['name'] = name; alpha_rows_global.append(rr)
                alpha_vals = [float(rr['alpha']) for rr in ar if np.isfinite(float(rr.get('alpha', np.nan)))]
                if alpha_vals: alpha_mean_img = float(np.mean(alpha_vals))
            gs = debug.get('gate_sum', []) or []
            gc = debug.get('gate_count', []) or []
            while len(gate_sum_global) < len(gs):
                E0 = int(len(gs[len(gate_sum_global)]))
                gate_sum_global.append(np.zeros((E0,), dtype=np.float64))
                gate_top1_global.append(np.zeros((E0,), dtype=np.float64))
                gate_count_global.append(0)
                gate_entropy_global.append(0.0)
            for li, arr in enumerate(gs):
                gate_sum_global[li] += np.asarray(arr, dtype=np.float64)
                if li < len(debug.get('gate_top1', [])):
                    gate_top1_global[li] += np.asarray(debug.get('gate_top1', [])[li], dtype=np.float64)
                gate_count_global[li] += int(gc[li]) if li < len(gc) else 0
                if li < len(debug.get('gate_entropy_sum', [])):
                    gate_entropy_global[li] += float(debug.get('gate_entropy_sum', [])[li])
            if gs:
                mats = []
                for li, arr in enumerate(gs):
                    cnt = max(1, int(gc[li]) if li < len(gc) else 1)
                    mats.append(np.asarray(arr, dtype=np.float64) / cnt)
                if mats:
                    gate_mean_img = np.mean(np.stack(mats, axis=0), axis=0)

        if int(getattr(args, 'save_paper_outputs', 1)) == 1:
            Image.fromarray(error_overlay(gt, pred, ignore_index=int(PAV_IGNORE_INDEX))).save(out_dir / 'error_overlay' / f"{name}.png")
            if (saved_visuals < max_visuals) or (name in visual_names):
                save_qual_panel(out_dir / 'paper_panels' / f"{name}_panel.png", rgb=img2d, depth=img3d.squeeze(-1), gt=gt, pred=pred, title=name, alpha=alpha_mean_img, gate_prob=gate_mean_img, ignore_index=int(PAV_IGNORE_INDEX))
                saved_visuals += 1

        if int(args.save_prob) == 1:
            # save each class prob as uint8
            for i, cid in enumerate(FG_CLASS_IDS):
                pmap = (np.clip(prob_fg[i], 0.0, 1.0) * 255.0).astype(np.uint8)
                Image.fromarray(pmap).save(out_dir / 'prob_fg' / f"{name}_c{cid}.png")

        # Metrics
        cm = confusion_matrix(gt, pred, num_classes=8, ignore_index=int(PAV_IGNORE_INDEX))
        cm_total += cm
        seam_mask = seam_band_mask(h, w, tile=int(args.tile_size), stride=int(args.tile_stride), band=int(getattr(args, 'seam_band', 8)))
        if seam_mask.any():
            gt_seam = gt.copy(); pred_seam = pred.copy(); gt_seam[~seam_mask] = int(PAV_IGNORE_INDEX)
            cm_seam_total += confusion_matrix(gt_seam, pred_seam, num_classes=8, ignore_index=int(PAV_IGNORE_INDEX))

        # boundary f1 per class
        bnd = boundary_f1_per_class(gt, pred, num_classes=8, ignore_index=int(PAV_IGNORE_INDEX))
        for c in range(8):
            if not np.isnan(bnd[c]):
                bnd_sum[c] += float(bnd[c])
                bnd_cnt[c] += 1.0

        per = per_class_metrics_from_cm(cm)

        # --- per-image metrics from per-image CM ---
        fg_idx = list(range(1, 8))
        mdice_all = _mean(per.dice)
        miou_all = _mean(per.iou)
        mf1_all = _mean(per.f1)
        mf2_all = _mean(per.f2)

        mdice_fg = _mean(per.dice[fg_idx])
        miou_fg = _mean(per.iou[fg_idx])
        mf1_fg = _mean(per.f1[fg_idx])
        mf2_fg = _mean(per.f2[fg_idx])
        acc = float(np.trace(cm) / max(cm.sum(), 1))

        rows.append({
            'name': name,
            'H': h,
            'W': w,
            'acc': acc,
            'mDice_all': mdice_all,
            'mIoU_all': miou_all,
            'mF1_all': mf1_all,
            'mF2_all': mf2_all,
            'mDice_fg': mdice_fg,
            'mIoU_fg': miou_fg,
            'mF1_fg': mf1_fg,
            'mF2_fg': mf2_fg,
            'CrackDice': float(per.dice[1]) if len(per.dice) > 1 else float('nan'),
            'alpha_mean': alpha_mean_img,
        })

    # Summary from total CM
    per_total = per_class_metrics_from_cm(cm_total)
    fg_idx = list(range(1, 8))
    mdice_all = _mean(per_total.dice)
    miou_all = _mean(per_total.iou)
    mf1_all = _mean(per_total.f1)
    mf2_all = _mean(per_total.f2)

    mdice_fg = _mean(per_total.dice[fg_idx])
    miou_fg = _mean(per_total.iou[fg_idx])
    mf1_fg = _mean(per_total.f1[fg_idx])
    mf2_fg = _mean(per_total.f2[fg_idx])

    overall = overall_metrics_from_cm(cm_total)
    acc = float(overall.get('acc', float('nan')))
    kappa = float(overall.get('kappa', float('nan')))
    mcc = float(overall.get('mcc', float('nan')))
    fwiou = float(freq_weighted_iou(per_total.iou, per_total.support))

    bnd_mean = np.full((8,), np.nan, dtype=np.float64)
    for c in range(8):
        if bnd_cnt[c] > 0:
            bnd_mean[c] = bnd_sum[c] / bnd_cnt[c]
    bnd_f1_fg = nanmean(bnd_mean[1:])
    crack_dice = float(per_total.dice[1]) if len(per_total.dice) > 1 else float('nan')
    seam_mdice_fg = foreground_mdice_from_cm(cm_seam_total) if cm_seam_total.sum() > 0 else float('nan')
    per_seam = per_class_metrics_from_cm(cm_seam_total) if cm_seam_total.sum() > 0 else None

    summary_lines = []
    summary_lines.append('=== Overall (from total confusion matrix) ===')
    summary_lines.append(f'PixelAcc: {acc:.6f}')
    summary_lines.append(f'mDice_all: {mdice_all:.6f} ({mdice_all * 100:.3f}%)')
    summary_lines.append(f'mIoU_all:  {miou_all:.6f} ({miou_all * 100:.3f}%)')
    summary_lines.append(f'mDice_fg: {mdice_fg:.6f}')
    summary_lines.append(f'CrackDice: {crack_dice:.6f}')
    summary_lines.append(f'BndF1_fg:  {bnd_f1_fg:.6f}')
    summary_lines.append(f'SeamDice_fg: {seam_mdice_fg:.6f}')
    summary_lines.append(f'mIoU_fg:  {miou_fg:.6f}')
    summary_lines.append(f'mF1_fg:   {mf1_fg:.6f}')
    summary_lines.append(f'mF2_fg:   {mf2_fg:.6f}')
    summary_lines.append(f'FWIoU:    {fwiou:.6f}')
    summary_lines.append(f'Kappa:    {kappa:.6f}')
    summary_lines.append(f'MCC:      {mcc:.6f}')
    summary_lines.append('')
    summary_lines.append('=== Per-class (Dice/IoU/F1/F2/BoundaryF1) ===')
    for cid in range(8):
        summary_lines.append(
            f'[{cid}] {CLASS_NAMES[cid]:<10} '
            f'Dice={per_total.dice[cid]:.4f} '
            f'IoU={per_total.iou[cid]:.4f} '
            f'F1={per_total.f1[cid]:.4f} '
            f'F2={per_total.f2[cid]:.4f} '
            f'BndF1={bnd_mean[cid]:.4f}'
        )

    (out_dir / 'summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')

    # ------------------------------------------------------------------
    # Baseline-compatible report file (predict_result.txt)
    # This is formatted similarly to common UNet baselines so that the
    # numbers can be copied into the same comparison table.
    # ------------------------------------------------------------------
    try:
        total_time = time.time() - t_all0
        avg_time = float(np.mean(np.asarray(infer_times, dtype=np.float64))) if infer_times else 0.0
        fps = (1.0 / avg_time) if avg_time > 0 else 0.0
    except Exception:
        total_time, avg_time, fps = 0.0, 0.0, 0.0

    report_lines: List[str] = []
    report_lines.append('Publication Metrics Report (PAVEMENT, deployment-aligned full-box protocol)')
    report_lines.append('=' * 88)
    report_lines.append(f"Split                : {split}")
    report_lines.append(f"Total Images         : {len(img_files)}")
    report_lines.append(f"Inference FPS        : {fps:.2f} fps")
    report_lines.append(f"Avg Time per Image   : {avg_time * 1000:.2f} ms")
    report_lines.append(f"Total Wall Time      : {total_time:.2f} s")
    report_lines.append(f"tile_size/stride     : {int(args.tile_size)}/{int(args.tile_stride)}")
    report_lines.append(f"prompt_mode          : {str(args.prompt_mode)}")
    report_lines.append(f"fg_thresh            : {float(args.fg_thresh)}")
    report_lines.append(f"stitch_mode          : {str(getattr(args, 'stitch_mode', 'logits'))}")
    report_lines.append(f"trainable_params     : {int(param_info.get('trainable_params', 0))}")
    report_lines.append(f"trainable_ratio      : {float(param_info.get('trainable_ratio', 0.0)):.6f}")
    report_lines.append('-' * 88)
    report_lines.append(f"mDice_fg             : {mdice_fg * 100:.4f}%")
    report_lines.append(f"Crack Dice           : {crack_dice * 100:.4f}%")
    report_lines.append(f"Boundary-F1_fg       : {bnd_f1_fg * 100:.4f}%")
    report_lines.append(f"Seam-band Dice_fg    : {seam_mdice_fg * 100:.4f}%")
    report_lines.append('-' * 88)
    report_lines.append(f"{'Class ID':<10} | {'Class Name':<15} | {'IoU (%)':<10} | {'Dice (%)':<10}")
    report_lines.append('-' * 88)

    for cid in range(8):
        report_lines.append(
            f"{cid:<10} | {CLASS_NAMES[cid]:<15} | {per_total.iou[cid] * 100:<10.2f} | {per_total.dice[cid] * 100:<10.2f}"
        )

    report_lines.append('-' * 88)
    report_lines.append(f"Mean IoU (mIoU)      : {miou_all * 100:.4f}%")
    report_lines.append(f"Mean Dice (mDice)    : {mdice_all * 100:.4f}%")
    report_lines.append('=' * 88)

    (out_dir / 'predict_result.txt').write_text('\n'.join(report_lines), encoding='utf-8')
    (out_dir / 'console_report_internal.txt').write_text('\n'.join(summary_lines + [''] + report_lines), encoding='utf-8')

    # Save per-image CSV
    import csv
    with open(out_dir / 'per_image_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['name'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Save confusion matrices and publication-oriented tables
    np.save(out_dir / 'confusion_matrix.npy', cm_total)
    np.save(out_dir / 'seam_confusion_matrix.npy', cm_seam_total)
    per_class_rows = []
    for cid in range(8):
        per_class_rows.append({'class_id': cid, 'class_name': CLASS_NAMES[cid], 'Dice': float(per_total.dice[cid]), 'BoundaryF1': float(bnd_mean[cid]) if np.isfinite(bnd_mean[cid]) else '', 'SeamDice': (float(per_seam.dice[cid]) if per_seam is not None else ''), 'support': int(per_total.support[cid])})
    write_csv(out_dir / 'per_class_dice.csv', per_class_rows)
    paper_metrics = {'split': split, 'n_images': len(img_files), 'mDice_fg': float(mdice_fg), 'CrackDice': float(crack_dice), 'BndF1_fg': float(bnd_f1_fg), 'SeamDice_fg': float(seam_mdice_fg), 'FPS': float(fps), 'avg_time_ms': float(avg_time * 1000.0), 'tile_size': int(args.tile_size), 'tile_stride': int(args.tile_stride), 'prompt_mode': str(args.prompt_mode), 'stitch_mode': str(getattr(args, 'stitch_mode', 'logits')), **param_info}
    save_json(out_dir / 'paper_metrics.json', paper_metrics)
    write_csv(out_dir / 'paper_table.csv', [paper_metrics])
    (out_dir / 'deployment_speed.txt').write_text('\n'.join([f'{k}: {v}' for k, v in paper_metrics.items()]), encoding='utf-8')
    if alpha_rows_global:
        write_csv(out_dir / 'alpha_per_tile.csv', alpha_rows_global, fieldnames=['name', 'tile_id', 'x0', 'y0', 'alpha'])
        if int(getattr(args, 'save_paper_outputs', 1)) == 1:
            plot_alpha_hist(out_dir / 'alpha_per_tile.csv', out_dir / 'paper_figures' / 'alpha_hist.png')
    if gate_sum_global:
        usage = []; top1 = []; ent_rows = []
        for li, arr in enumerate(gate_sum_global):
            cnt = max(1, int(gate_count_global[li]))
            usage.append(np.asarray(arr, dtype=np.float64) / cnt)
            top1.append(np.asarray(gate_top1_global[li], dtype=np.float64) / cnt)
            ent_rows.append({'block': li, 'entropy': float(gate_entropy_global[li] / cnt), 'count': int(cnt)})
        usage = np.stack(usage, axis=0); top1 = np.stack(top1, axis=0)
        np.save(out_dir / 'expert_usage_per_block.npy', usage)
        np.save(out_dir / 'expert_top1_per_block.npy', top1)
        write_csv(out_dir / 'expert_entropy_by_block.csv', ent_rows)
        expert_rows = []
        for li in range(usage.shape[0]):
            row = {'block': li}
            for ei in range(usage.shape[1]): row[f'E{ei+1}'] = float(usage[li, ei])
            expert_rows.append(row)
        write_csv(out_dir / 'expert_usage_per_block.csv', expert_rows)
        if int(getattr(args, 'save_paper_outputs', 1)) == 1:
            plot_expert_heatmap(usage, out_dir / 'paper_figures' / 'expert_usage_heatmap.png')

    mode = str(getattr(args, 'print_report', 'full'))
    if mode == 'summary':
        print('\n'.join(summary_lines))
    elif mode == 'report':
        print('\n'.join(report_lines))
    else:
        print('\n'.join(summary_lines + [''] + report_lines))
        print(f"\nSaved full console report to: {out_dir / 'console_report_internal.txt'}")


if __name__ == '__main__':
    main()
