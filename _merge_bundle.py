# -*- coding: utf-8 -*-
"""Merge per-class best specialist checkpoints into one ordinary checkpoint.

Why this exists
---------------
The final specialist training can save class-specific best states such as
``model_best_c1.pth`` (Crack), ``model_best_c5.pth`` (Marking), and
``model_best_c6.pth`` (Joint).  Multi-checkpoint logit fusion is useful for
analysis, but it is inconvenient for deployment and few-shot adaptation.

This script creates a *single* normal checkpoint by merging the line-specialist
head channels from those per-class checkpoints into one state_dict.  The output
checkpoint still has a plain ``model`` key, so it can be used by:

- _evaluate_core.py --ckpt model_final_single.pth
- finetune.py --resume model_final_single.pth
- _train_stage.py --init_from model_final_single.pth

It does not require the dataset and does not run inference.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Dict, Tuple

import torch


# Pavement fg class ids -> line-head row indices in SpecialistRefiner.
# SpecialistRefiner uses fg-channel indices (0=Crack, 4=Marking, 5=Joint),
# so the line head output rows are ordered as [Crack, Marking, Joint].
LINE_CLASS_TO_ROW = {1: 0, 5: 1, 6: 2}
ROW_KEYS = {
    "specialist_refiner.line_head.mix.1.weight",
    "specialist_refiner.line_head.mix.1.bias",
    "specialist_refiner.line_head.scale",
}
LINE_PREFIX = "specialist_refiner.line_head."


def parse_path_map(spec: str) -> Dict[int, str]:
    """Parse maps like ``1=path;5=path`` or ``1:path,5:path``.

    Semicolon + equals is recommended on Windows because paths may contain ':'
    after a drive letter.
    """
    spec = str(spec or "").strip().strip('"').strip("'")
    out: Dict[int, str] = {}
    if not spec:
        return out
    sep = ";" if ";" in spec else ","
    for part in spec.split(sep):
        part = part.strip().strip('"').strip("'")
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        elif ":" in part:
            k, v = part.split(":", 1)
        else:
            raise ValueError(f"Bad class map item: {part!r}. Expected '1=path' or '1:path'.")
        out[int(k.strip())] = v.strip().strip('"').strip("'")
    return out


def parse_weight_map(spec: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    spec = str(spec or "").strip().strip('"').strip("'")
    if not spec:
        return out
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            raise ValueError(f"Bad weight item: {part!r}. Expected '1:0.8'.")
        k, v = part.split(':', 1)
        out[int(k.strip())] = float(v.strip())
    return out


def load_ckpt(path: str) -> Tuple[dict, dict]:
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    elif isinstance(ckpt, dict) and 'ema_model' in ckpt:
        state = ckpt['ema_model']
    else:
        state = ckpt
        ckpt = {'model': state}
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} does not contain a state_dict-like model.")
    return ckpt, state


def clone_state(state: dict) -> dict:
    return {k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in state.items()}


def can_float_merge(t: torch.Tensor) -> bool:
    return torch.is_tensor(t) and (torch.is_floating_point(t) or torch.is_complex(t))


def merge_line_head(
    primary_state: dict,
    class_states: Dict[int, dict],
    row_weights: Dict[int, float],
    shared_alpha: float = 0.25,
) -> dict:
    merged = clone_state(primary_state)

    # 1) Lightly soup shared line-head feature extractor / BN stats.  This helps
    # rows copied from class-specific epochs see a compatible hidden feature map.
    alpha = float(shared_alpha)
    if alpha > 0:
        for k, v0 in list(primary_state.items()):
            if not str(k).startswith(LINE_PREFIX) or k in ROW_KEYS:
                continue
            if not can_float_merge(v0):
                continue
            vals = []
            for cid, st in class_states.items():
                vv = st.get(k, None)
                if torch.is_tensor(vv) and vv.shape == v0.shape and can_float_merge(vv):
                    vals.append(vv.to(dtype=v0.dtype))
            if vals:
                mean_v = torch.stack(vals, dim=0).mean(dim=0)
                merged[k] = (1.0 - alpha) * v0.detach().clone() + alpha * mean_v

    # 2) Merge per-output rows/channels from the checkpoint where that class was best.
    for cid, st in class_states.items():
        if int(cid) not in LINE_CLASS_TO_ROW:
            print(f"[merge] skip class {cid}: no line-head row mapping")
            continue
        row = int(LINE_CLASS_TO_ROW[int(cid)])
        w = float(row_weights.get(int(cid), 1.0))
        for k in ROW_KEYS:
            if k not in primary_state or k not in st:
                print(f"[merge] missing key {k} for class {cid}; skipped")
                continue
            base = merged[k]
            src = st[k]
            if not (torch.is_tensor(base) and torch.is_tensor(src) and base.shape == src.shape):
                print(f"[merge] shape mismatch for {k} class {cid}; skipped")
                continue
            out = base.detach().clone()
            if k.endswith('scale'):
                # shape: (1, 3, 1, 1)
                out[:, row:row + 1, ...] = (1.0 - w) * out[:, row:row + 1, ...] + w * src[:, row:row + 1, ...].to(dtype=out.dtype)
            else:
                # Conv weight (3,C,1,1) or bias (3,)
                out[row:row + 1, ...] = (1.0 - w) * out[row:row + 1, ...] + w * src[row:row + 1, ...].to(dtype=out.dtype)
            merged[k] = out
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary', required=True, help='Main/global specialist checkpoint, usually runs/.../model_best.pth')
    ap.add_argument('--class_ckpt_map', required=True, help='e.g. "1=...c1.pth;5=...c5.pth;6=...c6.pth"')
    ap.add_argument('--out', required=True, help='Output single merged checkpoint path')
    ap.add_argument('--row_weight_map', default='1:1.0,5:1.0,6:1.0', help='Blend strength per class row. 1.0=replace row from class-best.')
    ap.add_argument('--shared_alpha', type=float, default=0.25, help='How much to soup shared line-head layers from class-best ckpts.')
    ap.add_argument('--base_ckpt', default='', help='Optional base checkpoint. If provided, saved as base_model for exact one-file bundle fusion.')
    ap.add_argument('--fusion_weight_map', default='', help='Weights stored for bundle fusion, e.g. 1:0.75,5:0.50,6:0.65. Defaults to row_weight_map.')
    ap.add_argument('--include_class_bank', type=int, default=1, choices=[0,1], help='1: store class-best states inside the same file for exact bundle fusion.')
    ap.add_argument('--tag', default='geoformerx_single_merged', help='Text tag stored in the checkpoint metadata.')
    args = ap.parse_args()

    primary_ckpt, primary_state = load_ckpt(args.primary)
    base_ckpt = None
    base_state = None
    if str(getattr(args, 'base_ckpt', '')).strip():
        base_ckpt, base_state = load_ckpt(str(args.base_ckpt))
    path_map = parse_path_map(args.class_ckpt_map)
    weight_map = parse_weight_map(args.row_weight_map)
    fusion_weight_map = parse_weight_map(args.fusion_weight_map) if str(args.fusion_weight_map).strip() else dict(weight_map)
    if not path_map:
        raise ValueError('class_ckpt_map is empty')

    class_states = {}
    class_meta = {}
    for cid, path in path_map.items():
        ck, st = load_ckpt(path)
        class_states[int(cid)] = st
        class_meta[int(cid)] = {
            'path': str(path),
            'epoch': ck.get('epoch', None) if isinstance(ck, dict) else None,
            'class_best_dice': ck.get('class_best_dice', None) if isinstance(ck, dict) else None,
            'val_score': ck.get('val_score', None) if isinstance(ck, dict) else None,
        }

    merged_state = merge_line_head(primary_state, class_states, weight_map, shared_alpha=float(args.shared_alpha))

    out_ckpt = dict(primary_ckpt)
    out_ckpt['model'] = merged_state
    out_ckpt.pop('optimizer', None)  # make the final artifact lightweight/fresh for adaptation
    out_ckpt['merged_single_checkpoint'] = True
    if base_state is not None:
        out_ckpt['base_model'] = clone_state(base_state)
        if isinstance(base_ckpt, dict):
            out_ckpt['base_model_cfg'] = dict(base_ckpt.get('model_cfg', {}) or {})
    if int(getattr(args, 'include_class_bank', 1)) == 1:
        out_ckpt['class_bank'] = {int(cid): clone_state(st) for cid, st in class_states.items()}
        out_ckpt['class_bank_cfg'] = dict(primary_ckpt.get('model_cfg', {}) or {})
    out_ckpt['fusion_recipe'] = {
        'class_ckpt_weight_map': {int(k): float(v) for k, v in fusion_weight_map.items()},
        'class_ckpt_fuse_mode': 'delta',
    }
    out_ckpt['merge_recipe'] = {
        'tag': str(args.tag),
        'primary': str(args.primary),
        'base_ckpt': str(args.base_ckpt) if str(getattr(args, 'base_ckpt', '')).strip() else '',
        'class_ckpt_map': {int(k): str(v) for k, v in path_map.items()},
        'row_weight_map': {int(k): float(v) for k, v in weight_map.items()},
        'fusion_weight_map': {int(k): float(v) for k, v in fusion_weight_map.items()},
        'shared_alpha': float(args.shared_alpha),
        'line_class_to_row': dict(LINE_CLASS_TO_ROW),
        'class_meta': class_meta,
    }
    cfg = dict(out_ckpt.get('model_cfg', {}) or {})
    cfg['use_specialist_refiner'] = 1
    cfg.setdefault('fusion_mode', 'hybrid')
    cfg.setdefault('use_logit_refiner', 1)
    out_ckpt['model_cfg'] = cfg

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(out_path))
    print(f"[merge] wrote single merged checkpoint: {out_path}")
    print(f"[merge] classes merged: {sorted(path_map.keys())}; row_weight_map={weight_map}; shared_alpha={float(args.shared_alpha)}")


if __name__ == '__main__':
    main()
