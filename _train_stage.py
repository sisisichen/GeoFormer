# -*- coding: utf-8 -*-
"""
GeoFormerX pre-training script
"""
from __future__ import annotations

# ==========================
# Use a non-interactive Matplotlib backend for stable training logs
# ==========================
import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
import faulthandler
join = os.path.join
import time
import json
from collections import defaultdict
faulthandler.enable(all_threads=True)

import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:  # fallback when tensorboard is not installed
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def flush(self):
            pass
        def close(self):
            pass

try:
    from torchvision.transforms import Resize
except Exception:
    class Resize:
        def __init__(self, size, antialias=True):
            self.size = tuple(size)
            self.antialias = antialias

        def __call__(self, x):
            need_squeeze = (x.dim() == 3)
            if need_squeeze:
                x = x.unsqueeze(0)
            y = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
            return y.squeeze(0) if need_squeeze else y

from tqdm import tqdm

from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, sam_model_checkpoint
from segment_anything.utils.transforms import ResizeLongestSide
from model import GeoFormerX
from data.dataset import PavementMultiClassTileDB
from utils.multiclass_loss import multiclass_total_loss, logits_with_bg, foreground_mdice_hard
from utils.multiclass_metrics import confusion_matrix, per_class_metrics_from_cm, boundary_f1_per_class
from utils.publication_outputs import nanmean, plot_expert_heatmap, plot_training_curves, write_csv
from utils.logger import get_logger
from utils.moe_loss import moe_aux_loss

from utils.moe_preassign import build_routing_cache, PreassignConfig
from utils.sampler import build_task_mixed_sampler
from utils.train_runtime import (
    snapshot_model_state,
    restore_model_state,
    get_base_lrs,
    compute_epoch_lrs,
    set_optimizer_lrs,
    clip_grad_norm_and_get,
    gradients_finite,
    parameters_finite,
    sanitize_nonfinite_gradients,
)

# 注意：不再依赖 SegmentMetrics 来算 dsc，避免内部 0/0 nan


# ==========================
# Stable Dice/DSC calculation
# ==========================
def dice_batch(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_bin: (B,1,H,W)  0/1 或 bool
    gt_bin:   (B,1,H,W)  0/1
    return:   (B,)       每个样本 dice
    规则：
      - pred 与 gt 都全空 => dice=1
      - 其他情况按标准 dice（加 eps 防止 0/0）
    """
    pred = pred_bin.float()
    gt = gt_bin.float()

    pred_f = pred.flatten(1)
    gt_f = gt.flatten(1)

    inter = (pred_f * gt_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + gt_f.sum(dim=1)

    dice = (2 * inter + eps) / (denom + eps)
    dice = torch.where(denom == 0, torch.ones_like(dice), dice)
    return dice


# setup seeds
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.empty_cache()
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ==========================
# Optimizer helper: smaller LR for mask decoder and optionally unfrozen encoder
# ==========================
def get_unfrozen_block_ids(args, total_blocks: int = 12):
    n = int(max(0, getattr(args, 'unfreeze_last_n_blocks', 0)))
    n = min(n, int(total_blocks))
    if n <= 0:
        return []
    return list(range(int(total_blocks) - n, int(total_blocks)))


def build_optimizer(model_dp, args):
    """Build AdamW with param groups.
    - other trainable params use lr
    - mask_decoder uses lr * mask_decoder_lr_mult
    - partially-unfrozen encoder params use lr * encoder_lr_mult
    - early SAM adapter params can optionally use a smaller lr for stability
    """
    block_ids = get_unfrozen_block_ids(args)
    encoder_tokens = [f'image_encoder.blocks.{i}.' for i in block_ids]
    early_adapter_blocks = max(0, int(getattr(args, 'early_adapter_blocks', 0)))
    early_adapter_mult = float(getattr(args, 'early_adapter_lr_mult', 1.0))
    early_adapter_tokens = [f'sam.image_encoder.blocks.{i}.mlp.adapter_' for i in range(early_adapter_blocks)]

    mask_params = []
    encoder_params = []
    early_adapter_params = []
    specialist_params = []
    other_params = []
    for n, p in model_dp.named_parameters():
        if not p.requires_grad:
            continue
        if 'sam.mask_decoder' in n or ('mask_decoder' in n and 'sam.' in n):
            mask_params.append(p)
        elif ('sam.image_encoder.neck.' in n) or any(tok in n for tok in encoder_tokens):
            encoder_params.append(p)
        elif early_adapter_blocks > 0 and any(tok in n for tok in early_adapter_tokens):
            early_adapter_params.append(p)
        elif 'specialist_refiner' in n:
            specialist_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if other_params:
        groups.append({'params': other_params, 'lr': float(args.lr)})
    if early_adapter_params:
        groups.append({'params': early_adapter_params, 'lr': float(args.lr) * float(early_adapter_mult)})
    if specialist_params:
        groups.append({'params': specialist_params, 'lr': float(args.lr) * float(getattr(args, 'specialist_lr_mult', 1.0))})
    if mask_params:
        groups.append({'params': mask_params, 'lr': float(args.lr) * float(getattr(args, 'mask_decoder_lr_mult', 0.1))})
    if encoder_params:
        groups.append({'params': encoder_params, 'lr': float(args.lr) * float(getattr(args, 'encoder_lr_mult', 0.05))})

    return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))


def set_partial_encoder_trainability(model_module, enable_neck: bool = False, block_ids=None, enabled: bool = True):
    """Toggle trainability of the partially-unfrozen SAM encoder subset."""
    if block_ids is None:
        block_ids = []
    if hasattr(model_module, 'sam') and hasattr(model_module.sam, 'image_encoder'):
        enc = model_module.sam.image_encoder
        if bool(enable_neck) and hasattr(enc, 'neck'):
            for p in enc.neck.parameters():
                p.requires_grad = bool(enabled)
        for bid in list(block_ids):
            if 0 <= int(bid) < len(enc.blocks):
                for p in enc.blocks[int(bid)].parameters():
                    p.requires_grad = bool(enabled)


def apply_train_scope(model_dp, args, logger=None):
    """Restrict trainable parameters for stable final-stage polishing.

    all: keep the default GeoFormerX trainability.
    refiners: freeze the SAM/adapters/mask decoder and train only fusion/logit/specialist refiners.
    specialist: train only specialist_refiner.
    """
    scope = str(getattr(args, 'train_scope', 'all')).lower().strip()
    if scope in ('', 'all'):
        return
    if scope not in ('refiners', 'specialist'):
        raise ValueError(f'Unsupported train_scope={scope}')
    module = model_dp.module if hasattr(model_dp, 'module') else model_dp
    for p in module.parameters():
        p.requires_grad = False

    enabled = []
    def _enable(name):
        mod = getattr(module, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True
            enabled.append(name)

    if scope == 'refiners':
        _enable('fusion_2d3d')
        _enable('logit_refiner')
        _enable('specialist_refiner')
    elif scope == 'specialist':
        _enable('specialist_refiner')

    if logger is not None:
        n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f'Train scope applied: {scope}. Enabled modules={enabled}. Trainable params={n_train}')


def first_nonfinite_grad_param(model: torch.nn.Module):
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            try:
                max_abs = float(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
            except Exception:
                max_abs = None
            return n, max_abs
    return None, None


def parse_class_value_map(spec: str) -> dict:
    """Parse strings like '1:0.25,2:0.10'. Empty -> {}."""
    spec = '' if spec is None else str(spec).strip()
    if spec == '':
        return {}
    out = {}
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ':' not in chunk:
            raise ValueError(f"Invalid class map item: {chunk}. Expected class:value")
        k, v = chunk.split(':', 1)
        out[int(k.strip())] = float(v.strip())
    return out


def summarize_gate_batch(gates):
    stats = {}
    if not gates:
        return stats
    for li, g in enumerate(gates):
        if g is None:
            continue
        gd = g.detach()
        ent = -(gd * gd.clamp_min(1e-8).log()).sum(dim=1)
        top1 = torch.argmax(gd, dim=1)
        counts = torch.bincount(top1, minlength=gd.shape[1]).float() / max(1, gd.shape[0])
        stats[li] = {
            'mean_prob': gd.mean(dim=0).float().cpu().tolist(),
            'top1_load': counts.cpu().tolist(),
            'entropy': float(ent.mean().item()),
        }
    return stats


def default_epoch_cfg(args, rare_sampler_map, dice_class_weight_map, aux_ft_map, aux_bnd_map, aux_cldice_map=None):
    return {
        'stage': 'base',
        'rare_sampler_map': dict(rare_sampler_map),
        'focus_crop_prob': float(getattr(args, 'focus_crop_prob', 0.0)),
        'focus_crop_classes': list(getattr(args, 'focus_crop_classes', [])),
        'focus_crop_weights': list(getattr(args, 'focus_crop_weights', [])),
        'dice_class_weight_map': dict(dice_class_weight_map),
        'aux_ft_map': dict(aux_ft_map),
        'aux_bnd_map': dict(aux_bnd_map),
        'aux_cldice_map': dict(aux_cldice_map) if aux_cldice_map is not None else {},
        'ohem_ratio': float(getattr(args, 'ohem_ratio', 0.0)),
        'ce_bg_scale': float(getattr(args, 'ce_bg_scale', 0.5)),
        'crack_ft_lambda': float(getattr(args, 'crack_ft_lambda', 0.0)),
        'crack_bnd_lambda': float(getattr(args, 'crack_bnd_lambda', 0.0)),
        'crack_cldice_lambda': float(getattr(args, 'crack_cldice_lambda', 0.0)),
        'crack_cldice_iters': int(getattr(args, 'crack_cldice_iters', 3)),
    }


def get_crack_polish_cfg(epoch: int, num_epochs: int, base_cfg: dict, start_epoch: int = 20):
    """A mild late-stage crack polish on top of the GeoFormerX base recipe.

    Key idea from the thin-line refinement ablation:
    - small-area crack emphasis helped crack rise earlier,
    - but aggressive late reweighting + LR shrink on non-finite events flattened the run.
    So this schedule stays deliberately gentle and starts later.
    """
    cfg = dict(base_cfg)
    if int(epoch) < int(start_epoch):
        cfg['stage'] = 'base'
        return cfg
    cfg.update({
        'stage': 'crack_polish',
        'rare_sampler_map': {1: 1.9, 2: 2.5, 7: 0.8},
        'focus_crop_prob': 0.78,
        'focus_crop_classes': [1, 2, 7],
        'focus_crop_weights': [4.0, 3.8, 0.9],
        'dice_class_weight_map': {1: 2.0, 2: 1.9, 7: 1.1},
        'aux_ft_map': {2: 0.12, 7: 0.04},
        'aux_bnd_map': {2: 0.03},
        'ohem_ratio': 0.15625,
        'ce_bg_scale': 0.48,
        'crack_ft_lambda': max(float(base_cfg.get('crack_ft_lambda', 0.0)), 0.22),
        'crack_bnd_lambda': max(float(base_cfg.get('crack_bnd_lambda', 0.0)), 0.11),
        'crack_cldice_lambda': max(float(base_cfg.get('crack_cldice_lambda', 0.0)), 0.03),
        'crack_cldice_iters': min(int(base_cfg.get('crack_cldice_iters', 2)), 2),
    })
    return cfg


def get_pavement_final_push_cfg(epoch: int, num_epochs: int, base_cfg: dict):
    """A small curriculum designed from observed base/specialist trade-offs.

    Early: recover pothole/patch, Middle: balance, Late: polish crack/joint/marking.
    """
    cfg = dict(base_cfg)
    n = max(1, int(num_epochs))
    if epoch < int(round(0.30 * n)):
        cfg.update({
            'stage': 'stage_a_pothole_patch',
            'rare_sampler_map': {1: 1.3, 2: 3.2, 7: 0.8},
            'focus_crop_prob': 0.78,
            'focus_crop_classes': [1, 2, 7],
            'focus_crop_weights': [2.5, 5.5, 1.0],
            'dice_class_weight_map': {1: 1.55, 2: 2.45, 7: 1.1},
            'aux_ft_map': {2: 0.20, 7: 0.05},
            'aux_bnd_map': {2: 0.05},
            'ohem_ratio': 0.125,
            'ce_bg_scale': 0.50,
            'crack_ft_lambda': max(float(base_cfg.get('crack_ft_lambda', 0.0)), 0.18),
            'crack_bnd_lambda': max(float(base_cfg.get('crack_bnd_lambda', 0.0)), 0.10),
        })
    elif epoch < int(round(0.70 * n)):
        cfg.update({
            'stage': 'stage_b_balanced',
            'rare_sampler_map': {1: 1.8, 2: 2.6, 7: 0.8},
            'focus_crop_prob': 0.76,
            'focus_crop_classes': [1, 2, 7],
            'focus_crop_weights': [3.8, 4.2, 1.0],
            'dice_class_weight_map': {1: 1.95, 2: 2.0, 7: 1.15, 6: 1.05},
            'aux_ft_map': {2: 0.14, 6: 0.04, 7: 0.04},
            'aux_bnd_map': {2: 0.04, 6: 0.02},
            'ohem_ratio': 0.1875,
            'ce_bg_scale': 0.45,
            'crack_ft_lambda': 0.22,
            'crack_bnd_lambda': 0.11,
        })
    else:
        cfg.update({
            'stage': 'stage_c_thin_polish',
            'rare_sampler_map': {1: 2.4, 2: 1.8, 6: 0.7, 7: 0.7},
            'focus_crop_prob': 0.80,
            'focus_crop_classes': [1, 2, 6, 7],
            'focus_crop_weights': [5.0, 3.0, 1.6, 0.8],
            'dice_class_weight_map': {1: 2.4, 2: 1.6, 5: 1.10, 6: 1.25, 7: 1.10},
            'aux_ft_map': {2: 0.08, 5: 0.03, 6: 0.06, 7: 0.03},
            'aux_bnd_map': {2: 0.02, 5: 0.02, 6: 0.03},
            'ohem_ratio': 0.25,
            'ce_bg_scale': 0.40,
            'crack_ft_lambda': 0.24,
            'crack_bnd_lambda': 0.12,
        })
    return cfg


class ModelEMA:
    def __init__(self, model_module: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model_module.save_parameters().items()}

    @torch.no_grad()
    def reset_from_model(self, model_module: nn.Module):
        """Hard-reset EMA shadow to the current model weights.

        Useful when EMA starts several epochs after training began: without a reset,
        the shadow can still be dominated by the random initialization right when
        eval_with_ema is first enabled.
        """
        self.shadow = {k: v.detach().clone() for k, v in model_module.save_parameters().items()}

    @torch.no_grad()
    def update(self, model_module: nn.Module):
        cur = model_module.save_parameters()
        d = float(self.decay)
        for k, v in cur.items():
            vv = v.detach()
            if k not in self.shadow:
                self.shadow[k] = vv.clone()
                continue
            # EMA only makes sense for floating tensors. For integer / boolean buffers,
            # keep an exact copy of the latest value instead of trying to blend them.
            if (not torch.is_floating_point(vv)) and (not torch.is_complex(vv)):
                self.shadow[k] = vv.clone()
                continue
            if (not torch.is_floating_point(self.shadow[k])) and (not torch.is_complex(self.shadow[k])):
                self.shadow[k] = vv.clone()
                continue
            self.shadow[k].mul_(d).add_(vv.to(dtype=self.shadow[k].dtype), alpha=1.0 - d)

    def state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.shadow.items()}

    def copy_to(self, model_module: nn.Module):
        model_module.load_parameters(self.shadow)


# setup parser
parser = argparse.ArgumentParser("GeoFormerX training", add_help=False)
# model
parser.add_argument("--checkpoint", type=str, default="./checkpoints/sam",
                    help="path to SAM checkpoint folder")
parser.add_argument("--model_type", type=str, default="vit_b",
                    help="SAM model scale (e.g vit_b, vit_l, vit_h)")
parser.add_argument("--sam_image_size", type=int, default=256,
                    help="SAM encoder input size. Larger values use more VRAM and may improve fine-detail segmentation.")
parser.add_argument("--task_name", type=str, default="geoformerx")
parser.add_argument("--method", type=str, default="geoformerx", choices=["geoformerx"])
parser.add_argument("--bottleneck_dim", type=int, default=16)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--expert_num", type=int, default=8)
# MoE anti-collapse (gate regularization)
parser.add_argument("--moe_topk", type=int, default=2,
                    help="top-k experts used by the gate (0 = dense softmax, original behavior)")
parser.add_argument("--moe_temp", type=float, default=1.0,
                    help="gate softmax temperature (smaller -> peakier; >1 -> smoother)")
parser.add_argument("--moe_noise_std", type=float, default=0.0,
                    help="std of Gaussian noise added to gate logits during training (0 = off)")
parser.add_argument("--moe_lb_coef", type=float, default=0.01,
                    help="load-balance loss coefficient (0 = disable)")
parser.add_argument("--moe_ent_coef", type=float, default=0.01,
                    help="entropy bonus coefficient (0 = disable)")

# routing pre-assign (task-first, dynamic-K)
parser.add_argument("--moe_preassign", type=int, default=0, choices=[0,1],
                    help="Optional offline routing pre-assign. Keep OFF for the standard RGB-D pavement dataset. (default: 0)")
parser.add_argument("--moe_preassign_force", action="store_true",
                    help="Force rebuild the routing cache even if it already exists.")
parser.add_argument("--moe_route_sup_coef", type=float, default=0.05,
                    help="Routing supervision loss coefficient (0 = disable). Uses 'moe_target' from dataset cache.")
parser.add_argument("--moe_route_warmup", type=int, default=2,
                    help="Warmup epochs for routing supervision (linearly ramps from 0 to moe_route_sup_coef).")
parser.add_argument("--moe_route_cap_ratio", type=float, default=0.25,
                    help="Clip routing supervision term to at most (moe_route_cap_ratio * seg_loss) to avoid destabilizing training when coef is large.")
parser.add_argument("--moe_route_log", type=int, default=1, choices=[0,1],
                    help="Log routing supervision stats (route_loss/route_coef/route_term and invalid targets) to tqdm and TensorBoard. (default: 1)")
# data
parser.add_argument("--data_path", type=str, default="./data",
                    help="Dataset root. Expected: <data_path>/{train,val,test}/{image,label} and <data_path>/3Ddate")

# pavement tiling / prompts
parser.add_argument("--tile_size", type=int, default=256, help="Tile size (default: 256)")
parser.add_argument("--tile_stride", type=int, default=256, help="Tile stride (default: 256, no overlap)")

# --- Train-time tile sampling tricks (recommended for Crack / thin defects) ---
parser.add_argument(
    "--train_use_crack_crop",
    type=int,
    default=1,
    choices=[0, 1],
    help="1: enable crack-centered crop (train only) to boost Crack and reduce seam artifacts; 0: disable.",
)
parser.add_argument(
    "--crack_crop_prob",
    type=float,
    default=0.7,
    help="Probability to use crack-centered crop when crack pixels exist in the full label.",
)
parser.add_argument(
    "--tile_jitter",
    type=int,
    default=64,
    help="Random jitter (pixels) added to tile coords during training (0 disables). Helps avoid fixed seam at x=256.",
)
parser.add_argument(
    "--focus_crop_prob",
    type=float,
    default=0.0,
    help="Optional generalized class-aware centered crop probability. If 0, legacy crack crop args are used.",
)
parser.add_argument(
    "--focus_crop_classes",
    type=int,
    nargs='*',
    default=[],
    help="Class ids for class-aware centered crop, e.g. --focus_crop_classes 1 2 7",
)
parser.add_argument(
    "--focus_crop_weights",
    type=float,
    nargs='*',
    default=[],
    help="Sampling weights aligned with --focus_crop_classes.",
)
parser.add_argument("--num_fg_classes", type=int, default=7,
                    help="Number of foreground classes (exclude background). Default: 7")
parser.add_argument("--ce_weight", type=float, default=1.0, help="Cross-entropy weight.")
parser.add_argument("--dice_weight", type=float, default=1.0, help="Dice loss weight (foreground macro).")
parser.add_argument("--dice_present_only", type=int, default=1, choices=[0,1],
                    help="1: average dice only over classes present in current batch GT. Strongly recommended for rare classes.")
parser.add_argument("--dice_class_weights", type=str, default="",
                    help="Optional per-class dice weights, e.g. '1:1.5,2:2.0,7:1.2'.")
parser.add_argument("--use_dynamic_ce_weights", type=int, default=1, choices=[0,1],
                    help="1: inverse-frequency dynamic CE weights per batch; 0: no weighting.")
parser.add_argument("--ce_w_min", type=float, default=0.2, help="Min clamp for dynamic CE weights.")
parser.add_argument("--ce_w_max", type=float, default=10.0, help="Max clamp for dynamic CE weights.")
parser.add_argument("--ce_bg_scale", type=float, default=0.3, help="Multiply background CE weight by this factor.")

# --- OHEM for CE (optional; improves rare classes at the cost of extra compute) ---
parser.add_argument(
    "--ohem_ratio",
    type=float,
    default=0.0,
    help="Online hard example mining ratio for CE. 0 disables; typical values: 0.25.",
)
parser.add_argument("--prompt_mode", type=str, default="full", choices=["full", "gt_box"],
                    help="Prompt mode: 'full' uses full-tile box (prompt-free); 'gt_box' uses GT bbox (not deployment-realistic).")
parser.add_argument("--box_jitter", type=int, default=0, help="BBox jitter for gt_box mode.")

# expert routing
parser.add_argument("--force_class_expert", type=int, default=1, choices=[0, 1],
                    help="1: force each class to a dedicated expert during training; 0: let gate decide.")

# 2D+3D fusion + weaken style routing
parser.add_argument("--use_fusion_2d3d", type=int, default=1, choices=[0, 1],
                    help="1: enable trainable 2D+3D fusion head; 0: ignore the 3D channel.")
parser.add_argument("--fusion_hidden", type=int, default=16, help="Fusion gate hidden dim.")
parser.add_argument("--fusion_mode", type=str, default="hybrid", choices=["global", "hybrid", "fixed"],
                    help="global: scalar alpha; hybrid: scalar alpha times spatial alpha-map.")
parser.add_argument("--use_logit_refiner", type=int, default=1, choices=[0,1],
                    help="1: enable lightweight RGBD-aware residual logit refinement head.")
parser.add_argument("--refiner_hidden", type=int, default=32,
                    help="Hidden channels of the residual logit refiner.")
parser.add_argument("--unfreeze_encoder_neck", type=int, default=0, choices=[0,1],
                    help="1: unfreeze SAM image_encoder.neck with a small lr multiplier.")
parser.add_argument("--unfreeze_last_n_blocks", type=int, default=0,
                    help="number of last image-encoder transformer blocks to unfreeze (default: 0).")
parser.add_argument("--moe_style_scale", type=float, default=0.25,
                    help="Scale style embedding in MoE gate input (smaller -> weaker style routing).")
# env
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--device_ids", type=int, default=[0,1,2,3,4,5,6,7], nargs='+',
                    help="device ids assignment (e.g 0 1 2 3)")
parser.add_argument("--work_dir", type=str, default="./runs/geoformerx")
# train
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--pin_memory", type=int, default=0, choices=[0,1],
                    help="DataLoader pin_memory. On Windows, 0 is usually more stable.")
parser.add_argument("--persistent_workers", type=int, default=0, choices=[0,1],
                    help="keep DataLoader workers alive between epochs (only when num_workers>0).")
parser.add_argument("--train_cache_size", type=int, default=0,
                    help="per-worker LRU cache size for training full images. 0 disables cache for maximum stability.")
parser.add_argument("--val_cache_size", type=int, default=0,
                    help="per-worker LRU cache size for validation full images. 0 disables cache for maximum stability.")
parser.add_argument("--resume", type=str, default=None,
                    help="resume training from checkpoint")
# optimizer
parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                    help="learning rate (absolute lr default: 0.001)")
parser.add_argument("--weight_decay", type=float, default=0.01,
                    help="weight decay (default: 0.01)")
parser.add_argument("--mask_decoder_lr_mult", type=float, default=0.1,
                    help="learning-rate multiplier for SAM mask_decoder params (default: 0.1).")
parser.add_argument("--encoder_lr_mult", type=float, default=0.05,
                    help="learning-rate multiplier for partially-unfrozen image encoder params.")
parser.add_argument("--encoder_unfreeze_epoch", type=int, default=0,
                    help="delay unfreezing of encoder neck / last blocks until this epoch. 0 means unfreeze from epoch 0.")
parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                    help="global grad clipping (norm). 0 disables. (default: 1.0)")
parser.add_argument("--auto_resume", type=int, default=1, choices=[0,1],
                    help="auto resume from <work_dir>/<task_name>/model_latest.pth if exists and --resume is not set. (default: 1)")
parser.add_argument("--reset_optim", type=int, default=0, choices=[0, 1],
                    help="If 1, do NOT load optimizer state when resuming (useful when changing lr/weight_decay).")
parser.add_argument("--skip_nonfinite", type=int, default=1, choices=[0,1],
                    help="skip optimizer step when loss is NaN/Inf to prevent corruption. (default: 1)")
parser.add_argument("--use_amp", action="store_true", default=False,
                    help="whether to use amp")
parser.add_argument("--amp_init_scale", type=float, default=65536.0,
                    help="initial GradScaler scale when --use_amp is enabled.")
parser.add_argument("--amp_backoff", type=float, default=0.5,
                    help="on AMP overflow / non-finite recovery, multiply GradScaler scale by this factor.")
parser.add_argument("--amp_growth_interval", type=int, default=2000,
                    help="GradScaler growth interval.")
parser.add_argument("--lr_schedule", type=str, default="flatcosine", choices=["none", "cosine", "flatcosine"],
                    help="epoch-wise lr schedule. flatcosine keeps the base lr for a while, then decays near the end; better when you want the best model near epoch 50.")
parser.add_argument("--warmup_epochs", type=int, default=2,
                    help="number of warmup epochs for lr schedule (default: 2)")
parser.add_argument("--hold_epochs", type=int, default=0,
                    help="for flatcosine: keep base lr for this many epochs after warmup before decaying.")
parser.add_argument("--min_lr", type=float, default=1e-6,
                    help="minimum lr used by cosine schedule (default: 1e-6)")
parser.add_argument("--early_stop_patience", type=int, default=12,
                    help="stop if validation DSC does not improve for N epochs. 0 disables. (default: 12)")
parser.add_argument("--use_tqdm", type=int, default=1, choices=[0,1],
                    help="1: show tqdm progress bars; 0: epoch-only logging to avoid console spam.")
parser.add_argument("--overall_progress", type=int, default=1, choices=[0,1],
                    help="1: show an outer total progress bar across all train/val steps.")
parser.add_argument("--tqdm_mininterval", type=float, default=5.0,
                    help="minimum seconds between tqdm refreshes.")
parser.add_argument("--log_model_arch", type=int, default=0, choices=[0,1],
                    help="1: log full model architecture to output.log; 0: only log counts.")
parser.add_argument("--save_epoch_ckpt", type=int, default=0, choices=[0,1],
                    help="1: also save model_<epoch>.pth each epoch; 0: save latest/best/lowest only.")
parser.add_argument("--restore_on_nonfinite", type=int, default=1, choices=[0,1],
                    help="1: restore epoch-start weights and reduce lr when non-finite tensors appear.")
parser.add_argument("--nonfinite_reduce_lr", type=float, default=0.5,
                    help="multiply lr scale by this factor after a non-finite restore.")
parser.add_argument("--nonfinite_reduce_every", type=int, default=0,
                    help="Reduce lr_scale only once every N non-finite events within an epoch. 0 disables per-batch lr shrinking.")
parser.add_argument("--nonfinite_log_limit", type=int, default=3,
                    help="maximum number of detailed non-finite warnings printed per epoch.")
parser.add_argument("--abort_epoch_on_nonfinite", type=int, default=0, choices=[0,1],
                    help="1: abort the rest of the epoch after a non-finite batch; 0: restore and continue from the latest safe snapshot.")
parser.add_argument("--safe_snapshot_interval", type=int, default=25,
                    help="refresh the in-epoch safe model snapshot every N successful optimizer steps.")
parser.add_argument("--nonfinite_param_log", type=int, default=1, choices=[0,1],
                    help="1: log the first parameter name whose gradient becomes non-finite.")
parser.add_argument("--tb_batch_log", type=int, default=0, choices=[0,1],
                    help="1: write per-batch TensorBoard scalars; 0: epoch-only logging to reduce I/O.")
parser.add_argument("--debug_epoch_json", type=int, default=1, choices=[0,1],
                    help="1: append rich JSON diagnostics for each epoch.")
parser.add_argument("--paper_train_outputs", type=int, default=1, choices=[0,1],
                    help="1: save paper-facing training CSVs/plots: loss_components.csv, val_metrics_by_epoch.csv, alpha_by_epoch.csv and expert heatmaps.")
parser.add_argument("--debug_nonfinite_dump", type=int, default=1, choices=[0,1],
                    help="1: dump offending batch names/histograms when non-finite tensors appear.")
parser.add_argument("--task_balanced_sampling", type=int, default=1,
                    help="1: balance sampling across TaskFolders (recommended for highly imbalanced datasets); 0: plain shuffle")

# TaskFolder imbalance handling (mixed sampling)
parser.add_argument("--task_sampling_alpha", type=float, default=0.25,
                    help="mixing factor in [0,1] for TaskFolder reweighting. 0=natural, 1=fully balanced.")
parser.add_argument("--task_sampling_power", type=float, default=1.0,
                    help="strength of inverse-frequency reweighting: 1/(count**power). 1.0=standard, 0.5=milder.")
parser.add_argument("--task_sampling_num_samples", type=int, default=0,
                    help="samples drawn per epoch when using WeightedRandomSampler (0 => len(dataset)).")
parser.add_argument("--rare_sampler", type=int, default=0, choices=[0,1],
                    help="1: build a tile-level sampler that up-weights tiles containing rare classes.")
parser.add_argument("--rare_sampler_map", type=str, default="",
                    help="Class boosts for rare sampler, e.g. '1:1.5,2:3.0,7:0.8'.")

# MoE warmup (recommended)
parser.add_argument("--moe_warmup_epochs", type=int, default=5,
                    help="linearly warm up MoE auxiliary loss + gate noise for first N epochs (0 disables warmup).")

# Crack / thin-structure tuning (keeps evaluation prompts unchanged)
# These defaults are mild and mainly help very sparse crack tasks (Vehicle*_*Crack).
parser.add_argument("--crack_ft_lambda", type=float, default=0.15,
                    help="extra focal-tversky loss weight for tasks whose folder name contains 'Crack' (0 disables).")
parser.add_argument("--vehicle_crack_ft_extra", type=float, default=0.15,
                    help="additional focal-tversky weight for Vehicle*Crack tasks, added on top of crack_ft_lambda.")
parser.add_argument("--vehicle_crack_loss_boost", type=float, default=0.25,
                    help="multiply seg loss by (1+boost) for Vehicle*Crack samples (0 disables).")
parser.add_argument("--crack_alpha", type=float, default=0.3, help="focal-tversky alpha (FP weight).")
parser.add_argument("--crack_beta", type=float, default=0.7, help="focal-tversky beta (FN weight).")
parser.add_argument("--crack_gamma", type=float, default=0.75, help="focal-tversky gamma (focal exponent).")

# Optional: crack boundary auxiliary loss (helps thin structures)
parser.add_argument("--crack_bnd_lambda", type=float, default=0.05,
                    help="extra boundary dice loss weight for Crack (0 disables).")
parser.add_argument("--crack_bnd_kernel", type=int, default=3,
                    help="kernel size for crack boundary morphological gradient (odd int, e.g., 3/5).")
parser.add_argument("--aux_ft_map", type=str, default="",
                    help="Optional extra per-class focal-Tversky lambdas, e.g. '1:0.20,2:0.15'.")
parser.add_argument("--aux_bnd_map", type=str, default="",
                    help="Optional extra per-class boundary lambdas, e.g. '1:0.10,2:0.04'.")
parser.add_argument("--use_ema", type=int, default=1, choices=[0,1],
                    help="1: maintain EMA of trainable weights and evaluate/save with EMA.")
parser.add_argument("--ema_decay", type=float, default=0.999,
                    help="EMA decay for trainable weights.")
parser.add_argument("--ema_start_epoch", type=int, default=2,
                    help="Start updating EMA after this epoch.")
parser.add_argument("--eval_with_ema", type=int, default=1, choices=[0,1],
                    help="1: use EMA weights for validation/model_best when available.")
parser.add_argument("--ema_warm_start", type=int, default=1, choices=[0,1],
                    help="Reset EMA shadow to the current model at ema_start_epoch before first EMA-evaluated epoch.")
parser.add_argument("--best_metric", type=str, default="cm_mdice", choices=["batch_mdice", "cm_mdice", "weighted"],
                    help="Validation metric used to choose model_best. cm_mdice is usually closer to final stitched evaluation.")
parser.add_argument("--best_class_weights", type=str, default="",
                    help="Optional class weights for best_metric=weighted, e.g. '1:1.5,2:2.0'.")
parser.add_argument("--save_class_best_ckpt", type=int, default=0, choices=[0,1],
                    help="1: save model_best_c{class}.pth whenever a validation class Dice improves.")
parser.add_argument("--class_best_ids", type=str, default="1,5,6",
                    help="Comma-separated class ids for per-class best checkpoint saving, e.g. '1,5,6'.")
parser.add_argument("--pavement_final_push", type=int, default=0, choices=[0,1],
                    help="1: enable a 3-stage pavement curriculum tuned to preserve pothole/patch early and polish crack/joint late.")
parser.add_argument("--early_adapter_blocks", type=int, default=0,
                    help="Apply a smaller lr to adapter params in the first N SAM encoder blocks for stability.")
parser.add_argument("--early_adapter_lr_mult", type=float, default=0.10,
                    help="LR multiplier for early SAM adapter params when --early_adapter_blocks > 0.")
parser.add_argument("--grad_sanitize_nonfinite", type=int, default=1, choices=[0,1],
                    help="1: replace a small number of non-finite gradient tensors with zeros instead of restoring the full snapshot.")
parser.add_argument("--grad_sanitize_max_params", type=int, default=1,
                    help="Maximum number of gradient tensors allowed to be sanitized in one step.")
parser.add_argument("--grad_sanitize_fill", type=float, default=0.0,
                    help="Fill value used when sanitizing non-finite gradients.")
parser.add_argument("--crack_polish", type=int, default=0, choices=[0,1],
                    help="1: apply a modest crack-focused late schedule.")
parser.add_argument("--crack_polish_start", type=int, default=20,
                    help="Epoch index where crack-focused late schedule starts.")
parser.add_argument("--crack_small_area_boost", type=float, default=0.0,
                    help="Extra sampling boost for crack-positive tiles with very small crack area ratio.")
parser.add_argument("--crack_area_ref", type=float, default=0.02,
                    help="Reference crack area ratio used by crack small-area sampling.")
parser.add_argument("--crack_area_power", type=float, default=0.5,
                    help="Power used by crack small-area sampling. 0.5 is a gentle square-root boost.")
parser.add_argument("--crack_area_cap", type=float, default=3.0,
                    help="Maximum multiplicative thin-area emphasis for crack sampling.")
parser.add_argument("--crack_cldice_lambda", type=float, default=0.0,
                    help="Crack-only clDice / centerline auxiliary loss weight.")
parser.add_argument("--crack_cldice_iters", type=int, default=3,
                    help="Number of soft-skeleton iterations used by crack clDice.")


parser.add_argument("--aux_cldice_map", type=str, default='',
                    help="Per-class clDice lambdas, e.g. '5:0.02,6:0.02'.")
parser.add_argument("--line_group_ce_lambda", type=float, default=0.0,
                    help="Aux CE over crack/marking/joint subset.")
parser.add_argument("--surface_group_ce_lambda", type=float, default=0.0,
                    help="Aux CE over pothole/patch subset.")
parser.add_argument("--use_specialist_refiner", type=int, default=0, choices=[0,1],
                    help="Enable thin/surface specialist residual heads.")
parser.add_argument("--line_refiner_hidden", type=int, default=48,
                    help="Hidden channels for thin-class specialist head.")
parser.add_argument("--surface_refiner_hidden", type=int, default=32,
                    help="Hidden channels for surface-class specialist head.")
parser.add_argument("--specialist_scale_init", type=float, default=0.10,
                    help="Initial residual scale for specialist heads.")
parser.add_argument("--specialist_lr_mult", type=float, default=2.0,
                    help="LR multiplier for specialist heads.")
parser.add_argument("--train_scope", type=str, default="all", choices=["all", "refiners", "specialist"],
                    help="Final-stage stability switch. all=normal training; refiners=train only fusion/logit/specialist heads; specialist=train only specialist_refiner.")
parser.add_argument("--init_from", type=str, default=None,
                    help="Load model weights without resuming optimizer/epoch.")


def main(args):
    device = torch.device(args.device)

    checkpoint = join(args.checkpoint, sam_model_checkpoint[args.model_type])
    sam_model = sam_model_registry[args.model_type](
        image_size=int(getattr(args, 'sam_image_size', 256)),
        keep_resolution=True,
        checkpoint=checkpoint,
        num_multimask_outputs=int(args.num_fg_classes),
    )

    if args.method == "geoformerx":
        model = GeoFormerX(
            sam_model,
            args.bottleneck_dim,
            args.embedding_dim,
            args.expert_num,
            gate_topk=getattr(args, 'moe_topk', 2),
            gate_temperature=args.moe_temp,
            gate_noise_std=getattr(args, 'moe_noise_std', 0.0),
            style_bn=bool(getattr(args, 'moe_style_bn', 1)),
            style_dropout=getattr(args, 'moe_style_dropout', 0.10),
            style_scale=float(getattr(args, 'moe_style_scale', 0.25)),
            use_fusion_2d3d=bool(int(getattr(args, 'use_fusion_2d3d', 1)) == 1),
            fusion_hidden=int(getattr(args, 'fusion_hidden', 16)),
            fusion_mode=str(getattr(args, 'fusion_mode', 'global')),
            use_logit_refiner=bool(int(getattr(args, 'use_logit_refiner', 0)) == 1),
            refiner_hidden=int(getattr(args, 'refiner_hidden', 32)),
            use_specialist_refiner=bool(int(getattr(args, 'use_specialist_refiner', 0)) == 1),
            line_refiner_hidden=int(getattr(args, 'line_refiner_hidden', 48)),
            surface_refiner_hidden=int(getattr(args, 'surface_refiner_hidden', 32)),
            specialist_scale_init=float(getattr(args, 'specialist_scale_init', 0.10)),
            unfreeze_encoder_neck=bool(int(getattr(args, 'unfreeze_encoder_neck', 0)) == 1),
            unfreeze_last_n_blocks=int(getattr(args, 'unfreeze_last_n_blocks', 0)),
        ).to(device)
    else:
        raise NotImplementedError("Method {} not implemented!".format(args.method))

    model = nn.DataParallel(model, device_ids=args.device_ids)

    work_dir = join(args.work_dir, args.task_name)
    os.makedirs(work_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=work_dir)
    logger = get_logger(log_file=os.path.join(work_dir, 'output.log'))
    logger.info(f"args: {json.dumps(vars(args), indent=2)}")

    aux_ft_map = parse_class_value_map(getattr(args, 'aux_ft_map', ''))
    aux_bnd_map = parse_class_value_map(getattr(args, 'aux_bnd_map', ''))
    aux_cldice_map = parse_class_value_map(getattr(args, 'aux_cldice_map', ''))
    dice_class_weight_map = parse_class_value_map(getattr(args, 'dice_class_weights', ''))
    rare_sampler_map = parse_class_value_map(getattr(args, 'rare_sampler_map', ''))
    best_class_weight_map = parse_class_value_map(getattr(args, 'best_class_weights', ''))
    if len(getattr(args, 'focus_crop_weights', [])) not in (0, len(getattr(args, 'focus_crop_classes', []))):
        raise ValueError('--focus_crop_weights must be empty or same length as --focus_crop_classes')

    if int(getattr(args, "log_model_arch", 0)) == 1:
        logger.info("Model: %s" % str(model))
    logger.info("Number of total parameters: %d" % (sum(p.numel() for p in model.parameters())))
    logger.info("Number of trainable parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
    logger.info(
        f"Runtime knobs: sam_image_size={int(getattr(args, 'sam_image_size', 256))}, tile_size={int(getattr(args, 'tile_size', 256))}, "
        f"batch_size={int(getattr(args, 'batch_size', 16))}, use_amp={bool(getattr(args, 'use_amp', False))}, "
        f"unfreeze_encoder_neck={int(getattr(args, 'unfreeze_encoder_neck', 0))}, "
        f"unfreeze_last_n_blocks={int(getattr(args, 'unfreeze_last_n_blocks', 0))}, "
        f"encoder_lr_mult={float(getattr(args, 'encoder_lr_mult', 0.05))}, "
        f"encoder_unfreeze_epoch={int(getattr(args, 'encoder_unfreeze_epoch', 0))}, "
        f"use_specialist_refiner={int(getattr(args, 'use_specialist_refiner', 0))}, "
        f"specialist_lr_mult={float(getattr(args, 'specialist_lr_mult', 1.0))}, "
        f"train_scope={str(getattr(args, 'train_scope', 'all'))}, "
        f"lr_schedule={str(getattr(args, 'lr_schedule', 'none'))}, hold_epochs={int(getattr(args, 'hold_epochs', 0))}"
    )

    encoder_block_ids = get_unfrozen_block_ids(args)
    delayed_encoder_unfreeze = False
    if args.method == "geoformerx" and int(getattr(args, 'encoder_unfreeze_epoch', 0)) > 0 and (bool(int(getattr(args, 'unfreeze_encoder_neck', 0)) == 1) or len(encoder_block_ids) > 0):
        set_partial_encoder_trainability(model.module, bool(int(getattr(args, 'unfreeze_encoder_neck', 0)) == 1), encoder_block_ids, enabled=False)
        delayed_encoder_unfreeze = True
        logger.info(f"Delayed encoder unfreeze enabled: epoch {int(getattr(args, 'encoder_unfreeze_epoch', 0))}, blocks={encoder_block_ids}, neck={int(getattr(args, 'unfreeze_encoder_neck', 0))}")
        logger.info("Number of trainable parameters after delayed freeze: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    init_from_path = getattr(args, 'init_from', None)
    if init_from_path is not None and os.path.isfile(init_from_path):
        logger.info(f'Loading init_from checkpoint: {init_from_path}')
        init_ckpt = torch.load(init_from_path, map_location=device)
        if isinstance(init_ckpt, dict) and 'model' in init_ckpt:
            model.module.load_parameters(init_ckpt['model'])
        else:
            model.module.load_parameters(init_ckpt)
        logger.info('init_from loaded. Shared weights restored; optimizer/epoch are fresh.')

    apply_train_scope(model, args, logger=logger)
    logger.info("Number of trainable parameters after train_scope: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = build_optimizer(model, args)
    # Multi-class loss: CE(bg+fg) + soft Dice (macro over fg)
    logger.info("Loss: CE(bg+fg) + Dice(fg macro/present-only) + optional per-class auxiliary losses")

    
    # ==========================
    # MoE routing pre-assign (task-first, dynamic-K)
    # This writes <data_path>/train/_moe_routing_cache.json and dataset.py will attach `moe_target`.
    # ==========================
    if int(getattr(args, "moe_preassign", 0)) == 1 and args.method == "geoformerx":
        try:
            cfg = PreassignConfig(
                include_label_features=True,   # label-aware pre-assign (offline)
                num_workers=max(4, int(getattr(args, "num_workers", 8))),
            )
            cache_path = build_routing_cache(join(args.data_path, "train"), expert_num=args.expert_num, cfg=cfg, force_rebuild=args.moe_preassign_force)
            logger.info(f"MoE preassign cache ready: {cache_path}")
            # Write a pointer file so evaluation can apply training routing to ID/ODD without
            # changing command line.
            try:
                with open(os.path.join(work_dir, "moe_train_cache_path.txt"), "w", encoding="utf-8") as wf:
                    wf.write(os.path.normpath(cache_path))
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"MoE preassign failed (continue without it): {e}")
    collect_train_tile_stats = bool(int(getattr(args, 'rare_sampler', 0)) == 1 or int(getattr(args, 'debug_epoch_json', 0)) == 1)
    train_dataset = PavementMultiClassTileDB(
        data_root=args.data_path,
        split="train",
        train=True,
        tile_size=int(args.tile_size),
        tile_stride=int(args.tile_stride),
        modal="VehicleProfiler",
        prompt_mode=str(args.prompt_mode),
        box_jitter=int(args.box_jitter),
        use_crack_crop=bool(int(getattr(args, 'train_use_crack_crop', 1)) == 1),
        crack_crop_prob=float(getattr(args, 'crack_crop_prob', 0.7)),
        crack_class_id=1,
        focus_crop_prob=float(getattr(args, 'focus_crop_prob', 0.0)),
        focus_crop_classes=list(getattr(args, 'focus_crop_classes', [])),
        focus_crop_weights=list(getattr(args, 'focus_crop_weights', [])),
        tile_jitter=int(getattr(args, 'tile_jitter', 0)),
        cache_size=int(getattr(args, "train_cache_size", 0)),
        collect_tile_stats=collect_train_tile_stats,
        tile_stats_cache=True,
    )
    logger.info(f"Number of training tiles: {len(train_dataset)}")
    if collect_train_tile_stats:
        try:
            tile_summary = train_dataset.get_tile_presence_summary()
            logger.info(f"Train tile presence summary: {json.dumps(tile_summary, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"Failed to summarize train tile stats: {e}")

    base_epoch_cfg = default_epoch_cfg(args, rare_sampler_map, dice_class_weight_map, aux_ft_map, aux_bnd_map, aux_cldice_map)
    active_epoch_cfg = dict(base_epoch_cfg)

    def build_train_dataloader_for_epoch(epoch_cfg, log_change: bool = False):
        train_dataset.focus_crop_prob = float(epoch_cfg.get('focus_crop_prob', 0.0))
        train_dataset.focus_crop_classes = [int(x) for x in list(epoch_cfg.get('focus_crop_classes', []))]
        train_dataset.focus_crop_weights = [float(x) for x in list(epoch_cfg.get('focus_crop_weights', []))]
        sampler = None
        active_rare_map = dict(epoch_cfg.get('rare_sampler_map', {}))
        if int(getattr(args, 'rare_sampler', 0)) == 1 and len(active_rare_map) > 0:
            area_focus_map = {}
            if float(getattr(args, 'crack_small_area_boost', 0.0)) > 0.0:
                area_focus_map[1] = float(getattr(args, 'crack_small_area_boost', 0.0))
            sample_weights = train_dataset.build_presence_sample_weights(
                active_rare_map,
                area_focus_map=area_focus_map if len(area_focus_map) > 0 else None,
                area_focus_ref=float(getattr(args, 'crack_area_ref', 0.02)),
                area_focus_power=float(getattr(args, 'crack_area_power', 0.5)),
                area_focus_cap=float(getattr(args, 'crack_area_cap', 3.0)),
            )
            n_samples = int(args.task_sampling_num_samples) if int(args.task_sampling_num_samples) > 0 else len(train_dataset)
            sampler = WeightedRandomSampler(sample_weights, num_samples=n_samples, replacement=True)
            if log_change:
                logger.info(
                    f"Epoch curriculum [{epoch_cfg.get('stage','base')}] rare sampler: map={active_rare_map}, "
                    f"focus_classes={train_dataset.focus_crop_classes}, focus_weights={train_dataset.focus_crop_weights}, "
                    f"focus_prob={float(train_dataset.focus_crop_prob):.3f}, area_focus={area_focus_map}, "
                    f"min_w={float(sample_weights.min().item()):.4f}, mean_w={float(sample_weights.mean().item()):.4f}, max_w={float(sample_weights.max().item()):.4f}"
                )
        elif args.task_balanced_sampling == 1:
            sampler = build_task_mixed_sampler(
                train_dataset.task_folders,
                alpha=args.task_sampling_alpha,
                power=args.task_sampling_power,
                num_samples=args.task_sampling_num_samples,
            )
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=args.num_workers,
            pin_memory=bool(int(getattr(args, "pin_memory", 0)) == 1),
            persistent_workers=bool(int(getattr(args, "persistent_workers", 0)) == 1 and int(args.num_workers) > 0),
            drop_last=True,
        )

    train_dataloader = build_train_dataloader_for_epoch(active_epoch_cfg, log_change=True)

    val_dataset = PavementMultiClassTileDB(
        data_root=args.data_path,
        split="val",
        train=False,
        tile_size=int(args.tile_size),
        tile_stride=int(args.tile_stride),
        modal="VehicleProfiler",
        prompt_mode=str(args.prompt_mode),
        box_jitter=0,
        use_crack_crop=False,
        crack_crop_prob=0.0,
        crack_class_id=1,
        focus_crop_prob=0.0,
        focus_crop_classes=[],
        focus_crop_weights=[],
        tile_jitter=0,
        cache_size=int(getattr(args, "val_cache_size", 0)),
        collect_tile_stats=bool(int(getattr(args, 'debug_epoch_json', 0)) == 1),
        tile_stats_cache=True,
    )
    logger.info(f"Number of validation tiles: {len(val_dataset)}")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(int(getattr(args, "pin_memory", 0)) == 1),
        persistent_workers=bool(int(getattr(args, "persistent_workers", 0)) == 1 and int(args.num_workers) > 0),
        drop_last=False,
    )

    img_size = model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)

    num_epochs = args.num_epochs
    start_epoch = 0
    best_loss = 1e10
    best_dsc = 0
    best_epoch = -1
    best_class_dsc = [-1.0 for _ in range(int(getattr(args, 'num_fg_classes', 7)) + 1)]
    epochs_without_improve = 0
    lr_scale = 1.0
    loss_log = []
    lr_log = []
    dsc_log = []
    paper_loss_rows = []
    paper_epoch_rows = []
    paper_alpha_rows = []
    expert_usage_history = []
    last_epoch_stage = None

    # -------------------------
    # Resume (explicit --resume or auto-resume from model_latest)
    # -------------------------
    resume_path = None
    if args.resume is not None and os.path.isfile(args.resume):
        resume_path = args.resume
    elif int(getattr(args, 'auto_resume', 1)) == 1:
        cand = join(work_dir, 'model_latest.pth')
        if os.path.isfile(cand):
            resume_path = cand

    if resume_path is not None:
        logger.info(f'Loading resume checkpoint: {resume_path}')
        checkpoint = torch.load(resume_path, map_location=device)
        if 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch']) + 1
        if 'model' in checkpoint:
            model.module.load_parameters(checkpoint['model'])
        if int(getattr(args, 'reset_optim', 0)) == 1:
            logger.info('reset_optim=1: optimizer state will NOT be loaded (using fresh optimizer settings).')
        else:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
        if args.use_amp and ('scaler' in checkpoint) and ('scaler' not in locals()):
            pass
        logger.info(f'Resume done. start_epoch={start_epoch}')
        try:
            lr_scale = float(checkpoint.get("train_state", {}).get("lr_scale", 1.0))
        except Exception:
            lr_scale = 1.0

    scaler = None
    amp_scale_value = float(getattr(args, 'amp_init_scale', 65536.0))
    amp_growth_interval = int(getattr(args, 'amp_growth_interval', 2000))

    def _build_grad_scaler(init_scale: float | None = None):
        scale = float(amp_scale_value if init_scale is None else init_scale)
        scale = max(1.0, float(scale))
        return torch.amp.GradScaler('cuda', init_scale=scale, growth_interval=amp_growth_interval)

    if args.use_amp:
        scaler = _build_grad_scaler()

        # load scaler state if present in resume checkpoint
        try:
            if int(getattr(args, 'reset_optim', 0)) == 0 and 'checkpoint' in locals() and isinstance(checkpoint, dict) and ('scaler' in checkpoint):
                scaler.load_state_dict(checkpoint['scaler'])
                try:
                    amp_scale_value = float(scaler.get_scale())
                except Exception:
                    pass
                logger.info(f'AMP scaler state loaded from checkpoint. current_scale={amp_scale_value:.1f}')
        except Exception as e:
            logger.warning(f'Failed to load AMP scaler state: {e}')

    model_ema = ModelEMA(model.module, decay=float(getattr(args, 'ema_decay', 0.999))) if int(getattr(args, 'use_ema', 0)) == 1 else None
    ema_reset_done = False
    if model_ema is not None and resume_path is not None:
        try:
            if isinstance(checkpoint, dict) and ('ema_model' in checkpoint):
                model_ema.shadow = {k: v.to(device) for k, v in checkpoint['ema_model'].items()}
                ema_reset_done = True
                logger.info('EMA state loaded from checkpoint.')
        except Exception as e:
            logger.warning(f'Failed to load EMA state: {e}')

    epoch_debug_path = join(work_dir, 'epoch_debug.jsonl')
    nonfinite_debug_path = join(work_dir, 'nonfinite_batches.jsonl')
    if start_epoch == 0:
        if int(getattr(args, 'debug_epoch_json', 0)) == 1:
            open(epoch_debug_path, 'w', encoding='utf-8').close()
        if int(getattr(args, 'debug_nonfinite_dump', 0)) == 1:
            open(nonfinite_debug_path, 'w', encoding='utf-8').close()

    base_lrs = get_base_lrs(optimizer)

    def _reset_optimizer_for_epoch(cur_epoch: int, reset_scaler: bool = False):
        nonlocal optimizer, scaler, amp_scale_value
        optimizer = build_optimizer(model, args)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["initial_lr"] = float(base_lr)
        cur_lrs = compute_epoch_lrs(
            base_lrs=base_lrs,
            epoch=cur_epoch,
            num_epochs=num_epochs,
            schedule=str(getattr(args, "lr_schedule", "none")),
            warmup_epochs=int(getattr(args, "warmup_epochs", 0)),
            hold_epochs=int(getattr(args, "hold_epochs", 0)),
            min_lr=float(getattr(args, "min_lr", 1e-6)),
            lr_scale=float(lr_scale),
        )
        set_optimizer_lrs(optimizer, cur_lrs)
        if args.use_amp and (reset_scaler or scaler is None):
            scaler = _build_grad_scaler(amp_scale_value)

    def _amp_backoff(reason: str = 'nonfinite'):
        nonlocal scaler, amp_scale_value
        if (not args.use_amp) or (scaler is None):
            return
        try:
            cur_scale = float(scaler.get_scale())
        except Exception:
            cur_scale = float(amp_scale_value)
        new_scale = max(1.0, cur_scale * float(getattr(args, 'amp_backoff', 0.5)))
        amp_scale_value = float(new_scale)
        scaler = _build_grad_scaler(amp_scale_value)
        logger.info(f'AMP scaler backoff after {reason}: {cur_scale:.1f} -> {amp_scale_value:.1f}')

    start_time = time.time()
    num_classes = int(getattr(args, 'num_fg_classes', 7)) + 1
    overall_pbar = None
    if int(getattr(args, 'use_tqdm', 1)) == 1 and int(getattr(args, 'overall_progress', 1)) == 1:
        total_steps_all = max(0, num_epochs - start_epoch) * (len(train_dataloader) + len(val_dataloader))
        overall_pbar = tqdm(total=total_steps_all, desc='Overall', position=0, dynamic_ncols=True, mininterval=float(getattr(args, 'tqdm_mininterval', 5.0)))
    encoder_has_been_unfrozen = (not delayed_encoder_unfreeze)
    for epoch in range(start_epoch, num_epochs):
        if int(getattr(args, 'crack_polish', 0)) == 1:
            active_epoch_cfg = get_crack_polish_cfg(epoch, num_epochs, base_epoch_cfg, start_epoch=int(getattr(args, 'crack_polish_start', 15)))
        elif int(getattr(args, 'pavement_final_push', 0)) == 1:
            active_epoch_cfg = get_pavement_final_push_cfg(epoch, num_epochs, base_epoch_cfg)
        else:
            active_epoch_cfg = dict(base_epoch_cfg)
        stage_changed = str(active_epoch_cfg.get('stage', 'base')) != str(last_epoch_stage)
        train_dataloader = build_train_dataloader_for_epoch(active_epoch_cfg, log_change=stage_changed or epoch == int(start_epoch))
        last_epoch_stage = str(active_epoch_cfg.get('stage', 'base'))

        if (model_ema is not None) and (not ema_reset_done) and bool(int(getattr(args, 'ema_warm_start', 1)) == 1) and epoch >= int(getattr(args, 'ema_start_epoch', 0)):
            model_ema.reset_from_model(model.module)
            ema_reset_done = True
            logger.info(f"EMA warm-start reset at epoch {epoch}. Shadow now matches current model before EMA evaluation/updates.")

        if delayed_encoder_unfreeze and (not encoder_has_been_unfrozen) and epoch >= int(getattr(args, 'encoder_unfreeze_epoch', 0)):
            set_partial_encoder_trainability(model.module, bool(int(getattr(args, 'unfreeze_encoder_neck', 0)) == 1), encoder_block_ids, enabled=True)
            optimizer = build_optimizer(model, args)
            base_lrs = get_base_lrs(optimizer)
            logger.info(f"Partial encoder unfrozen at epoch {epoch}. trainable params now: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            encoder_has_been_unfrozen = True
        if torch.cuda.is_available() and str(device).startswith('cuda'):
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        epoch_lrs = compute_epoch_lrs(
            base_lrs=base_lrs,
            epoch=epoch,
            num_epochs=num_epochs,
            schedule=str(getattr(args, "lr_schedule", "none")),
            warmup_epochs=int(getattr(args, "warmup_epochs", 0)),
            hold_epochs=int(getattr(args, "hold_epochs", 0)),
            min_lr=float(getattr(args, "min_lr", 1e-6)),
            lr_scale=float(lr_scale),
        )
        set_optimizer_lrs(optimizer, epoch_lrs)
        lr = optimizer.param_groups[0]['lr']

        # ==========================
        # Train
        # ==========================
        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        step = 0
        success_steps = 0
        epoch_nonfinite = 0
        epoch_sanitized = 0
        nonfinite_logs = 0
        epoch_aborted = False
        safe_model_state = snapshot_model_state(model.module)
        safe_snapshot_every = max(1, int(getattr(args, "safe_snapshot_interval", 25)))
        current_batch_debug = None

        epoch_part_sum = defaultdict(float)
        train_pixel_hist = np.zeros((num_classes,), dtype=np.float64)
        train_present_hist = np.zeros((num_classes,), dtype=np.float64)
        focus_counter = defaultdict(int)
        alpha_count = 0
        alpha_sum = 0.0
        alpha_sq_sum = 0.0
        alpha_min = float('inf')
        alpha_max = float('-inf')
        refine_abs_sum = 0.0
        refine_abs_n = 0
        line_delta_abs_sum = 0.0
        line_delta_abs_n = 0
        surface_delta_abs_sum = 0.0
        surface_delta_abs_n = 0
        gate_sum = {}
        gate_top1 = {}
        gate_ent_sum = defaultdict(float)
        gate_n = defaultdict(int)
        grad_norm_sum = 0.0
        grad_norm_max = 0.0

        model.train()
        pbar_train = tqdm(
            train_dataloader,
            desc=f"Epoch [{epoch}/{num_epochs}] Train",
            disable=int(getattr(args, "use_tqdm", 1)) == 0,
            mininterval=float(getattr(args, "tqdm_mininterval", 5.0)),
            dynamic_ncols=True,
            position=1 if overall_pbar is not None else 0,
            leave=False if overall_pbar is not None else True,
        )

        def _dump_nonfinite(reason: str, value=None):
            if int(getattr(args, 'debug_nonfinite_dump', 0)) != 1:
                return
            payload = {
                'epoch': int(epoch),
                'step': int(step),
                'reason': str(reason),
                'value': None if value is None else (float(value) if isinstance(value, (int, float)) else str(value)),
                'batch': current_batch_debug,
            }
            try:
                with open(nonfinite_debug_path, 'a', encoding='utf-8') as wf:
                    wf.write(json.dumps(payload, ensure_ascii=False) + '\n')
            except Exception:
                pass

        def _handle_nonfinite(reason: str, value=None):
            nonlocal optimizer, scaler, lr_scale, epoch_nonfinite, nonfinite_logs, epoch_aborted
            epoch_nonfinite += 1
            culprit_name, culprit_abs = (None, None)
            if str(reason).startswith('grad') and int(getattr(args, 'nonfinite_param_log', 1)) == 1:
                culprit_name, culprit_abs = first_nonfinite_grad_param(model)
            _dump_nonfinite(reason, value)
            if nonfinite_logs < int(getattr(args, "nonfinite_log_limit", 3)):
                msg = f"Non-finite {reason} at epoch={epoch}, step={step}"
                if value is not None:
                    msg += f": {value}"
                if culprit_name is not None:
                    msg += f" | grad_param={culprit_name}"
                    if culprit_abs is not None:
                        msg += f" max_abs={culprit_abs:.4e}"
                logger.warning(msg)
                nonfinite_logs += 1
            if int(getattr(args, "restore_on_nonfinite", 1)) == 1:
                restore_model_state(model.module, safe_model_state)
                if args.use_amp:
                    _amp_backoff(reason=str(reason))
                reduce_every = int(getattr(args, "nonfinite_reduce_every", 0))
                if reduce_every > 0 and (epoch_nonfinite % reduce_every == 0):
                    min_scale = float(getattr(args, "min_lr", 1e-6)) / max(max(base_lrs), 1e-12)
                    lr_scale = max(min_scale, float(lr_scale) * float(getattr(args, "nonfinite_reduce_lr", 0.5)))
                _reset_optimizer_for_epoch(epoch, reset_scaler=False)
                epoch_aborted = bool(int(getattr(args, 'abort_epoch_on_nonfinite', 0)) == 1)
            else:
                optimizer.zero_grad(set_to_none=True)

        for data, label in pbar_train:
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # Prepare a debug snapshot early, before any tensor is mutated.
            batch_names = data.get('name', [])
            if isinstance(batch_names, str):
                batch_names = [batch_names]
            elif torch.is_tensor(batch_names):
                batch_names = [str(x) for x in batch_names.cpu().tolist()]
            else:
                batch_names = [str(x) for x in list(batch_names)]

            focus_vals = data.get('focus_class', [])
            if torch.is_tensor(focus_vals):
                focus_vals_list = [int(x) for x in focus_vals.detach().cpu().tolist()]
            elif isinstance(focus_vals, (list, tuple)):
                focus_vals_list = [int(x) for x in focus_vals]
            elif focus_vals is None:
                focus_vals_list = []
            else:
                focus_vals_list = [int(focus_vals)]

            valid_dbg = label != 255
            if torch.any(valid_dbg):
                dbg_hist = torch.bincount(label[valid_dbg].reshape(-1), minlength=num_classes).cpu().tolist()
            else:
                dbg_hist = [0 for _ in range(num_classes)]
            current_batch_debug = {
                'names': batch_names,
                'focus_class': focus_vals_list,
                'label_hist': dbg_hist,
            }

            # Resize to SAM image size if needed.
            if data["img"].shape[-1] != img_size:
                data["box"] = box_transform.apply_boxes_torch(
                    data["box"].reshape(-1, 2, 2),
                    data["img"].shape[-2:]
                ).reshape(-1, 4)
                data["img"] = img_transform(data["img"])

            data["img"] = data["img"].to(device, non_blocking=True)
            data["box"] = data["box"].to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # Forward (GeoFormerX can optionally return gates/fusion/refine debug tensors).
            alpha_map = None
            refine_delta = None
            specialist_aux = None
            if args.method == "geoformerx":
                if args.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(data, return_gates=True, return_alpha=True, return_refine=True, return_specialist=True)
                else:
                    out = model(data, return_gates=True, return_alpha=True, return_refine=True, return_specialist=True)
                mask_pred = out["masks"]
                gates = out.get("gates", [])
                alpha_map = out.get('fusion_alpha', None)
                refine_delta = out.get('refine_delta', None)
                specialist_aux = out.get('specialist_aux', None)
            else:
                if args.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        mask_pred = model(data)
                else:
                    mask_pred = model(data)
                gates = []

            if int(getattr(args, "skip_nonfinite", 1)) == 1 and (not torch.isfinite(mask_pred).all()):
                _handle_nonfinite("mask_pred")
                if epoch_aborted:
                    break
                continue

            if mask_pred.shape[-2:] != label.shape[-2:]:
                mask_pred = F.interpolate(mask_pred, size=label.shape[-2:], mode="bilinear", align_corners=False)
                if refine_delta is not None:
                    refine_delta = F.interpolate(refine_delta, size=label.shape[-2:], mode="bilinear", align_corners=False)
                if alpha_map is not None:
                    alpha_map = F.interpolate(alpha_map, size=label.shape[-2:], mode='bilinear', align_corners=False)

            active_dice_class_weight_map = dict(active_epoch_cfg.get('dice_class_weight_map', {}))
            active_aux_ft_map = dict(active_epoch_cfg.get('aux_ft_map', {}))
            active_aux_bnd_map = dict(active_epoch_cfg.get('aux_bnd_map', {}))
            active_aux_cldice_map = dict(active_epoch_cfg.get('aux_cldice_map', {}))
            loss, loss_parts = multiclass_total_loss(
                logits_fg=mask_pred.float(),
                labels=label.long(),
                ignore_index=255,
                ce_weight=float(getattr(args, 'ce_weight', 1.0)),
                dice_weight=float(getattr(args, 'dice_weight', 1.0)),
                use_dynamic_ce_weights=bool(int(getattr(args, 'use_dynamic_ce_weights', 1)) == 1),
                ce_w_min=float(getattr(args, 'ce_w_min', 0.2)),
                ce_w_max=float(getattr(args, 'ce_w_max', 10.0)),
                ce_bg_scale=float(active_epoch_cfg.get('ce_bg_scale', getattr(args, 'ce_bg_scale', 0.3))),
                ohem_ratio=float(active_epoch_cfg.get('ohem_ratio', getattr(args, 'ohem_ratio', 0.0))),
                dice_present_only=bool(int(getattr(args, 'dice_present_only', 1)) == 1),
                dice_class_weights=(active_dice_class_weight_map if len(active_dice_class_weight_map) > 0 else None),
                crack_ft_lambda=float(active_epoch_cfg.get('crack_ft_lambda', getattr(args, 'crack_ft_lambda', 0.0))),
                crack_class_id=1,
                crack_alpha=float(getattr(args, 'crack_alpha', 0.3)),
                crack_beta=float(getattr(args, 'crack_beta', 0.7)),
                crack_gamma=float(getattr(args, 'crack_gamma', 0.75)),
                crack_bnd_lambda=float(active_epoch_cfg.get('crack_bnd_lambda', getattr(args, 'crack_bnd_lambda', 0.0))),
                crack_bnd_kernel=int(getattr(args, 'crack_bnd_kernel', 3)),
                aux_ft_map=(active_aux_ft_map if len(active_aux_ft_map) > 0 else None),
                aux_bnd_map=(active_aux_bnd_map if len(active_aux_bnd_map) > 0 else None),
                aux_cldice_map=(active_aux_cldice_map if len(active_aux_cldice_map) > 0 else None),
                crack_cldice_lambda=float(active_epoch_cfg.get('crack_cldice_lambda', getattr(args, 'crack_cldice_lambda', 0.0))),
                crack_cldice_iters=int(active_epoch_cfg.get('crack_cldice_iters', getattr(args, 'crack_cldice_iters', 3))),
                line_group_ce_lambda=float(getattr(args, 'line_group_ce_lambda', 0.0)),
                surface_group_ce_lambda=float(getattr(args, 'surface_group_ce_lambda', 0.0)),
            )

            if args.moe_warmup_epochs and args.moe_warmup_epochs > 0:
                moe_scale = min(1.0, float(epoch + 1) / float(args.moe_warmup_epochs))
            else:
                moe_scale = 1.0
            moe_out = moe_aux_loss(
                gates,
                lb_coef=args.moe_lb_coef * moe_scale,
                ent_coef=args.moe_ent_coef * moe_scale,
                topk_for_load=1,
            )

            route_loss = None
            route_term = None
            invalid_tgt = 0
            if args.moe_route_sup_coef > 0 and gates and isinstance(data, dict) and ("moe_target" in data):
                try:
                    tgt = data["moe_target"].to(device, non_blocking=True).long()
                    E = int(gates[0].shape[1])
                    invalid_tgt = int(((tgt < 0) | (tgt >= E)).sum().item())
                    tgt = tgt.clamp_(0, E - 1)
                    nlls = []
                    for g in gates:
                        if g is None:
                            continue
                        g_clamp = torch.clamp(g, min=1e-8)
                        p = g_clamp.gather(1, tgt.view(-1, 1)).squeeze(1)
                        nlls.append((-torch.log(p)).mean())
                    if nlls:
                        route_loss = torch.stack(nlls).mean()
                except Exception:
                    route_loss = None
                    invalid_tgt = 0

            if args.moe_route_sup_coef > 0 and args.moe_route_warmup > 0:
                w = min(1.0, float(epoch) / float(max(1, args.moe_route_warmup)))
            else:
                w = 1.0
            route_coef = float(args.moe_route_sup_coef) * float(w)
            if (route_loss is not None) and (route_coef > 0):
                route_term = route_loss * float(route_coef)
                cap = float(getattr(args, 'moe_route_cap_ratio', 0.25)) * loss.detach()
                route_term = torch.minimum(route_term, cap)
            else:
                route_term = 0.0

            total_loss = loss + moe_out.total.to(loss.device) + (route_term if isinstance(route_term, torch.Tensor) else 0.0)
            if int(getattr(args, 'skip_nonfinite', 1)) == 1 and (not torch.isfinite(total_loss)):
                _handle_nonfinite('total_loss', float(total_loss.detach().item()) if total_loss.numel() == 1 else None)
                if epoch_aborted:
                    break
                continue

            if args.use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
            else:
                total_loss.backward()

            sanitized = {'num_params': 0, 'num_values': 0, 'first_name': None, 'first_max_abs': None}
            if int(getattr(args, 'grad_sanitize_nonfinite', 1)) == 1:
                sanitized = sanitize_nonfinite_gradients(model, fill=float(getattr(args, 'grad_sanitize_fill', 0.0)))
                if int(sanitized.get('num_params', 0)) > 0:
                    if int(sanitized.get('num_params', 0)) <= int(getattr(args, 'grad_sanitize_max_params', 2)):
                        epoch_sanitized += 1
                        if nonfinite_logs < int(getattr(args, 'nonfinite_log_limit', 3)):
                            msg = (
                                f"Sanitized non-finite gradients at epoch={epoch}, step={step}: "
                                f"params={int(sanitized.get('num_params', 0))}, values={int(sanitized.get('num_values', 0))}"
                            )
                            if sanitized.get('first_name') is not None:
                                msg += f" | first_param={sanitized.get('first_name')}"
                            logger.warning(msg)
                            nonfinite_logs += 1
                    else:
                        _handle_nonfinite('gradients', 'sanitize_limit_exceeded')
                        optimizer.zero_grad(set_to_none=True)
                        if epoch_aborted:
                            break
                        continue

            grad_norm = clip_grad_norm_and_get(model, float(getattr(args, 'grad_clip_norm', 1.0)))
            if (not torch.isfinite(grad_norm)) or (not gradients_finite(model)):
                _handle_nonfinite('gradients', float(grad_norm.detach().item()) if torch.isfinite(grad_norm) else 'nan')
                optimizer.zero_grad(set_to_none=True)
                if epoch_aborted:
                    break
                continue

            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
                try:
                    amp_scale_value = float(scaler.get_scale())
                except Exception:
                    pass
            else:
                optimizer.step()

            if int(getattr(args, 'skip_nonfinite', 1)) == 1 and (not parameters_finite(model)):
                _handle_nonfinite('parameters_after_step')
                optimizer.zero_grad(set_to_none=True)
                if epoch_aborted:
                    break
                continue

            if model_ema is not None and epoch >= int(getattr(args, 'ema_start_epoch', 0)):
                model_ema.update(model.module)

            if success_steps == 0 or ((success_steps + 1) % safe_snapshot_every == 0):
                safe_model_state = snapshot_model_state(model.module)

            epoch_loss += float(total_loss.detach().item())
            epoch_seg_loss += float(loss.detach().item())
            success_steps += 1
            grad_norm_val = float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
            grad_norm_sum += grad_norm_val
            grad_norm_max = max(grad_norm_max, grad_norm_val)
            lr = optimizer.state_dict()['param_groups'][0]['lr']

            # Debug accumulators.
            valid = label != 255
            if torch.any(valid):
                hist = torch.bincount(label[valid].reshape(-1), minlength=num_classes).detach().cpu().numpy().astype(np.float64)
                train_pixel_hist += hist
                train_present_hist += (hist > 0).astype(np.float64)
            for fv in focus_vals_list:
                focus_counter[int(fv)] += 1
            for k, v in loss_parts.items():
                try:
                    epoch_part_sum[k] += float(v.detach().item()) if torch.is_tensor(v) else float(v)
                except Exception:
                    pass
            epoch_part_sum['moe_total'] += float(moe_out.total.detach().item())
            epoch_part_sum['moe_lb'] += float(moe_out.lb.detach().item())
            epoch_part_sum['moe_ent'] += float(moe_out.ent.detach().item())
            epoch_part_sum['route_term'] += float(route_term.detach().item()) if hasattr(route_term, 'detach') else float(route_term)
            epoch_part_sum['route_loss'] += float(route_loss.detach().item()) if hasattr(route_loss, 'detach') else 0.0
            epoch_part_sum['route_invalid_tgt'] += float(invalid_tgt)
            if alpha_map is not None:
                a = alpha_map.detach().float()
                alpha_sum += float(a.mean().item()) * a.shape[0]
                alpha_sq_sum += float((a.mean(dim=(1,2,3)) ** 2).sum().item())
                alpha_min = min(alpha_min, float(a.min().item()))
                alpha_max = max(alpha_max, float(a.max().item()))
                alpha_count += int(a.shape[0])
            if refine_delta is not None:
                rd = refine_delta.detach().float()
                refine_abs_sum += float(rd.abs().mean().item()) * rd.shape[0]
                refine_abs_n += int(rd.shape[0])
            if isinstance(specialist_aux, dict):
                if specialist_aux.get('line_delta', None) is not None:
                    ld = specialist_aux['line_delta'].detach().float()
                    line_delta_abs_sum += float(ld.abs().mean().item()) * ld.shape[0]
                    line_delta_abs_n += int(ld.shape[0])
                if specialist_aux.get('surface_delta', None) is not None:
                    sd = specialist_aux['surface_delta'].detach().float()
                    surface_delta_abs_sum += float(sd.abs().mean().item()) * sd.shape[0]
                    surface_delta_abs_n += int(sd.shape[0])
            if gates:
                for li, g in enumerate(gates):
                    gd = g.detach().float()
                    if li not in gate_sum:
                        gate_sum[li] = torch.zeros(gd.shape[1], dtype=torch.float64)
                        gate_top1[li] = torch.zeros(gd.shape[1], dtype=torch.float64)
                    gate_sum[li] += gd.sum(dim=0).cpu().double()
                    gate_top1[li] += torch.bincount(torch.argmax(gd, dim=1), minlength=gd.shape[1]).cpu().double()
                    gate_ent_sum[li] += float((-(gd * gd.clamp_min(1e-8).log()).sum(dim=1)).sum().item())
                    gate_n[li] += int(gd.shape[0])

            mem_gb = 0.0
            if torch.cuda.is_available() and str(device).startswith('cuda'):
                try:
                    mem_gb = float(torch.cuda.memory_allocated(device) / (1024 ** 3))
                except Exception:
                    mem_gb = 0.0
            pbar_train.set_postfix({
                'lr': f'{lr:.2e}',
                'loss': float(loss.item()),
                'moe': float(moe_out.total.detach().item()) if hasattr(moe_out.total, 'item') else 0.0,
                'route': float(route_term.detach().item()) if hasattr(route_term, 'detach') else float(route_term),
                'memGB': f'{mem_gb:.1f}',
            })
            if overall_pbar is not None:
                overall_pbar.update(1)
                overall_pbar.set_postfix({'epoch': f'{epoch+1}/{num_epochs}', 'phase': 'train', 'lr': f'{lr:.2e}', 'memGB': f'{mem_gb:.1f}'})
            if int(getattr(args, 'tb_batch_log', 0)) == 1:
                epoch_1000x = int((epoch + step / len(train_dataloader)) * 1000)
                log_writer.add_scalar('batch/lr', lr, epoch_1000x)
                log_writer.add_scalar('batch/loss', float(loss.item()), epoch_1000x)
                log_writer.add_scalar('batch/moe_total', float(moe_out.total.detach().item()), epoch_1000x)
                log_writer.add_scalar('batch/moe_lb', float(moe_out.lb.detach().item()), epoch_1000x)
                log_writer.add_scalar('batch/moe_ent', float(moe_out.ent.detach().item()), epoch_1000x)
                if alpha_map is not None:
                    log_writer.add_scalar('batch/fusion_alpha_mean', float(alpha_map.detach().mean().item()), epoch_1000x)
                if refine_delta is not None:
                    log_writer.add_scalar('batch/refine_abs_mean', float(refine_delta.detach().abs().mean().item()), epoch_1000x)
                if isinstance(specialist_aux, dict) and specialist_aux.get('line_delta', None) is not None:
                    log_writer.add_scalar('batch/line_delta_abs_mean', float(specialist_aux['line_delta'].detach().abs().mean().item()), epoch_1000x)
                if isinstance(specialist_aux, dict) and specialist_aux.get('surface_delta', None) is not None:
                    log_writer.add_scalar('batch/surface_delta_abs_mean', float(specialist_aux['surface_delta'].detach().abs().mean().item()), epoch_1000x)
                if int(getattr(args, 'moe_route_log', 1)) == 1:
                    try:
                        log_writer.add_scalar('batch/route_term', float(route_term.detach().item()) if hasattr(route_term, 'detach') else float(route_term), epoch_1000x)
                        log_writer.add_scalar('batch/route_loss', float(route_loss.detach().item()) if hasattr(route_loss, 'detach') else 0.0, epoch_1000x)
                        log_writer.add_scalar('batch/route_coef', float(route_coef), epoch_1000x)
                        log_writer.add_scalar('batch/route_invalid_tgt', float(invalid_tgt), epoch_1000x)
                    except Exception:
                        pass

        if success_steps > 0:
            epoch_loss /= success_steps
            epoch_seg_loss /= success_steps
        else:
            epoch_loss = float('nan')
            epoch_seg_loss = float('nan')
        lr_log.append(lr)
        loss_log.append(epoch_loss)
        log_writer.add_scalar('epoch/lr', lr, epoch + 1)
        if np.isfinite(epoch_loss):
            log_writer.add_scalar('epoch/loss', epoch_loss, epoch + 1)
            log_writer.add_scalar('epoch/seg_loss', epoch_seg_loss, epoch + 1)
        if success_steps > 0:
            for k, v in epoch_part_sum.items():
                log_writer.add_scalar(f'epoch_dbg/{k}', float(v / success_steps), epoch + 1)
            log_writer.add_scalar('epoch_dbg/grad_norm_mean', float(grad_norm_sum / success_steps), epoch + 1)
            log_writer.add_scalar('epoch_dbg/grad_norm_max', float(grad_norm_max), epoch + 1)
            if alpha_count > 0:
                alpha_mean = alpha_sum / alpha_count
                alpha_var = max(0.0, alpha_sq_sum / alpha_count - alpha_mean ** 2)
                log_writer.add_scalar('epoch_dbg/fusion_alpha_mean', float(alpha_mean), epoch + 1)
                log_writer.add_scalar('epoch_dbg/fusion_alpha_std', float(alpha_var ** 0.5), epoch + 1)
            if refine_abs_n > 0:
                log_writer.add_scalar('epoch_dbg/refine_abs_mean', float(refine_abs_sum / refine_abs_n), epoch + 1)
            if line_delta_abs_n > 0:
                log_writer.add_scalar('epoch_dbg/line_delta_abs_mean', float(line_delta_abs_sum / line_delta_abs_n), epoch + 1)
            if surface_delta_abs_n > 0:
                log_writer.add_scalar('epoch_dbg/surface_delta_abs_mean', float(surface_delta_abs_sum / surface_delta_abs_n), epoch + 1)

        # save raw latest/lowest checkpoints
        ckpt = {
            'model': model.module.save_parameters(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'moe_cfg': {
                'moe_topk': int(getattr(args, 'moe_topk', 2)),
                'moe_temp': float(getattr(args, 'moe_temp', 1.0)),
                'moe_noise_std': float(getattr(args, 'moe_noise_std', 0.0)),
                'moe_style_bn': int(bool(getattr(args, 'moe_style_bn', 1))),
                'moe_style_dropout': float(getattr(args, 'moe_style_dropout', 0.10)),
                'fusion_mode': str(getattr(args, 'fusion_mode', 'global')), 'use_fusion_2d3d': int(getattr(args, 'use_fusion_2d3d', 1)), 'moe_style_scale': float(getattr(args, 'moe_style_scale', 0.25)), 'lr_schedule': str(getattr(args, 'lr_schedule', 'none')), 'hold_epochs': int(getattr(args, 'hold_epochs', 0)),
            },
            'model_cfg': {
                'sam_image_size': int(getattr(args, 'sam_image_size', 256)),
                'use_logit_refiner': int(getattr(args, 'use_logit_refiner', 0)),
                'refiner_hidden': int(getattr(args, 'refiner_hidden', 32)),
                'use_specialist_refiner': int(getattr(args, 'use_specialist_refiner', 0)),
                'line_refiner_hidden': int(getattr(args, 'line_refiner_hidden', 48)),
                'surface_refiner_hidden': int(getattr(args, 'surface_refiner_hidden', 32)),
                'specialist_scale_init': float(getattr(args, 'specialist_scale_init', 0.10)),
                'train_scope': str(getattr(args, 'train_scope', 'all')),
                'fusion_mode': str(getattr(args, 'fusion_mode', 'global')), 'use_fusion_2d3d': int(getattr(args, 'use_fusion_2d3d', 1)), 'moe_style_scale': float(getattr(args, 'moe_style_scale', 0.25)), 'lr_schedule': str(getattr(args, 'lr_schedule', 'none')), 'hold_epochs': int(getattr(args, 'hold_epochs', 0)),
                'unfreeze_encoder_neck': int(getattr(args, 'unfreeze_encoder_neck', 0)),
                'unfreeze_last_n_blocks': int(getattr(args, 'unfreeze_last_n_blocks', 0)),
            },
            'loss_cfg': {
                'crack_ft_lambda': float(getattr(args, 'crack_ft_lambda', 0.0)),
                'vehicle_crack_ft_extra': float(getattr(args, 'vehicle_crack_ft_extra', 0.0)),
                'vehicle_crack_loss_boost': float(getattr(args, 'vehicle_crack_loss_boost', 0.0)),
                'crack_alpha': float(getattr(args, 'crack_alpha', 0.3)),
                'crack_beta': float(getattr(args, 'crack_beta', 0.7)),
                'crack_gamma': float(getattr(args, 'crack_gamma', 0.75)),
                'aux_ft_map': aux_ft_map,
                'aux_bnd_map': aux_bnd_map,
                'aux_cldice_map': aux_cldice_map,
                'dice_class_weight_map': dice_class_weight_map,
            },
            'force_preassign': False,
            'train_state': {
                'lr_scale': float(lr_scale),
            },
        }
        if model_ema is not None:
            ckpt['ema_model'] = model_ema.state_dict()
        if args.use_amp and (scaler is not None):
            try:
                ckpt['scaler'] = scaler.state_dict()
            except Exception:
                pass
        torch.save(ckpt, join(work_dir, 'model_latest.pth'))
        if np.isfinite(epoch_loss) and (epoch_loss < best_loss):
            best_loss = epoch_loss
            torch.save(ckpt, join(work_dir, 'model_lowest.pth'))
        if int(getattr(args, 'save_epoch_ckpt', 0)) == 1:
            torch.save(ckpt, join(work_dir, f'model_{epoch}.pth'))

        # ==========================
        # Eval (optionally with EMA weights)
        # ==========================
        raw_eval_backup = None
        use_ema_eval = (model_ema is not None) and (int(getattr(args, 'eval_with_ema', 1)) == 1) and (epoch >= int(getattr(args, 'ema_start_epoch', 0)))
        if use_ema_eval:
            raw_eval_backup = snapshot_model_state(model.module)
            model_ema.copy_to(model.module)

        epoch_dsc_batch = 0.0
        size = 0
        val_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        val_bnd_sum = np.zeros((num_classes,), dtype=np.float64)
        val_bnd_cnt = np.zeros((num_classes,), dtype=np.float64)
        val_alpha_sum = 0.0
        val_alpha_n = 0
        model.eval()
        pbar_val = tqdm(
            val_dataloader,
            desc=f'Epoch [{epoch}/{num_epochs}] Val',
            disable=int(getattr(args, 'use_tqdm', 1)) == 0,
            mininterval=float(getattr(args, 'tqdm_mininterval', 5.0)),
            dynamic_ncols=True,
            position=1 if overall_pbar is not None else 0,
            leave=False if overall_pbar is not None else True,
        )

        with torch.no_grad():
            for data, label in pbar_val:
                if data['img'].shape[-1] != img_size:
                    data['box'] = box_transform.apply_boxes_torch(
                        data['box'].reshape(-1, 2, 2),
                        data['img'].shape[-2:]
                    ).reshape(-1, 4)
                    data['img'] = img_transform(data['img'])

                data['img'] = data['img'].to(device, non_blocking=True)
                data['box'] = data['box'].to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                if args.method == 'geoformerx':
                    out = model(data, return_alpha=True, return_refine=True, return_specialist=True)
                    mask_pred = out['masks']
                    alpha_val = out.get('fusion_alpha', None)
                else:
                    mask_pred = model(data)
                    alpha_val = None

                if int(getattr(args, 'skip_nonfinite', 1)) == 1 and (not torch.isfinite(mask_pred).all()):
                    logger.warning(f'Validation produced non-finite mask_pred at epoch={epoch}; skipping current batch.')
                    continue
                if mask_pred.shape[-2:] != label.shape[-2:]:
                    mask_pred = F.interpolate(mask_pred, size=label.shape[-2:], mode='bilinear', align_corners=False)
                    if alpha_val is not None:
                        alpha_val = F.interpolate(alpha_val, size=label.shape[-2:], mode='bilinear', align_corners=False)
                logits_all = logits_with_bg(mask_pred.float())
                pred = torch.argmax(logits_all, dim=1)
                mdice = foreground_mdice_hard(
                    pred=pred,
                    gt=label.long(),
                    num_fg=int(getattr(args, 'num_fg_classes', 7)),
                    ignore_index=255,
                    present_only=True,
                )
                epoch_dsc_batch += float(mdice.item()) * float(label.shape[0])
                size += int(label.shape[0])
                gt_np = label.detach().cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                val_cm += confusion_matrix(
                    gt=gt_np,
                    pred=pred_np,
                    num_classes=num_classes,
                    ignore_index=255,
                )
                try:
                    for bi in range(gt_np.shape[0]):
                        bnd = boundary_f1_per_class(gt_np[bi], pred_np[bi], num_classes=num_classes, ignore_index=255)
                        for ci in range(num_classes):
                            if not np.isnan(bnd[ci]):
                                val_bnd_sum[ci] += float(bnd[ci])
                                val_bnd_cnt[ci] += 1.0
                except Exception:
                    pass
                if alpha_val is not None:
                    val_alpha_sum += float(alpha_val.detach().mean().item()) * int(alpha_val.shape[0])
                    val_alpha_n += int(alpha_val.shape[0])
                val_mem_gb = 0.0
                if torch.cuda.is_available() and str(device).startswith('cuda'):
                    try:
                        val_mem_gb = float(torch.cuda.memory_allocated(device) / (1024 ** 3))
                    except Exception:
                        val_mem_gb = 0.0
                pbar_val.set_postfix({'mdice_fg': float(mdice.item()), 'memGB': f'{val_mem_gb:.1f}'})
                if overall_pbar is not None:
                    overall_pbar.update(1)
                    overall_pbar.set_postfix({'epoch': f'{epoch+1}/{num_epochs}', 'phase': 'val', 'memGB': f'{val_mem_gb:.1f}'})

        if use_ema_eval and raw_eval_backup is not None:
            restore_model_state(model.module, raw_eval_backup)

        epoch_dsc_batch = epoch_dsc_batch / max(size, 1)
        per_cls = per_class_metrics_from_cm(val_cm)
        fg_support = per_cls.support[1:]
        fg_dice = per_cls.dice[1:]
        present_fg = fg_support > 0
        epoch_dsc_cm = float(np.mean(fg_dice[present_fg])) if np.any(present_fg) else 1.0
        val_bnd_mean = np.full((num_classes,), np.nan, dtype=np.float64)
        for ci in range(num_classes):
            if val_bnd_cnt[ci] > 0:
                val_bnd_mean[ci] = val_bnd_sum[ci] / val_bnd_cnt[ci]
        epoch_bnd_f1_fg = nanmean(val_bnd_mean[1:])
        epoch_crack_dice = float(per_cls.dice[1]) if len(per_cls.dice) > 1 else float('nan')
        if str(getattr(args, 'best_metric', 'cm_mdice')) == 'batch_mdice':
            epoch_score = epoch_dsc_batch
        elif str(getattr(args, 'best_metric', 'cm_mdice')) == 'weighted' and len(best_class_weight_map) > 0:
            weights = []
            values = []
            for cid in range(1, num_classes):
                if fg_support[cid - 1] <= 0:
                    continue
                values.append(float(fg_dice[cid - 1]))
                weights.append(float(best_class_weight_map.get(cid, 1.0)))
            epoch_score = float(np.average(values, weights=weights)) if len(values) > 0 else epoch_dsc_cm
        else:
            epoch_score = epoch_dsc_cm

        # Optional per-class best checkpoints.  This is useful for final class-wise
        # delta fusion: one 50-epoch run can expose the best crack/marking/joint
        # states even when the global weighted score peaks earlier.
        if int(getattr(args, 'save_class_best_ckpt', 0)) == 1:
            try:
                class_best_ids = [int(x) for x in str(getattr(args, 'class_best_ids', '1,5,6')).replace(';', ',').split(',') if str(x).strip()]
            except Exception:
                class_best_ids = [1, 5, 6]
            improved_class_ids = []
            for cid in class_best_ids:
                if 0 <= int(cid) < num_classes and float(per_cls.dice[int(cid)]) > float(best_class_dsc[int(cid)]):
                    best_class_dsc[int(cid)] = float(per_cls.dice[int(cid)])
                    improved_class_ids.append(int(cid))
            if improved_class_ids:
                class_state = model_ema.state_dict() if use_ema_eval and (model_ema is not None) else model.module.save_parameters()
                for cid in improved_class_ids:
                    torch.save(
                        {
                            'model': class_state,
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'class_best_id': int(cid),
                            'class_best_dice': float(best_class_dsc[int(cid)]),
                            'val_score': float(epoch_score),
                            'val_batch_mdice': float(epoch_dsc_batch),
                            'val_cm_mdice': float(epoch_dsc_cm),
                            'ema_used': bool(use_ema_eval),
                            'model_cfg': {
                                'sam_image_size': int(getattr(args, 'sam_image_size', 256)),
                                'use_logit_refiner': int(getattr(args, 'use_logit_refiner', 0)),
                                'refiner_hidden': int(getattr(args, 'refiner_hidden', 32)),
                                'use_specialist_refiner': int(getattr(args, 'use_specialist_refiner', 0)),
                                'line_refiner_hidden': int(getattr(args, 'line_refiner_hidden', 48)),
                                'surface_refiner_hidden': int(getattr(args, 'surface_refiner_hidden', 32)),
                                'specialist_scale_init': float(getattr(args, 'specialist_scale_init', 0.10)),
                                'train_scope': str(getattr(args, 'train_scope', 'all')),
                                'fusion_mode': str(getattr(args, 'fusion_mode', 'global')), 'use_fusion_2d3d': int(getattr(args, 'use_fusion_2d3d', 1)), 'moe_style_scale': float(getattr(args, 'moe_style_scale', 0.25)),
                                'lr_schedule': str(getattr(args, 'lr_schedule', 'none')),
                                'hold_epochs': int(getattr(args, 'hold_epochs', 0)),
                                'unfreeze_encoder_neck': int(getattr(args, 'unfreeze_encoder_neck', 0)),
                                'unfreeze_last_n_blocks': int(getattr(args, 'unfreeze_last_n_blocks', 0)),
                            },
                        },
                        join(work_dir, f'model_best_c{int(cid)}.pth'),
                    )
                logger.info('Updated per-class best checkpoint(s): ' + ', '.join([f'c{cid}={best_class_dsc[cid]:.6f}' for cid in improved_class_ids]))

        dsc_log.append(epoch_score)
        log_writer.add_scalar('epoch/dsc_batch', epoch_dsc_batch, epoch + 1)
        log_writer.add_scalar('epoch/dsc_cm', epoch_dsc_cm, epoch + 1)
        log_writer.add_scalar('epoch/score', epoch_score, epoch + 1)
        log_writer.add_scalar('epoch/val_mDice_fg', epoch_dsc_cm, epoch + 1)
        log_writer.add_scalar('epoch/val_CrackDice', epoch_crack_dice, epoch + 1)
        log_writer.add_scalar('epoch/val_BndF1_fg', epoch_bnd_f1_fg, epoch + 1)
        for cid in range(1, num_classes):
            log_writer.add_scalar(f'epoch/val_dice_c{cid}', float(per_cls.dice[cid]), epoch + 1)
            log_writer.add_scalar(f'epoch/val_iou_c{cid}', float(per_cls.iou[cid]), epoch + 1)
        if val_alpha_n > 0:
            log_writer.add_scalar('epoch_dbg/val_fusion_alpha_mean', float(val_alpha_sum / val_alpha_n), epoch + 1)

        if int(getattr(args, 'paper_train_outputs', 1)) == 1:
            loss_row = {'epoch': int(epoch + 1), 'train_total_loss': float(epoch_loss) if np.isfinite(epoch_loss) else float('nan'), 'train_seg_loss': float(epoch_seg_loss) if np.isfinite(epoch_seg_loss) else float('nan'), 'lr': float(lr)}
            for k, v in epoch_part_sum.items():
                loss_row[str(k)] = float(v / max(1, success_steps))
            paper_loss_rows.append(loss_row)
            write_csv(os.path.join(work_dir, 'loss_components.csv'), paper_loss_rows)

            metric_row = {
                'epoch': int(epoch + 1),
                'train_total_loss': float(epoch_loss) if np.isfinite(epoch_loss) else float('nan'),
                'train_seg_loss': float(epoch_seg_loss) if np.isfinite(epoch_seg_loss) else float('nan'),
                'lr': float(lr),
                'val_mDice_fg': float(epoch_dsc_cm),
                'val_CrackDice': float(epoch_crack_dice),
                'val_BndF1_fg': float(epoch_bnd_f1_fg),
                'val_score': float(epoch_score),
            }
            for cid in range(1, num_classes):
                metric_row[f'Dice_c{cid}'] = float(per_cls.dice[cid])
            paper_epoch_rows.append(metric_row)
            write_csv(os.path.join(work_dir, 'val_metrics_by_epoch.csv'), paper_epoch_rows)

            alpha_row = {'epoch': int(epoch + 1)}
            if alpha_count > 0:
                alpha_mean_train = alpha_sum / alpha_count
                alpha_var_train = max(0.0, alpha_sq_sum / alpha_count - alpha_mean_train ** 2)
                alpha_row.update({'train_alpha_mean': float(alpha_mean_train), 'train_alpha_std': float(alpha_var_train ** 0.5), 'train_alpha_min': float(alpha_min), 'train_alpha_max': float(alpha_max)})
            if val_alpha_n > 0:
                alpha_row['val_alpha_mean'] = float(val_alpha_sum / val_alpha_n)
            paper_alpha_rows.append(alpha_row)
            write_csv(os.path.join(work_dir, 'alpha_by_epoch.csv'), paper_alpha_rows)

            if gate_sum:
                try:
                    usage = []
                    for li in sorted(gate_sum.keys()):
                        usage.append((gate_sum[li] / max(1, gate_n[li])).numpy())
                    usage = np.stack(usage, axis=0).astype(np.float64)
                    expert_usage_history.append(usage)
                    np.savez(os.path.join(work_dir, 'expert_usage_by_epoch.npz'), usage=np.stack(expert_usage_history, axis=0))
                    np.save(os.path.join(work_dir, 'expert_usage_latest.npy'), usage)
                    plot_expert_heatmap(usage, os.path.join(work_dir, 'expert_usage_latest.png'))
                except Exception:
                    pass
            try:
                plot_training_curves(os.path.join(work_dir, 'val_metrics_by_epoch.csv'), os.path.join(work_dir, 'paper_training_curves.png'))
            except Exception:
                pass

        if epoch_score > best_dsc:
            best_dsc = epoch_score
            best_epoch = epoch
            epochs_without_improve = 0
            moe_train_cache = None
            try:
                p_txt = os.path.join(work_dir, 'moe_train_cache_path.txt')
                if os.path.exists(p_txt):
                    with open(p_txt, 'r', encoding='utf-8') as rf:
                        moe_train_cache = rf.read().strip()
            except Exception:
                moe_train_cache = None
            best_state = model_ema.state_dict() if use_ema_eval and (model_ema is not None) else model.module.save_parameters()
            torch.save(
                {
                    'model': best_state,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'force_preassign': bool(int(getattr(args, 'moe_preassign', 0)) == 1 and float(getattr(args, 'moe_route_sup_coef', 0.0)) > 0),
                    'moe_train_cache': moe_train_cache,
                    'val_score': float(epoch_score),
                    'val_batch_mdice': float(epoch_dsc_batch),
                    'val_cm_mdice': float(epoch_dsc_cm),
                    'ema_used': bool(use_ema_eval),
                    'model_cfg': {
                        'sam_image_size': int(getattr(args, 'sam_image_size', 256)),
                        'use_logit_refiner': int(getattr(args, 'use_logit_refiner', 0)),
                        'refiner_hidden': int(getattr(args, 'refiner_hidden', 32)),
                        'use_specialist_refiner': int(getattr(args, 'use_specialist_refiner', 0)),
                        'line_refiner_hidden': int(getattr(args, 'line_refiner_hidden', 48)),
                        'surface_refiner_hidden': int(getattr(args, 'surface_refiner_hidden', 32)),
                        'specialist_scale_init': float(getattr(args, 'specialist_scale_init', 0.10)),
                        'train_scope': str(getattr(args, 'train_scope', 'all')),
                        'fusion_mode': str(getattr(args, 'fusion_mode', 'global')), 'use_fusion_2d3d': int(getattr(args, 'use_fusion_2d3d', 1)), 'moe_style_scale': float(getattr(args, 'moe_style_scale', 0.25)), 'lr_schedule': str(getattr(args, 'lr_schedule', 'none')), 'hold_epochs': int(getattr(args, 'hold_epochs', 0)),
                        'unfreeze_encoder_neck': int(getattr(args, 'unfreeze_encoder_neck', 0)),
                        'unfreeze_last_n_blocks': int(getattr(args, 'unfreeze_last_n_blocks', 0)),
                    },
                },
                join(work_dir, 'model_best.pth'),
            )
        else:
            epochs_without_improve += 1

        plt.figure()
        plt.plot(loss_log)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(join(work_dir, 'train_loss.png'))
        plt.close()
        plt.figure()
        plt.plot(lr_log)
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.savefig(join(work_dir, 'lr.png'))
        plt.close()
        plt.figure()
        plt.plot(dsc_log)
        plt.title('Validation Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.savefig(join(work_dir, 'val_dsc.png'))
        plt.close()

        epoch_debug = {
            'epoch': int(epoch),
            'lr': float(lr),
            'stage': str(active_epoch_cfg.get('stage', 'base')),
            'epoch_loss': float(epoch_loss) if np.isfinite(epoch_loss) else None,
            'epoch_seg_loss': float(epoch_seg_loss) if np.isfinite(epoch_seg_loss) else None,
            'epoch_nonfinite': int(epoch_nonfinite),
            'epoch_sanitized': int(epoch_sanitized),
            'epoch_aborted': bool(epoch_aborted),
            'grad_norm_mean': float(grad_norm_sum / max(1, success_steps)),
            'grad_norm_max': float(grad_norm_max),
            'train_pixel_hist': train_pixel_hist.tolist(),
            'train_present_hist': train_present_hist.tolist(),
            'focus_counter': {str(k): int(v) for k, v in sorted(focus_counter.items())},
            'loss_parts': {k: float(v / max(1, success_steps)) for k, v in epoch_part_sum.items()},
            'val_batch_mdice': float(epoch_dsc_batch),
            'val_cm_mdice': float(epoch_dsc_cm),
            'val_score': float(epoch_score),
            'val_CrackDice': float(epoch_crack_dice),
            'val_BndF1_fg': float(epoch_bnd_f1_fg),
            'val_per_class_dice': {str(cid): float(per_cls.dice[cid]) for cid in range(num_classes)},
            'val_per_class_iou': {str(cid): float(per_cls.iou[cid]) for cid in range(num_classes)},
            'val_support': {str(cid): float(per_cls.support[cid]) for cid in range(num_classes)},
            'best_class_dice_so_far': {str(cid): float(best_class_dsc[cid]) for cid in range(num_classes)},
            'ema_used_for_eval': bool(use_ema_eval),
            'train_steps': int(success_steps),
            'train_steps_total': int(len(train_dataloader)),
            'val_steps_total': int(len(val_dataloader)),
        }
        if torch.cuda.is_available() and str(device).startswith('cuda'):
            try:
                epoch_debug['gpu_peak_alloc_gb'] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
                epoch_debug['gpu_peak_reserved_gb'] = float(torch.cuda.max_memory_reserved(device) / (1024 ** 3))
            except Exception:
                pass
        if alpha_count > 0:
            alpha_mean = alpha_sum / alpha_count
            alpha_var = max(0.0, alpha_sq_sum / alpha_count - alpha_mean ** 2)
            epoch_debug['train_fusion_alpha'] = {
                'mean': float(alpha_mean),
                'std': float(alpha_var ** 0.5),
                'min': float(alpha_min),
                'max': float(alpha_max),
            }
        if refine_abs_n > 0:
            epoch_debug['refine_abs_mean'] = float(refine_abs_sum / refine_abs_n)
        if val_alpha_n > 0:
            epoch_debug['val_fusion_alpha_mean'] = float(val_alpha_sum / val_alpha_n)
        if gate_sum:
            gate_debug = {}
            for li in sorted(gate_sum.keys()):
                n = max(1, gate_n[li])
                gate_debug[str(li)] = {
                    'mean_prob': (gate_sum[li] / n).tolist(),
                    'top1_load': (gate_top1[li] / n).tolist(),
                    'entropy': float(gate_ent_sum[li] / n),
                }
            epoch_debug['gate'] = gate_debug
        if int(getattr(args, 'debug_epoch_json', 0)) == 1:
            with open(epoch_debug_path, 'a', encoding='utf-8') as wf:
                wf.write(json.dumps(epoch_debug, ensure_ascii=False) + '\n')

        peak_alloc_gb = 0.0
        peak_reserved_gb = 0.0
        if torch.cuda.is_available() and str(device).startswith('cuda'):
            try:
                peak_alloc_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
                peak_reserved_gb = float(torch.cuda.max_memory_reserved(device) / (1024 ** 3))
            except Exception:
                peak_alloc_gb = 0.0
                peak_reserved_gb = 0.0
        logger.info(
            f"Epoch [{epoch}] [{str(active_epoch_cfg.get('stage','base'))}] - LR: {lr:.6g}, TotalLoss: {epoch_loss if np.isfinite(epoch_loss) else float('nan')}, "
            f"SegLoss: {epoch_seg_loss if np.isfinite(epoch_seg_loss) else float('nan')}, "
            f"BatchDSC: {epoch_dsc_batch:.6f}, CmDSC: {epoch_dsc_cm:.6f}, Score: {epoch_score:.6f}, "
            f"Crack: {float(per_cls.dice[1]):.4f}, BndF1_fg: {epoch_bnd_f1_fg:.4f}, Pothole: {float(per_cls.dice[2]):.4f}, "
            f"Patch: {float(per_cls.dice[4]):.4f}, Manhole: {float(per_cls.dice[7]):.4f}, "
            f"peak_alloc={peak_alloc_gb:.2f}GB, peak_reserved={peak_reserved_gb:.2f}GB, "
            f"nonfinite_steps: {epoch_nonfinite}{' (restored)' if epoch_nonfinite > 0 else ''}, sanitized_steps: {epoch_sanitized}, lr_scale={lr_scale:.6f}"
        )
        log_writer.flush()

        if int(getattr(args, 'early_stop_patience', 0)) > 0 and epochs_without_improve >= int(getattr(args, 'early_stop_patience', 0)):
            logger.info(
                f"Early stopping triggered at epoch={epoch}. "
                f"No score improvement for {epochs_without_improve} epochs. Best epoch={best_epoch}, best score={best_dsc:.6f}"
            )
            break

    if overall_pbar is not None:
        overall_pbar.close()
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    logger.info(f"Best epoch: {best_epoch}, Best score: {best_dsc}")
    logger.info(f"Time cost: {total_time_str}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
