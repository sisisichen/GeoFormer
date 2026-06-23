# -*- coding: utf-8 -*-
"""Default reproduction recipe for GeoFormerX.

This file intentionally centralizes the long, publication-oriented training and
inference settings. End users should normally call only ``train.py`` and
``evaluate.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence


def _s(x) -> str:
    return str(x)


def python_executable() -> str:
    return sys.executable or "python"


def base_stage_args(
    data_path: str,
    checkpoint: str,
    work_dir: str,
    model_type: str = "vit_b",
    device: str = "cuda:0",
    device_ids: Sequence[int] = (0,),
    num_epochs: int = 50,
    batch_size: int = 28,
) -> List[str]:
    return [
        python_executable(), "_train_stage.py",
        "--data_path", data_path,
        "--checkpoint", checkpoint,
        "--model_type", model_type,
        "--sam_image_size", "256",
        "--method", "geoformerx",
        "--work_dir", work_dir,
        "--task_name", "base",
        "--bottleneck_dim", "16",
        "--embedding_dim", "16",
        "--expert_num", "8",
        "--moe_topk", "2",
        "--moe_temp", "1.0",
        "--moe_noise_std", "0.0",
        "--fusion_hidden", "16",
        "--fusion_mode", "hybrid",
        "--use_logit_refiner", "1",
        "--refiner_hidden", "32",
        "--use_specialist_refiner", "0",
        "--moe_style_scale", "0.25",
        "--batch_size", _s(batch_size),
        "--num_epochs", _s(num_epochs),
        "--num_workers", "0",
        "--pin_memory", "0",
        "--persistent_workers", "0",
        "--train_cache_size", "0",
        "--val_cache_size", "0",
        "--tb_batch_log", "0",
        "--prompt_mode", "full",
        "--use_fusion_2d3d", "1",
        "--device", device,
        "--device_ids", *[_s(i) for i in device_ids],
        "--auto_resume", "0",
        "--reset_optim", "1",
        "--lr", "0.0003",
        "--weight_decay", "0.01",
        "--mask_decoder_lr_mult", "0.25",
        "--unfreeze_encoder_neck", "1",
        "--unfreeze_last_n_blocks", "2",
        "--encoder_lr_mult", "0.03",
        "--encoder_unfreeze_epoch", "8",
        "--task_balanced_sampling", "0",
        "--rare_sampler", "1",
        "--rare_sampler_map", "1:1.8,2:1.8,4:1.2,5:1.1,6:1.2,7:0.8",
        "--crack_small_area_boost", "0.15",
        "--crack_area_ref", "0.02",
        "--crack_area_power", "0.5",
        "--crack_area_cap", "1.7",
        "--ohem_ratio", "0.0625",
        "--ce_bg_scale", "0.55",
        "--dice_present_only", "1",
        "--dice_class_weights", "1:1.8,2:1.5,4:1.1,5:1.2,6:1.3,7:0.8",
        "--train_use_crack_crop", "1",
        "--crack_crop_prob", "0.45",
        "--focus_crop_prob", "0.65",
        "--focus_crop_classes", "1", "2", "4", "5", "6",
        "--focus_crop_weights", "4.0", "2.0", "1.2", "1.2", "1.4",
        "--tile_jitter", "64",
        "--crack_ft_lambda", "0.12",
        "--crack_bnd_lambda", "0.06",
        "--crack_cldice_lambda", "0.012",
        "--crack_cldice_iters", "2",
        "--aux_ft_map", "2:0.06,4:0.03,5:0.03,6:0.04",
        "--aux_bnd_map", "2:0.02,4:0.02,5:0.02,6:0.02",
        "--aux_cldice_map", "5:0.008,6:0.008",
        "--line_group_ce_lambda", "0.10",
        "--surface_group_ce_lambda", "0.05",
        "--lr_schedule", "flatcosine",
        "--warmup_epochs", "1",
        "--hold_epochs", "32",
        "--min_lr", "0.00001",
        "--early_stop_patience", "0",
        "--use_tqdm", "1",
        "--overall_progress", "1",
        "--log_model_arch", "0",
        "--save_epoch_ckpt", "0",
        "--restore_on_nonfinite", "1",
        "--nonfinite_reduce_lr", "0.98",
        "--nonfinite_reduce_every", "0",
        "--safe_snapshot_interval", "25",
        "--nonfinite_param_log", "1",
        "--debug_epoch_json", "1",
        "--debug_nonfinite_dump", "1",
        "--grad_sanitize_nonfinite", "1",
        "--grad_sanitize_max_params", "99999",
        "--grad_sanitize_fill", "0.0",
        "--use_ema", "1",
        "--ema_decay", "0.995",
        "--ema_start_epoch", "0",
        "--ema_warm_start", "1",
        "--eval_with_ema", "1",
        "--best_metric", "weighted",
        "--best_class_weights", "1:2.2,2:1.2,4:0.9,5:1.4,6:1.5,7:0.6",
    ]


def specialist_stage_args(
    data_path: str,
    checkpoint: str,
    work_dir: str,
    init_from: str,
    model_type: str = "vit_b",
    device: str = "cuda:0",
    device_ids: Sequence[int] = (0,),
    num_epochs: int = 50,
    batch_size: int = 24,
) -> List[str]:
    return [
        python_executable(), "_train_stage.py",
        "--data_path", data_path,
        "--checkpoint", checkpoint,
        "--model_type", model_type,
        "--sam_image_size", "256",
        "--method", "geoformerx",
        "--work_dir", work_dir,
        "--task_name", "line_specialist",
        "--init_from", init_from,
        "--bottleneck_dim", "16",
        "--embedding_dim", "16",
        "--expert_num", "8",
        "--moe_topk", "2",
        "--moe_temp", "1.0",
        "--moe_noise_std", "0.0",
        "--fusion_hidden", "16",
        "--fusion_mode", "hybrid",
        "--use_logit_refiner", "1",
        "--refiner_hidden", "32",
        "--use_specialist_refiner", "1",
        "--line_refiner_hidden", "64",
        "--surface_refiner_hidden", "32",
        "--specialist_scale_init", "0.04",
        "--specialist_lr_mult", "3.0",
        "--train_scope", "specialist",
        "--moe_style_scale", "0.25",
        "--batch_size", _s(batch_size),
        "--num_epochs", _s(num_epochs),
        "--num_workers", "0",
        "--pin_memory", "0",
        "--persistent_workers", "0",
        "--train_cache_size", "0",
        "--val_cache_size", "0",
        "--tb_batch_log", "0",
        "--prompt_mode", "full",
        "--use_fusion_2d3d", "1",
        "--device", device,
        "--device_ids", *[_s(i) for i in device_ids],
        "--auto_resume", "0",
        "--reset_optim", "1",
        "--lr", "0.00004",
        "--weight_decay", "0.005",
        "--mask_decoder_lr_mult", "0.0",
        "--unfreeze_encoder_neck", "0",
        "--unfreeze_last_n_blocks", "0",
        "--encoder_lr_mult", "0.0",
        "--task_balanced_sampling", "0",
        "--rare_sampler", "1",
        "--rare_sampler_map", "1:2.2,2:0.6,4:0.6,5:1.6,6:1.8,7:0.4",
        "--crack_small_area_boost", "0.20",
        "--crack_area_ref", "0.02",
        "--crack_area_power", "0.5",
        "--crack_area_cap", "1.7",
        "--ohem_ratio", "0.0625",
        "--ce_bg_scale", "0.55",
        "--dice_present_only", "1",
        "--dice_class_weights", "1:2.8,2:0.5,4:0.5,5:1.8,6:2.1,7:0.3",
        "--train_use_crack_crop", "1",
        "--crack_crop_prob", "0.45",
        "--focus_crop_prob", "0.70",
        "--focus_crop_classes", "1", "5", "6",
        "--focus_crop_weights", "5.0", "2.0", "2.4",
        "--tile_jitter", "64",
        "--crack_ft_lambda", "0.12",
        "--crack_bnd_lambda", "0.06",
        "--crack_cldice_lambda", "0.018",
        "--crack_cldice_iters", "2",
        "--aux_ft_map", "5:0.04,6:0.05",
        "--aux_bnd_map", "5:0.02,6:0.025",
        "--aux_cldice_map", "5:0.010,6:0.012",
        "--line_group_ce_lambda", "0.14",
        "--surface_group_ce_lambda", "0.00",
        "--crack_polish", "0",
        "--lr_schedule", "flatcosine",
        "--warmup_epochs", "1",
        "--hold_epochs", "46",
        "--min_lr", "0.00001",
        "--early_stop_patience", "0",
        "--use_tqdm", "1",
        "--overall_progress", "1",
        "--log_model_arch", "0",
        "--save_epoch_ckpt", "0",
        "--save_class_best_ckpt", "1",
        "--class_best_ids", "1,5,6",
        "--restore_on_nonfinite", "1",
        "--nonfinite_reduce_lr", "0.98",
        "--nonfinite_reduce_every", "0",
        "--safe_snapshot_interval", "25",
        "--nonfinite_param_log", "1",
        "--debug_epoch_json", "1",
        "--debug_nonfinite_dump", "1",
        "--grad_sanitize_nonfinite", "1",
        "--grad_sanitize_max_params", "99999",
        "--grad_sanitize_fill", "0.0",
        "--use_ema", "1",
        "--ema_decay", "0.995",
        "--ema_start_epoch", "0",
        "--ema_warm_start", "1",
        "--eval_with_ema", "1",
        "--best_metric", "weighted",
        "--best_class_weights", "1:3.0,2:0.4,4:0.4,5:2.0,6:2.2,7:0.2",
    ]


def merge_args(work_dir: str) -> List[str]:
    root = Path(work_dir)
    base = root / "base" / "model_best.pth"
    spec = root / "line_specialist"
    final = root / "model_final.pth"
    return [
        python_executable(), "_merge_bundle.py",
        "--primary", _s(spec / "model_best.pth"),
        "--base_ckpt", _s(base),
        "--class_ckpt_map", f"1={spec / 'model_best_c1.pth'};5={spec / 'model_best_c5.pth'};6={spec / 'model_best_c6.pth'}",
        "--row_weight_map", "1:1.0,5:1.0,6:1.0",
        "--fusion_weight_map", "1:0.75,5:0.50,6:0.65",
        "--shared_alpha", "0.25",
        "--include_class_bank", "1",
        "--out", _s(final),
    ]


def evaluation_args(
    data_path: str,
    checkpoint: str,
    ckpt: str,
    out_dir: str,
    split: str = "test",
    model_type: str = "vit_b",
    device: str = "cuda:0",
    device_ids: Sequence[int] = (0,),
    batch_size: int = 10,
) -> List[str]:
    return [
        python_executable(), "_evaluate_core.py",
        "--data_path", data_path,
        "--split", split,
        "--checkpoint", checkpoint,
        "--model_type", model_type,
        "--model", "geoformerx",
        "--expert_num", "8",
        "--num_fg_classes", "7",
        "--bottleneck_dim", "16",
        "--embedding_dim", "16",
        "--moe_topk", "2",
        "--moe_temp", "1.0",
        "--moe_noise_std", "0.0",
        "--moe_style_scale", "0.25",
        "--fusion_hidden", "16",
        "--fusion_mode", "auto",
        "--use_logit_refiner", "-1",
        "--refiner_hidden", "32",
        "--use_specialist_refiner", "-1",
        "--line_refiner_hidden", "64",
        "--surface_refiner_hidden", "32",
        "--tile_size", "256",
        "--tile_stride", "128",
        "--batch_size", _s(batch_size),
        "--prompt_mode", "full",
        "--use_fusion_2d3d", "1",
        "--tta_hflip", "1",
        "--blend", "hann",
        "--print_report", "full",
        "--device", device,
        "--device_ids", *[_s(i) for i in device_ids],
        "--ckpt", ckpt,
        "--use_bundle_fusion", "1",
        "--class_ckpt_weight_map", "1:0.75,5:0.50,6:0.65",
        "--class_ckpt_fuse_mode", "delta",
        "--class_logit_bias_map", "1:0.02,5:0.01,6:0.02",
        "--out_dir", out_dir,
    ]


def _set_cli_arg(cmd: List[str], name: str, value: str) -> List[str]:
    out = list(cmd)
    if name in out:
        i = out.index(name)
        if i + 1 < len(out):
            out[i + 1] = str(value)
        else:
            out.append(str(value))
    else:
        out.extend([name, str(value)])
    return out


def apply_ablation_variant(cmd: List[str], variant: str = "full") -> List[str]:
    """Apply one of the three paper ablations to a train/eval command.

    Variants used in the manuscript:
    - full: unchanged GeoFormerX
    - no_depth: remove geometric input use throughout fusion/refinement
    - no_geometry_gate: keep RGB-D but replace reliability gate with fixed alpha=1
    - no_style_routing: keep MoE but remove sample-wise latent style from the router
    """
    v = str(variant or "full").lower().strip().replace("-", "_")
    out = list(cmd)
    if v in ("", "full", "geoformerx"):
        return out
    if v in ("no_depth", "wo_depth", "w_o_depth"):
        out = _set_cli_arg(out, "--use_fusion_2d3d", "0")
        return out
    if v in ("no_geometry_gate", "wo_geometry_gate", "fixed_fusion"):
        out = _set_cli_arg(out, "--use_fusion_2d3d", "1")
        out = _set_cli_arg(out, "--fusion_mode", "fixed")
        return out
    if v in ("no_style_routing", "wo_style", "no_style"):
        out = _set_cli_arg(out, "--moe_style_scale", "0.0")
        return out
    raise ValueError(f"Unsupported ablation variant: {variant}")
