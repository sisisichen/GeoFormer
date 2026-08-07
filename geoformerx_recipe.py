# -*- coding: utf-8 -*-
"""Canonical GeoFormerX-G8-D0-S0 training and evaluation commands."""

from __future__ import annotations

import sys
from typing import List, Sequence


def _s(value) -> str:
    return str(value)


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
    seed: int = 2028,
) -> List[str]:
    """Resolve the single 50-epoch canonical training run."""
    return [
        python_executable(), "_train_stage.py",
        "--data_path", data_path,
        "--checkpoint", checkpoint,
        "--model_type", model_type,
        "--sam_image_size", "256",
        "--method", "geoformerx",
        "--work_dir", work_dir,
        "--task_name", "base",
        "--seed", _s(seed),
        "--bottleneck_dim", "16",
        "--embedding_dim", "16",
        "--expert_num", "4",
        "--fusion_gate_variant", "G8",
        "--adapter_variant", "S0",
        "--static_bottleneck_dim", "42",
        "--moe_topk", "2",
        "--moe_temp", "1.0",
        "--moe_noise_std", "0.0",
        "--moe_lb_coef", "0.0",
        "--moe_ent_coef", "0.0",
        "--moe_route_sup_coef", "0.0",
        "--moe_preassign", "0",
        "--force_class_expert", "0",
        "--fusion_hidden", "16",
        "--fusion_mode", "hybrid",
        "--use_fusion_2d3d", "1",
        "--use_logit_refiner", "1",
        "--refiner_hidden", "32",
        "--use_specialist_refiner", "0",
        "--tile_size", "256",
        "--tile_stride", "128",
        "--prompt_mode", "full",
        "--box_jitter", "0",
        "--batch_size", _s(batch_size),
        "--num_epochs", _s(num_epochs),
        "--num_workers", "0",
        "--pin_memory", "0",
        "--persistent_workers", "0",
        "--train_cache_size", "0",
        "--val_cache_size", "0",
        "--device", device,
        "--device_ids", *[_s(i) for i in device_ids],
        "--auto_resume", "0",
        "--reset_optim", "1",
        "--lr", "0.0003",
        "--weight_decay", "0.01",
        "--mask_decoder_lr_mult", "0.25",
        "--grad_clip_norm", "1.0",
        "--unfreeze_encoder_neck", "0",
        "--unfreeze_last_n_blocks", "0",
        "--encoder_unfreeze_epoch", "0",
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
        "--use_ema", "1",
        "--ema_decay", "0.995",
        "--ema_start_epoch", "0",
        "--ema_warm_start", "1",
        "--eval_with_ema", "1",
        "--best_metric", "cm_mdice",
        "--save_epoch_ckpt", "0",
        "--restore_on_nonfinite", "1",
        "--safe_snapshot_interval", "25",
        "--grad_sanitize_nonfinite", "1",
        "--grad_sanitize_max_params", "99999",
        "--grad_sanitize_fill", "0.0",
        "--use_tqdm", "1",
        "--overall_progress", "1",
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
        "--expert_num", "4",
        "--num_fg_classes", "7",
        "--bottleneck_dim", "16",
        "--embedding_dim", "16",
        "--fusion_gate_variant", "G8",
        "--adapter_variant", "S0",
        "--static_bottleneck_dim", "42",
        "--fusion_hidden", "16",
        "--fusion_mode", "hybrid",
        "--use_fusion_2d3d", "1",
        "--use_logit_refiner", "1",
        "--refiner_hidden", "32",
        "--use_specialist_refiner", "0",
        "--tile_size", "256",
        "--tile_stride", "128",
        "--batch_size", _s(batch_size),
        "--prompt_mode", "full",
        "--tta_hflip", "1",
        "--blend", "hann",
        "--prefer_ema", "1",
        "--stitch_mode", "logits",
        "--print_report", "full",
        "--device", device,
        "--device_ids", *[_s(i) for i in device_ids],
        "--ckpt", ckpt,
        "--out_dir", out_dir,
    ]


def _set_cli_arg(cmd: List[str], name: str, value: str) -> List[str]:
    out = list(cmd)
    if name in out:
        index = out.index(name)
        out[index + 1] = str(value)
    else:
        out.extend([name, str(value)])
    return out


def apply_ablation_variant(cmd: List[str], variant: str = "full") -> List[str]:
    """Resolve compact architecture controls used in the manuscript screening."""
    value = str(variant or "full").lower().strip().replace("-", "_")
    out = list(cmd)
    if value in {"", "full", "g8_d0_s0"}:
        return out
    if value in {"no_range", "intensity_only"}:
        out = _set_cli_arg(out, "--use_fusion_2d3d", "0")
        return _set_cli_arg(out, "--use_logit_refiner", "0")
    if value in {"no_geometry_gate", "g0"}:
        return _set_cli_arg(out, "--fusion_gate_variant", "G0")
    if value in {"moe_adapter", "m0"}:
        return _set_cli_arg(out, "--adapter_variant", "M0")
    if value in {"decoder_only", "a0"}:
        return _set_cli_arg(out, "--adapter_variant", "A0")
    raise ValueError(f"Unsupported ablation variant: {variant}")
