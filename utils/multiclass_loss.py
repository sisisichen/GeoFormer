# -*- coding: utf-8 -*-
"""Multi-class losses for the pavement 2D+3D segmentation setting.

We train a model that outputs **foreground** class logits (7 channels: 1..7).
For a stable and fair multi-class objective we build an 8-class logit tensor by
prepending a fixed background logit (all zeros), then apply:
  - Cross Entropy (with optional dynamic class weights)
  - Soft Dice (macro over foreground classes)

This keeps the model output contract (7 logits) while allowing a standard
multi-class competition among classes at each pixel.

All functions support an ignore_index (255).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _flatten(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x.flatten(1)
    if x.dim() == 3:
        return x.flatten(1)
    raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")


def focal_tversky_from_probs(
    prob: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Focal-Tversky computed from **probabilities** (recommended for softmax setups).

    Args:
        prob:   (B,1,H,W) in [0,1]
        target: (B,1,H,W) binary {0,1}

    Returns:
        scalar loss (mean over batch), empty-aware.
    """
    p = prob.float().clamp(0.0, 1.0)
    t = (target > 0).float()

    pf = _flatten(p)
    tf = _flatten(t)

    tp = (pf * tf).sum(dim=1)
    fp = (pf * (1.0 - tf)).sum(dim=1)
    fn = ((1.0 - pf) * tf).sum(dim=1)

    tversky = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)

    # empty-aware: if both empty => loss=0
    # IMPORTANT: gamma < 1 has an infinite derivative at zero.  In late training
    # some easy samples can reach tversky ~= 1.0; pow(1 - tversky, 0.75) is then
    # finite in the forward pass but can produce Inf/NaN gradients.  Clamp the
    # pow base and zero the true-empty samples explicitly.  This is the main
    # stabilizer for 50-epoch final polishing.
    empty = (tf.sum(dim=1) == 0) & (pf.sum(dim=1) == 0)
    tversky = torch.where(empty, torch.ones_like(tversky), tversky)

    base = (1.0 - tversky).clamp_min(float(eps))
    loss = torch.pow(base, float(gamma))
    loss = torch.where(empty, torch.zeros_like(loss), loss)
    return loss.mean()


def _soft_morph_gradient(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Differentiable morphological gradient: dilation(x) - erosion(x)."""
    if k <= 1:
        return torch.zeros_like(x)
    pad = k // 2
    dil = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
    ero = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=pad)
    return (dil - ero).clamp_min(0.0)


def _dice_from_probs(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary soft dice from probabilities."""
    p = prob.float().clamp(0.0, 1.0)
    t = (target > 0).float()
    pf = _flatten(p)
    tf = _flatten(t)
    inter = (pf * tf).sum(dim=1)
    denom = pf.sum(dim=1) + tf.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    empty = (tf.sum(dim=1) == 0) & (pf.sum(dim=1) == 0)
    dice = torch.where(empty, torch.ones_like(dice), dice)
    return dice.mean()




def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Soft erosion used by clDice-style skeletonization."""
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(x.shape)}")
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def _soft_skel(x: torch.Tensor, iters: int = 3) -> torch.Tensor:
    x = x.float().clamp(0.0, 1.0)
    img = x
    img1 = _soft_open(img)
    skel = torch.relu(img - img1)
    for _ in range(max(0, int(iters) - 1)):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = torch.relu(img - img1)
        skel = skel + torch.relu(delta - skel * delta)
    return skel


def cldice_from_probs(
    prob: torch.Tensor,
    target: torch.Tensor,
    iters: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft clDice / centerline Dice for thin structures.

    This is especially useful for crack-like classes where topology / continuity
    matters more than filled area. Empty-aware: if both target and prediction are
    empty for a sample, that sample contributes zero loss.
    """
    p = prob.float().clamp(0.0, 1.0)
    t = (target > 0).float()

    skel_p = _soft_skel(p, iters=int(iters))
    skel_t = _soft_skel(t, iters=int(iters))

    pf = _flatten(p)
    tf = _flatten(t)
    spf = _flatten(skel_p)
    stf = _flatten(skel_t)

    empty = (tf.sum(dim=1) == 0) & (pf.sum(dim=1) == 0)

    tprec = (spf * tf).sum(dim=1) / (spf.sum(dim=1) + eps)
    tsens = (stf * pf).sum(dim=1) / (stf.sum(dim=1) + eps)
    cl = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    cl = torch.where(empty, torch.ones_like(cl), cl)
    return cl.mean()


def grouped_subset_ce_loss(
    logits_all: torch.Tensor,
    labels: torch.Tensor,
    class_ids,
    ignore_index: int = 255,
) -> torch.Tensor:
    """CE over a subset of confusing classes only.

    Example: class_ids=[2,4] trains pothole-vs-patch discrimination only on
    pixels whose GT is pothole or patch.
    """
    class_ids = [int(c) for c in list(class_ids)]
    if len(class_ids) <= 1:
        return logits_all.sum() * 0.0
    valid = labels != int(ignore_index)
    mask = valid & torch.zeros_like(valid, dtype=torch.bool)
    for cid in class_ids:
        mask = mask | ((labels == int(cid)) & valid)
    if mask.sum() == 0:
        return logits_all.sum() * 0.0
    cls_logits = logits_all[:, class_ids, :, :]  # (B,K,H,W)
    mapping = torch.full((int(logits_all.shape[1]),), -1, device=labels.device, dtype=torch.long)
    for i, cid in enumerate(class_ids):
        mapping[int(cid)] = int(i)
    sub_labels = mapping[labels.clamp(min=0, max=int(logits_all.shape[1]) - 1)]
    return F.cross_entropy(cls_logits, sub_labels, ignore_index=-1, reduction='none')[mask].mean()


def logits_with_bg(logits_fg: torch.Tensor) -> torch.Tensor:
    """Prepend a background logit channel (zeros)."""
    if logits_fg.dim() != 4:
        raise ValueError(f"Expected logits_fg as (B,C,H,W), got {tuple(logits_fg.shape)}")
    b, _, h, w = logits_fg.shape
    bg = torch.zeros((b, 1, h, w), dtype=logits_fg.dtype, device=logits_fg.device)
    return torch.cat([bg, logits_fg], dim=1)


def dynamic_ce_weights(
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    w_min: float = 0.2,
    w_max: float = 10.0,
    bg_scale: float = 0.3,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """Compute inverse-frequency class weights from a label map.

    Args:
        labels: (B,H,W) long
        num_classes: total classes incl. background
        ignore_index: pixels with this label are ignored
        w_min/w_max: clamp range
        bg_scale: multiply background weight by this factor (down-weight bg)

    Returns:
        weight tensor (num_classes,) on the same device, or None if labels empty.
    """
    if labels.dim() != 3:
        raise ValueError(f"Expected labels as (B,H,W), got {tuple(labels.shape)}")

    valid = labels != int(ignore_index)
    if valid.sum() == 0:
        return None

    flat = labels[valid].view(-1)
    flat = torch.clamp(flat, 0, num_classes - 1)

    counts = torch.bincount(flat, minlength=num_classes).float()
    total = counts.sum().clamp(min=eps)
    freq = counts / total

    inv = 1.0 / (freq + eps)
    inv = inv / inv.mean().clamp(min=eps)
    inv = torch.clamp(inv, min=float(w_min), max=float(w_max))

    # down-weight background
    inv[0] = inv[0] * float(bg_scale)
    inv = inv / inv.mean().clamp(min=eps)

    return inv


def soft_dice_per_class(
    prob: torch.Tensor,
    target_onehot: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute soft Dice for each class.

    Args:
        prob: (B,C,H,W) probabilities
        target_onehot: (B,C,H,W) one-hot targets
        valid_mask: (B,1,H,W) float mask {0,1} for valid pixels

    Returns:
        dice: (C,) per-class dice
    """
    if prob.shape != target_onehot.shape:
        raise ValueError(f"prob and target_onehot shape mismatch: {prob.shape} vs {target_onehot.shape}")

    prob = prob * valid_mask
    tgt = target_onehot * valid_mask

    inter = (prob * tgt).sum(dim=(0, 2, 3))
    denom = prob.sum(dim=(0, 2, 3)) + tgt.sum(dim=(0, 2, 3))

    dice = (2.0 * inter + eps) / (denom + eps)
    dice = torch.where(denom < eps, torch.ones_like(dice), dice)
    return dice


def multiclass_ce_dice_loss(
    logits_fg: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 255,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    use_dynamic_ce_weights: bool = True,
    ce_w_min: float = 0.2,
    ce_w_max: float = 10.0,
    ce_bg_scale: float = 0.3,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """CE + soft dice (foreground macro) for 7-logit foreground model."""
    if logits_fg.dim() != 4:
        raise ValueError(f"Expected logits_fg as (B,7,H,W), got {tuple(logits_fg.shape)}")
    if labels.dim() != 3:
        raise ValueError(f"Expected labels as (B,H,W), got {tuple(labels.shape)}")

    logits_all = logits_with_bg(logits_fg)  # (B,8,H,W)
    num_classes = logits_all.shape[1]

    # Safety guard: avoid CUDA "t >= 0 && t < n_classes" asserts
    # If some pixels contain unexpected label values (e.g., noisy masks), we
    # remap them to ignore_index.
    labels = labels.long()
    invalid = (labels != int(ignore_index)) & ((labels < 0) | (labels >= int(num_classes)))
    # Avoid CUDA assert by remapping invalid labels to ignore_index.
    # Use torch.where to avoid GPU->CPU sync (no .item() / bool checks).
    labels = torch.where(invalid, torch.full_like(labels, int(ignore_index)), labels)
    invalid_cnt = invalid.sum().detach()

    weight = None
    if use_dynamic_ce_weights:
        weight = dynamic_ce_weights(
            labels=labels,
            num_classes=num_classes,
            ignore_index=ignore_index,
            w_min=ce_w_min,
            w_max=ce_w_max,
            bg_scale=ce_bg_scale,
            eps=eps,
        )

    ce = F.cross_entropy(logits_all, labels, weight=weight, ignore_index=int(ignore_index))

    # dice on probabilities
    prob = torch.softmax(logits_all, dim=1)
    valid = (labels != int(ignore_index)).unsqueeze(1).float()

    labels_clamped = labels.clone()
    labels_clamped[labels_clamped == int(ignore_index)] = 0
    onehot = F.one_hot(labels_clamped.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    dice_all = soft_dice_per_class(prob, onehot, valid, eps=eps)
    # exclude background (idx=0)
    dice_fg = dice_all[1:]
    dice_loss = 1.0 - dice_fg.mean()

    total = float(ce_weight) * ce + float(dice_weight) * dice_loss
    parts = {
        'loss_ce': ce.detach(),
        'loss_dice': dice_loss.detach(),
    }
    parts['label_invalid_cnt'] = invalid_cnt
    if weight is not None:
        parts['ce_w_bg'] = weight[0].detach()
        # log mean fg weight for debug
        parts['ce_w_fg_mean'] = weight[1:].mean().detach()

    return total, parts


def multiclass_total_loss(
    logits_fg: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 255,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    use_dynamic_ce_weights: bool = True,
    ce_w_min: float = 0.2,
    ce_w_max: float = 10.0,
    ce_bg_scale: float = 0.3,
    # OHEM (pixel hard mining) for CE
    ohem_ratio: float = 0.0,
    # Dice options
    dice_present_only: bool = False,
    dice_class_weights=None,
    # Crack auxiliary loss (thin structure)
    crack_ft_lambda: float = 0.0,
    crack_class_id: int = 1,
    crack_alpha: float = 0.3,
    crack_beta: float = 0.7,
    crack_gamma: float = 0.75,
    # Crack boundary auxiliary loss
    crack_bnd_lambda: float = 0.0,
    crack_bnd_kernel: int = 3,
    # Generic per-class aux maps {class_id: lambda}
    aux_ft_map=None,
    aux_bnd_map=None,
    aux_cldice_map=None,
    # Crack centerline / topology auxiliary
    crack_cldice_lambda: float = 0.0,
    crack_cldice_iters: int = 3,
    line_group_ce_lambda: float = 0.0,
    surface_group_ce_lambda: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Unified training loss for 7-logit foreground model.

    Loss = CE(bg+fg) + Dice(fg macro) + optional Crack auxiliary losses.

    Notes
    -----
    - Crack aux uses **softmax probability** of the crack class, which is
      better aligned with the multi-class competition than a standalone sigmoid.
    - Boundary aux uses a differentiable morphological gradient.
    """

    if logits_fg.dim() != 4:
        raise ValueError(f"Expected logits_fg as (B,7,H,W), got {tuple(logits_fg.shape)}")
    if labels.dim() != 3:
        raise ValueError(f"Expected labels as (B,H,W), got {tuple(labels.shape)}")

    logits_all = logits_with_bg(logits_fg)  # (B,8,H,W)
    num_classes = logits_all.shape[1]

    # Safety guard for invalid labels
    labels = labels.long()
    invalid = (labels != int(ignore_index)) & ((labels < 0) | (labels >= int(num_classes)))
    labels = torch.where(invalid, torch.full_like(labels, int(ignore_index)), labels)
    invalid_cnt = invalid.sum().detach()

    weight = None
    if use_dynamic_ce_weights:
        weight = dynamic_ce_weights(
            labels=labels,
            num_classes=num_classes,
            ignore_index=ignore_index,
            w_min=ce_w_min,
            w_max=ce_w_max,
            bg_scale=ce_bg_scale,
            eps=eps,
        )

    # -----------------
    # CE (optionally OHEM)
    # -----------------
    if float(ohem_ratio) > 0.0:
        ce_map = F.cross_entropy(
            logits_all,
            labels,
            weight=weight,
            ignore_index=int(ignore_index),
            reduction='none',
        )  # (B,H,W)
        valid = labels != int(ignore_index)
        ce_valid = ce_map[valid]
        if ce_valid.numel() == 0:
            ce = ce_map.mean() * 0.0
        else:
            k = int(max(1, round(float(ohem_ratio) * float(ce_valid.numel()))))
            topk, _ = torch.topk(ce_valid, k=k, largest=True, sorted=False)
            ce = topk.mean()
    else:
        ce = F.cross_entropy(logits_all, labels, weight=weight, ignore_index=int(ignore_index))

    # -----------------
    # Dice (foreground macro / present-only)
    # -----------------
    prob_all = torch.softmax(logits_all, dim=1)
    valid_m = (labels != int(ignore_index)).unsqueeze(1).float()

    labels_clamped = labels.clone()
    labels_clamped[labels_clamped == int(ignore_index)] = 0
    onehot = F.one_hot(labels_clamped.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    dice_all = soft_dice_per_class(prob_all, onehot, valid_m, eps=eps)
    dice_fg = dice_all[1:]
    present_fg = ((onehot[:, 1:] * valid_m).sum(dim=(0, 2, 3)) > 0)

    if dice_class_weights is None:
        dice_w = torch.ones_like(dice_fg)
    elif torch.is_tensor(dice_class_weights):
        dcw = dice_class_weights.to(device=logits_fg.device, dtype=dice_fg.dtype).flatten()
        if dcw.numel() == num_classes:
            dice_w = dcw[1:]
        elif dcw.numel() == (num_classes - 1):
            dice_w = dcw
        else:
            raise ValueError(f"dice_class_weights tensor must have length {num_classes} or {num_classes-1}, got {dcw.numel()}")
    elif isinstance(dice_class_weights, dict):
        dice_w = torch.ones_like(dice_fg)
        for cid, wv in dice_class_weights.items():
            cid_i = int(cid)
            if 1 <= cid_i < num_classes:
                dice_w[cid_i - 1] = float(wv)
    else:
        tmp = torch.as_tensor(dice_class_weights, device=logits_fg.device, dtype=dice_fg.dtype).flatten()
        if tmp.numel() == num_classes:
            dice_w = tmp[1:]
        elif tmp.numel() == (num_classes - 1):
            dice_w = tmp
        else:
            raise ValueError(f"dice_class_weights must be dict/tensor/list with length {num_classes} or {num_classes-1}")

    if bool(dice_present_only):
        use_mask = present_fg
    else:
        use_mask = torch.ones_like(present_fg, dtype=torch.bool)

    if torch.any(use_mask):
        dice_mean = (dice_fg[use_mask] * dice_w[use_mask]).sum() / dice_w[use_mask].sum().clamp(min=eps)
    else:
        dice_mean = torch.ones([], device=logits_fg.device, dtype=prob_all.dtype)
    dice_loss = 1.0 - dice_mean

    total = float(ce_weight) * ce + float(dice_weight) * dice_loss
    parts: Dict[str, torch.Tensor] = {
        'loss_ce': ce.detach(),
        'loss_dice': dice_loss.detach(),
        'dice_present_count': present_fg.float().sum().detach(),
        'label_invalid_cnt': invalid_cnt,
    }
    if weight is not None:
        parts['ce_w_bg'] = weight[0].detach()
        parts['ce_w_fg_mean'] = weight[1:].mean().detach()

    # -----------------
    # Auxiliary per-class losses (Focal-Tversky + Boundary Dice)
    # -----------------
    ft_map = {}
    if float(crack_ft_lambda) > 0.0:
        ft_map[int(crack_class_id)] = ft_map.get(int(crack_class_id), 0.0) + float(crack_ft_lambda)
    if aux_ft_map:
        for cid, val in dict(aux_ft_map).items():
            ft_map[int(cid)] = ft_map.get(int(cid), 0.0) + float(val)

    bnd_map = {}
    if float(crack_bnd_lambda) > 0.0:
        bnd_map[int(crack_class_id)] = bnd_map.get(int(crack_class_id), 0.0) + float(crack_bnd_lambda)
    if aux_bnd_map:
        for cid, val in dict(aux_bnd_map).items():
            bnd_map[int(cid)] = bnd_map.get(int(cid), 0.0) + float(val)

    for cid, lam in sorted(ft_map.items()):
        if lam <= 0.0:
            continue
        if cid < 0 or cid >= num_classes:
            continue
        cls_prob = prob_all[:, cid:cid + 1]
        cls_gt = (labels == cid).unsqueeze(1).float()
        cls_loss = focal_tversky_from_probs(
            cls_prob,
            cls_gt,
            alpha=float(crack_alpha),
            beta=float(crack_beta),
            gamma=float(crack_gamma),
            eps=eps,
        )
        total = total + float(lam) * cls_loss
        parts[f'loss_aux_ft_c{cid}'] = cls_loss.detach()
        if cid == int(crack_class_id):
            parts['loss_crack_ft'] = cls_loss.detach()

    for cid, lam in sorted(bnd_map.items()):
        if lam <= 0.0:
            continue
        if cid < 0 or cid >= num_classes:
            continue
        cls_prob = prob_all[:, cid:cid + 1]
        cls_gt = (labels == cid).unsqueeze(1).float()
        bnd_p = _soft_morph_gradient(cls_prob, k=int(crack_bnd_kernel))
        bnd_t = _soft_morph_gradient(cls_gt, k=int(crack_bnd_kernel))
        bnd_dice = _dice_from_probs(bnd_p, bnd_t, eps=eps)
        bnd_loss = 1.0 - bnd_dice
        total = total + float(lam) * bnd_loss
        parts[f'loss_aux_bnd_c{cid}'] = bnd_loss.detach()
        parts[f'aux_bnd_dice_c{cid}'] = bnd_dice.detach()
        if cid == int(crack_class_id):
            parts['loss_crack_bnd'] = bnd_loss.detach()
            parts['crack_bnd_dice'] = bnd_dice.detach()

    if float(crack_cldice_lambda) > 0.0 and 0 <= int(crack_class_id) < num_classes:
        cls_prob = prob_all[:, int(crack_class_id):int(crack_class_id) + 1]
        cls_gt = (labels == int(crack_class_id)).unsqueeze(1).float()
        crack_cl = cldice_from_probs(cls_prob, cls_gt, iters=int(crack_cldice_iters), eps=eps)
        crack_cl_loss = 1.0 - crack_cl
        total = total + float(crack_cldice_lambda) * crack_cl_loss
        parts['loss_crack_cldice'] = crack_cl_loss.detach()
        parts['crack_cldice'] = crack_cl.detach()

    if aux_cldice_map:
        for cid, lam in sorted(dict(aux_cldice_map).items()):
            cid_i = int(cid)
            lam = float(lam)
            if lam <= 0.0 or cid_i < 1 or cid_i >= num_classes or cid_i == int(crack_class_id):
                continue
            cls_prob = prob_all[:, cid_i:cid_i + 1]
            cls_gt = (labels == cid_i).unsqueeze(1).float()
            cls_cl = cldice_from_probs(cls_prob, cls_gt, iters=int(crack_cldice_iters), eps=eps)
            cls_cl_loss = 1.0 - cls_cl
            total = total + lam * cls_cl_loss
            parts[f'loss_aux_cldice_c{cid_i}'] = cls_cl_loss.detach()
            parts[f'aux_cldice_c{cid_i}'] = cls_cl.detach()

    if float(line_group_ce_lambda) > 0.0:
        line_ce = grouped_subset_ce_loss(logits_all, labels, class_ids=[1, 5, 6], ignore_index=ignore_index)
        total = total + float(line_group_ce_lambda) * line_ce
        parts['loss_line_group_ce'] = line_ce.detach()

    if float(surface_group_ce_lambda) > 0.0:
        surf_ce = grouped_subset_ce_loss(logits_all, labels, class_ids=[2, 4], ignore_index=ignore_index)
        total = total + float(surface_group_ce_lambda) * surf_ce
        parts['loss_surface_group_ce'] = surf_ce.detach()

    return total, parts


@torch.no_grad()
def foreground_mdice_hard(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_fg: int = 7,
    ignore_index: int = 255,
    eps: float = 1e-6,
    present_only: bool = True,
) -> torch.Tensor:
    """Mean Dice over foreground classes using hard labels.

    Args:
        pred: (B,H,W) long in {0..num_fg}
        gt:   (B,H,W) long in {0..num_fg} or ignore_index
        present_only: if True, average only over classes that appear in GT.

    Returns:
        scalar tensor
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have same shape, got {pred.shape} vs {gt.shape}")

    valid = gt != int(ignore_index)
    mdices = []
    for cid in range(1, num_fg + 1):
        gt_c = (gt == cid) & valid
        if present_only and gt_c.sum() == 0:
            continue
        pred_c = (pred == cid) & valid
        inter = (pred_c & gt_c).sum().float()
        denom = pred_c.sum().float() + gt_c.sum().float()
        if denom < eps:
            d = torch.tensor(1.0, device=pred.device)
        else:
            d = (2.0 * inter + eps) / (denom + eps)
        mdices.append(d)

    if len(mdices) == 0:
        return torch.tensor(1.0, device=pred.device)
    return torch.stack(mdices).mean()
