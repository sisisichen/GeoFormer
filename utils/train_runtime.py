import copy
import math
from typing import Any, Dict, Iterable, List

import torch


def recursive_to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: recursive_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def snapshot_model_state(model_module) -> Dict[str, Any]:
    return recursive_to_cpu(model_module.save_parameters())


def restore_model_state(model_module, state: Dict[str, Any]) -> None:
    if state is not None:
        model_module.load_parameters(state)


def get_base_lrs(optimizer: torch.optim.Optimizer) -> List[float]:
    base_lrs: List[float] = []
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = float(group["lr"])
        base_lrs.append(float(group["initial_lr"]))
    return base_lrs


def compute_epoch_lrs(
    base_lrs: List[float],
    epoch: int,
    num_epochs: int,
    schedule: str = "none",
    warmup_epochs: int = 0,
    hold_epochs: int = 0,
    min_lr: float = 1e-6,
    lr_scale: float = 1.0,
) -> List[float]:
    schedule = str(schedule).lower()
    warmup_epochs = max(0, int(warmup_epochs))
    num_epochs = max(1, int(num_epochs))
    lr_scale = float(lr_scale)
    min_lr = float(min_lr)

    if schedule == "none":
        return [max(0.0, float(b) * lr_scale) for b in base_lrs]

    if schedule not in {"cosine", "flatcosine"}:
        raise ValueError(f"Unsupported lr schedule: {schedule}")

    hold_epochs = max(0, int(hold_epochs))

    if warmup_epochs > 0 and epoch < warmup_epochs:
        warmup_factor = float(epoch + 1) / float(max(1, warmup_epochs))
        return [max(min_lr, float(b) * warmup_factor * lr_scale) for b in base_lrs]

    if schedule == "flatcosine" and epoch < (warmup_epochs + hold_epochs):
        return [max(min_lr, float(b) * lr_scale) for b in base_lrs]

    tail_epochs = max(1, num_epochs - warmup_epochs - hold_epochs)
    tail_epoch = max(0, epoch - warmup_epochs - hold_epochs)
    progress = float(tail_epoch) / float(max(1, tail_epochs - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return [max(min_lr, (min_lr + (float(b) - min_lr) * cosine) * lr_scale) for b in base_lrs]


def set_optimizer_lrs(optimizer: torch.optim.Optimizer, lrs: Iterable[float]) -> None:
    for group, lr in zip(optimizer.param_groups, lrs):
        group["lr"] = float(lr)


def gradients_finite(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if (p.grad is not None) and (not torch.isfinite(p.grad).all()):
            return False
    return True


def parameters_finite(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.requires_grad and (not torch.isfinite(p).all()):
            return False
    return True


def clip_grad_norm_and_get(model: torch.nn.Module, max_norm: float = 0.0) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    if float(max_norm) > 0:
        return torch.nn.utils.clip_grad_norm_(params, float(max_norm))

    device = params[0].grad.device
    total = torch.zeros([], device=device)
    for p in params:
        total = total + p.grad.detach().pow(2).sum()
    return total.sqrt()


def sanitize_nonfinite_gradients(model: torch.nn.Module, fill: float = 0.0):
    """Replace non-finite gradient values in-place and report what was changed.

    This is useful when a tiny number of gradient tensors become NaN/Inf while the
    rest of the model remains healthy. Instead of discarding the whole batch, we can
    zero (or fill) the offending entries and continue the optimizer step.
    """
    num_params = 0
    num_values = 0
    first_name = None
    first_max_abs = None
    fill = float(fill)
    for n, p in model.named_parameters():
        g = p.grad
        if g is None:
            continue
        bad = ~torch.isfinite(g)
        if not bool(bad.any()):
            continue
        if first_name is None:
            first_name = str(n)
            try:
                first_max_abs = float(torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
            except Exception:
                first_max_abs = None
        num_params += 1
        try:
            num_values += int(bad.sum().item())
        except Exception:
            pass
        g.data = torch.nan_to_num(g.data, nan=fill, posinf=fill, neginf=fill)
    return {
        'num_params': int(num_params),
        'num_values': int(num_values),
        'first_name': first_name,
        'first_max_abs': first_max_abs,
    }
