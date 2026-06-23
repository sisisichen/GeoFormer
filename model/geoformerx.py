import os
import math
from typing import Tuple

import torch
import torch.nn as nn

from segment_anything.modeling import Sam
from segment_anything.modeling.common import MLPBlock
from data.datainfo import *


def _safe_spatial_std(x: torch.Tensor, dim, keepdim: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """Stable std-like statistic with well-defined gradients at zero variance.

    torch.std can backpropagate NaN when variance is exactly zero on some inputs.
    This implementation uses sqrt(mean((x-mu)^2) + eps) instead.
    """
    mu = x.mean(dim=dim, keepdim=True)
    var = (x - mu).pow(2).mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(var + float(eps))



class Fusion2D3D(nn.Module):
    """Trainable 2D-3D fusion.

    Modes
    -----
    - global: scalar alpha per image
    - hybrid: scalar alpha * spatial alpha-map, which is better for local defects
      such as potholes / manholes that only occupy part of the tile.
    - fixed: no learned reliability gate; depth is injected with alpha=1.0.
      This mode is used only for the w/o Geometry Gate ablation.

    Notes
    -----
    The previous hybrid initialization used bias=-2 for both gates and multiplied
    them directly, which makes the initial effective alpha about 0.12 * 0.12 ~= 0.014.
    That nearly disables the 3D branch at the beginning of training. Here we use a
    milder initialization so the RGBD branch can participate earlier.
    """

    def __init__(self, hidden: int = 16, mode: str = "global", global_init_bias: float = -0.5, spatial_init_bias: float = 0.0):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in {"global", "hybrid", "fixed"}:
            raise ValueError(f"Unsupported fusion mode: {mode}")

        self.depth_proj = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        if self.mode == "fixed":
            self.global_gate = None
        else:
            self.global_gate = nn.Sequential(
                nn.Linear(8, int(hidden)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden), 1),
                nn.Sigmoid(),
            )
            with torch.no_grad():
                if isinstance(self.global_gate[-2], nn.Linear):
                    self.global_gate[-2].bias.fill_(float(global_init_bias))

        if self.mode == "hybrid":
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(5, int(hidden), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(int(hidden)),
                nn.GELU(),
                nn.Conv2d(int(hidden), int(hidden), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(int(hidden)),
                nn.GELU(),
                nn.Conv2d(int(hidden), 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            with torch.no_grad():
                if isinstance(self.spatial_gate[-2], nn.Conv2d):
                    self.spatial_gate[-2].bias.fill_(float(spatial_init_bias))
        else:
            self.spatial_gate = None

    @staticmethod
    def _stats(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=(2, 3))
        sigma = _safe_spatial_std(x, dim=(2, 3), keepdim=False)
        return torch.cat([mu, sigma], dim=1)

    @staticmethod
    def _depth_edge(d: torch.Tensor) -> torch.Tensor:
        # local residual acts like a cheap edge/relief cue without hard-coded kernels
        return d - torch.nn.functional.avg_pool2d(d, kernel_size=3, stride=1, padding=1)

    def forward(self, rgbd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgbd.ndim != 4:
            raise ValueError(f"Fusion2D3D expects (B,C,H,W), got {tuple(rgbd.shape)}")
        if rgbd.shape[1] != 4:
            raise ValueError(f"Fusion2D3D expects 4 channels (RGBD), got C={rgbd.shape[1]}")

        rgb = rgbd[:, :3]
        d = rgbd[:, 3:4]
        if self.mode == "fixed":
            alpha_g = torch.ones((rgb.shape[0], 1, 1, 1), device=rgb.device, dtype=rgb.dtype)
        else:
            rgb_stats = self._stats(rgb)
            d_stats = self._stats(d)
            stats = torch.cat([rgb_stats, d_stats], dim=1)
            alpha_g = self.global_gate(stats).view(-1, 1, 1, 1)

        if self.mode == "hybrid":
            d_edge = self._depth_edge(d)
            alpha_s = self.spatial_gate(torch.cat([rgb, d, d_edge], dim=1))
            alpha = alpha_g * alpha_s
        else:
            alpha = alpha_g.expand(-1, 1, rgb.shape[-2], rgb.shape[-1])

        fused = rgb + alpha * self.depth_proj(d)
        return fused, alpha


class LogitRefiner(nn.Module):
    """Lightweight RGBD-aware residual refiner on top of SAM logits.

    Motivation
    ----------
    SAM decoder logits are already strong on dominant classes, but small/rare
    defects can benefit from a shallow local correction head that sees the raw
    RGBD tile together with the current logits.
    """

    def __init__(self, num_logits: int, in_img_channels: int = 4, hidden: int = 32):
        super().__init__()
        self.in_img_channels = int(in_img_channels)
        self.net = nn.Sequential(
            nn.Conv2d(int(num_logits) + int(in_img_channels), int(hidden), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(hidden)),
            nn.GELU(),
            nn.Conv2d(int(hidden), int(hidden), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(hidden)),
            nn.GELU(),
            nn.Conv2d(int(hidden), int(num_logits), kernel_size=1, bias=True),
        )
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, logits: torch.Tensor, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if img.shape[-2:] != logits.shape[-2:]:
            img = torch.nn.functional.interpolate(img, size=logits.shape[-2:], mode='bilinear', align_corners=False)
        if img.shape[1] != self.in_img_channels:
            if img.shape[1] == 1 and self.in_img_channels == 4:
                img = img.repeat(1, 4, 1, 1)
            elif img.shape[1] == 3 and self.in_img_channels == 4:
                pad = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3]), device=img.device, dtype=img.dtype)
                img = torch.cat([img, pad], dim=1)
            else:
                raise ValueError(f"LogitRefiner expects {self.in_img_channels} image channels, got {img.shape[1]}")
        delta = self.net(torch.cat([logits, img], dim=1))
        return logits + delta, delta


def _align_img_channels(img: torch.Tensor, in_img_channels: int = 4) -> torch.Tensor:
    if img.shape[1] == in_img_channels:
        return img
    if img.shape[1] == 1 and int(in_img_channels) == 4:
        return img.repeat(1, 4, 1, 1)
    if img.shape[1] == 3 and int(in_img_channels) == 4:
        pad = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3]), device=img.device, dtype=img.dtype)
        return torch.cat([img, pad], dim=1)
    raise ValueError(f"Expected {in_img_channels} image channels, got {img.shape[1]}")


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, padding=1, dilation=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LineRefinerHead(nn.Module):
    """Refine thin linear classes (Crack / Marking / Joint) with strip convolutions.

    The main decoder is shared across all classes. Thin classes benefit from
    orientation-aware residual filters that are cheap but biased toward long,
    sparse structures.
    """

    def __init__(self, in_img_channels: int = 4, hidden: int = 48, out_ch: int = 3, scale_init: float = 0.10):
        super().__init__()
        self.in_img_channels = int(in_img_channels)
        in_ch = int(in_img_channels) + int(out_ch) + 2  # image + selected logits + rgb/depth edges
        h = int(hidden)
        self.stem = ConvBNAct(in_ch, h, kernel_size=3, padding=1)
        self.b3 = ConvBNAct(h, h, kernel_size=3, padding=1)
        self.bh = ConvBNAct(h, h, kernel_size=(1, 9), padding=(0, 4))
        self.bv = ConvBNAct(h, h, kernel_size=(9, 1), padding=(4, 0))
        self.bd = ConvBNAct(h, h, kernel_size=3, padding=2, dilation=2)
        self.mix = nn.Sequential(
            ConvBNAct(h * 4, h, kernel_size=1, padding=0),
            nn.Conv2d(h, int(out_ch), kernel_size=1, bias=True),
        )
        self.scale = nn.Parameter(torch.full((1, int(out_ch), 1, 1), float(scale_init)))
        with torch.no_grad():
            nn.init.zeros_(self.mix[-1].weight)
            nn.init.zeros_(self.mix[-1].bias)

    @staticmethod
    def _edge_maps(img: torch.Tensor) -> torch.Tensor:
        rgb = img[:, :3].mean(dim=1, keepdim=True)
        dep = img[:, 3:4] if img.shape[1] >= 4 else rgb
        rgb_e = rgb - torch.nn.functional.avg_pool2d(rgb, kernel_size=3, stride=1, padding=1)
        dep_e = dep - torch.nn.functional.avg_pool2d(dep, kernel_size=3, stride=1, padding=1)
        return torch.cat([rgb_e, dep_e], dim=1)

    def forward(self, logits_sel: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-2:] != logits_sel.shape[-2:]:
            img = torch.nn.functional.interpolate(img, size=logits_sel.shape[-2:], mode='bilinear', align_corners=False)
        img = _align_img_channels(img, self.in_img_channels)
        x = torch.cat([img, logits_sel, self._edge_maps(img)], dim=1)
        x = self.stem(x)
        x = torch.cat([self.b3(x), self.bh(x), self.bv(x), self.bd(x)], dim=1)
        delta = self.mix(x)
        return delta * self.scale


class SurfaceRefinerHead(nn.Module):
    """Refine pothole / patch with depth-aware local relief cues."""

    def __init__(self, in_img_channels: int = 4, hidden: int = 32, out_ch: int = 2, scale_init: float = 0.10):
        super().__init__()
        self.in_img_channels = int(in_img_channels)
        in_ch = int(in_img_channels) + int(out_ch) + 2
        h = int(hidden)
        self.stem = ConvBNAct(in_ch, h, kernel_size=3, padding=1)
        self.d1 = ConvBNAct(h, h, kernel_size=3, padding=1, dilation=1)
        self.d2 = ConvBNAct(h, h, kernel_size=3, padding=2, dilation=2)
        self.d3 = ConvBNAct(h, h, kernel_size=3, padding=3, dilation=3)
        self.mix = nn.Sequential(
            ConvBNAct(h * 3, h, kernel_size=1, padding=0),
            nn.Conv2d(h, int(out_ch), kernel_size=1, bias=True),
        )
        self.scale = nn.Parameter(torch.full((1, int(out_ch), 1, 1), float(scale_init)))
        with torch.no_grad():
            nn.init.zeros_(self.mix[-1].weight)
            nn.init.zeros_(self.mix[-1].bias)

    @staticmethod
    def _relief_maps(img: torch.Tensor) -> torch.Tensor:
        dep = img[:, 3:4] if img.shape[1] >= 4 else img[:, :1]
        smooth = torch.nn.functional.avg_pool2d(dep, kernel_size=5, stride=1, padding=2)
        edge = dep - torch.nn.functional.avg_pool2d(dep, kernel_size=3, stride=1, padding=1)
        return torch.cat([dep - smooth, edge], dim=1)

    def forward(self, logits_sel: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-2:] != logits_sel.shape[-2:]:
            img = torch.nn.functional.interpolate(img, size=logits_sel.shape[-2:], mode='bilinear', align_corners=False)
        img = _align_img_channels(img, self.in_img_channels)
        x = torch.cat([img, logits_sel, self._relief_maps(img)], dim=1)
        x = self.stem(x)
        x = torch.cat([self.d1(x), self.d2(x), self.d3(x)], dim=1)
        delta = self.mix(x)
        return delta * self.scale


class SpecialistRefiner(nn.Module):
    """Class-group-specific residual refinement.

    - line head: crack / marking / joint
    - surface head: pothole / patch
    """

    def __init__(
        self,
        num_logits: int,
        in_img_channels: int = 4,
        line_hidden: int = 48,
        surface_hidden: int = 32,
        scale_init: float = 0.10,
        line_indices: tuple[int, ...] = (0, 4, 5),
        surface_indices: tuple[int, ...] = (1, 3),
    ):
        super().__init__()
        self.num_logits = int(num_logits)
        self.line_indices = tuple(int(x) for x in line_indices)
        self.surface_indices = tuple(int(x) for x in surface_indices)
        self.line_head = LineRefinerHead(in_img_channels=int(in_img_channels), hidden=int(line_hidden), out_ch=len(self.line_indices), scale_init=float(scale_init))
        self.surface_head = SurfaceRefinerHead(in_img_channels=int(in_img_channels), hidden=int(surface_hidden), out_ch=len(self.surface_indices), scale_init=float(scale_init))

    def forward(self, logits: torch.Tensor, img: torch.Tensor) -> tuple[torch.Tensor, dict]:
        out = logits
        aux = {}
        if len(self.line_indices) > 0:
            idx = torch.as_tensor(self.line_indices, device=logits.device, dtype=torch.long)
            delta_line = self.line_head(torch.index_select(logits, 1, idx), img)
            out = out.clone()
            out[:, idx] = out[:, idx] + delta_line
            aux['line_delta'] = delta_line
        if len(self.surface_indices) > 0:
            idx = torch.as_tensor(self.surface_indices, device=logits.device, dtype=torch.long)
            delta_surface = self.surface_head(torch.index_select(logits, 1, idx), img)
            if 'line_delta' not in aux:
                out = out.clone()
            out[:, idx] = out[:, idx] + delta_surface
            aux['surface_delta'] = delta_surface
        return out, aux


def get_index(map_idx):
    idx = map_idx
    for i in range(list(map_idx.items())[-1][0]):
        if i not in idx:
            idx[i] = 0
    idx = dict(sorted(idx.items(), key=lambda x: x[0]))
    return torch.LongTensor(list(idx.values()))


class MoEAdaptMLPBlock(nn.Module):
    """MoE Adapter injected into each ViT MLP block.

    Routing is computed per-sample (B,E) and applied to all tokens in that image.

    Gate input (concat, dim = 4*embedding_dim):
      - task_embed     : route_embed[4]   (L4 task leaf)
      - dataset_embed  : route_embed[5]   (dataset prefix id)
      - modal_embed    : modal_embed      (L0 / modality id)
      - style_embed    : learned from token statistics in bottleneck space
    """

    def __init__(
        self,
        mlp: MLPBlock,
        embedding_dim: int = 16,
        bottleneck_dim: int = 16,
        expert_num: int = 4,
        gate_topk: int = 2,
        gate_temperature: float = 1.0,
        gate_noise: float = 0.0,
        style_bn: bool = True,
        style_dropout: float = 0.10,
        style_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.embedding_dim = int(embedding_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.expert_num = int(expert_num)

        self.gate_topk = int(gate_topk) if gate_topk is not None else 0
        self.gate_temperature = float(gate_temperature) if gate_temperature is not None else 1.0
        self.gate_noise = float(gate_noise) if gate_noise is not None else 0.0
        self.style_bn = bool(style_bn)
        self.style_dropout = float(style_dropout) if style_dropout is not None else 0.0
        # Weaken "style-based expert" by scaling down style embedding before the gate.
        # (User request: weaken style routing, not completely remove.)
        self.style_scale = float(style_scale) if style_scale is not None else 1.0

        # token-space down projection (C -> bottleneck)
        self.adapter_down = nn.Sequential(
            nn.Linear(768, bottleneck_dim),
            nn.GELU(),
        )

        # experts in bottleneck space
        self.adapter_up = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(bottleneck_dim, bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(bottleneck_dim, 768),
                )
                for _ in range(expert_num)
            ]
        )

        # Gate input dim is fixed: 4 * embedding_dim (task, dataset, modal, style)
        self.adapter_gate = nn.Linear(embedding_dim * 4, expert_num, bias=True)

        # Style embed from bottleneck statistics: [mean || std] -> embedding_dim
        self.style_proj = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, embedding_dim),
            nn.Tanh(),
        )

        self.dropout = nn.Dropout(p=self.style_dropout) if self.style_dropout > 0 else nn.Identity()

        # init small so gate doesn't dominate early, but MUST be non-zero on task/dataset/modal dims
        # Otherwise, those embeddings receive zero gradient and routing collapses to style-only.
        nn.init.zeros_(self.adapter_gate.weight)
        nn.init.zeros_(self.adapter_gate.bias)
        with torch.no_grad():
            E = int(self.embedding_dim)
            # gate input order: [task | dataset | modal | style]
            # give tiny random weights to the first 3 segments; keep style segment at 0 initially
            self.adapter_gate.weight[:, :3*E].normal_(mean=0.0, std=0.02)
            self.adapter_gate.weight[:, :3*E].mul_(0.1)
            self.adapter_gate.weight[:, 3*E:].zero_()


    @staticmethod
    def _safe_zero_like(ref: torch.Tensor, d: int) -> torch.Tensor:
        return torch.zeros((ref.shape[0], d), device=ref.device, dtype=ref.dtype)

    def _compute_style_embed(self, x_bn: torch.Tensor) -> torch.Tensor:
        """x_bn: (B,H,W,bottleneck) -> (B, embedding_dim)"""
        mu = x_bn.mean(dim=(1, 2))
        if self.style_bn:
            sigma = _safe_spatial_std(x_bn, dim=(1, 2), keepdim=False)
        else:
            sigma = (x_bn - mu[:, None, None, :]).abs().mean(dim=(1, 2))
        feat = torch.cat([mu, sigma], dim=-1)
        return self.style_proj(feat)

    def _apply_topk(self, probs: torch.Tensor) -> torch.Tensor:
        k = self.gate_topk
        if k is None or k <= 0 or k >= probs.shape[1]:
            return probs
        _, topk_idx = torch.topk(probs, k=k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(dim=-1, index=topk_idx, value=1.0)
        probs = probs * mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
        return probs

    def forward(self, x: torch.Tensor, modal: torch.Tensor, route: tuple, force_expert: torch.Tensor = None):
        # ---- 1) Original MLP ----
        x_res = self.mlp(x)

        # ---- 2) Down proj ----
        x_bn = self.adapter_down(x)  # (B,H,W,bottleneck)

        # ---- 3) Gate input ----
        task_embed = route[4] if len(route) > 4 else self._safe_zero_like(modal, self.embedding_dim)
        dataset_embed = route[5] if len(route) > 5 else self._safe_zero_like(modal, self.embedding_dim)

        style_embed = self._compute_style_embed(x_bn)
        style_embed = self.dropout(style_embed)
        style_embed = style_embed * self.style_scale

        gate_in = torch.cat([task_embed, dataset_embed, modal, style_embed], dim=-1)
        temp = max(1e-6, self.gate_temperature)
        logits = self.adapter_gate(gate_in / temp)
        if self.gate_noise > 0:
            logits = logits + torch.randn_like(logits) * self.gate_noise

        probs = torch.softmax(logits, dim=-1)
        probs = self._apply_topk(probs)


        # ---- 3b) Optional: force expert routing (few-shot) ----
        # force_expert: LongTensor [B], values in [0..expert_num-1] or -1 for "no forcing"
        if force_expert is not None:
            try:
                fe = force_expert
                if not torch.is_tensor(fe):
                    fe = torch.as_tensor(fe, device=probs.device)
                fe = fe.to(device=probs.device, dtype=torch.long).view(-1)
                if fe.numel() == probs.shape[0]:
                    mask = fe >= 0
                    if mask.any():
                        forced = torch.zeros_like(probs[mask])
                        forced.scatter_(1, fe[mask].unsqueeze(1).clamp(0, self.expert_num - 1), 1.0)
                        probs = probs.clone()
                        probs[mask] = forced
            except Exception:
                # keep original probs on any failure
                pass
        # ---- 4) Mixture ----
        out = 0.0
        for i in range(self.expert_num):
            out = out + probs[:, i].view(-1, 1, 1, 1) * self.adapter_up[i](x_bn)

        x_out = x_res + out
        return x_out, probs



class GeoFormerX(nn.Module):
    """Applies Tree MoE Adapter to SAM's image encoder.

    Args:
        sam: segment anything model, see 'segment_anything' dir
        bottleneck_dim: bottleneck dimension of adapter
        embedding_dim: modal and route embedding dimension
        expert_num: number of experts in MoE adapter
        pos: which layer to apply adapter
    """

    def __init__(
        self,
        sam: Sam,
        bottleneck_dim: int,
        embedding_dim: int,
        expert_num: int,
        pos: list = None,
        # routing knobs
        gate_topk: int = 2,
        gate_temperature: float = 1.0,
        gate_noise_std: float = 0.0,
        style_bn: bool = True,
        style_dropout: float = 0.10,
        style_scale: float = 0.25,
        # 2D+3D fusion
        use_fusion_2d3d: bool = True,
        fusion_hidden: int = 16,
        fusion_mode: str = "global",
        # local residual refinement head
        use_logit_refiner: bool = False,
        refiner_hidden: int = 32,
        # class-group specialist refinement
        use_specialist_refiner: bool = False,
        line_refiner_hidden: int = 48,
        surface_refiner_hidden: int = 32,
        specialist_scale_init: float = 0.10,
        # optional partial unfreeze to use more capacity / VRAM
        unfreeze_encoder_neck: bool = False,
        unfreeze_last_n_blocks: int = 0,
    ):
        super().__init__()

        assert bottleneck_dim > 0
        assert embedding_dim > 0
        assert expert_num > 0

        # assign Adapter layer position (all layers by default)
        if pos:
            self.pos = pos
        else:
            self.pos = list(range(len(sam.image_encoder.blocks)))

        # freeze SAM image and prompt encoder
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False

        # Optional partial unfreeze for the upper encoder. This is the cleanest way
        # to spend extra VRAM on actual model capacity instead of just inflating the
        # batch size, which can reduce the number of optimizer steps per epoch.
        self.unfreeze_encoder_neck = bool(unfreeze_encoder_neck)
        self.unfreeze_last_n_blocks = int(max(0, unfreeze_last_n_blocks))
        total_blocks = len(sam.image_encoder.blocks)
        if self.unfreeze_last_n_blocks > total_blocks:
            self.unfreeze_last_n_blocks = total_blocks
        self.unfreeze_block_ids = list(range(total_blocks - self.unfreeze_last_n_blocks, total_blocks)) if self.unfreeze_last_n_blocks > 0 else []

        if self.unfreeze_encoder_neck and hasattr(sam.image_encoder, 'neck'):
            for param in sam.image_encoder.neck.parameters():
                param.requires_grad = True
        for bid in self.unfreeze_block_ids:
            for param in sam.image_encoder.blocks[bid].parameters():
                param.requires_grad = True

        # modality and route embedding index
        modal_index = get_index(modal_map_idx)
        route_index_1 = get_index(route_level_1_map_idx)
        route_index_2 = get_index(route_level_2_map_idx)
        route_index_3 = get_index(route_level_3_map_idx)

        sam.image_encoder.register_buffer('modal_index', modal_index, False)
        sam.image_encoder.register_buffer('route_index_1', route_index_1, False)
        sam.image_encoder.register_buffer('route_index_2', route_index_2, False)
        sam.image_encoder.register_buffer('route_index_3', route_index_3, False)

        modal_embed = nn.Embedding(len(modal_map_idx), embedding_dim)
        route_embed_0 = nn.Embedding(1, embedding_dim)
        route_embed_1 = nn.Embedding(len(route_level_1_map_idx), embedding_dim)
        route_embed_2 = nn.Embedding(len(route_level_2_map_idx), embedding_dim)
        route_embed_3 = nn.Embedding(len(route_level_3_map_idx), embedding_dim)
        route_embed_4 = nn.Embedding(len(task_list)+1, embedding_dim)
        dataset_buckets = int(os.getenv("MOE_DATASET_BUCKETS", "1024"))
        route_embed_5 = nn.Embedding(dataset_buckets, embedding_dim)
        ## ---- IMPORTANT: non-zero init so gate can learn to use task/dataset/modal ----
        # Small random init keeps early routing close to uniform while allowing gradients to flow.
        nn.init.normal_(modal_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(route_embed_0.weight, mean=0.0, std=0.02)
        nn.init.normal_(route_embed_1.weight, mean=0.0, std=0.02)
        nn.init.normal_(route_embed_2.weight, mean=0.0, std=0.02)
        nn.init.normal_(route_embed_3.weight, mean=0.0, std=0.02)
        nn.init.normal_(route_embed_4.weight, mean=0.0, std=0.02)
        nn.init.normal_(route_embed_5.weight, mean=0.0, std=0.02)

        sam.image_encoder.modal_embed = modal_embed
        sam.image_encoder.route_embed = nn.ModuleList([
            route_embed_0, route_embed_1, 
            route_embed_2, route_embed_3, 
            route_embed_4, 
            route_embed_5,
        ])

        # apply Adapter to SAM image encoder
        for idx, blk in enumerate(sam.image_encoder.blocks):
            if idx not in self.pos:
                continue

            # create moe adapter layers
            blk.mlp = MoEAdaptMLPBlock(
                blk.mlp,
                embedding_dim=embedding_dim,
                bottleneck_dim=bottleneck_dim,
                expert_num=expert_num,
                gate_topk=gate_topk,
                gate_temperature=gate_temperature,
                gate_noise=gate_noise_std,
                style_bn=style_bn,
                style_dropout=style_dropout,
                style_scale=style_scale,
            )

        self.sam = sam

        # Trainable 2D+3D fusion head (kept outside frozen SAM encoders).
        self.use_fusion_2d3d = bool(use_fusion_2d3d)
        self.fusion_2d3d = Fusion2D3D(hidden=int(fusion_hidden), mode=str(fusion_mode)) if self.use_fusion_2d3d else None
        self.use_logit_refiner = bool(use_logit_refiner)
        self.logit_refiner = LogitRefiner(num_logits=int(sam.mask_decoder.num_multimask_outputs), in_img_channels=4, hidden=int(refiner_hidden)) if self.use_logit_refiner else None
        self.use_specialist_refiner = bool(use_specialist_refiner)
        self.specialist_refiner = SpecialistRefiner(
            num_logits=int(sam.mask_decoder.num_multimask_outputs),
            in_img_channels=4,
            line_hidden=int(line_refiner_hidden),
            surface_hidden=int(surface_refiner_hidden),
            scale_init=float(specialist_scale_init),
        ) if self.use_specialist_refiner else None

    def save_parameters(self) -> dict:
        r"""save both adapter and mask decoder parameters.
        """
        # We save a *subset* of GeoFormerX parameters (not full SAM):
        #   - MoE adapters + gate + style_proj
        #   - task/modal embeddings
        #   - SAM mask_decoder (trainable)
        #   - 2D3D fusion head
        state_dict = self.state_dict()

        keep: dict = {}
        for k, v in state_dict.items():
            # fusion / local refinement heads (trainable, outside SAM)
            if k.startswith('fusion_2d3d') or k.startswith('logit_refiner') or k.startswith('specialist_refiner'):
                keep[k] = v
                continue

            # everything under SAM lives under 'sam.' prefix
            if not k.startswith('sam.'):
                continue

            # keep adapters / routing / style
            if ('adapter' in k) or ('adapter_gate' in k) or ('style_proj' in k):
                keep[k] = v
                continue

            # keep modality/task embeddings
            if ('modal_embed' in k) or ('route_embed' in k):
                keep[k] = v
                continue

            # keep mask decoder
            if 'mask_decoder' in k:
                keep[k] = v
                continue

            # optional partially-unfrozen encoder pieces
            if self.unfreeze_encoder_neck and k.startswith('sam.image_encoder.neck.'):
                keep[k] = v
                continue
            if any(k.startswith(f'sam.image_encoder.blocks.{bid}.') for bid in self.unfreeze_block_ids):
                keep[k] = v
                continue

        return keep

    def load_parameters(self, state_dict) -> None:
        r"""load both adapter and mask decoder parameters.
        """
        cur = self.state_dict()
        load_dict = {}
        used, skipped_missing, skipped_shape = 0, 0, 0

        for k, v in cur.items():
            if k not in state_dict:
                skipped_missing += 1
                continue
            vv = state_dict[k]
            if vv.shape != v.shape:
                skipped_shape += 1
                continue
            load_dict[k] = vv
            used += 1

        cur.update(load_dict)
        self.load_state_dict(cur, strict=False)

        if os.getenv("MOE_VERBOSE_LOAD", "0") == "1":
            print(f"[LoadParameters] used={used}  skipped_missing={skipped_missing}  skipped_shape={skipped_shape}")

    def forward(
        self,
        data,
        return_gates: bool = False,
        return_iou: bool = False,
        return_alpha: bool = False,
        return_refine: bool = False,
        return_specialist: bool = False,
    ):
        img, box = data['img'], data['box']
        modal, route = data['modal'], data['route']
        # The raw image is also used by optional logit refiners.  When the
        # w/o Depth ablation disables 2D-3D fusion, drop the depth channel here
        # as well so that no later refiner can accidentally use geometry.
        raw_img_for_refine = img
        if img.ndim == 4 and img.shape[1] == 4 and not self.use_fusion_2d3d:
            raw_img_for_refine = img[:, :3]

        # modal and route embedding
        B = img.shape[0]

        modal_index = self.sam.image_encoder.modal_index[modal]
        modal_embed = self.sam.image_encoder.modal_embed(modal_index)

        # route indices can be (l1,l2,l3,l4) or (l1,l2,l3,l4,dataset...)
        if len(route) == 4:
            route_1, route_2, route_3, route_4 = route
            dataset_idx = None
        elif len(route) == 5:
            route_1, route_2, route_3, route_4, dataset_idx = route
        else:
            raise ValueError(f"Invalid route tuple length: {len(route)}")
        route_index_0 = torch.zeros(B, dtype=torch.long, device=img.device)
        route_embed_0 = self.sam.image_encoder.route_embed[0](route_index_0)
        route_index_1 = self.sam.image_encoder.route_index_1[route_1]
        route_embed_1 = self.sam.image_encoder.route_embed[1](route_index_1)
        route_index_2 = self.sam.image_encoder.route_index_2[route_2]
        route_embed_2 = self.sam.image_encoder.route_embed[2](route_index_2)
        route_index_3 = self.sam.image_encoder.route_index_3[route_3]
        route_embed_3 = self.sam.image_encoder.route_embed[3](route_index_3)

        route_embed_4 = self.sam.image_encoder.route_embed[4](route_4)
        if dataset_idx is None:
            route_embed_5 = torch.zeros_like(route_embed_4)
        else:
            # dataset_idx already hashed to bucket in dataset.py
            route_embed_5 = self.sam.image_encoder.route_embed[5](dataset_idx)
        route_embed = (route_embed_0, route_embed_1, route_embed_2, route_embed_3, route_embed_4, route_embed_5)

        # prompt encoder
        if len(box.shape) == 2:
            box = box[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=box,
            masks=None,
        )

        # --------------------------
        # 2D+3D fusion (optional)
        # --------------------------
        alpha = None
        if img.shape[1] == 4:
            if self.use_fusion_2d3d and (self.fusion_2d3d is not None):
                img, alpha = self.fusion_2d3d(img)
            else:
                # fallback: drop 3D channel
                img = img[:, :3]
        elif img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        elif img.shape[1] != 3:
            raise ValueError(f"GeoFormerX expects input channels in [1,3,4], got C={img.shape[1]}")

        # --------------------------
        # Important
        # SAM 的 preprocess 使用的 pixel_mean/pixel_std 是 0..255 量纲的 ImageNet 统计值。
        # 但本项目的数据加载器会把图像归一化到 0..1。
        # 如果直接把 0..1 输入喂给 SAM，会导致归一化后的张量几乎是常数（动态范围极小），
        # 从而出现 Dice/IoU 极低的“假收敛”。
        #
        # 这里在模型内部做自动兼容：
        # - 若输入是 float 且最大值 <= 2，则视为 0..1，clamp 后 *255。
        # - 若输入已经像 0..255，则保持不变。
        # 用纯 Tensor 逻辑实现，避免 forward 里触发 GPU->CPU 同步。
        # --------------------------
        if img.dtype.is_floating_point:
            b = img.shape[0]
            maxv = img.detach().flatten(1).amax(dim=1)  # (B,)
            scale_mask = (maxv <= 2.0).to(dtype=img.dtype).view(b, 1, 1, 1)
            img_01 = torch.clamp(img, 0.0, 1.0) * 255.0
            img = img_01 * scale_mask + img * (1.0 - scale_mask)

        # adapter image encoder
        input_image = self.sam.preprocess(img)  # (B,3,img_size,img_size)
        image_embedding, expert_activation = self.sam.image_encoder(
            input_image, modal_embed, route_embed, force_expert=data.get('force_expert', None)
        )  # (B, 256, 64, 64)

        # predicted masks
        mask_predictions, iou_pred = self.sam.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.sam.prompt_encoder.get_dense_pe(), # (B, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=True,
            modal=modal_embed,
            route=route_embed,
          )

        refine_delta = None
        if self.use_logit_refiner and (self.logit_refiner is not None):
            mask_predictions, refine_delta = self.logit_refiner(mask_predictions, raw_img_for_refine)

        specialist_aux = None
        if self.use_specialist_refiner and (self.specialist_refiner is not None):
            mask_predictions, specialist_aux = self.specialist_refiner(mask_predictions, raw_img_for_refine)

        if return_gates or return_iou or return_alpha or return_refine or return_specialist:
            out = {"masks": mask_predictions}
            if return_gates:
                out["gates"] = expert_activation
            if return_iou:
                out["iou_pred"] = iou_pred
            if return_alpha:
                out["fusion_alpha"] = alpha
            if return_refine:
                out["refine_delta"] = refine_delta
            if return_specialist:
                out["specialist_aux"] = specialist_aux
            return out

        return mask_predictions
