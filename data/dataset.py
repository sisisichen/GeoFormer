# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

# Routing taxonomy for platform/class metadata.
from .datainfo import (
    modal_dict, modal_map,
    task_idx,
    route_level_1_map, route_level_2_map, route_level_3_map,
    route_level_1_dict, route_level_2_dict, route_level_3_dict,
    TASK_FOLDER_META
)

from pavement_config import (
    IMG_EXTS as PAV_IMG_EXTS,
    MASK_EXT as PAV_MASK_EXT,
    CLASS_RGB_VALUES as PAV_CLASS_RGB_VALUES,
    IGNORE_INDEX as PAV_IGNORE_INDEX,
    DIST_THRESHOLD as PAV_DIST_THRESHOLD,
    TILE_SIZE as PAV_TILE_SIZE,
    TILE_STRIDE as PAV_TILE_STRIDE,
    CLASSID_TO_TASK,
    classid_to_task_folder,
    default_force_expert_map,
)


PAV_CLASS_RGB_VALUES_NP = np.array(PAV_CLASS_RGB_VALUES, dtype=np.float32)  # (C,3)


def _safe_norm_img(img: np.ndarray) -> np.ndarray:
    """
    统一把 img 变成 float32, 且尽量归一化到 [0,1]
    兼容:
      - HxW (灰度/深度) -> 1xHxW
      - HxWxC -> CxHxW
    """
    if img is None:
        raise ValueError("Loaded image is None")

    # dtype 处理
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        # 常见深度/工业相机 16bit
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
        # 若明显是 0-255 浮点（偶发），做一个保守归一化
        if img.max() > 1.5:
            img = img / 255.0

    # 维度处理
    if img.ndim == 2:
        # HxW -> 1xHxW
        img = img[None, :, :]
    elif img.ndim == 3:
        # HxWxC -> CxHxW
        if img.shape[2] in (1, 3, 4):
            img = img.transpose(2, 0, 1)
        else:
            # 异常通道排列：尽量不猜，直接报错更安全
            raise ValueError(f"Unexpected image shape: {img.shape} (expected HxWxC with C in 1/3/4)")
    else:
        raise ValueError(f"Unexpected image ndim: {img.ndim}, shape={img.shape}")

    return img


def _safe_load_gt(gt: np.ndarray) -> np.ndarray:
    """
    GT 统一成 (H, W) 的 int/long 语义标签
    兼容 (H,W,1)。
    """
    if gt is None:
        raise ValueError("Loaded gt is None")

    if gt.ndim == 3 and gt.shape[-1] == 1:
        gt = gt.squeeze(-1)
    if gt.ndim != 2:
        raise ValueError(f"Unexpected gt shape: {gt.shape} (expected HxW or HxWx1)")
    return gt



# =========================================================
# 2D+3D pavement dataset helpers (PNG/JPG + BMP palette mask)
# =========================================================


def map_dl2_to_dl3(filename: str) -> str:
    """DL2xxxx.ext -> DL3xxxx.ext (keeps the rest identical)."""
    if filename.startswith("DL2"):
        return "DL3" + filename[3:]
    return filename.replace("DL2", "DL3", 1)


def find_3d_path(image3d_root: str, split: str, image2d_name: str) -> str:
    """Robust 2D->3D file lookup.

    Search order:
      1) root/split/image/<DL3...>
      2) root/split/<DL3...>
      3) root/<DL3...>
      4) recursive walk as fallback
    """
    name3d = map_dl2_to_dl3(image2d_name)
    base3d, ext3d = os.path.splitext(name3d)

    cand_names = [name3d]
    for e in PAV_IMG_EXTS:
        if e != ext3d.lower():
            cand_names.append(base3d + e)

    candidates: List[str] = []
    for fn in cand_names:
        candidates.extend([
            os.path.join(image3d_root, split, "image", fn),
            os.path.join(image3d_root, split, fn),
            os.path.join(image3d_root, fn),
        ])
    for p in candidates:
        if os.path.exists(p):
            return p

    targets = set(cand_names)
    for root, _, files in os.walk(image3d_root):
        for fn in files:
            if fn in targets:
                return os.path.join(root, fn)

    raise FileNotFoundError(
        f"[3D NOT FOUND] 2D={image2d_name} -> 3D candidates={cand_names} under {image3d_root}"
    )


def rgb_to_label_nearest(mask_rgb: np.ndarray, dist_threshold: float, ignore_index: int) -> np.ndarray:
    """Nearest-color encoding for palette masks, but memory-safe.

    This implementation works on *unique RGB colors* instead of broadcasting over
    every pixel against the whole palette, which avoids very large temporary arrays
    on high-resolution road images.
    Returns uint8 labels for compact caching/storage.
    """
    flat = mask_rgb.reshape(-1, 3).astype(np.float32, copy=False)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    palette = PAV_CLASS_RGB_VALUES_NP.astype(np.float32, copy=False)
    dist = np.linalg.norm(uniq[:, None, :] - palette[None, :, :], axis=-1)
    label_u = np.argmin(dist, axis=-1).astype(np.uint8)
    min_dist = np.min(dist, axis=-1)
    if ignore_index is not None:
        label_u[min_dist > float(dist_threshold)] = np.uint8(ignore_index)
    label = label_u[inv].reshape(mask_rgb.shape[:2])
    return label


def _pad_to_multiple(
    img_hwc: np.ndarray,
    lbl_hw: Optional[np.ndarray],
    tile: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Pad bottom/right so H and W become multiples of `tile`.

    - image: reflect pad (keeps texture continuity)
    - label: edge pad (or constant background if label is None)
    """
    h, w = img_hwc.shape[:2]
    pad_h = (tile - (h % tile)) % tile
    pad_w = (tile - (w % tile)) % tile
    if pad_h == 0 and pad_w == 0:
        return img_hwc, lbl_hw

    img_pad = np.pad(img_hwc, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    if lbl_hw is None:
        return img_pad, None
    lbl_pad = np.pad(lbl_hw, ((0, pad_h), (0, pad_w)), mode="edge")
    return img_pad, lbl_pad


def _tile_grid(h: int, w: int, tile: int, stride: int) -> List[Tuple[int, int]]:
    """Return a list of (x0,y0) for tiling.

    This version is safe for stride < tile: it never produces coordinates where
    x0+tile > w or y0+tile > h. It also ensures the last tile touches the
    bottom/right borders.

    We assume `h` and `w` are already padded to be >= tile.
    """
    if h < tile or w < tile:
        raise ValueError(f"Invalid padded size h={h}, w={w} for tile={tile}")

    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))

    # ensure coverage of the border
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)

    coords: List[Tuple[int, int]] = []
    for y0 in ys:
        for x0 in xs:
            coords.append((int(x0), int(y0)))
    return coords




def _compute_full_box(tile_size: int) -> np.ndarray:
    return np.array([0, 0, tile_size, tile_size], dtype=np.float32)


def _compute_box_from_binary_mask(bin_mask: np.ndarray, jitter: int = 0, train: bool = False) -> np.ndarray:
    """Compute bbox from binary mask; empty returns full tile box."""
    y_idx, x_idx = np.where(bin_mask > 0)
    H, W = bin_mask.shape
    if len(y_idx) == 0:
        return np.array([0, 0, W, H], dtype=np.float32)
    x_min, x_max = int(np.min(x_idx)), int(np.max(x_idx))
    y_min, y_max = int(np.min(y_idx)), int(np.max(y_idx))
    if train and int(jitter) > 0:
        j = int(jitter)
        x_min = max(0, x_min - np.random.randint(0, j + 1))
        x_max = min(W, x_max + np.random.randint(0, j + 1))
        y_min = max(0, y_min - np.random.randint(0, j + 1))
        y_max = min(H, y_max + np.random.randint(0, j + 1))
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


class _LRUCache:
    """A tiny per-worker LRU cache to avoid re-reading the same image 14 times.

    Notes:
      - DataLoader workers have separate Dataset instances, so this cache is per-worker.
      - Keep it small to avoid RAM issues.
    """

    def __init__(self, max_items: int = 0):
        self.max_items = int(max_items)
        self._d: "OrderedDict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()

    def get(self, key: str):
        if self.max_items <= 0:
            return None
        if key not in self._d:
            return None
        v = self._d.pop(key)
        self._d[key] = v
        return v

    def put(self, key: str, value):
        if self.max_items <= 0:
            return
        if key in self._d:
            self._d.pop(key)
        self._d[key] = value
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)


def _compute_box_from_mask(gt_hw: np.ndarray, train: bool, jitter: int = 20) -> np.ndarray:
    """
    从 mask 计算 bbox；空 mask 返回整图 bbox
    bbox 格式: [x_min, y_min, x_max, y_max]
    """
    y_indices, x_indices = np.where(gt_hw > 0)
    H, W = gt_hw.shape

    if len(y_indices) == 0:
        return np.array([0, 0, W, H], dtype=np.float32)

    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))

    if train and int(jitter) > 0:
        # 随机扰动增强鲁棒性（对细长目标，过大 jitter 会显著恶化 box prompt）
        j = int(jitter)
        x_min = max(0, x_min - np.random.randint(0, j + 1))
        x_max = min(W, x_max + np.random.randint(0, j + 1))
        y_min = max(0, y_min - np.random.randint(0, j + 1))
        y_max = min(H, y_max + np.random.randint(0, j + 1))

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def _fallback_find_level(task_name: str, level_dict: dict, level_map: dict) -> int:
    """
    回退：如果 TASK_FOLDER_META 缺失，则按 “task 属于哪个 L1/L2/L3 的列表” 来找索引。
    找不到返回 -1。
    """
    for k, v in level_dict.items():
        if task_name in v:
            return level_map.get(k, -1)
    return -1


def resolve_task_folder_meta(task_folder_name: str):
    """
    返回:
      modal_idx, l1_idx, l2_idx, l3_idx, l4_idx
    解析优先级:
      1) TASK_FOLDER_META 精确匹配
      2) folder split: L0 + '_' + L4
      3) L1/L2/L3 用回退查表
    """
    # 1) 精确匹配
    if task_folder_name in TASK_FOLDER_META:
        l0_str, l1_str, l2_str, l3_str, l4_str = TASK_FOLDER_META[task_folder_name]
    else:
        # 2) split 解析
        parts = task_folder_name.split('_')
        if len(parts) >= 2:
            l0_str = parts[0]
            l4_str = '_'.join(parts[1:])
        else:
            # 极端兜底
            l0_str = parts[0] if parts else 'Handheld'
            l4_str = task_folder_name

        # 3) 用回退查表推断 L1/L2/L3（不保证 100% 正确，但尽量不断训练）
        l1_str = None
        l2_str = None
        l3_str = None
        # 先根据 task 找对应的 key
        for k, v in route_level_1_dict.items():
            if l4_str in v:
                l1_str = k
                break
        for k, v in route_level_2_dict.items():
            if l4_str in v:
                l2_str = k
                break
        for k, v in route_level_3_dict.items():
            if l4_str in v:
                l3_str = k
                break

        # 如果仍为空，后面再转 idx 时会回退成 -1
        l1_str = l1_str or 'Unknown'
        l2_str = l2_str or 'Unknown'
        l3_str = l3_str or 'Unknown'

    # 映射到 idx
    modal_idx = modal_map.get(modal_dict.get(l0_str, l0_str), -1)
    l4_idx = task_idx.get(l4_str, -1)

    l1_idx = route_level_1_map.get(l1_str, -1)
    l2_idx = route_level_2_map.get(l2_str, -1)
    l3_idx = route_level_3_map.get(l3_str, -1)

    # 如果 L1/L2/L3 因为 Unknown 失败，再用回退查表（按 task list 归属找）
    if l1_idx == -1:
        l1_idx = _fallback_find_level(l4_str, route_level_1_dict, route_level_1_map)
    if l2_idx == -1:
        l2_idx = _fallback_find_level(l4_str, route_level_2_dict, route_level_2_map)
    if l3_idx == -1:
        l3_idx = _fallback_find_level(l4_str, route_level_3_dict, route_level_3_map)

    return modal_idx, l1_idx, l2_idx, l3_idx, l4_idx


class PavementMultiClassTileDB(Dataset):
    """Online tiling **multi-class** dataset for the 2D+3D pavement data.

    The dataset returns each tile once with a full multi-class label map (0..7, ignore=255),
    and can pre-compute tile-level class statistics for rare-class sampling/debug.

    Output
    ------
    data: dict
        - img:   FloatTensor (4, tile, tile) in [0,1]
        - box:   FloatTensor (4,) prompt box
        - modal: int
        - route: tuple(int,int,int,int,int)  (l1,l2,l3,l4,dataset_bucket)
        - name / tile_xy / orig_size: debug info
        - focus_class: training-time centered crop target (-1 if none)
    label: LongTensor (tile, tile) values in {0..7} or IGNORE_INDEX

    Notes
    -----
    - prompt_mode='full' is recommended for fair evaluation (no GT leakage).
    - prompt_mode='gt_box' is for oracle/debug only.
    - For multi-class training the historical crack-only crop is generalized here to
      class-aware focus-crop so that rare classes such as pothole/manhole can also
      be centered during training.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        train: bool = True,
        tile_size: int = PAV_TILE_SIZE,
        tile_stride: int = PAV_TILE_STRIDE,
        modal: str = 'VehicleProfiler',
        prompt_mode: str = 'full',
        box_jitter: int = 0,
        # ---- legacy crack crop args (kept for backward compatibility) ----
        use_crack_crop: bool = False,
        crack_crop_prob: float = 0.7,
        crack_class_id: int = 1,
        # ---- generalized focus-crop args ----
        focus_crop_prob: float = 0.0,
        focus_crop_classes: Optional[Sequence[int]] = None,
        focus_crop_weights: Optional[Sequence[float]] = None,
        tile_jitter: int = 0,
        threed_root: Optional[str] = None,
        cache_size: int = 32,
        # ---- optional tile statistics for rare-class sampler / debug ----
        collect_tile_stats: bool = False,
        tile_stats_cache: bool = True,
    ) -> None:
        super().__init__()

        self.data_root = os.path.abspath(data_root)
        self.split = str(split)
        self.train = bool(train)
        self.tile_size = int(tile_size)
        self.tile_stride = int(tile_stride)
        self.modal = str(modal)
        self.prompt_mode = str(prompt_mode)
        self.box_jitter = int(box_jitter)
        self.tile_jitter = int(tile_jitter)
        self.collect_tile_stats = bool(collect_tile_stats)
        self.tile_stats_cache = bool(tile_stats_cache)

        # Backward-compatible focus crop defaults.
        self.use_crack_crop = bool(use_crack_crop)
        self.crack_crop_prob = float(crack_crop_prob)
        self.crack_class_id = int(crack_class_id)
        if focus_crop_classes is None or len(list(focus_crop_classes)) == 0:
            if self.use_crack_crop:
                focus_crop_classes = [self.crack_class_id]
            else:
                focus_crop_classes = []
        if focus_crop_prob <= 0.0 and self.use_crack_crop:
            focus_crop_prob = self.crack_crop_prob
        self.focus_crop_prob = float(max(0.0, focus_crop_prob))
        self.focus_crop_classes = [int(c) for c in list(focus_crop_classes)]
        if focus_crop_weights is None or len(list(focus_crop_weights)) == 0:
            self.focus_crop_weights = [1.0 for _ in self.focus_crop_classes]
        else:
            ws = [float(w) for w in list(focus_crop_weights)]
            if len(ws) != len(self.focus_crop_classes):
                raise ValueError('focus_crop_weights must have same length as focus_crop_classes')
            self.focus_crop_weights = ws

        self.image_dir = os.path.join(self.data_root, self.split, 'image')
        self.label_dir = os.path.join(self.data_root, self.split, 'label')
        self.threed_root = os.path.join(self.data_root, '3Ddate') if threed_root is None else os.path.abspath(threed_root)

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f'Image dir not found: {self.image_dir}')
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f'Label dir not found: {self.label_dir}')
        if not os.path.isdir(self.threed_root):
            raise FileNotFoundError(f'3D root not found: {self.threed_root}')

        # Use a single TaskFolder meta (road/asphalt etc.).
        # We reuse class 1 (Crack) folder purely for the L0-L4 indices.
        self.task_folder = classid_to_task_folder(1, modal=self.modal)
        self.meta = resolve_task_folder_meta(self.task_folder)

        # Build sample list: (image_name, x0, y0, orig_w, orig_h)
        self.samples: List[Tuple[str, int, int, int, int]] = []
        self._coords_by_image: Dict[str, List[Tuple[int, int]]] = {}
        self._orig_size_by_image: Dict[str, Tuple[int, int]] = {}

        img_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(PAV_IMG_EXTS)]
        img_files.sort()
        if len(img_files) == 0:
            raise RuntimeError(f'No images found under {self.image_dir} (exts={PAV_IMG_EXTS})')

        for fn in img_files:
            img_path = os.path.join(self.image_dir, fn)
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:
                raise RuntimeError(f'Failed to open image header: {img_path}, error={e}')

            pad_h = (self.tile_size - (h % self.tile_size)) % self.tile_size
            pad_w = (self.tile_size - (w % self.tile_size)) % self.tile_size
            Hp, Wp = h + pad_h, w + pad_w
            coords = _tile_grid(Hp, Wp, tile=self.tile_size, stride=self.tile_stride)
            self._coords_by_image[fn] = coords
            self._orig_size_by_image[fn] = (int(w), int(h))
            for (x0, y0) in coords:
                self.samples.append((fn, int(x0), int(y0), int(w), int(h)))

        # For compatibility with existing samplers/loggers.
        self.task_folders: List[str] = [self.task_folder for _ in range(len(self.samples))]

        self._cache = _LRUCache(max_items=int(cache_size))
        self.tile_presence: Optional[np.ndarray] = None  # (N, C) uint8
        self.tile_pixel_count: Optional[np.ndarray] = None  # (N, C) int32
        if self.collect_tile_stats:
            self._load_or_build_tile_stats()

    def __len__(self) -> int:
        return len(self.samples)

    def _stats_cache_path(self) -> str:
        cache_dir = os.path.join(self.data_root, '_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(
            cache_dir,
            f'pav_tile_stats_{self.split}_ts{self.tile_size}_st{self.tile_stride}.npz',
        )

    def _load_or_build_tile_stats(self) -> None:
        cache_path = self._stats_cache_path()
        if self.tile_stats_cache and os.path.exists(cache_path):
            try:
                obj = np.load(cache_path)
                presence = obj['presence'].astype(np.uint8, copy=False)
                pixel_count = obj['pixel_count'].astype(np.int32, copy=False)
                if presence.shape[0] == len(self.samples):
                    self.tile_presence = presence
                    self.tile_pixel_count = pixel_count
                    return
            except Exception:
                pass

        presence_rows: List[np.ndarray] = []
        pixel_rows: List[np.ndarray] = []
        for image_name, coords in self._coords_by_image.items():
            base = os.path.splitext(image_name)[0]
            mask_path = os.path.join(self.label_dir, base + PAV_MASK_EXT)
            mask_rgb = np.array(Image.open(mask_path).convert('RGB'), dtype=np.uint8)
            label = rgb_to_label_nearest(mask_rgb, PAV_DIST_THRESHOLD, PAV_IGNORE_INDEX).astype(np.uint8, copy=False)
            _, label_p = _pad_to_multiple(mask_rgb, label, tile=self.tile_size)
            for (x0, y0) in coords:
                tile_lbl = label_p[y0:y0 + self.tile_size, x0:x0 + self.tile_size]
                valid = tile_lbl != int(PAV_IGNORE_INDEX)
                binc = np.bincount(tile_lbl[valid].reshape(-1).astype(np.int64), minlength=len(PAV_CLASS_RGB_VALUES)).astype(np.int32)
                pres = (binc > 0).astype(np.uint8)
                presence_rows.append(pres)
                pixel_rows.append(binc)

        if len(presence_rows) != len(self.samples):
            raise RuntimeError(
                f'Tile stats length mismatch: stats={len(presence_rows)} samples={len(self.samples)}'
            )
        self.tile_presence = np.stack(presence_rows, axis=0).astype(np.uint8, copy=False)
        self.tile_pixel_count = np.stack(pixel_rows, axis=0).astype(np.int32, copy=False)
        if self.tile_stats_cache:
            np.savez_compressed(cache_path, presence=self.tile_presence, pixel_count=self.tile_pixel_count)

    def get_tile_presence_summary(self) -> Dict[str, object]:
        if self.tile_presence is None:
            self._load_or_build_tile_stats()
        assert self.tile_presence is not None
        assert self.tile_pixel_count is not None
        total_tiles = int(self.tile_presence.shape[0])
        per_class = {}
        area_denom = float(self.tile_size * self.tile_size)
        for cid in range(1, len(PAV_CLASS_RGB_VALUES)):
            present = self.tile_presence[:, cid].astype(np.int64)
            pixels = self.tile_pixel_count[:, cid].astype(np.float64)
            n_present = int(present.sum())
            mean_area = float(pixels[present > 0].mean() / area_denom) if n_present > 0 else 0.0
            per_class[int(cid)] = {
                'tile_present': n_present,
                'tile_present_ratio': float(n_present / max(1, total_tiles)),
                'mean_area_ratio_when_present': mean_area,
            }
        return {
            'num_tiles': total_tiles,
            'per_class': per_class,
        }

    def build_presence_sample_weights(
        self,
        class_weight_map: Dict[int, float],
        area_focus_map: Optional[Dict[int, float]] = None,
        area_focus_ref: float = 0.02,
        area_focus_power: float = 0.5,
        area_focus_cap: float = 3.0,
    ) -> torch.Tensor:
        if self.tile_presence is None:
            self._load_or_build_tile_stats()
        assert self.tile_presence is not None
        assert self.tile_pixel_count is not None
        w = np.ones((self.tile_presence.shape[0],), dtype=np.float32)
        for cid, boost in class_weight_map.items():
            cid_i = int(cid)
            if cid_i <= 0 or cid_i >= self.tile_presence.shape[1]:
                continue
            w += float(boost) * self.tile_presence[:, cid_i].astype(np.float32)

        if area_focus_map:
            denom = float(self.tile_size * self.tile_size)
            area_focus_ref = float(max(area_focus_ref, 1.0 / max(1.0, denom)))
            area_focus_power = float(max(area_focus_power, 0.0))
            area_focus_cap = float(max(area_focus_cap, 1.0))
            for cid, boost in dict(area_focus_map).items():
                cid_i = int(cid)
                if cid_i <= 0 or cid_i >= self.tile_presence.shape[1]:
                    continue
                present = self.tile_presence[:, cid_i].astype(np.float32)
                area = self.tile_pixel_count[:, cid_i].astype(np.float32) / denom
                thin_mul = np.ones_like(area, dtype=np.float32)
                valid = area > 0
                if np.any(valid):
                    ratio = area_focus_ref / np.maximum(area[valid], 1.0 / denom)
                    ratio = np.power(ratio.astype(np.float32), area_focus_power, dtype=np.float32)
                    ratio = np.clip(ratio, 1.0, area_focus_cap).astype(np.float32)
                    thin_mul[valid] = ratio
                w += float(boost) * present * (thin_mul - 1.0)
        return torch.as_tensor(w, dtype=torch.float32)

    def _load_triplet(self, image_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ck = f'{self.split}::{image_name}'
        cached = self._cache.get(ck)
        if cached is not None:
            return cached

        base = os.path.splitext(image_name)[0]
        img2d_path = os.path.join(self.image_dir, image_name)
        img3d_path = find_3d_path(self.threed_root, self.split, image_name)
        mask_path = os.path.join(self.label_dir, base + PAV_MASK_EXT)

        rgb = np.array(Image.open(img2d_path).convert('RGB'), dtype=np.uint8)
        d = np.array(Image.open(img3d_path).convert('L'), dtype=np.uint8)[:, :, None]

        if d.shape[0] != rgb.shape[0] or d.shape[1] != rgb.shape[1]:
            d_pil = Image.fromarray(d.squeeze(-1))
            d_pil = d_pil.resize((rgb.shape[1], rgb.shape[0]), resample=Image.NEAREST)
            d = np.array(d_pil, dtype=np.uint8)[:, :, None]

        img4_u8 = np.concatenate([rgb, d], axis=-1).astype(np.uint8, copy=False)

        mask_rgb = np.array(Image.open(mask_path).convert('RGB'), dtype=np.uint8)
        label = rgb_to_label_nearest(mask_rgb, PAV_DIST_THRESHOLD, PAV_IGNORE_INDEX).astype(np.uint8, copy=False)

        self._cache.put(ck, (img4_u8, label, rgb))
        return img4_u8, label, rgb

    def __getitem__(self, index: int):
        image_name, x0, y0, orig_w, orig_h = self.samples[index]

        img4, label_hw, _rgb_uint8 = self._load_triplet(image_name)

        # pad-to-multiple for tiling
        img4_p, label_p = _pad_to_multiple(img4, label_hw, tile=self.tile_size)

        # -------------------------------------------------
        # Train-time crop augmentation
        # -------------------------------------------------
        Hp, Wp = label_p.shape[:2]
        ts = self.tile_size
        max_x0 = max(0, int(Wp - ts))
        max_y0 = max(0, int(Hp - ts))
        focus_class = -1

        if self.train:
            # (1) class-aware centered crop for rare classes
            if self.focus_crop_classes and (np.random.rand() < float(self.focus_crop_prob)):
                available_classes: List[int] = []
                available_weights: List[float] = []
                class_pixels: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                for cid, cw in zip(self.focus_crop_classes, self.focus_crop_weights):
                    ys, xs = np.where(label_p == int(cid))
                    if len(xs) > 0:
                        available_classes.append(int(cid))
                        available_weights.append(float(max(cw, 1e-6)))
                        class_pixels[int(cid)] = (ys, xs)
                if available_classes:
                    probs = np.asarray(available_weights, dtype=np.float64)
                    probs = probs / probs.sum()
                    chosen_idx = int(np.random.choice(len(available_classes), p=probs))
                    focus_class = int(available_classes[chosen_idx])
                    ys, xs = class_pixels[focus_class]
                    k = np.random.randint(0, len(xs))
                    cx, cy = int(xs[k]), int(ys[k])
                    x0 = int(cx - ts // 2)
                    y0 = int(cy - ts // 2)

            # (2) Random jitter around the base grid
            if int(self.tile_jitter) > 0:
                j = int(self.tile_jitter)
                x0 = int(x0 + np.random.randint(-j, j + 1))
                y0 = int(y0 + np.random.randint(-j, j + 1))

        # clamp to valid region
        x0 = int(np.clip(int(x0), 0, max_x0))
        y0 = int(np.clip(int(y0), 0, max_y0))

        tile_img = img4_p[y0:y0 + ts, x0:x0 + ts, :]
        tile_lbl = label_p[y0:y0 + ts, x0:x0 + ts]

        tile_chw = tile_img.transpose(2, 0, 1).astype(np.float32, copy=False) / 255.0

        # prompt box
        if self.prompt_mode == 'gt_box':
            # union of all foreground classes (exclude ignore)
            fg = ((tile_lbl > 0) & (tile_lbl != int(PAV_IGNORE_INDEX))).astype(np.uint8)
            box = _compute_box_from_binary_mask(fg, jitter=self.box_jitter, train=self.train)
        else:
            box = _compute_full_box(ts)

        modal_idx, l1_idx, l2_idx, l3_idx, l4_idx = self.meta
        dataset_idx = 0

        data: Dict[str, object] = {
            'img': torch.from_numpy(tile_chw).float(),
            'box': torch.from_numpy(box).float(),
            'modal': int(modal_idx),
            'route': (int(l1_idx), int(l2_idx), int(l3_idx), int(l4_idx), int(dataset_idx)),
            'name': f'{self.split}/{image_name}::x{x0}y{y0}',
            'task_folder': self.task_folder,
            'tile_xy': (int(x0), int(y0)),
            'orig_size': (int(orig_h), int(orig_w)),
            'focus_class': int(focus_class),
        }

        return data, torch.from_numpy(tile_lbl.copy()).long()


class PavementFullImageDB(Dataset):
    """Full-image dataset for evaluation & visualization (keeps original size)."""

    def __init__(
        self,
        data_root: str,
        split: str = "test",
        modal: str = "VehicleProfiler",
        threed_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_root = os.path.abspath(data_root)
        self.split = str(split)
        self.modal = str(modal)

        self.image_dir = os.path.join(self.data_root, self.split, "image")
        self.label_dir = os.path.join(self.data_root, self.split, "label")
        self.threed_root = os.path.join(self.data_root, "3Ddate") if threed_root is None else os.path.abspath(threed_root)

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label dir not found: {self.label_dir}")
        if not os.path.isdir(self.threed_root):
            raise FileNotFoundError(f"3D root not found: {self.threed_root}")

        self.images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(PAV_IMG_EXTS)]
        self.images.sort()
        if len(self.images) == 0:
            raise RuntimeError(f"No images found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_name = self.images[index]
        base = os.path.splitext(image_name)[0]

        img2d_path = os.path.join(self.image_dir, image_name)
        img3d_path = find_3d_path(self.threed_root, self.split, image_name)
        mask_path = os.path.join(self.label_dir, base + PAV_MASK_EXT)

        rgb = np.array(Image.open(img2d_path).convert("RGB"), dtype=np.uint8)
        d = np.array(Image.open(img3d_path).convert("L"), dtype=np.uint8)[:, :, None]
        if d.shape[0] != rgb.shape[0] or d.shape[1] != rgb.shape[1]:
            d_pil = Image.fromarray(d.squeeze(-1))
            d_pil = d_pil.resize((rgb.shape[1], rgb.shape[0]), resample=Image.NEAREST)
            d = np.array(d_pil, dtype=np.uint8)[:, :, None]

        img4 = np.concatenate([rgb, d], axis=-1).astype(np.float32) / 255.0

        mask_rgb = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        label = rgb_to_label_nearest(mask_rgb, PAV_DIST_THRESHOLD, PAV_IGNORE_INDEX)

        return {
            "name": image_name,
            "rgb": rgb,
            "img4": img4,
            "label": label,
            "orig_size": (int(rgb.shape[0]), int(rgb.shape[1])),
        }
