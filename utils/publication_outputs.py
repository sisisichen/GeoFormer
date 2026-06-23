# -*- coding: utf-8 -*-
"""Publication-oriented metrics, plots and visual panels for GeoFormerX."""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import List, Mapping, Sequence
import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
from pavement_config import CLASS_COLORS, CLASS_NAMES
from utils.multiclass_metrics import per_class_metrics_from_cm

def ensure_dir(path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def count_parameters(model) -> dict:
    module = model.module if hasattr(model, 'module') else model
    total = int(sum(p.numel() for p in module.parameters()))
    trainable = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    ratio = float(trainable / max(total, 1))
    return {'total_params': total, 'trainable_params': trainable, 'trainable_ratio': ratio, 'trainable_param_ratio': ratio}

def nanmean(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=np.float64); arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float('nan')

def foreground_mdice_from_cm(cm: np.ndarray) -> float:
    per = per_class_metrics_from_cm(cm)
    return nanmean(per.dice[1:])

def seam_band_mask(h: int, w: int, tile: int = 256, stride: int = 256, band: int = 8) -> np.ndarray:
    band = int(max(0, band)); mask = np.zeros((int(h), int(w)), dtype=bool)
    if band <= 0: return mask
    tile, stride = int(tile), int(stride)
    if tile <= 0 or stride <= 0: return mask
    xs = list(range(0, max(1, w - tile + 1), stride)); ys = list(range(0, max(1, h - tile + 1), stride))
    if not xs or xs[-1] != w - tile: xs.append(max(0, w - tile))
    if not ys or ys[-1] != h - tile: ys.append(max(0, h - tile))
    bx, by = set(), set()
    for x in xs:
        if 0 < x < w: bx.add(int(x))
        xe = int(x + tile)
        if 0 < xe < w: bx.add(xe)
    for y in ys:
        if 0 < y < h: by.add(int(y))
        ye = int(y + tile)
        if 0 < ye < h: by.add(ye)
    for x in bx: mask[:, max(0, x-band):min(w, x+band+1)] = True
    for y in by: mask[max(0, y-band):min(h, y+band+1), :] = True
    return mask

def label_to_color(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label); out = np.zeros((*label.shape[:2], 3), dtype=np.uint8)
    for cid, rgb in enumerate(CLASS_COLORS): out[label == cid] = np.array(rgb, dtype=np.uint8)
    return out

def error_overlay(gt: np.ndarray, pred: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    gt = np.asarray(gt); pred = np.asarray(pred); valid = gt != int(ignore_index)
    gt_fg = (gt > 0) & valid; pred_fg = (pred > 0) & valid
    tp = gt_fg & pred_fg & (gt == pred)
    fp = pred_fg & (~gt_fg | (gt != pred))
    fn = gt_fg & (~pred_fg | (gt != pred))
    out = np.full((*gt.shape[:2], 3), 245, dtype=np.uint8)
    out[tp] = np.array([0,170,80], dtype=np.uint8)
    out[fp] = np.array([220,40,40], dtype=np.uint8)
    out[fn] = np.array([40,90,220], dtype=np.uint8)
    out[~valid] = np.array([210,210,210], dtype=np.uint8)
    return out

def _fit(img: Image.Image, size: tuple[int,int]) -> Image.Image:
    img = img.convert('RGB'); canvas = Image.new('RGB', size, 'white')
    img.thumbnail(size, Image.BICUBIC); canvas.paste(img, ((size[0]-img.size[0])//2, (size[1]-img.size[1])//2)); return canvas

def save_qual_panel(out_path, rgb, depth, gt, pred, title: str = '', alpha=None, gate_prob=None, ignore_index: int = 255) -> None:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    cell=(210,130); header_h=34; footer_h=34 if (alpha is not None or gate_prob is not None) else 0
    labels=['RGB','Depth','Ground truth','GeoFormerX','Error overlay']
    imgs=[Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert('RGB'), Image.fromarray(np.asarray(depth).squeeze().astype(np.uint8)).convert('L').convert('RGB'), Image.fromarray(label_to_color(gt)), Image.fromarray(label_to_color(pred)), Image.fromarray(error_overlay(gt,pred,ignore_index))]
    canvas=Image.new('RGB',(cell[0]*len(imgs), header_h+cell[1]+footer_h),'white'); draw=ImageDraw.Draw(canvas)
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',14); font_b=ImageFont.truetype('DejaVuSans-Bold.ttf',15)
    except Exception: font=font_b=None
    if title: draw.text((8,5), title, fill=(0,0,0), font=font_b)
    for i,(lab,im) in enumerate(zip(labels,imgs)):
        x=i*cell[0]; draw.text((x+6,18), lab, fill=(0,0,0), font=font)
        canvas.paste(_fit(im,cell),(x,header_h)); draw.rectangle([x,header_h,x+cell[0]-1,header_h+cell[1]-1], outline=(90,90,90), width=1)
    if footer_h:
        txt=[]
        if alpha is not None and np.isfinite(alpha): txt.append(f'fusion alpha = {float(alpha):.3f}')
        if gate_prob is not None:
            arr=np.asarray(gate_prob,dtype=np.float64)
            if arr.size: txt.append(f'top expert = E{int(np.nanargmax(arr))+1}')
        draw.text((8,header_h+cell[1]+7),' | '.join(txt), fill=(0,0,0), font=font_b)
    canvas.save(out_path)

def write_csv(path, rows: List[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None: fieldnames=list(rows[0].keys()) if rows else ['empty']
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(fieldnames)); w.writeheader(); [w.writerow(dict(r)) for r in rows]

def save_json(path, payload: Mapping) -> None:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')

def plot_training_curves(csv_path, out_path) -> None:
    if plt is None or not Path(csv_path).exists(): return
    rows=[]
    with open(csv_path,'r',encoding='utf-8') as f:
        for r in csv.DictReader(f): rows.append(r)
    if not rows: return
    epochs=[int(float(r.get('epoch',i+1))) for i,r in enumerate(rows)]
    fig=plt.figure(figsize=(7.2,4.2), dpi=160); ax1=fig.add_subplot(111)
    if 'train_total_loss' in rows[0]:
        ax1.plot(epochs,[float(r.get('train_total_loss','nan')) for r in rows], label='Train loss'); ax1.set_ylabel('Train loss')
    ax2=ax1.twinx()
    for key,label in [('val_mDice_fg','mDice_fg'),('val_CrackDice','Crack Dice'),('val_BndF1_fg','BndF1_fg')]:
        if key in rows[0]: ax2.plot(epochs,[float(r.get(key,'nan')) for r in rows], label=label)
    ax1.set_xlabel('Epoch'); ax2.set_ylabel('Validation metric')
    lines,labels=ax1.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels(); ax1.legend(lines+lines2,labels+labels2,loc='best',fontsize=8)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def plot_alpha_hist(alpha_csv, out_path) -> None:
    if plt is None or not Path(alpha_csv).exists(): return
    vals=[]
    with open(alpha_csv,'r',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try: vals.append(float(r.get('alpha','nan')))
            except Exception: pass
    vals=np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if vals.size==0: return
    fig=plt.figure(figsize=(5.2,3.6),dpi=180); ax=fig.add_subplot(111); ax.hist(vals,bins=30,edgecolor='black',linewidth=0.4)
    ax.set_xlabel('Fusion gate alpha'); ax.set_ylabel('Number of tiles'); ax.set_title('Distribution of geometry gate alpha')
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def plot_expert_heatmap(usage: np.ndarray, out_path) -> None:
    if plt is None: return
    arr=np.asarray(usage,dtype=np.float64)
    if arr.ndim!=2 or arr.size==0: return
    fig=plt.figure(figsize=(7.2,4.8),dpi=180); ax=fig.add_subplot(111); im=ax.imshow(arr,aspect='auto')
    ax.set_xlabel('Expert index'); ax.set_ylabel('Adapted block'); ax.set_xticks(np.arange(arr.shape[1])); ax.set_xticklabels([f'E{i+1}' for i in range(arr.shape[1])]); ax.set_yticks(np.arange(arr.shape[0])); ax.set_yticklabels([f'B{i+1}' for i in range(arr.shape[0])]); ax.set_title('Average expert routing probability')
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
