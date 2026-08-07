# GeoFormerX

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model Card](https://img.shields.io/badge/model-card-informational)](MODEL_CARD.md)
[![Citation](https://img.shields.io/badge/citation-CFF-lightgrey)](CITATION.cff)

Official PyTorch release for **GeoFormerX: Parameter-efficient Segment Anything
adaptation for automated pavement inspection with intensity-range imagery**.

GeoFormerX adapts a frozen SAM ViT-B encoder to paired grayscale-intensity and
range-coded pavement rasters. The final `GeoFormerX-G8-D0-S0` model combines
automatic full-tile prompting, G8 range-contribution gating, a single-path S0
residual adapter in every ViT block, D0 seven-foreground-logit decoding, and
overlap-aware reconstruction.

> The auxiliary raster is treated as coded input, not as metrically calibrated
> depth. Results are limited to PaIR-Pave10K; no external-domain or physical
> metrology claim is made.

## Highlights

- **Parameter-efficient SAM adaptation:** trains 5.39M of 92.07M parameters
  (5.86%) while keeping the original SAM image and prompt encoders frozen.
- **Intensity-range contribution control:** uses six replicated-intensity
  statistics plus two range statistics and an enabled spatial modulation branch.
- **Static S0 adapters:** inserts one `768 -> 42 -> 42 -> 768` residual path in
  all 12 ViT-B blocks, with no router, top-k operation, or expert branches.
- **Automatic semantic prompting:** applies the deterministic box
  `[0, 0, 256, 256]` to every tile without manual or label-derived localization.
- **One-pass eight-class output:** predicts seven foreground logits and prepends
  a fixed zero-valued background logit before loss, reconstruction, and argmax.
- **Audited evaluation:** reports a frozen 1,000-image test, paired image-level
  inference, three formal seeds, boundary metrics, and seam-region metrics.

## Architecture

<p align="center">
  <a href="assets/figures/fig1_overall_architecture.png"><img src="assets/figures/fig1_overall_architecture.png" alt="Overall GeoFormerX-G8-D0-S0 architecture" width="100%"></a><br>
  <sub><a href="assets/figures/fig1_overall_architecture.png">Open the full-resolution figure</a></sub>
</p>

| Component | Canonical setting |
| --- | --- |
| Input | Grayscale intensity `I` + range-coded raster `Q` |
| Backbone | SAM ViT-B; original image and prompt encoders frozen |
| Fusion | G8 global descriptor + spatial modulation |
| Adapter | S0 in blocks 0-11; `768 -> 42 -> 42 -> 768` |
| Decoder | Seven foreground logits + fixed zero background |
| Tiling | `256 x 256`, stride `128`, three tiles per `512 x 256` image |
| Prompt | Fixed full-tile box `[0, 0, 256, 256]` |
| Parameters | 92,073,368 total; 5,394,508 trainable |

### G8 fusion and S0 adaptation

<p align="center">
  <a href="assets/figures/fig2_g8_s0_modules.png"><img src="assets/figures/fig2_g8_s0_modules.png" alt="G8 range-contribution module and S0 static residual adapter" width="100%"></a><br>
  <sub><a href="assets/figures/fig2_g8_s0_modules.png">Open the full-resolution figure</a></sub>
</p>

The G8 gate controls the model-internal contribution of `Q`; it is not an
estimate of physical sensor reliability. The S0 adapter was selected over the
routed M0 alternative because the latter did not meet the prespecified `0.005`
gain threshold and did not show repeatable expert differentiation.

## Dataset and locked protocol

PaIR-Pave10K contains 10,000 paired samples at `512 x 256` pixels. The original
7,000-image training and 1,000-image test memberships remain unchanged. The
original 2,000-image validation set is split with seed 2026 into `source_val`
and `adaptation_pool`, each with 1,000 images.

<p align="center">
  <a href="assets/figures/fig3_data_partition.png"><img src="assets/figures/fig3_data_partition.png" alt="Audited PaIR-Pave10K data partition and locked subset use" width="100%"></a><br>
  <sub><a href="assets/figures/fig3_data_partition.png">Open the full-resolution figure</a></sub>
</p>

- `train`: architecture screening and formal model fitting.
- `source_val`: architecture and checkpoint selection.
- `adaptation_pool`: reserved and unaccessed in the reported study.
- `test`: opened read-only after architecture, checkpoints, baselines, and the
  statistical plan were frozen.

## Architecture selection

<p align="center">
  <a href="assets/figures/fig4_architecture_screening.png"><img src="assets/figures/fig4_architecture_screening.png" alt="Prespecified source-validation architecture screening" width="100%"></a><br>
  <sub><a href="assets/figures/fig4_architecture_screening.png">Open the full-resolution figure</a></sub>
</p>

| Stage | Selected | Alternatives | Source-val foreground macro Dice |
| --- | --- | --- | ---: |
| Gate | G8 | G4, G0 | 0.7524 |
| Decoder | D0 | D1 | 0.7524 |
| Adapter | S0 | M0, A0 | 0.7486 +/- 0.0019 |

<p align="center">
  <a href="assets/figures/fig5_per_class_source_val.png"><img src="assets/figures/fig5_per_class_source_val.png" alt="Two-seed per-class source-validation diagnostics" width="100%"></a><br>
  <sub><a href="assets/figures/fig5_per_class_source_val.png">Open the full-resolution figure</a></sub>
</p>

<p align="center">
  <a href="assets/figures/fig6_static_adapter_diagnostics.png"><img src="assets/figures/fig6_static_adapter_diagnostics.png" alt="Static-adapter selection diagnostics" width="100%"></a><br>
  <sub><a href="assets/figures/fig6_static_adapter_diagnostics.png">Open the full-resolution figure</a></sub>
</p>

These two diagnostic figures report source-validation behavior only; they are
not locked-test comparisons.

## Frozen-test results

The primary paper comparison uses seed 2028 and the same 1,000 frozen test IDs
for all four methods.

| Method | Input | FG mDice | mIoU | Macro F1 | Boundary-F1 | Seam Dice |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **GeoFormerX** | I + Q | **0.7605** | **0.6910** | **0.7889** | **0.7154** | **0.7659** |
| FrozenSAM matched | I only | 0.6920 | 0.6464 | 0.7289 | 0.6273 | 0.6927 |
| SegFormer-B2 | I only | 0.7104 | 0.6598 | 0.7449 | 0.6403 | 0.7354 |
| CMX | I + Q | 0.7763 | 0.7107 | 0.8027 | 0.7181 | 0.7743 |

Paired image-level inference found practically meaningful improvements over
matched intensity-only FrozenSAM (`+0.0685`, Holm-adjusted `p = 0.0003`) and
SegFormer-B2 (`+0.0502`, `p = 0.0014`). The GeoFormerX-CMX difference was
inconclusive (`-0.0157`, `p = 0.2877`), so GeoFormerX is presented as a
parameter-efficient competitive alternative rather than an unqualified winner.

Across seeds 2026, 2027, and 2028, test foreground macro Dice was
`0.7633 +/- 0.0053`.

### Qualitative comparison

<p align="center">
  <a href="assets/figures/fig7_locked_test_qualitative.png"><img src="assets/figures/fig7_locked_test_qualitative.png" alt="Locked-test qualitative comparison for deterministic sample DL2D00006080" width="100%"></a><br>
  <sub><a href="assets/figures/fig7_locked_test_qualitative.png">Open the full-resolution figure</a></sub>
</p>

The displayed sample is deterministic position 600/1000, selected by an equally
spaced ID rule rather than by performance.

## Release contents

```text
GeoFormer/
  assets/figures/          manuscript figures 1-7
  configs/                 canonical G8-D0-S0 configuration
  data/                    intensity-range dataset reader and taxonomy
  docs/DATA_FORMAT.md      expected intensity, range, and mask layout
  examples/                tiny synthetic smoke-test dataset
  model/geoformerx.py      G8 fusion, S0/M0/A0 adapters, and D0 model path
  scripts/                 demo and SAM-checkpoint utilities
  segment_anything/        minimal vendored SAM model-building subset
  utils/                   losses, metrics, sampling, and publication outputs
  train.py                 one-command canonical training wrapper
  evaluate.py              tiled locked-protocol evaluation wrapper
  MODEL_CARD.md
  CITATION.cff
```

## Installation

```bash
git clone https://github.com/sisisichen/GeoFormer.git
cd GeoFormer
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for your CUDA/CPU platform first, then the remaining packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data layout

The public loader keeps the source dataset's historical `3Ddate` directory and
`DL2 -> DL3` filename mapping, but interprets the files as grayscale intensity
and range-coded rasters:

```text
data/pavement_rgbd/
  train/
    image/DL2xxxx.png
    label/DL2xxxx.bmp
  val/
    image/DL2xxxx.png
    label/DL2xxxx.bmp
  test/
    image/DL2xxxx.png
    label/DL2xxxx.bmp
  3Ddate/
    train/image/DL3xxxx.png
    val/image/DL3xxxx.png
    test/image/DL3xxxx.png
```

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for palette rules, the audited
ignore color, and validation checks.

## Quick smoke test

The bundled example is synthetic and verifies only the data, palette, overlay,
and metric paths.

```bash
pip install numpy pillow
python scripts/demo.py
```

Expected outputs:

```text
runs/demo/summary.json
runs/demo/metrics.csv
runs/demo/predictions/*_pred.png
runs/demo/overlays/*_overlay.png
```

## Training

Download the SAM ViT-B checkpoint:

```bash
python scripts/download_sam_checkpoint.py --model_type vit_b --out_dir checkpoints/sam
```

Run the canonical seed-2028 recipe:

```bash
python train.py \
  --data_path data/pavement_rgbd \
  --checkpoint checkpoints/sam \
  --work_dir runs/geoformerx_seed2028 \
  --seed 2028 \
  --device cuda:0
```

Resolve and inspect the complete command without training:

```bash
python train.py --dry_run
```

The selected checkpoint is copied to `WORK_DIR/model_final.pth`. The canonical
settings are also recorded in
[configs/GeoFormerX_G8_D0_S0.yaml](configs/GeoFormerX_G8_D0_S0.yaml).

## Evaluation

```bash
python evaluate.py \
  --data_path data/pavement_rgbd \
  --checkpoint checkpoints/sam \
  --ckpt runs/geoformerx_seed2028/model_final.pth \
  --split test \
  --out_dir runs/geoformerx_seed2028/eval_test \
  --device cuda:0
```

The wrapper uses `256 x 256` tiles, stride 128, Hann-weighted logit averaging,
horizontal-flip TTA, and a final eight-class argmax.

## Class palette

| ID | Class | Functional group | RGB |
| ---: | --- | --- | --- |
| 0 | Background | Background | `255, 255, 255` |
| 1 | Crack | Active distress | `255, 0, 0` |
| 2 | Pothole | Active distress | `0, 255, 0` |
| 3 | Sealed Crack | Repair or treatment | `140, 40, 225` |
| 4 | Patch | Repair or treatment | `0, 190, 255` |
| 5 | Road Marking | Surface object or feature | `0, 0, 255` |
| 6 | Expansion Joint | Structural feature | `140, 70, 0` |
| 7 | Manhole Cover | Surface object or asset | `255, 100, 50` |

The audited color `#008C5A` maps to `IGNORE_INDEX = 255`. Any other undeclared
mask color raises an error.

## Reproducibility and limitations

- Large checkpoints, private data, TensorBoard logs, and formal-run archives are
  intentionally excluded from Git; store them under `checkpoints/`,
  `data/pavement_rgbd/`, or `runs/`.
- The auxiliary channel has no recovered physical unit, scale factor, or invalid
  value convention and must not be interpreted as calibrated depth.
- Road-section, route, date, campaign, and annotation-provenance metadata are not
  available, preventing verified group-based external validation.
- The fixed-resolution tiling interface is not evidence of unrestricted
  large-image or field deployment.
- The vendored SAM subset is covered by the notice in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{bao2026geoformerx,
  title  = {GeoFormerX: Parameter-efficient Segment Anything adaptation for
            automated pavement inspection with intensity-range imagery},
  author = {Bao, Longsheng and Chen, Si and Bao, Yuyang and Li, Baoxian and
            Zhao, Jiakang and Yu, Ling},
  year   = {2026},
  note   = {Manuscript and official PyTorch implementation}
}
```
