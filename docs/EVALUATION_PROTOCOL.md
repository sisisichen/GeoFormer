# GeoFormerX Final Evaluation Protocol

This document defines `GeoFormerX_final_paper_evaluation_v1`, the canonical
fixed-checkpoint protocol used for the reported GeoFormerX-G8-D0-S0 test
results.

## 1. Input and tiling

The evaluator reads the paired grayscale-intensity raster and range-coded
raster, resizes a mismatched range raster to the intensity geometry with
nearest-neighbor resampling, concatenates the two channels, and scales the
8-bit values by `1/255`. Each `512 x 256` image is evaluated at three
overlapping `256 x 256` tile locations with stride 128. Every tile receives the
deterministic full-tile prompt `[0, 0, 256, 256]`.

## 2. Horizontal-flip TTA

Each tile is evaluated twice: once in its original orientation and once after a
horizontal reflection along the spatial width dimension. The reflected
prediction is flipped back to the original orientation before fusion.

## 3. Logit fusion

For tile `t`, the test-time-augmentation fusion is

\[
L_t = \frac{1}{2}\left[L_t^{\mathrm{orig}} +
\operatorname{Flip}^{-1}\left(L_t^{\mathrm{flip}}\right)\right].
\]

The arithmetic mean is applied to logits before any tile-level semantic
decision. GeoFormerX first averages its seven foreground-logit channels and
then prepends the fixed zero background channel to form eight-class logits.

## 4. Hann-weighted reconstruction

The one-dimensional Hann vector is floored at `1e-3`; its outer product forms
the two-dimensional tile window. At complete-image position `x`, the
reconstructed logit is

\[
L(x) = \frac{\sum_t w_t(x)L_t(x)}{\sum_t w_t(x)}.
\]

Both weighted logits and weights are accumulated at the tile's complete-image
coordinates. The accumulated denominator is protected with the evaluator's
`1e-6` floor before division. The reconstructed tensor is then cropped to the
original image dimensions.

## 5. Final semantic decision

The final class map is computed only after complete-image reconstruction:

\[
\hat{y}(x) = \operatorname*{argmax}_c L_c(x).
\]

The canonical prediction path does not apply a foreground threshold. Softmax
probabilities are computed only after argmax for optional probability outputs
and therefore do not affect the predicted class map.

## 6. Structured condition records

For each image, the evaluator writes the semantic map and a
`GeoFormerX.condition_record.v1` record. For foreground class `c`, the pixel
count, presence field, and relative coverage are

\[
n_{i,c}=\sum_{x\in\Omega_i}\mathbf{1}[\hat y_i(x)=c],\qquad
p_{i,c}=\mathbf{1}[n_{i,c}>0],\qquad
r_{i,c}=\frac{n_{i,c}}{|\Omega_i|}.
\]

`Omega_i` contains valid pixels only; audited `ignore_index=255` pixels are
excluded. The records are saved as `condition_records.json` and
`condition_records.csv`. Relative coverage is dimensionless and is not a
physical area or severity estimate.

## 7. Timing scope

GeoFormerX complete-image latency is `106.6 ms/image` on one NVIDIA RTX 4090 D
in FP32. The mean includes intensity/range reading and preprocessing, all three
overlapping tiles, the original and horizontally reflected forward passes for
each tile, inverse flipping, 1:1 TTA logit fusion, Hann-weighted complete-image
reconstruction, final argmax, and the CUDA synchronization immediately before
the timer stops.

The measurement excludes ground-truth reading, metric computation,
foreground-probability export, optional diagnostic collection, and
prediction-PNG saving. Canonical launches therefore use `--collect_debug 0`.
All 1,000 test images contribute to the mean; the first image was not removed
as a warm-up. This is software inference latency, not a field end-to-end
throughput measurement.

## 8. Canonical command

Run the public wrapper with the appropriate local data and checkpoint paths:

```bash
python evaluate.py \
  --data_path data/pavement_rgbd \
  --checkpoint checkpoints/sam \
  --ckpt runs/geoformerx_seed2028/model_final.pth \
  --split test \
  --out_dir runs/geoformerx_seed2028/eval_test \
  --device cuda:0
```

The resolved evaluator command must contain:

```text
--tta_hflip 1
--blend hann
--stitch_mode logits
--collect_debug 0
```

## 9. Canonical versus noncanonical settings

The canonical protocol requires horizontal-flip TTA, 1:1 logit fusion after
unflipping, Hann blending, logit stitching, and the full G8-D0-S0 architecture.
The existing `--stitch_mode hard` and architecture-ablation controls remain
available for controlled experiments. Such launches print
`NONCANONICAL_EVALUATION_PROTOCOL` and must not be labeled as canonical paper
reproductions.

## 10. Reproducibility manifest

Every `evaluate.py` launch, including a dry run, writes
`resolved_evaluation_protocol.json` to `out_dir`. It records the resolved
protocol, canonical status, evaluator and configuration SHA-256 hashes,
checkpoint path and SHA-256 when the checkpoint is available, launcher and
resolved evaluator commands, UTC date/time, and PyTorch version. This manifest
documents the launch and does not alter evaluation outputs.
