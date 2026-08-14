# GeoFormerX Canonical Training Protocol

This document records the `GeoFormerX-G8-D0-S0` recipe used by the final
2026-08-14 manuscript revision. The executable source of truth is
[`geoformerx_recipe.py`](../geoformerx_recipe.py), with the same values recorded
in [`configs/GeoFormerX_G8_D0_S0.yaml`](../configs/GeoFormerX_G8_D0_S0.yaml).

## Data-use boundary

<div align="center">
<table align="center">
  <thead>
    <tr><th align="center">Subset</th><th align="center">Images</th><th align="center">Permitted use</th></tr>
  </thead>
  <tbody>
    <tr><td align="center"><code>train</code></td><td align="center">7,000</td><td align="center">Architecture screening and model fitting</td></tr>
    <tr><td align="center"><code>source_val</code></td><td align="center">1,000</td><td align="center">Architecture and checkpoint selection</td></tr>
    <tr><td align="center"><code>adaptation_pool</code></td><td align="center">1,000</td><td align="center">Reserved; not used in the reported study</td></tr>
    <tr><td align="center"><code>test</code></td><td align="center">1,000</td><td align="center">Final fixed-checkpoint comparison only</td></tr>
  </tbody>
</table>
</div>

The original 2,000-image validation split is divided with seed 2026 into the
disjoint `source_val` and `adaptation_pool` subsets. An earlier manuscript had
already reported results on an official test split, so the revised evaluation
is not described as historically unseen or fully blind.

## Canonical optimization settings

<div align="center">
<table align="center">
  <thead>
    <tr><th align="center">Item</th><th align="center">Canonical setting</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Optimizer</td><td align="center">AdamW; weight decay 0.01</td></tr>
    <tr><td align="center">Learning rates</td><td align="center">New modules 3e-4; SAM mask decoder 7.5e-5</td></tr>
    <tr><td align="center">Schedule</td><td align="center">1 warm-up + 32 hold + cosine decay; 50 epochs</td></tr>
    <tr><td align="center">Minimum learning rates</td><td align="center">New modules 1e-5; decoder 2.5e-6</td></tr>
    <tr><td align="center">Batch</td><td align="center">28 tiles/step; accumulation 1</td></tr>
    <tr><td align="center">Precision</td><td align="center">FP32; no AMP</td></tr>
    <tr><td align="center">Gradient control</td><td align="center">Global norm clipping at 1.0</td></tr>
    <tr><td align="center">EMA</td><td align="center">Decay 0.995 from epoch 0</td></tr>
    <tr><td align="center">Selection</td><td align="center">Highest source-validation foreground macro Dice; earliest exact tie</td></tr>
  </tbody>
</table>
</div>

Training uses replacement weighted sampling, rare-presence and small-crack
boosts, and a class-aware centered-crop probability of 0.65. No stochastic
flip, rotation, or color-jitter augmentation is applied. Tile-coordinate jitter
is a crop-location control and is not an image reflection or photometric
augmentation.

## Active loss definition

The model predicts seven foreground logits. A deterministic zero-valued
background channel is prepended before the eight-class loss is evaluated.

<div align="center">
<table align="center">
  <thead>
    <tr><th align="center">Term</th><th align="center">Canonical definition</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Eight-class CE</td><td align="center">Weight 1.0; batch-wise inverse-frequency weights clamped to 0.2-10.0; background scale 0.55; OHEM fraction 0.0625</td></tr>
    <tr><td align="center">Foreground soft Dice</td><td align="center">Weight 1.0; present-only; class weights 1:1.8, 2:1.5, 3:1.0, 4:1.1, 5:1.2, 6:1.3, 7:0.8</td></tr>
    <tr><td align="center">Crack-sensitive</td><td align="center">Focal-Tversky 0.12; boundary Dice 0.06; clDice 0.012</td></tr>
    <tr><td align="center">Other rare classes</td><td align="center">Focal-Tversky: Pothole 0.06, Patch 0.03, Road Marking 0.03, Expansion Joint 0.04; boundary Dice 0.02 each</td></tr>
    <tr><td align="center">Line-group CE</td><td align="center">Three-class CE among Crack (1), Road Marking (5), and Expansion Joint (6), restricted to pixels with one of those labels; weight 0.10</td></tr>
    <tr><td align="center">Surface-group CE</td><td align="center">Two-class CE between Pothole (2) and Patch (4), restricted to pixels with one of those labels; weight 0.05</td></tr>
  </tbody>
</table>
</div>

For a group with ordered class IDs \(C=(c_0,\ldots,c_{m-1})\), the implementation
selects the matching channels directly from the eight-class semantic logits and
remaps ground-truth class \(c_j\) to local target \(j\). It computes ordinary
unweighted cross-entropy only on subset-valid pixels:

\[
\mathcal{L}_{\mathrm{group}} =
\frac{1}{|\Omega_C|}\sum_{x\in\Omega_C}
-\log\frac{\exp z_{c_{j(x)}}(x)}{\sum_{c\in C}\exp z_c(x)}.
\]

Background, out-of-subset classes, and `ignore_index=255` do not contribute.
There is no independent group head, probability aggregation, OHEM, dynamic
class weighting, or background scaling in either group term. If a batch has no
subset-valid pixel, the implementation returns a differentiable zero.

The exact code is in
[`grouped_subset_ce_loss`](../utils/multiclass_loss.py) and the two terms are
added directly to the total loss with `--line_group_ce_lambda 0.10` and
`--surface_group_ce_lambda 0.05`.

## Formal seeds

<div align="center">
<table align="center">
  <thead>
    <tr><th align="center">Seed</th><th align="center">Best epoch</th><th align="center">Source-val FG mDice</th><th align="center">Test FG mDice</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">2026</td><td align="center">48</td><td align="center">0.7565</td><td align="center">0.7693</td></tr>
    <tr><td align="center">2027</td><td align="center">25</td><td align="center">0.7547</td><td align="center">0.7599</td></tr>
    <tr><td align="center">2028</td><td align="center">26</td><td align="center">0.7562</td><td align="center">0.7605</td></tr>
    <tr><td align="center">Mean +/- SD</td><td align="center">-</td><td align="center">0.7558 +/- 0.0010</td><td align="center">0.7633 +/- 0.0053</td></tr>
  </tbody>
</table>
</div>

Seed 2028 is the prespecified primary paper model; no seed was selected from
test performance.
