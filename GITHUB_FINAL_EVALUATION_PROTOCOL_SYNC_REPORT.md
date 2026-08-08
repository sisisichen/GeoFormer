# GitHub Final-Evaluation Protocol Synchronization Report

## Outcome

**PASS.** The public evaluation entry point, configuration, documentation, and
CPU regression tests now describe the final fixed-checkpoint paper protocol:
horizontal-flip TTA, 1:1 logit fusion after inverse flipping, Hann-weighted
overlapping-logit reconstruction, and complete-image argmax after weight
normalization.

No training, model inference, checkpoint mutation, result regeneration, data
download, or weight download was performed.

## Formal evidence

The evidence was inspected outside the public repository and was not copied
into it:

- Final worker: `.final_locked_test_v1_work/locked_test_worker.py`
  - SHA-256: `adcd4f9d3a88e99aa7a659aefecf6655971ee627b7762521b99fbd93246e51b8`
  - Matches the required final-evaluator hash prefix `adcd4f9d`.
  - Calls inference with `tta_hflip=True`, `blend="hann"`, and
    `stitch_mode="logits"`.
- Formal evaluator:
  `.baseline_decision_freeze_v2_work/inputs/_evaluate_core.py`
  - SHA-256: `35f18384e424de1f6a890532d6762b1e8a3dea86794336afed0a2c4ae9623f73`
  - Restores reflected logits along spatial width, merges them 0.5/0.5,
    accumulates logits and a two-dimensional Hann window, applies the `1e-6`
    denominator floor, normalizes, crops, and then applies argmax.

The public `_evaluate_core.py` was originally not byte-identical to the formal
evaluator because the public release removes unrelated private/legacy
packaging. Its pre-sync working-tree SHA-256 was
`ca2db85b1be50c13eae2ee109e13b88f66c151b44e1b4d295c5271ffe7a23012`.
The protocol-critical helpers and mathematical statements were already
equivalent, so the evaluator received comments only. Its post-sync working-tree
SHA-256 is `4b1bdfb7cb9db9f077160f80cce412d3cd2277e38e5772dcc01894a9dacae77b`.

## Twelve-point evaluator comparison

| Requirement | Public/formal comparison | Status |
|---|---|---|
| Horizontal flip uses spatial width only | Both flip tensor dimension 3 and transform boxes horizontally | Match |
| Restore reflected prediction before fusion | Both inverse-flip reflected logits on dimension 3 | Match |
| Strict 0.5/0.5 merge | Both use `0.5 * (original + restored)` | Match |
| Merge before argmax | Both merge foreground logits before eight-class construction and reconstruction | Match |
| Two-dimensional Hann weighting | Both use the outer product of the floored 1D Hann vector | Match |
| Accumulate weighted logits in image coordinates | Same coordinate-slice accumulation | Match |
| Accumulate Hann weights | Same `cnt_acc` coordinate-slice accumulation | Match |
| Divide by accumulated weights | Same channel-broadcast normalization | Match |
| Argmax after complete reconstruction | Same post-normalization, post-crop argmax | Match |
| Zero-denominator protection | Same `np.maximum(cnt_acc, 1e-6)` | Match |
| Padding and crop | `safe_tile_coords` and `pad_to_multiple` ASTs match; same original-size crop | Match |
| No prediction-changing softmax/threshold/postprocessing | Canonical `fg_thresh` remains disabled (`-1`); optional softmax is computed only after argmax for saved probabilities | Match |

## Canonical parameters

- Tile size: 256
- Tile stride: 128
- Tiles per `512 x 256` image: 3
- Horizontal-flip TTA: enabled
- TTA merge: `0.5_original_plus_0.5_unflipped_hflip`
- Blend: Hann
- Stitch mode: logits
- Final decision: argmax after weight-normalized complete-image reconstruction

`geoformerx_recipe.py` already fixed `--tta_hflip 1`, `--blend hann`, and
`--stitch_mode logits`, so it was inspected and intentionally left unchanged.
The YAML configuration now records the explicit TTA merge rule.

## Public interface and documentation changes

- `evaluate.py` writes `resolved_evaluation_protocol.json` on every launch,
  including dry runs. It records protocol fields, canonical status, evaluator
  and configuration hashes, checkpoint path/hash when available, launcher and
  resolved commands, UTC time, and PyTorch version.
- Existing hard-stitching and architecture-ablation experiments are retained.
  They emit `NONCANONICAL_EVALUATION_PROTOCOL` and are not marked as paper
  reproductions.
- README and model card now state the exact TTA merge, Hann normalization,
  historical-test disclosure, and latency scope.
- `docs/EVALUATION_PROTOCOL.md` provides the implementation-aligned equations,
  canonical command, setting classification, timing scope, and manifest rules.

## Latency evidence

The final worker's `106.6 ms/image` value on one NVIDIA RTX 4090 D in FP32
includes intensity/range reading and preprocessing, three overlapping tiles,
original and horizontal-flip forward passes, inverse flip, 1:1 logit fusion,
Hann reconstruction, final argmax, and the final CUDA synchronization. It
excludes ground-truth reading, metrics, and prediction-PNG saving. All 1,000
test images contribute to the mean, with no first-image warm-up exclusion. It
is software inference latency, not field end-to-end throughput.

## Changed files

- `_evaluate_core.py`
- `configs/GeoFormerX_G8_D0_S0.yaml`
- `evaluate.py`
- `README.md`
- `MODEL_CARD.md`
- `docs/EVALUATION_PROTOCOL.md`
- `tests/test_final_evaluation_protocol.py`
- `GITHUB_FINAL_EVALUATION_PROTOCOL_SYNC_REPORT.md`
- `GITHUB_PROTOCOL_CHANGED_FILES.csv`

## CPU verification

- `pytest -q tests/test_final_evaluation_protocol.py`: `6 passed`
- Canonical dry run: manifest created with `canonical=true`; resolved command
  displayed all three required flags.
- Noncanonical dry run: manifest created with `canonical=false`; explicit
  `NONCANONICAL_EVALUATION_PROTOCOL` warning printed.

The tests load neither a real model nor real data. They exercise command
resolution, asymmetric flip/unflip fusion, the production NumPy Hann helper,
weight-normalized synthetic reconstruction, source ordering of reconstruction
and argmax, manifest contents, and noncanonical warnings.

## Scientific behavior

Scientific prediction behavior was **not changed**. The public evaluator's
protocol-critical mathematics already matched the formal evaluator. Changes
are limited to comments, explicit configuration metadata, launch provenance,
warnings, documentation, and CPU-only regression coverage. No formal result
number or result artifact was modified.

## Items requiring human confirmation

None.
