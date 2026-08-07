# GeoFormerX Model Card

## Model summary

GeoFormerX-G8-D0-S0 is a parameter-efficient SAM ViT-B adaptation for
eight-class pavement-condition and surface-object segmentation from paired
grayscale-intensity and range-coded rasters.

The original SAM image and prompt encoders remain frozen. Trainable components
are the G8 range-contribution module, one S0 static residual adapter in each of
the 12 ViT-B blocks, the seven-logit SAM mask decoder, and a residual logit
refiner. The model has 92,073,368 parameters, of which 5,394,508 are trainable
(5.8589%).

## Intended use

- Research on automated pavement-condition and surface-object segmentation.
- Reproduction of the PaIR-Pave10K source-validation and frozen-test protocol.
- Controlled study of parameter-efficient foundation-model adaptation for
  paired intensity-range imagery.
- Generation of class-resolved maps for research and human-reviewed inspection
  workflows.

## Out-of-scope use

- Safety-critical maintenance decisions without qualified human review.
- Physical depth, height, area, or volume measurement from the range-coded input.
- Claims of cross-device, cross-region, or field generalization without new
  external validation.
- Use of manual, detector-generated, or ground-truth-derived prompts as though
  they were the reported automatic inference protocol.
- Treating the bundled synthetic example data as model-quality evidence.

## Inputs

- One grayscale-intensity raster `I` at `512 x 256` pixels. Files stored as three
  channels must satisfy `R = G = B`; the channel is replicated only at the SAM
  interface.
- One paired 8-bit range-coded raster `Q` at the same resolution. The input is
  coded data rather than calibrated metric depth.
- For supervised training, one RGB palette mask with eight legal class colors
  and the audited ignore color `#008C5A`.

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Outputs

GeoFormerX predicts seven foreground logit channels:

1. Crack
2. Pothole
3. Sealed Crack
4. Patch
5. Road Marking
6. Expansion Joint
7. Manhole Cover

A fixed zero-valued background logit is prepended before loss computation,
overlap-aware reconstruction, and the final eight-class argmax.

## Architecture and inference

- SAM ViT-B with original image-encoder and prompt-encoder parameters frozen.
- G8 descriptor: six replicated-intensity statistics and two range statistics,
  with global and spatial contribution control.
- S0 adapter in blocks 0-11: `768 -> 42 -> 42 -> 768` with GELU activations.
- No router, top-k operation, routing embedding, or expert branch in the final
  architecture.
- Deterministic full-tile prompt `[0, 0, 256, 256]`.
- Three overlapping `256 x 256` tiles per image, stride 128, followed by
  Hann-weighted logit averaging and argmax.

## Training data and protocol

PaIR-Pave10K contains 10,000 paired samples. The reported protocol uses 7,000
`train`, 1,000 `source_val`, 1,000 reserved `adaptation_pool`, and 1,000 frozen
`test` images. Architecture and checkpoint selection used only `train` and
`source_val`; `adaptation_pool` was not accessed.

The private dataset and trained checkpoints are not distributed in this
repository. The bundled miniature dataset is synthetic and intended only for
software smoke testing.

## Evaluation summary

For the fixed seed-2028 comparison on the 1,000-image test split, GeoFormerX
obtained foreground macro Dice 0.7605, mIoU 0.6910, macro F1 0.7889,
Boundary-F1 0.7154, and seam-region Dice 0.7659. Across three formal seeds, test
foreground macro Dice was 0.7633 +/- 0.0053.

Paired inference showed significant improvements over matched intensity-only
FrozenSAM and SegFormer-B2. The difference from CMX was inconclusive; therefore,
the model is described as a parameter-efficient competitive alternative, not an
unqualified state-of-the-art winner.

## Limitations

- The range-coded raster lacks a recovered physical unit, scale factor,
  invalid-value convention, and verified metrological registration.
- Results are within-dataset only; no independent device, region, road section,
  or acquisition campaign is evaluated.
- Pothole is extremely rare in PaIR-Pave10K, so rare-class estimates require
  careful interpretation.
- Tiling may affect thin structures and seam regions even though overlap-aware
  reconstruction is used.
- Annotation software, annotator qualifications, adjudication rules, and
  inter-annotator agreement are unavailable.

## License

The release code is distributed under the MIT License, except for vendored or
derived components listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
