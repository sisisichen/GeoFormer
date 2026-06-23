# GeoFormer Model Card

## Model Summary

GeoFormer is a SAM-adapted RGB-D pavement defect segmentation model. It predicts
seven foreground road-surface classes plus background from paired RGB and
3D/depth inputs.

## Intended Use

- Research on pavement defect segmentation and RGB-D road-surface parsing.
- Reproducibility checks for the associated GeoFormer training and evaluation
  recipe.
- Controlled experiments on geometry-gated fusion, task-aware MoE routing, and
  lightweight class-group refinement.

## Out-of-Scope Use

- Safety-critical inspection decisions without domain validation.
- Direct deployment on new sensors, lighting conditions, or road materials
  without recalibration and human review.
- Using the bundled synthetic example data as evidence of model accuracy.

## Inputs

- RGB pavement images.
- Aligned 3D/depth images using the repository's `DL2` to `DL3` filename
  convention.
- RGB palette masks for supervised training.

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the required layout and mask
palette.

## Outputs

GeoFormer outputs foreground logits for:

1. Crack
2. Pothole
3. Seal
4. Patch
5. Marking
6. Joint
7. Manhole

Background is handled through the multi-class fusion and post-processing path.

## Training Data

The release package does not include private training data. It includes a tiny
synthetic RGB-D example dataset for smoke testing only.

## Checkpoints

Large trained weights and SAM checkpoints are intentionally excluded from the
repository. Use `scripts/download_sam_checkpoint.py` to download the base SAM
checkpoint, and store trained GeoFormer weights under `runs/` or
`checkpoints/`.

## Limitations

- Performance depends on RGB-depth alignment and consistent depth normalization.
- Palette masks with unexpected colors are mapped to the ignore index when they
  exceed the configured color-distance threshold.
- Small or rare defects may require class-balanced sampling and specialist-stage
  fine-tuning for stable results.
- The bundled demo validates software behavior, not segmentation quality.

## License

GeoFormer release code is distributed under the MIT License, except for vendored
or derived third-party components listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
