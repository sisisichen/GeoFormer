# Data Format

GeoFormerX expects paired grayscale-intensity images, range-coded rasters, and
RGB semantic masks. The source dataset's historical `3Ddate` directory name is
retained for compatibility; it does not imply calibrated metric depth.

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

The filename mapping replaces the `DL2` prefix with `DL3`:

```text
train/image/DL20001.png
3Ddate/train/image/DL30001.png
train/label/DL20001.bmp
```

## Intensity modality

The intensity input contains one independent information channel. Single-channel
files are loaded directly. Three-channel files must have identical `R`, `G`, and
`B` values; the grayscale channel is replicated only when forming SAM's
three-channel interface.

## Range-coded modality

The auxiliary file is an 8-bit grayscale PNG. Its values are treated as digital
numbers. Do not convert them to distance, height, or volume unless an independent
calibration is available.

Paired files should have the same `512 x 256` dimensions. The loader supports
nearest-neighbor resizing for compatibility, but matching dimensions should be
verified before a formal run.

## Label palette

| Class ID | Name | RGB |
| ---: | --- | --- |
| 0 | Background | `255, 255, 255` |
| 1 | Crack | `255, 0, 0` |
| 2 | Pothole | `0, 255, 0` |
| 3 | Sealed Crack | `140, 40, 225` |
| 4 | Patch | `0, 190, 255` |
| 5 | Road Marking | `0, 0, 255` |
| 6 | Expansion Joint | `140, 70, 0` |
| 7 | Manhole Cover | `255, 100, 50` |

The audited extra color `#008C5A` maps to `IGNORE_INDEX = 255`. Any other
undeclared color raises an error; nearest-color guessing is not used.

## Tiling and prompting

- Source image: `512 x 256`.
- Tile size: `256 x 256`.
- Stride: 128 pixels.
- Tiles per source image: 3.
- Prompt: fixed box `[0, 0, 256, 256]` for every tile.
- Reconstruction: Hann-weighted logit averaging, then eight-class argmax.

## Practical checks

Before a long run:

```bash
python scripts/demo.py
python train.py --data_path data/pavement_rgbd --checkpoint checkpoints/sam --dry_run
python evaluate.py --dry_run
```

The demo uses synthetic data only. It validates the software path, not the
reported segmentation accuracy.
