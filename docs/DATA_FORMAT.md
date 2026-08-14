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
three-channel interface. The formal loader therefore supplies a two-channel
`[intensity, range]` tensor to GeoFormerX; the model performs intensity
replication internally and retains a four-channel compatibility path only for
older checkpoints and tools.

## Range-coded modality

The auxiliary file is an 8-bit grayscale PNG. Its values are treated as digital
numbers. Do not convert them to distance, height, or volume unless an independent
calibration is available.

Paired files should have the same `512 x 256` dimensions. The loader supports
nearest-neighbor resizing for compatibility, but matching dimensions should be
verified before a formal run.

## Label palette

<div align="center">
<table align="center">
  <thead>
    <tr><th align="center">Class ID</th><th align="center">Name</th><th align="center">RGB</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">0</td><td align="center">Background</td><td align="center"><code>255, 255, 255</code></td></tr>
    <tr><td align="center">1</td><td align="center">Crack</td><td align="center"><code>255, 0, 0</code></td></tr>
    <tr><td align="center">2</td><td align="center">Pothole</td><td align="center"><code>0, 255, 0</code></td></tr>
    <tr><td align="center">3</td><td align="center">Sealed Crack</td><td align="center"><code>140, 40, 225</code></td></tr>
    <tr><td align="center">4</td><td align="center">Patch</td><td align="center"><code>0, 190, 255</code></td></tr>
    <tr><td align="center">5</td><td align="center">Road Marking</td><td align="center"><code>0, 0, 255</code></td></tr>
    <tr><td align="center">6</td><td align="center">Expansion Joint</td><td align="center"><code>140, 70, 0</code></td></tr>
    <tr><td align="center">7</td><td align="center">Manhole Cover</td><td align="center"><code>255, 100, 50</code></td></tr>
  </tbody>
</table>
</div>

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
