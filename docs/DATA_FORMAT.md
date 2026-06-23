# Data Format

GeoFormer expects paired 2D RGB images, RGB label masks, and 3D/depth images.

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

The 2D-to-3D filename mapping replaces the `DL2` prefix with `DL3`. For example:

```text
train/image/DL20001.png
3Ddate/train/image/DL30001.png
train/label/DL20001.bmp
```

## Label Palette

| Class ID | Name | RGB |
| --- | --- | --- |
| 0 | Background | 255, 255, 255 |
| 1 | Crack | 255, 0, 0 |
| 2 | Pothole | 0, 255, 0 |
| 3 | Seal | 140, 40, 225 |
| 4 | Patch | 0, 190, 255 |
| 5 | Marking | 0, 0, 255 |
| 6 | Joint | 140, 70, 0 |
| 7 | Manhole | 255, 100, 50 |

Unknown colors farther than the configured threshold are mapped to `IGNORE_INDEX=255`.

## Practical Checks

Before long training runs, verify:

```bash
python scripts/demo.py
python train.py --data_path data/pavement_rgbd --checkpoint checkpoints/sam --dry_run
```

The demo uses synthetic data only. It is a smoke test for layout and metrics, not a trained-model quality benchmark.
