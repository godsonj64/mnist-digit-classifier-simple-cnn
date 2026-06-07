# Data

## Option A – Automatic download (default)

When `data.use_torchvision_mnist: true` is set in `configs/default.yaml`, `torchvision` will automatically download the MNIST dataset (~11 MB) to `data/mnist/` the first time you run training. No extra steps required.

## Option B – Custom image-folder dataset

If you have your own greyscale digit images, set `data.use_torchvision_mnist: false` and arrange them as:

```
data/mnist/
  train/
    0/   ← images of digit 0
    1/   ← images of digit 1
    …
    9/
  test/
    0/
    …
    9/
```

`torchvision.datasets.ImageFolder` is used to load this structure, so any image format supported by Pillow (PNG, JPEG, BMP, …) is accepted.

## Class mapping

| Folder name | Digit |
|-------------|-------|
| 0 | 0 |
| 1 | 1 |
| … | … |
| 9 | 9 |

## Data statistics (standard MNIST)

| Split | Images |
|-------|--------|
| Train | 60 000 |
| Test  | 10 000 |

Images are 28 × 28 greyscale; the data loader resizes them to the `image_size` specified in the config (default 32 × 32) and converts them to 3-channel RGB so both models accept the same input.
