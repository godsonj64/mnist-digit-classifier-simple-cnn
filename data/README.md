# Data

This project uses the `image_folder` format. Each class lives in its own subfolder:

```
data/
  train/
    0/ ... 9/
  val/
    0/ ... 9/
```

Images are grayscale digits but are converted to 3 channels and resized to 32x32 at load
time so they work with both the small CNN and MobileNetV3-Small.

## Generate from MNIST

The dataset module can download MNIST via torchvision and write out the image-folder
layout automatically:

```bash
python -m src.dataset --prepare --data-dir data
```

This creates `data/train` and `data/val` populated with PNG files. If you already have your
own digit images in the layout above, you can skip this step.
