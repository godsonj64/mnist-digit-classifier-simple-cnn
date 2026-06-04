# Data layout

This project uses the **image folder** format. Organize your images so each digit
class has its own subfolder inside `train/` and `val/`.

```
data/
├── train/
│   ├── 0/
│   │   ├── img001.png
│   │   └── ...
│   ├── 1/
│   ├── 2/
│   ├── ...
│   └── 9/
└── val/
    ├── 0/
    ├── 1/
    ├── ...
    └── 9/
```

- There should be exactly **10 class folders** named `0` through `9`.
- Images are grayscale MNIST digits, automatically resized to 28x28.
- Common image formats (`.png`, `.jpg`, `.jpeg`, `.bmp`) are supported.

## Getting MNIST as image folders

If you have MNIST in another format, you can export it to this layout using
`torchvision`:

```python
from torchvision.datasets import MNIST
from pathlib import Path

for split, train_flag in [("train", True), ("val", False)]:
    ds = MNIST(root="raw", train=train_flag, download=True)
    for idx, (img, label) in enumerate(ds):
        out = Path("data") / split / str(label)
        out.mkdir(parents=True, exist_ok=True)
        img.save(out / f"{idx}.png")
```
