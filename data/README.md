# Data

This project uses an **image-folder** dataset layout. Each class has its own
subfolder, and images for that class live inside it.

```
data/
  train/
    0/ *.png
    1/ *.png
    ...
    9/ *.png
  val/
    0/ *.png
    ...
    9/ *.png
```

- There are 10 classes, one per digit (0-9).
- Folder names are used as class labels.
- Images can be grayscale; they are converted to the size in the config.

## Automatic download

If the folders above are missing and `data.auto_download` is `true` in
`configs/default.yaml`, the code downloads MNIST via torchvision and writes
it into this image-folder layout automatically. No manual steps required.
