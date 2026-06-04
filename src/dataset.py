import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size):
    """Create the image preprocessing pipeline (resize, to 3 channels, normalize)."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_dataloaders(cfg):
    """Create training and validation data loaders from the image-folder layout."""
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    tfm = build_transforms(data_cfg["image_size"])

    train_dir = os.path.join(data_cfg["data_dir"], data_cfg["train_split"])
    val_dir = os.path.join(data_cfg["data_dir"], data_cfg["val_split"])

    train_ds = datasets.ImageFolder(train_dir, transform=tfm)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )
    return train_loader, val_loader, train_ds.classes


def prepare_mnist(data_dir):
    """Download MNIST and write it out as PNG files in the image-folder layout."""
    from PIL import Image

    os.makedirs(data_dir, exist_ok=True)
    raw_dir = os.path.join(data_dir, "_mnist_raw")

    splits = {
        "train": datasets.MNIST(raw_dir, train=True, download=True),
        "val": datasets.MNIST(raw_dir, train=False, download=True),
    }

    for split, ds in splits.items():
        print(f"Writing {split} split ({len(ds)} images)...")
        for cls in range(10):
            os.makedirs(os.path.join(data_dir, split, str(cls)), exist_ok=True)
        for idx in range(len(ds)):
            img, label = ds[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img.numpy())
            out_path = os.path.join(data_dir, split, str(label), f"{idx}.png")
            img.save(out_path)
    print("Done. Data written to", data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Download and lay out MNIST")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    if args.prepare:
        prepare_mnist(args.data_dir)
    else:
        print("Pass --prepare to download MNIST into the image-folder layout.")
