import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _build_transforms(image_size):
    """Resize, convert to 3-channel tensors, and normalize images."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _has_images(folder):
    """Check whether a folder contains class subfolders with files."""
    if not os.path.isdir(folder):
        return False
    for name in os.listdir(folder):
        sub = os.path.join(folder, name)
        if os.path.isdir(sub) and any(
            os.path.isfile(os.path.join(sub, f)) for f in os.listdir(sub)
        ):
            return True
    return False


def _export_mnist_to_image_folder(root, split, train):
    """Download MNIST and save its images into class subfolders."""
    out_dir = os.path.join(root, split)
    os.makedirs(out_dir, exist_ok=True)
    raw = datasets.MNIST(root=os.path.join(root, "_raw"), train=train, download=True)
    for cls in range(10):
        os.makedirs(os.path.join(out_dir, str(cls)), exist_ok=True)
    for idx, (img, label) in enumerate(raw):
        path = os.path.join(out_dir, str(label), f"{idx:06d}.png")
        img.save(path)


def ensure_dataset(cfg):
    """Make sure train/val image folders exist, downloading MNIST if needed."""
    train_dir = cfg["data"]["train_dir"]
    val_dir = cfg["data"]["val_dir"]
    root = cfg["data"]["root"]
    auto = cfg["data"].get("auto_download", True)

    if not _has_images(train_dir):
        if not auto:
            raise FileNotFoundError(f"No training data found in {train_dir}")
        _export_mnist_to_image_folder(root, "train", train=True)
    if not _has_images(val_dir):
        if not auto:
            raise FileNotFoundError(f"No validation data found in {val_dir}")
        _export_mnist_to_image_folder(root, "val", train=False)


def build_dataloaders(cfg):
    """Create training and validation data loaders from image folders."""
    ensure_dataset(cfg)
    image_size = cfg["data"]["image_size"]
    tf = _build_transforms(image_size)

    train_ds = datasets.ImageFolder(cfg["data"]["train_dir"], transform=tf)
    val_ds = datasets.ImageFolder(cfg["data"]["val_dir"], transform=tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
