"""Dataset loading and preprocessing for MNIST digit classification."""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(image_size: int, train: bool) -> transforms.Compose:
    """
    Build a transform pipeline.

    MNIST images are greyscale; we convert to RGB so both the baseline CNN
    and EfficientNet-B0 (which expects 3 channels) see the same input.
    """
    common = [
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),  # 1-ch → 3-ch
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.1307, 0.1307, 0.1307],
            std=[0.3081, 0.3081, 0.3081],
        ),
    ]
    if train:
        augment = [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
        return transforms.Compose(augment + common)
    return transforms.Compose(common)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_torchvision_datasets(
    root: str,
    image_size: int,
    val_split: float,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Download (if needed) and return (train_dataset, val_dataset, test_dataset)
    using the built-in torchvision MNIST dataset.
    """
    full_train = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=get_transforms(image_size, train=True),
    )
    # For validation we use a non-augmented view of the same underlying data
    full_train_val = datasets.MNIST(
        root=root,
        train=True,
        download=False,
        transform=get_transforms(image_size, train=False),
    )

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    # Use the same indices for both augmented (train) and clean (val) views
    indices = torch.randperm(n_total).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = Subset(full_train, train_indices)
    val_ds = Subset(full_train_val, val_indices)

    test_ds = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=get_transforms(image_size, train=False),
    )
    return train_ds, val_ds, test_ds


def build_image_folder_datasets(
    root: str,
    image_size: int,
    val_split: float,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load from a custom ImageFolder structure:
      root/train/<class_name>/*.png
      root/test/<class_name>/*.png
    """
    train_path = os.path.join(root, "train")
    test_path = os.path.join(root, "test")

    full_train = datasets.ImageFolder(
        root=train_path,
        transform=get_transforms(image_size, train=True),
    )
    full_train_val = datasets.ImageFolder(
        root=train_path,
        transform=get_transforms(image_size, train=False),
    )

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    indices = torch.randperm(n_total).tolist()
    train_ds = Subset(full_train, indices[:n_train])
    val_ds = Subset(full_train_val, indices[n_train:])

    test_ds = datasets.ImageFolder(
        root=test_path,
        transform=get_transforms(image_size, train=False),
    )
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader) from the config.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    root = data_cfg["root"]
    image_size = data_cfg["image_size"]
    val_split = data_cfg["val_split"]
    num_workers = data_cfg["num_workers"]
    batch_size = train_cfg["batch_size"]

    if data_cfg.get("use_torchvision_mnist", True):
        train_ds, val_ds, test_ds = build_torchvision_datasets(
            root, image_size, val_split
        )
    else:
        train_ds, val_ds, test_ds = build_image_folder_datasets(
            root, image_size, val_split
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
