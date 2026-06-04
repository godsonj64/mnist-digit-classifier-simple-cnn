"""Dataset and dataloader construction for image-folder MNIST data."""
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size):
    """Build the preprocessing pipeline: grayscale, resize, tensor, normalize."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )


def build_dataloaders(cfg):
    """Create training and validation dataloaders from the image folders."""
    data_cfg = cfg["data"]
    tfm = build_transforms(data_cfg["image_size"])

    train_path = os.path.join(data_cfg["root"], data_cfg["train_dir"])
    val_path = os.path.join(data_cfg["root"], data_cfg["val_dir"])

    train_ds = datasets.ImageFolder(train_path, transform=tfm)
    val_ds = datasets.ImageFolder(val_path, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader
