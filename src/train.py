"""Training script for the MNIST digit classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

# Make sure the src package is importable when called directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import ensure_dir, get_device, get_logger, load_config, save_checkpoint, set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    train_cfg = cfg["training"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg["weight_decay"]
    name = train_cfg.get("optimizer", "adam").lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=wd
        )
    raise ValueError(f"Unknown optimizer '{name}'.")


def build_scheduler(
    cfg: dict,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
):
    train_cfg = cfg["training"]
    name = train_cfg.get("scheduler", "cosine").lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    if name == "step":
        return StepLR(
            optimizer,
            step_size=train_cfg.get("step_size", 7),
            gamma=train_cfg.get("gamma", 0.1),
        )
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler '{name}'.")


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run a single training epoch; return average loss."""
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> float:
    """Return top-1 accuracy on *loader*."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST digit classifier")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    logger = get_logger("mnist.train")
    set_seed(cfg["project"]["seed"])
    device = get_device()
    logger.info(f"Using device: {device}")

    output_dir = cfg["project"]["output_dir"]
    ensure_dir(output_dir)

    # Data
    logger.info("Building data loaders …")
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # Model
    model = build_model(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg['model']['name']}  |  trainable params: {num_params:,}")

    # Training objects
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    num_epochs = cfg["training"]["epochs"]
    scheduler = build_scheduler(cfg, optimizer, num_epochs)

    # Early stopping state
    patience = cfg["training"].get("early_stopping_patience", 5)
    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_ckpt_path = f"{output_dir}/best_model.pt"

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        # *** REQUIRED FORMAT – do not change ***
        print(f"epoch {epoch}/{num_epochs} loss={loss:.4f} val_acc={val_acc:.4f}")

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_acc, best_ckpt_path)
            logger.info(f"  ↑ New best val_acc={best_val_acc:.4f} – checkpoint saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {patience} epochs)."
                )
                break

    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}")
    logger.info(f"Best checkpoint saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()
