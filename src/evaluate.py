"""Evaluate a trained MNIST digit classifier on the test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import get_device, get_logger, load_checkpoint, load_config, set_seed


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader, device: torch.device):
    """Collect all predictions and ground-truth labels from *loader*."""
    model.eval()
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MNIST digit classifier")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/best_model.pt",
        help="Path to model checkpoint (.pt)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("mnist.evaluate")
    set_seed(cfg["project"]["seed"])
    device = get_device()
    logger.info(f"Using device: {device}")

    # Data
    _, _, test_loader = get_dataloaders(cfg)

    # Model
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device)
    logger.info(
        f"Loaded checkpoint from '{args.checkpoint}' "
        f"(saved at epoch {ckpt.get('epoch', '?')}, "
        f"val_acc={ckpt.get('val_acc', float('nan')):.4f})"
    )

    # Inference
    preds, labels = run_inference(model, test_loader, device)

    # Metrics
    accuracy = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro")

    logger.info("=" * 50)
    logger.info(f"Test accuracy : {accuracy:.4f}  ({accuracy * 100:.2f} %)")
    logger.info(f"Macro F1 score: {macro_f1:.4f}")
    logger.info("=" * 50)

    class_names = [str(i) for i in range(cfg["model"]["num_classes"])]
    report = classification_report(labels, preds, target_names=class_names)
    logger.info("Per-class report:\n" + report)


if __name__ == "__main__":
    main()
