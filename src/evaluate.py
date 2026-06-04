import argparse

import torch

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import compute_metrics, load_config, resolve_device


def evaluate(config_path):
    """Evaluate a saved model and print accuracy and F1."""
    cfg = load_config(config_path)
    device = resolve_device(cfg["train"]["device"])

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    checkpoint = cfg["train"]["checkpoint"]
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_pred.extend(preds.tolist())
            all_true.extend(labels.tolist())

    metrics = compute_metrics(all_true, all_pred)
    print(f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MNIST digit classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
