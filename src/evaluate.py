"""Evaluate a trained checkpoint on the validation set."""
import argparse

import torch

from src.dataset import build_dataloaders
from src.model import build_model
from src.train import evaluate
from src.utils import get_device, load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate the MNIST classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"]["device"])

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    metrics = evaluate(model, val_loader, device)
    print(f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
