import argparse
import os

import torch
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import get_device, load_config


def evaluate(config_path):
    """Run evaluation on the validation set and print accuracy and F1."""
    cfg = load_config(config_path)
    device = get_device(cfg["train"]["device"])

    _, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    ckpt_path = os.path.join(cfg["output"]["dir"], cfg["output"]["checkpoint_name"])
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    print(f"accuracy={acc:.4f} f1={f1:.4f}")
    return {"accuracy": acc, "f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    evaluate(args.config)
