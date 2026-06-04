import argparse
import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import ensure_dir, get_device, load_config, set_seed


def evaluate_accuracy(model, loader, device):
    """Compute validation accuracy over a data loader."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    return accuracy_score(targets, preds)


def train(config_path):
    """Train the model for the configured number of epochs."""
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    out_dir = ensure_dir(cfg["output"]["dir"])
    ckpt_path = os.path.join(out_dir, cfg["output"]["checkpoint_name"])
    total_epochs = cfg["train"]["epochs"]
    best_acc = 0.0

    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        epoch_loss = running_loss / max(n_batches, 1)
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"epoch {epoch}/{total_epochs} loss={epoch_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)

    print(f"Best val_acc={best_acc:.4f}. Checkpoint saved to {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
