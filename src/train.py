import argparse

import torch
import torch.nn as nn

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import (
    compute_metrics,
    ensure_dir,
    load_config,
    resolve_device,
    set_seed,
)


def evaluate_model(model, loader, device):
    """Run the model over a dataset and return predictions and metrics."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            all_pred.extend(preds.tolist())
            all_true.extend(labels.tolist())
    return compute_metrics(all_true, all_pred)


def train(config_path):
    """Train the model and save the best checkpoint."""
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    epochs = cfg["train"]["epochs"]
    checkpoint = cfg["train"]["checkpoint"]
    ensure_dir(checkpoint)
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        metrics = evaluate_model(model, val_loader, device)
        acc = metrics["accuracy"]

        print(f"epoch {epoch}/{epochs} loss={avg_loss:.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state": model.state_dict(), "config": cfg}, checkpoint)

    if best_acc < 0:
        torch.save({"model_state": model.state_dict(), "config": cfg}, checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Train MNIST digit classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
