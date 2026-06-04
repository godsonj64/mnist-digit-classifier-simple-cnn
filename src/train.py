"""Training loop for the MNIST digit classifier."""
import argparse
import os

import torch
import torch.nn as nn

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import (
    compute_metrics,
    ensure_dir,
    get_device,
    load_config,
    set_seed,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run a single training pass and return the average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model and return accuracy and F1 metrics."""
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())
    return compute_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Train the MNIST digit classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    epochs = cfg["train"]["epochs"]
    ckpt_dir = cfg["train"]["checkpoint_dir"]
    ensure_dir(ckpt_dir)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        acc = metrics["accuracy"]
        print(f"epoch {epoch}/{epochs} loss={loss:.4f} val_acc={acc:.4f}", flush=True)

        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {"model_state": model.state_dict(), "config": cfg},
                os.path.join(ckpt_dir, "best.pt"),
            )

    torch.save(
        {"model_state": model.state_dict(), "config": cfg},
        os.path.join(ckpt_dir, "last.pt"),
    )


if __name__ == "__main__":
    main()
