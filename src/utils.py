"""Utility helpers for configuration, reproducibility, and metrics."""
import os
import random

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score


def load_config(path):
    """Load a YAML configuration file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Make results reproducible by fixing all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred):
    """Return the requested device, falling back to CPU if CUDA is unavailable."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_metrics(y_true, y_pred):
    """Compute accuracy and macro F1 score from true and predicted labels."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": float(acc), "f1": float(f1)}


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
