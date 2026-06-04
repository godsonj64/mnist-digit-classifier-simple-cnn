import os
import random

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score


def load_config(path):
    """Load a YAML config file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Make runs reproducible by fixing all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device):
    """Choose a compute device, auto-detecting CUDA when requested."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def compute_metrics(y_true, y_pred):
    """Return accuracy and macro-F1 for predictions."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": float(acc), "f1": float(f1)}


def ensure_dir(path):
    """Create the parent directory for a file path if it does not exist."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
