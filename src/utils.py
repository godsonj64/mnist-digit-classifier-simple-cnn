import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    """Read a YAML config file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Make results reproducible by fixing all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference="auto"):
    """Choose a compute device: GPU if available, otherwise CPU."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path
