"""Smoke tests verifying that the models run a forward pass correctly."""
import torch

from src.model import build_model


def _base_cfg(name):
    return {
        "data": {"num_classes": 10, "image_size": 28},
        "model": {"name": name, "pretrained": False},
    }


def test_small_cnn_forward():
    model = build_model(_base_cfg("small_cnn"))
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)


def test_efficientnet_b0_forward():
    model = build_model(_base_cfg("efficientnet_b0"))
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)
