import torch

from src.model import build_model


def _cfg(name):
    return {"model": {"name": name, "num_classes": 10, "pretrained": False}}


def test_small_cnn_forward():
    model = build_model(_cfg("small_cnn"))
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_mobilenet_forward():
    model = build_model(_cfg("mobilenet_v3_small"))
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
