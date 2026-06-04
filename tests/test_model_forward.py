import torch

from src.model import SmallCNN, build_mobilenet_v3_small


def test_small_cnn_forward():
    """SmallCNN should output one score per class for each image."""
    model = SmallCNN(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_mobilenet_forward():
    """MobileNetV3-Small should output one score per class for each image."""
    model = build_mobilenet_v3_small(num_classes=10, pretrained=False)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
