import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """A compact convolutional network trained from scratch."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_mobilenet_v3_small(num_classes=10, pretrained=True):
    """Build MobileNetV3-Small and replace its head for our classes."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_model(cfg):
    """Construct the model selected in the config."""
    name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]
    if name == "mobilenet_v3_small":
        return build_mobilenet_v3_small(
            num_classes=num_classes,
            pretrained=cfg["model"].get("pretrained", True),
        )
    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name}")
