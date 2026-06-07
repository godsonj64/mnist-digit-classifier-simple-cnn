"""Model definitions: baseline CNN and EfficientNet-B0 transfer-learning variant."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


# ---------------------------------------------------------------------------
# Baseline: small CNN trained from scratch
# ---------------------------------------------------------------------------

class BaselineCNN(nn.Module):
    """
    A lightweight 3-block convolutional network (~90 k parameters).

    Architecture:
        Conv(3→32) → BN → ReLU → MaxPool
        Conv(32→64) → BN → ReLU → MaxPool
        Conv(64→128) → BN → ReLU → AdaptiveAvgPool
        Flatten → Dropout → Linear(128 → num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32 → 16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16 → 8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # → 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Transfer-learning variant: EfficientNet-B0
# ---------------------------------------------------------------------------

class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 pre-trained on ImageNet with its head replaced for
    MNIST's 10 classes.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        # Replace the default 1000-class head
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """
    Instantiate the correct model based on ``cfg['model']['name']``.

    Supported values
    ----------------
    ``baseline``        – BaselineCNN (default)
    ``efficientnet_b0`` – EfficientNet-B0 with optional ImageNet weights
    """
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "baseline").lower()
    num_classes = model_cfg.get("num_classes", 10)

    if name == "baseline":
        return BaselineCNN(num_classes=num_classes)

    if name == "efficientnet_b0":
        return EfficientNetB0Classifier(
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )

    raise ValueError(
        f"Unknown model name '{name}'. Choose 'baseline' or 'efficientnet_b0'."
    )
