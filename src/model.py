"""Model definitions: a small baseline CNN and an EfficientNet-B0 adaptation."""
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """A compact convolutional network trained from scratch on grayscale digits."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_efficientnet_b0(num_classes=10, pretrained=True):
    """Build EfficientNet-B0 adapted for 1-channel input and the given class count."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    net = models.efficientnet_b0(weights=weights)

    # Adapt the first conv layer to accept single-channel grayscale images.
    old_conv = net.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    if pretrained:
        # Average the pretrained RGB weights to initialize the grayscale filter.
        with_no_grad = old_conv.weight.data.mean(dim=1, keepdim=True)
        new_conv.weight.data.copy_(with_no_grad)
    net.features[0][0] = new_conv

    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes)
    return net


def build_model(cfg):
    """Build the model selected in the configuration."""
    model_cfg = cfg["model"]
    num_classes = cfg["data"]["num_classes"]
    name = model_cfg["name"].lower()

    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    if name == "efficientnet_b0":
        return build_efficientnet_b0(
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
        )
    raise ValueError(f"Unknown model name: {model_cfg['name']}")
