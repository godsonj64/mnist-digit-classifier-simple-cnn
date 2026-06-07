"""Unit tests for model forward passes and dataset utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import BaselineCNN, EfficientNetB0Classifier, build_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 4
C, H, W = 3, 32, 32
NUM_CLASSES = 10


@pytest.fixture()
def dummy_input() -> torch.Tensor:
    """A random batch of 3-channel 32×32 images."""
    return torch.randn(BATCH, C, H, W)


@pytest.fixture()
def baseline_model() -> BaselineCNN:
    return BaselineCNN(num_classes=NUM_CLASSES)


@pytest.fixture()
def efficientnet_model() -> EfficientNetB0Classifier:
    # pretrained=False so tests run offline without downloading weights
    return EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=False)


# ---------------------------------------------------------------------------
# Baseline CNN
# ---------------------------------------------------------------------------

class TestBaselineCNN:
    def test_output_shape(self, baseline_model, dummy_input):
        """Output should be (batch, num_classes)."""
        out = baseline_model(dummy_input)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_output_dtype(self, baseline_model, dummy_input):
        """Output should be float32 logits."""
        out = baseline_model(dummy_input)
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, baseline_model, dummy_input):
        out = baseline_model(dummy_input)
        assert not torch.isnan(out).any()

    def test_trainable_parameters(self, baseline_model):
        num_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
        # Baseline should be lightweight
        assert num_params < 1_000_000, f"Too many params: {num_params:,}"

    def test_eval_mode_deterministic(self, baseline_model, dummy_input):
        baseline_model.eval()
        with torch.no_grad():
            out1 = baseline_model(dummy_input)
            out2 = baseline_model(dummy_input)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# EfficientNet-B0
# ---------------------------------------------------------------------------

class TestEfficientNetB0:
    def test_output_shape(self, efficientnet_model, dummy_input):
        out = efficientnet_model(dummy_input)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_output_dtype(self, efficientnet_model, dummy_input):
        out = efficientnet_model(dummy_input)
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, efficientnet_model, dummy_input):
        out = efficientnet_model(dummy_input)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_baseline_factory(self):
        cfg = {"model": {"name": "baseline", "num_classes": 10}}
        model = build_model(cfg)
        assert isinstance(model, BaselineCNN)

    def test_efficientnet_factory(self):
        cfg = {
            "model": {
                "name": "efficientnet_b0",
                "num_classes": 10,
                "pretrained": False,
                "freeze_backbone": False,
            }
        }
        model = build_model(cfg)
        assert isinstance(model, EfficientNetB0Classifier)

    def test_unknown_model_raises(self):
        cfg = {"model": {"name": "resnet999", "num_classes": 10}}
        with pytest.raises(ValueError, match="Unknown model name"):
            build_model(cfg)

    def test_forward_pass_via_factory(self):
        cfg = {"model": {"name": "baseline", "num_classes": 10}}
        model = build_model(cfg)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_baseline_gradients_flow(self, baseline_model, dummy_input):
        """Ensure all parameters receive gradients during a backward pass."""
        baseline_model.train()
        out = baseline_model(dummy_input)
        loss = out.sum()
        loss.backward()
        for name, param in baseline_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
