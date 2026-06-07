"""Export the trained model to ONNX and TorchScript formats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import onnx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_model
from src.utils import ensure_dir, get_device, get_logger, load_checkpoint, load_config, set_seed


def export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    path: str,
    opset_version: int,
    num_classes: int,
) -> None:
    """Trace the model with a dummy tensor and write an ONNX file."""
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    # Verify the exported model
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)


def export_torchscript(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    path: str,
) -> None:
    """Trace and save the model as a self-contained TorchScript archive."""
    model.eval()
    traced = torch.jit.trace(model, dummy_input)
    traced.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MNIST model")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/best_model.pt",
        help="Path to model checkpoint (.pt)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("mnist.export")
    set_seed(cfg["project"]["seed"])
    device = get_device()

    export_cfg = cfg["export"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    # Build & load model (always export on CPU for maximum compatibility)
    model = build_model(cfg)
    load_checkpoint(model, args.checkpoint, torch.device("cpu"))
    model.eval()
    model.cpu()

    image_size = data_cfg["image_size"]
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # ONNX
    onnx_path = export_cfg["onnx_path"]
    ensure_dir(str(Path(onnx_path).parent))
    logger.info(f"Exporting ONNX → {onnx_path}")
    export_onnx(
        model,
        dummy_input,
        onnx_path,
        opset_version=export_cfg.get("opset_version", 17),
        num_classes=model_cfg["num_classes"],
    )
    logger.info("ONNX export verified ✓")

    # TorchScript
    ts_path = export_cfg["torchscript_path"]
    ensure_dir(str(Path(ts_path).parent))
    logger.info(f"Exporting TorchScript → {ts_path}")
    export_torchscript(model, dummy_input, ts_path)
    logger.info("TorchScript export complete ✓")

    logger.info("All exports finished successfully.")


if __name__ == "__main__":
    main()
