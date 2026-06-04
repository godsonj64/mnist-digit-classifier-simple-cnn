import argparse

import torch

from src.model import build_model
from src.utils import ensure_dir, load_config, resolve_device


def export(config_path):
    """Export the trained model to ONNX and TorchScript formats."""
    cfg = load_config(config_path)
    device = resolve_device(cfg["train"]["device"])

    model = build_model(cfg).to(device)
    checkpoint = cfg["train"]["checkpoint"]
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    image_size = cfg["data"]["image_size"]
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    onnx_path = cfg["export"]["onnx_path"]
    ts_path = cfg["export"]["torchscript_path"]
    ensure_dir(onnx_path)
    ensure_dir(ts_path)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=cfg["export"]["opset"],
    )
    print(f"Saved ONNX model to {onnx_path}")

    scripted = torch.jit.trace(model, dummy)
    scripted.save(ts_path)
    print(f"Saved TorchScript model to {ts_path}")


def main():
    parser = argparse.ArgumentParser(description="Export MNIST digit classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    export(args.config)


if __name__ == "__main__":
    main()
