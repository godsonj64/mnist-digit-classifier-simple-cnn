"""Export a trained checkpoint to PyTorch and ONNX formats."""
import argparse
import os

import torch

from src.model import build_model
from src.utils import ensure_dir, get_device, load_config


def main():
    parser = argparse.ArgumentParser(description="Export the MNIST classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"]["device"])

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = cfg["export"]["output_dir"]
    ensure_dir(out_dir)

    # 1. PyTorch export
    pt_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Saved PyTorch model to {pt_path}")

    # 2. ONNX export
    image_size = cfg["data"]["image_size"]
    dummy = torch.randn(1, 1, image_size, image_size, device=device)
    onnx_path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=cfg["export"]["onnx_opset"],
    )
    print(f"Saved ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
