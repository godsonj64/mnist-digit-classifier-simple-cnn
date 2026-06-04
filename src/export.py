import argparse
import os

import torch

from src.model import build_model
from src.utils import ensure_dir, get_device, load_config


def export(config_path):
    """Save the trained model in PyTorch (.pt) and ONNX (.onnx) formats."""
    cfg = load_config(config_path)
    device = get_device(cfg["train"]["device"])

    model = build_model(cfg).to(device)
    ckpt_path = os.path.join(cfg["output"]["dir"], cfg["output"]["checkpoint_name"])
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = ensure_dir(cfg["output"]["dir"])
    image_size = cfg["data"]["image_size"]
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    pt_path = os.path.join(out_dir, cfg["export"]["pytorch_name"])
    torch.save(model.state_dict(), pt_path)
    print("Saved PyTorch model to", pt_path)

    onnx_path = os.path.join(out_dir, cfg["export"]["onnx_name"])
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=cfg["export"]["onnx_opset"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print("Saved ONNX model to", onnx_path)
    return pt_path, onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    export(args.config)
