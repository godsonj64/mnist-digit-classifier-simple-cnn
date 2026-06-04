# MNIST Digit Classifier (Simple CNN)

This project trains an image classifier that recognizes handwritten digits (0-9) from the
classic MNIST dataset. It uses a **MobileNetV3-Small** model with transfer learning as the
recommended model, and a **small CNN trained from scratch** as a baseline.

## Task

- **Task type:** image classification
- **Classes:** 10 (digits 0 through 9)
- **Metrics:** accuracy, F1 score
- **Export formats:** PyTorch (`.pt`), ONNX (`.onnx`)
- **Epochs:** 5

## Dataset format

The code expects an `image_folder` layout (one subfolder per class):

```
data/
  train/
    0/ img1.png img2.png ...
    1/ ...
    ...
    9/ ...
  val/
    0/ ...
    ...
    9/ ...
```

See `data/README.md` for how to produce this layout from MNIST automatically.

## Quick start

```bash
pip install -r requirements.txt

# (optional) download MNIST and create the image_folder layout
python -m src.dataset --prepare --data-dir data

# train
bash scripts/run_train.sh

# evaluate
python -m src.evaluate --config configs/default.yaml

# export to PyTorch + ONNX
python -m src.export --config configs/default.yaml
```

## Configuration

All settings live in `configs/default.yaml`. Switch between the recommended model and the
baseline by changing the `model.name` field to `mobilenet_v3_small` or `small_cnn`.

## Output

During training one line is printed per epoch in this exact format:

```
epoch 1/5 loss=0.3124 val_acc=0.9521
```

The best checkpoint is saved to `outputs/best_model.pt`.
