# MNIST Digit Classifier (Simple CNN)

This project trains an image-recognition model to identify handwritten digits (0-9)
using the classic MNIST dataset arranged as an image folder.

- **Task:** Image classification (10 classes)
- **Recommended model:** MobileNetV3-Small (transfer learning)
- **Baseline model:** Small CNN trained from scratch
- **Metrics:** accuracy, F1
- **Export formats:** ONNX, TorchScript
- **Epochs:** 5

## Dataset layout

The code expects an `image_folder` layout (see `data/README.md`):

```
data/
  train/
    0/ img001.png ...
    1/ ...
    ...
    9/ ...
  val/
    0/ ...
    ...
    9/ ...
```

If no dataset is found, the training and evaluation scripts automatically
download MNIST and write it into the image-folder layout for you.

## Quick start

```bash
pip install -r requirements.txt

# Train
bash scripts/run_train.sh

# Evaluate
python -m src.evaluate --config configs/default.yaml

# Export to ONNX + TorchScript
python -m src.export --config configs/default.yaml
```

## Configuration

All settings live in `configs/default.yaml`. You can switch between the
MobileNetV3-Small transfer-learning model and the small baseline CNN by
changing `model.name` to `mobilenet_v3_small` or `small_cnn`.

## Docker

```bash
docker build -t mnist-classifier .
docker run --rm mnist-classifier
```

## Training output

During training, one line is printed per epoch in this exact format:

```
epoch 1/5 loss=0.3210 val_acc=0.9512
```
