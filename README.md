# MNIST Digit Classifier (Simple CNN)

A simple image classifier that recognizes handwritten digits (0-9) from the classic MNIST dataset using PyTorch.

## Overview

- **Task:** Image classification (10 classes, digits 0-9)
- **Dataset format:** Image folder (one subfolder per class)
- **Recommended model:** EfficientNet-B0 (transfer learning) — adapted for single-channel grayscale input
- **Baseline model:** A small CNN trained from scratch
- **Metrics:** Accuracy and F1 score
- **Export formats:** PyTorch (`.pt`) and ONNX (`.onnx`)
- **Epochs:** 20

## Project layout

```
.
├── README.md
├── requirements.txt
├── Dockerfile
├── configs/default.yaml
├── data/README.md
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   └── utils.py
├── tests/test_model_forward.py
└── scripts/run_train.sh
```

## Quick start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your data as described in [`data/README.md`](data/README.md). You need
   `data/train/<class>/*.png` and `data/val/<class>/*.png`.

3. Train the model:

   ```bash
   bash scripts/run_train.sh
   ```

   During training you will see one line per epoch like:

   ```
   epoch 1/20 loss=0.4123 val_acc=0.9512
   ```

4. Evaluate the trained model:

   ```bash
   python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt
   ```

5. Export the model to PyTorch and ONNX:

   ```bash
   python -m src.export --config configs/default.yaml --checkpoint checkpoints/best.pt
   ```

## Choosing the model

Set `model.name` in `configs/default.yaml` to either:

- `efficientnet_b0` — the recommended transfer-learning model.
- `small_cnn` — the baseline CNN trained from scratch.

## Running tests

```bash
pytest tests/
```
