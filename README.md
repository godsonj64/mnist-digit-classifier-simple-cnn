# MNIST Digit Classifier (Simple CNN)

A lightweight convolutional neural network (CNN) that recognises handwritten digits (0–9) from the classic MNIST dataset. The project also ships an EfficientNet-B0 transfer-learning variant so you can compare both approaches side-by-side.

## Project layout

```
.
├── configs/
│   └── default.yaml        # All hyper-parameters in one place
├── data/
│   └── README.md           # How to download / structure the dataset
├── scripts/
│   └── run_train.sh        # One-command training launcher
├── src/
│   ├── dataset.py          # Dataset loading & augmentation
│   ├── model.py            # CNN baseline + EfficientNet-B0 variant
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Accuracy & F1 evaluation
│   ├── export.py           # ONNX & TorchScript export
│   └── utils.py            # Shared helpers (logging, seeding, …)
├── tests/
│   └── test_model_forward.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Prepare data

See `data/README.md`. The dataset is downloaded automatically if you use the default config.

### 3 – Train

```bash
bash scripts/run_train.sh
# or, with custom overrides:
python src/train.py --config configs/default.yaml --epochs 20
```

### 4 – Evaluate

```bash
python src/evaluate.py --config configs/default.yaml --checkpoint outputs/best_model.pt
```

### 5 – Export

```bash
python src/export.py --config configs/default.yaml --checkpoint outputs/best_model.pt
# Writes outputs/model.onnx and outputs/model_torchscript.pt
```

## Docker

```bash
docker build -t mnist-classifier .
docker run --rm -v $(pwd)/outputs:/app/outputs mnist-classifier bash scripts/run_train.sh
```

## Metrics

| Metric | Description |
|--------|-------------|
| accuracy | Fraction of digits predicted correctly |
| f1 | Macro-averaged F1 score across all 10 digit classes |

## Model options

| Key | Description |
|-----|-------------|
| `baseline` | Small 3-layer CNN trained from scratch (~90 k parameters) |
| `efficientnet_b0` | EfficientNet-B0 with ImageNet pre-trained weights (transfer learning) |

Set `model.name` in `configs/default.yaml` to switch between them.
