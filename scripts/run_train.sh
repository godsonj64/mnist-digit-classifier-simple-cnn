#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_train.sh  –  Train the MNIST digit classifier
#
# Usage:
#   bash scripts/run_train.sh                  # default settings
#   MODEL=efficientnet_b0 bash scripts/run_train.sh   # swap model via env var
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
EPOCHS="${EPOCHS:-20}"

echo "========================================="
echo " MNIST Digit Classifier – Training"
echo "========================================="
echo " Config : ${CONFIG}"
echo " Epochs : ${EPOCHS}"
echo "-----------------------------------------"

python src/train.py --config "${CONFIG}" --epochs "${EPOCHS}"

echo "-----------------------------------------"
echo " Training finished. Running evaluation …"
echo "-----------------------------------------"

python src/evaluate.py \
    --config "${CONFIG}" \
    --checkpoint outputs/best_model.pt

echo "-----------------------------------------"
echo " Exporting model …"
echo "-----------------------------------------"

python src/export.py \
    --config "${CONFIG}" \
    --checkpoint outputs/best_model.pt

echo "========================================="
echo " All done!"
echo "  Checkpoint : outputs/best_model.pt"
echo "  ONNX       : outputs/model.onnx"
echo "  TorchScript: outputs/model_torchscript.pt"
echo "========================================="
