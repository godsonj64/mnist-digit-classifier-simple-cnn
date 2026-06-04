#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/default.yaml}

if [ ! -d "data/train" ]; then
  echo "No data found, preparing MNIST..."
  python -m src.dataset --prepare --data-dir data
fi

python -m src.train --config "$CONFIG"
