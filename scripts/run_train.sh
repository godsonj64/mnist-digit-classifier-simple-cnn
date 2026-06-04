#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

python -m src.train --config "$CONFIG"
