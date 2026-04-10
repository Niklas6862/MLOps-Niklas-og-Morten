#!/usr/bin/env bash
# scripts/inference.sh — run single-image inference.
#
# Usage:
#   bash scripts/inference.sh path/to/image.jpg
#   IMAGE_PATH=data/sample.jpg MODEL_DIR=models/artifacts bash scripts/inference.sh

set -euo pipefail

IMAGE_PATH="${1:-${IMAGE_PATH:-}}"
MODEL_DIR="${MODEL_DIR:-models/artifacts}"
TOP_K="${TOP_K:-3}"

if [ -z "$IMAGE_PATH" ]; then
    echo "Usage: bash scripts/inference.sh <image_path>"
    exit 1
fi

echo "==> Running inference on '$IMAGE_PATH' …"
python inference.py "$IMAGE_PATH" --model-dir "$MODEL_DIR" --top-k "$TOP_K"
