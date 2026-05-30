#!/usr/bin/env bash
set -euo pipefail

IMAGE_PATH="${1:-${IMAGE_PATH:-}}"
MODEL_DIR="${MODEL_DIR:-models/artifacts}"
TOP_K="${TOP_K:-3}"

if [ -z "$IMAGE_PATH" ]; then
    echo "Usage: bash scripts/inference.sh <image_path>"
    exit 1
fi

python inference.py "$IMAGE_PATH" --model-dir "$MODEL_DIR" --top-k "$TOP_K"
