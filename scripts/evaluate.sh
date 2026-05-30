#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/artifacts}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-models/artifacts/eval_results.json}"

python evaluate.py --model-dir "$MODEL_DIR" --split "$SPLIT" --output "$OUTPUT" "$@"
