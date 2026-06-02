#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/artifacts}"
OUTPUT="${OUTPUT:-models/artifacts/pruning_report.json}"

python pruning.py \
    --model-dir "$MODEL_DIR" \
    --sweep \
    --output "$OUTPUT" \
    "$@"
