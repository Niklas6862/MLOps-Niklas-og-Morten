#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/artifacts_compressed}"
OUTPUT="${OUTPUT:-models/artifacts/finetune_pruned_report.json}"

python finetune_pruned.py \
    --model-dir "$MODEL_DIR" \
    --output "$OUTPUT" \
    "$@"
