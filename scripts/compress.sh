#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/artifacts}"
METHOD="${METHOD:-dynamic_quant}"
PRUNE_AMOUNT="${PRUNE_AMOUNT:-0.3}"
OUTPUT="${OUTPUT:-models/artifacts/compression_report.json}"

python compress.py \
    --model-dir "$MODEL_DIR" \
    --method "$METHOD" \
    --prune-amount "$PRUNE_AMOUNT" \
    --output "$OUTPUT" \
    "$@"
