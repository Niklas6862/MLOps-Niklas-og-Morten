#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/artifacts}"
OUTPUT="${OUTPUT:-models/artifacts/batch_sweep_report.json}"
PLOT="${PLOT:-models/artifacts/batch_sweep_plot.png}"

python batch_size_sweep.py \
    --model-dir "$MODEL_DIR" \
    --output "$OUTPUT" \
    --plot "$PLOT" \
    "$@"
