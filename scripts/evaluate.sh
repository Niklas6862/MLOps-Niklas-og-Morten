#!/usr/bin/env bash
# scripts/evaluate.sh — evaluate a saved model on the test split.
#
# Usage:
#   bash scripts/evaluate.sh
#   bash scripts/evaluate.sh --model-dir models/artifacts --split test

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/artifacts}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-models/artifacts/eval_results.json}"

echo "==> Evaluating model at '$MODEL_DIR' on split '$SPLIT' …"
python evaluate.py --model-dir "$MODEL_DIR" --split "$SPLIT" --output "$OUTPUT" "$@"
echo "==> Evaluation results saved to $OUTPUT"
