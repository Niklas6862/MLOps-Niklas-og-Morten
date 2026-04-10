#!/usr/bin/env bash
# scripts/train.sh — run the full fine-tuning pipeline locally.
#
# Usage:
#   bash scripts/train.sh
#   bash scripts/train.sh --config configs/base.yaml configs/training.yaml

set -euo pipefail

echo "==> Starting training pipeline …"
python train.py "$@"
echo "==> Training complete. Artifacts saved to models/artifacts/"
