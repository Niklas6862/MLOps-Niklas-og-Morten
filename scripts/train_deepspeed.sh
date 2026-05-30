#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:-zero2}

case "${STAGE}" in
    zero2)
        DS_CONFIG="configs/deepspeed_zero2.json"
        ;;
    zero3)
        DS_CONFIG="configs/deepspeed_zero3.json"
        ;;
    *)
        echo "Unknown stage '${STAGE}'. Use 'zero2' or 'zero3'." >&2
        exit 1
        ;;
esac

N_GPUS=${N_GPUS:-$(python -c "import torch; print(max(torch.cuda.device_count(), 1))")}

echo "Launching DeepSpeed ${STAGE} training on ${N_GPUS} GPU(s) with config ${DS_CONFIG}"

deepspeed --num_gpus="${N_GPUS}" train.py \
    --config configs/base.yaml \
             configs/data.yaml \
             configs/model.yaml \
             configs/training.yaml \
             configs/training_deepspeed.yaml
