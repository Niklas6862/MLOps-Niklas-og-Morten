#!/usr/bin/env bash
# Launch DDP + AMP training on a single node.
#
# Usage:
#   bash scripts/train_ddp.sh           # uses all available GPUs
#   bash scripts/train_ddp.sh 2         # use 2 GPUs
#   bash scripts/train_ddp.sh 1         # single GPU (still goes through torchrun)
#
# Multi-node example (run on every node, adjusting --node_rank and --master_addr):
#   NODE_RANK=0 MASTER_ADDR=<node0-ip> bash scripts/train_ddp.sh 8
#   NODE_RANK=1 MASTER_ADDR=<node0-ip> bash scripts/train_ddp.sh 8
set -euo pipefail

N_GPUS=${1:-$(python -c "import torch; print(max(torch.cuda.device_count(), 1))")}
NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching DDP training: ${NNODES} node(s), ${N_GPUS} GPU(s) per node"

if [ "${NNODES}" -gt 1 ]; then
    torchrun \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --nproc_per_node="${N_GPUS}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        train_ddp.py "$@"
else
    torchrun \
        --standalone \
        --nproc_per_node="${N_GPUS}" \
        train_ddp.py "$@"
fi
