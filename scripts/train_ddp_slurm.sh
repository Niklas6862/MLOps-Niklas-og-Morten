#!/bin/bash
#SBATCH --job-name=train_ddp
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs models/artifacts data/raw

module load cuda/12.1 2>/dev/null || true

if [ ! -d ".venv" ]; then
    echo "Setting up Python environment..."
    pip install uv --quiet
    uv venv .venv
    uv pip install -e ".[dev]" --quiet
fi

source .venv/bin/activate

MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://172.24.198.42:5050}
export MLFLOW_TRACKING_URI

N_GPUS=${SLURM_GPUS_ON_NODE:-2}

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "GPUs:      ${N_GPUS}"
echo "Started:   $(date)"

torchrun \
    --standalone \
    --nproc_per_node="${N_GPUS}" \
    train_ddp.py

echo "Finished: $(date)"
