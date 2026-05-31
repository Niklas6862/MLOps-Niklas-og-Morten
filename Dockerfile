# syntax=docker/dockerfile:1.4
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml ./

RUN uv venv .venv && \
    uv pip install --python .venv/bin/python \
    torch \
    torchvision \
    transformers \
    datasets \
    accelerate \
    mlflow \
    pyyaml \
    pillow \
    numpy \
    scikit-learn \
    scipy

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/models/hf_cache \
    HF_HOME=/app/models/hf_cache

COPY configs/  ./configs/
COPY src/       ./src/
COPY scripts/  ./scripts/
COPY train.py train_amp.py evaluate.py inference.py compress.py batch_inference.py detect_drift.py ./

RUN mkdir -p data/raw data/processed models/artifacts models/hf_cache mlruns

CMD ["python", "train.py"]
