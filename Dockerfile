# syntax=docker/dockerfile:1.4
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – dependency builder
#   Uses the official uv image so we don't need to install uv separately.
#   All packages are installed into a virtual env that is copied to the runtime
#   stage, keeping the final image lean.
# ─────────────────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-slim AS builder

WORKDIR /app

# Copy only dependency files first — Docker caches this layer until they change
COPY pyproject.toml ./

# Install all project dependencies (no dev extras) into a local .venv
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
        scikit-learn

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the pre-built venv from builder
COPY --from=builder /app/.venv /app/.venv

# Prepend venv to PATH so every python/pip call uses it
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/models/hf_cache \
    HF_HOME=/app/models/hf_cache

# Copy source code and configs (ordered by change frequency)
COPY configs/ ./configs/
COPY src/      ./src/
COPY train.py evaluate.py inference.py ./

# Create runtime directories
RUN mkdir -p data/raw data/processed models/artifacts models/hf_cache mlruns

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Default command: run training
CMD ["python", "train.py"]
