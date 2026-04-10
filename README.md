# Image Classifier Fine-Tuning — MLOps Project

A production-style, reproducible MLOps pipeline for fine-tuning a **Vision Transformer (ViT)** image classifier using HuggingFace Transformers on the [beans](https://huggingface.co/datasets/beans) dataset.

**Model:** `google/vit-base-patch16-224`
**Dataset:** `beans` — 3-class plant disease classification (angular\_leaf\_spot, bean\_rust, healthy)
**Authors:** Niklas & Morten

---

## Repository Structure

```
.
├── configs/                # YAML configuration files
│   ├── base.yaml           # Project-level settings (seed, MLflow URI, etc.)
│   ├── data.yaml           # Dataset name, splits, image column
│   ├── model.yaml          # Model name, num_labels, HF cache dir
│   └── training.yaml       # Batch size, LR, epochs, eval strategy, …
│
├── src/                    # All library code (imported by entry points)
│   ├── config.py           # YAML loading + deep-merge
│   ├── data.py             # Dataset loading, transforms, collate_fn
│   ├── model.py            # Model + processor loading
│   ├── train.py            # TrainingArguments builder + compute_metrics
│   ├── eval.py             # Detailed per-class metrics + JSON serialiser
│   ├── infer.py            # Single-image forward pass
│   └── utils.py            # Seeding, logging, path helpers, label maps
│
├── train.py                # Entry point: fine-tune the model
├── evaluate.py             # Entry point: evaluate on test split
├── inference.py            # Entry point: classify a single image
│
├── tests/                  # pytest test suite
│   ├── conftest.py         # Shared fixtures
│   ├── test_config.py      # Config loading + merging
│   ├── test_data.py        # Transforms, collate_fn, utils
│   └── test_model.py       # Metrics, inference, results serialisation
│
├── notebooks/
│   └── exploration.ipynb   # Dataset exploration + forward-pass demo
│
├── scripts/
│   ├── train.sh            # Shell wrapper for train.py
│   ├── evaluate.sh         # Shell wrapper for evaluate.py
│   └── inference.sh        # Shell wrapper for inference.py
│
├── data/                   # Local dataset cache (git-ignored)
├── models/                 # Saved model artefacts + HF cache (git-ignored)
├── mlruns/                 # MLflow experiment data (git-ignored)
│
├── Dockerfile              # Multi-stage build (builder → runtime)
├── docker-compose.yml      # trainer + mlflow UI services
├── .dockerignore
│
├── .github/workflows/
│   └── ci.yml              # GitHub Actions: lint → test → validate → docker
│
├── Jenkinsfile             # Jenkins: checkout → setup → lint → test → docker
│
├── pyproject.toml          # uv / hatchling project config + Ruff + pytest
└── .pre-commit-config.yaml # Ruff lint/format + file hygiene hooks
```

---

## Why This Structure Supports Reproducibility

| Practice | Where |
|---|---|
| Fixed random seed | `configs/base.yaml` → `src/utils.set_seed()` |
| Config-driven training (no hardcoded values) | `configs/*.yaml` loaded by every entry point |
| Versioned model artefacts | `models/artifacts/` written by `train.py` |
| Deterministic CUDA ops | `src/utils.set_seed()` sets `cudnn.deterministic=True` |
| Experiment tracking | MLflow auto-logs params, metrics, and artefacts |
| Dependency pinning | `pyproject.toml` + `uv` for reproducible installs |
| Layer-cached Docker image | Builder stage caches deps separately from source |

---

## Setup

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)

### Local install

```bash
# 1. Clone the repo
git clone https://github.com/Niklas6862/MLOps-Niklas-og-Morten-1.git
cd MLOps-Niklas-og-Morten-1

# 2. Create virtual environment and install all dependencies
uv venv
uv pip install -e ".[dev]"

# 3. Activate (Linux/macOS)
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 4. (Optional) Install pre-commit hooks
pre-commit install
```

---

## Training

```bash
# Default — uses all four config files
python train.py

# Or via the shell wrapper
bash scripts/train.sh

# Override config files explicitly
python train.py --config configs/base.yaml configs/data.yaml configs/model.yaml configs/training.yaml
```

Training artefacts are saved to `models/artifacts/`.
Metrics and parameters are logged to MLflow under `mlruns/`.

**Key config knobs (`configs/training.yaml`):**

| Key | Default | Description |
|---|---|---|
| `num_epochs` | `3` | Training epochs |
| `learning_rate` | `2e-5` | AdamW LR |
| `per_device_train_batch_size` | `16` | Batch size per GPU/CPU |
| `fp16` | `false` | Enable mixed precision (requires CUDA) |

---

## Evaluation

```bash
# Evaluate on the test split (default)
python evaluate.py

# Specify model directory and split
python evaluate.py --model-dir models/artifacts --split test --output models/artifacts/eval_results.json

# Via shell wrapper
bash scripts/evaluate.sh
```

Results are written as JSON to `models/artifacts/eval_results.json`.

---

## Inference

```bash
# Single-image classification
python inference.py path/to/leaf_photo.jpg

# Custom model directory and top-k
python inference.py path/to/leaf_photo.jpg --model-dir models/artifacts --top-k 3

# Via shell wrapper
bash scripts/inference.sh path/to/leaf_photo.jpg
```

Example output:

```json
[
  { "label": "healthy",            "score": 0.9821 },
  { "label": "bean_rust",          "score": 0.0143 },
  { "label": "angular_leaf_spot",  "score": 0.0036 }
]
```

---

## MLflow Experiment Tracking

```bash
# Start the MLflow UI (after at least one training run)
mlflow ui --backend-store-uri mlruns

# Or via Docker Compose
docker compose up mlflow
```

Open [http://localhost:5000](http://localhost:5000) to browse experiments, compare runs, and download artefacts.

---

## Docker

### Build

```bash
docker build -t image-classifier:latest .
```

### Run training

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlruns:/app/mlruns \
  image-classifier:latest python train.py
```

### Docker Compose (trainer + MLflow UI)

```bash
# Run training and then view results
docker compose up trainer
docker compose up mlflow        # then open http://localhost:5000
```

---

## Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific file
pytest tests/test_config.py -v
```

---

## Linting & Formatting

```bash
# Check
ruff check src/ tests/ train.py evaluate.py inference.py

# Auto-fix
ruff check --fix src/ tests/ train.py evaluate.py inference.py

# Format
ruff format src/ tests/ train.py evaluate.py inference.py
```

---

## CI/CD

### GitHub Actions

Workflows are defined in [.github/workflows/ci.yml](.github/workflows/ci.yml).

On every push / PR to `main`:

1. **Lint** — Ruff check + format check
2. **Test** — pytest with coverage
3. **Validate** — required files + YAML integrity
4. **Docker Build** — builds the image (main branch only)

### Jenkins

The [Jenkinsfile](Jenkinsfile) mirrors the same stages and publishes JUnit test results.

---

## Configuration Reference

All configs live in `configs/`. Later files override earlier ones when merged.

| File | Responsibility |
|---|---|
| `configs/base.yaml` | Seed, output paths, MLflow URI, experiment name |
| `configs/data.yaml` | HuggingFace dataset name, split names, image/label columns |
| `configs/model.yaml` | Pre-trained model name, `num_labels`, cache dir |
| `configs/training.yaml` | Batch size, LR, epochs, eval strategy, save settings |

To use a custom dataset, change `dataset.name` in `configs/data.yaml` to any HuggingFace image-classification dataset that exposes a `ClassLabel` feature.

---

## Assumptions

- Dataset is downloaded automatically from HuggingFace Hub on first run.
- GPU is optional — training works on CPU (slow) and CUDA (fast).
- `fp16: false` by default for CPU compatibility; set to `true` if you have CUDA.
- MLflow logs locally to `mlruns/`; swap `mlflow_tracking_uri` for a remote server URI.
- The `evaluate.py` root script avoids importing the `evaluate` HuggingFace package to prevent module-name shadowing; accuracy is computed with numpy instead.
