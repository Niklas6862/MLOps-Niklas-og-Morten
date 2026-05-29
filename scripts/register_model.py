"""Register the latest trained model to the MLflow Model Registry.

Reads the run ID written by train.py from ``models/artifacts/run_id.txt``,
registers the logged ``model`` artifact, and transitions it to *Staging*.

Run from project root::

    python scripts/register_model.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.tracking
from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
]
MODEL_DIR = Path("models/artifacts")
MODEL_NAME = "image-classifier"


def main() -> None:
    cfg = load_config(*DEFAULT_CONFIGS)
    base_cfg = cfg.get("project", {})

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", base_cfg.get("mlflow_tracking_uri", "mlruns")
    )
    mlflow.set_tracking_uri(tracking_uri)

    run_id_file = MODEL_DIR / "run_id.txt"
    if not run_id_file.exists():
        logger.error("run_id.txt not found at %s — was train.py successful?", run_id_file)
        sys.exit(1)

    run_id = run_id_file.read_text().strip()
    logger.info("Registering model from MLflow run %s", run_id)

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    logger.info("Registered '%s' version %s", MODEL_NAME, result.version)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.info("Model version %s transitioned to Staging.", result.version)


if __name__ == "__main__":
    main()
