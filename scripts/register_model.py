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

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", base_cfg.get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)

    run_id_file = MODEL_DIR / "run_id.txt"
    if not run_id_file.exists():
        logger.error("run_id.txt not found at %s — was train.py successful?", run_id_file)
        sys.exit(1)

    run_id = run_id_file.read_text().strip()
    if not run_id:
        logger.error("run_id.txt is empty — was train.py successful?")
        sys.exit(1)

    logger.info("Registering model from MLflow run %s", run_id)

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    if result is None or result.version is None:
        logger.error(
            "mlflow.register_model returned no version — model URI may be wrong: %s", model_uri
        )
        sys.exit(1)

    version = str(result.version)
    logger.info("Registered '%s' version %s", MODEL_NAME, version)

    client = mlflow.tracking.MlflowClient()

    # transition_model_version_stage is deprecated in MLflow 2.9+; use aliases instead.
    try:
        client.set_registered_model_alias(name=MODEL_NAME, alias="staging", version=version)
        logger.info("Model version %s aliased as 'staging'.", version)
    except Exception as exc:
        logger.warning(
            "set_registered_model_alias failed (%s), falling back to stage transition.", exc
        )
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        logger.info("Model version %s transitioned to Staging.", version)


if __name__ == "__main__":
    main()
