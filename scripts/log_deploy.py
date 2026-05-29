"""Log a deployment event to MLflow and promote the model to Production.

Reads the run ID from ``models/artifacts/run_id.txt``, tags the training run
as deployed, and transitions the latest *Staging* version of the registered
model to *Production* (archiving any existing Production versions).

Run from project root::

    python scripts/log_deploy.py
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
        logger.error("run_id.txt not found — was the full pipeline completed?")
        sys.exit(1)

    run_id = run_id_file.read_text().strip()

    # Tag the training run with deployment metadata
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags(
            {
                "deployed": "true",
                "deploy_jenkins_build": os.getenv("JENKINS_BUILD_NUMBER", "local"),
                "deploy_env": "production",
            }
        )
    logger.info("Deployment tags logged to MLflow run %s.", run_id)

    # Promote the latest Staging version to Production
    client = mlflow.tracking.MlflowClient()
    staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not staging_versions:
        logger.warning("No model version found in Staging for '%s' — nothing to promote.", MODEL_NAME)
        sys.exit(0)

    latest = staging_versions[0]
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(
        "Model '%s' version %s promoted to Production.", MODEL_NAME, latest.version
    )


if __name__ == "__main__":
    main()
