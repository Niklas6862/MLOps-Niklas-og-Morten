"""AMP training entry point — single GPU with Automatic Mixed Precision.

Usage::

    python train_amp.py
    python train_amp.py --config configs/base.yaml configs/data.yaml configs/model.yaml configs/training.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import mlflow
import mlflow.transformers
import torch
import yaml
from src.config import load_config
from src.data import collate_fn, load_image_dataset, preprocess_dataset
from src.model import load_model_and_processor
from src.utils import ensure_dir, get_label_mappings, set_seed, setup_logging
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMP fine-tuning of a ViT image classifier")
    parser.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS, metavar="PATH")
    return parser.parse_args()


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            preds = model(pixel_values=pixel_values).logits.argmax(dim=-1)
            correct += int((preds == labels).sum())
            total += labels.size(0)
    return correct / total if total else 0.0


def _build_model_card(cfg: dict, metrics: dict, run_id: str, amp: bool) -> dict:
    base = cfg.get("project", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    return {
        "model_name": model_cfg.get("name"),
        "dataset": data_cfg.get("name"),
        "task": "image-classification",
        "training_strategy": "AMP (FP16)" if amp else "FP32",
        "mlflow_run_id": run_id,
        "git_commit": os.getenv("GIT_COMMIT_HASH", "unknown"),
        "docker_image": os.getenv("DOCKER_IMAGE_TAG", "local"),
        "jenkins_build": os.getenv("JENKINS_BUILD_NUMBER", "local"),
        "hyperparameters": {
            "num_epochs": training_cfg.get("num_epochs", 3),
            "learning_rate": training_cfg.get("learning_rate", 2e-5),
            "per_device_batch_size": training_cfg.get("per_device_train_batch_size", 16),
            "seed": base.get("seed", 42),
            "amp_enabled": amp,
        },
        "train_metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)

    base_cfg = cfg.get("project", {})
    data_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    setup_logging(base_cfg.get("log_level", "INFO"))
    set_seed(base_cfg.get("seed", 42))

    output_dir = Path(training_cfg["output_dir"])
    ensure_dir(output_dir)
    ensure_dir(data_cfg.get("cache_dir", "data/raw"))
    ensure_dir(model_cfg.get("cache_dir", "models/hf_cache"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # AMP requires CUDA; falls back to FP32 transparently on CPU
    use_amp = device.type == "cuda"
    logger.info("Device: %s | AMP: %s", device, use_amp)

    raw_ds = load_image_dataset(data_cfg)
    label2id, id2label = get_label_mappings(raw_ds)
    logger.info("Label mapping: %s", id2label)

    model, processor = load_model_and_processor(model_cfg, label2id, id2label)
    model = model.to(device)

    processed_ds = preprocess_dataset(raw_ds, processor, data_cfg)
    processed_ds.set_format("torch", columns=["pixel_values", "labels"])

    batch_size = training_cfg.get("per_device_train_batch_size", 16)
    eval_batch_size = training_cfg.get("per_device_eval_batch_size", 32)
    num_workers = training_cfg.get("dataloader_num_workers", 2)

    train_loader = DataLoader(
        processed_ds["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_amp,
    )
    val_loader = DataLoader(
        processed_ds["validation"],
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_amp,
    )

    num_epochs = training_cfg.get("num_epochs", 3)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * training_cfg.get("warmup_ratio", 0.1))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 2e-5),
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # GradScaler is a no-op when enabled=False, so safe to keep for CPU runs
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", base_cfg.get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(base_cfg.get("experiment_name", "image-classifier"))

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "jenkins_build_number": os.getenv("JENKINS_BUILD_NUMBER", "local"),
                "docker_image": os.getenv("DOCKER_IMAGE_TAG", "local"),
                "git_commit": os.getenv("GIT_COMMIT_HASH", "unknown"),
                "training_strategy": "AMP" if use_amp else "FP32",
            }
        )
        mlflow.log_params(
            {
                "model_name": model_cfg["name"],
                "dataset": data_cfg["name"],
                "seed": base_cfg.get("seed", 42),
                "num_epochs": num_epochs,
                "learning_rate": training_cfg.get("learning_rate", 2e-5),
                "per_device_batch_size": batch_size,
                "amp_enabled": use_amp,
            }
        )

        run_id_path = output_dir / "run_id.txt"
        run_id_path.write_text(run.info.run_id)
        logger.info("MLflow run ID: %s", run.info.run_id)

        logging_steps = training_cfg.get("logging_steps", 10)
        best_val_acc = 0.0
        best_metrics: dict = {}

        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(train_loader, 1):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()

                if step % logging_steps == 0:
                    logger.info(
                        "Epoch %d/%d | step %d/%d | loss %.4f",
                        epoch,
                        num_epochs,
                        step,
                        len(train_loader),
                        loss.item(),
                    )

            avg_loss = epoch_loss / len(train_loader)
            val_acc = _evaluate(model, val_loader, device)
            logger.info("Epoch %d | train_loss=%.4f | val_acc=%.4f", epoch, avg_loss, val_acc)
            mlflow.log_metrics({"train_loss": avg_loss, "eval_accuracy": val_acc}, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_metrics = {"train_loss": avg_loss, "eval_accuracy": val_acc}

        logger.info("Saving model to '%s' …", output_dir)
        model.save_pretrained(str(output_dir))
        processor.save_pretrained(str(output_dir))

        mlflow.transformers.log_model(
            transformers_model={"model": model, "image_processor": processor},
            artifact_path="model",
            task="image-classification",
        )
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        card = _build_model_card(cfg, best_metrics, run.info.run_id, use_amp)
        card_path = output_dir / "model_card.yaml"
        with open(card_path, "w") as fh:
            yaml.dump(card, fh, default_flow_style=False)
        mlflow.log_artifact(str(card_path), artifact_path="model_card")

        logger.info("Done. Best val accuracy: %.4f", best_val_acc)


if __name__ == "__main__":
    main()
