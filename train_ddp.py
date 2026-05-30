"""DDP + AMP training entry point — launch with torchrun.

torchrun --standalone --nproc_per_node=<N> train_ddp.py
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import mlflow
import mlflow.transformers
import torch
import torch.distributed as dist
import yaml
from src.config import load_config
from src.data import collate_fn, load_image_dataset, preprocess_dataset
from src.model import load_model_and_processor
from src.utils import ensure_dir, get_label_mappings, set_seed, setup_logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP + AMP fine-tuning of a ViT image classifier")
    parser.add_argument(
        "--config",
        nargs="+",
        default=DEFAULT_CONFIGS,
        metavar="PATH",
        help="One or more YAML config files (merged left-to-right)",
    )
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "LOCAL_RANK env var not found.  Launch with torchrun, e.g.:\n"
            "  torchrun --standalone --nproc_per_node=<N_GPUS> train_ddp.py\n"
            "  bash scripts/train_ddp.sh [N_GPUS]"
        )
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def evaluate_distributed(
    model: DDP,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = torch.zeros(1, device=device)
    total = torch.zeros(1, device=device)
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum()
            total += labels.size(0)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return (correct / total).item()


def _build_model_card(cfg: dict, metrics: dict, run_id: str, world_size: int) -> dict:
    base = cfg.get("project", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    return {
        "model_name": model_cfg.get("name"),
        "dataset": data_cfg.get("name"),
        "task": "image-classification",
        "training_strategy": f"DDP ({world_size} GPU(s)) + AMP",
        "mlflow_run_id": run_id,
        "git_commit": os.getenv("GIT_COMMIT_HASH", "unknown"),
        "docker_image": os.getenv("DOCKER_IMAGE_TAG", "local"),
        "jenkins_build": os.getenv("JENKINS_BUILD_NUMBER", "local"),
        "hyperparameters": {
            "num_epochs": training_cfg.get("num_epochs", 3),
            "learning_rate": training_cfg.get("learning_rate", 2e-5),
            "per_device_batch_size": training_cfg.get("per_device_train_batch_size", 16),
            "global_batch_size": training_cfg.get("per_device_train_batch_size", 16) * world_size,
            "seed": base.get("seed", 42),
            "world_size": world_size,
        },
        "train_metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
    }


def main() -> None:  # noqa: C901
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    cfg = load_config(*args.config)
    base_cfg = cfg.get("project", {})
    data_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    if is_main:
        setup_logging(base_cfg.get("log_level", "INFO"))
        logger.info("DDP world_size=%d | local_rank=%d", world_size, local_rank)

    set_seed(base_cfg.get("seed", 42) + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    if is_main:
        ensure_dir(Path(training_cfg["output_dir"]))
        ensure_dir(data_cfg.get("cache_dir", "data/raw"))
        ensure_dir(model_cfg.get("cache_dir", "models/hf_cache"))
        logger.info("Device: %s | AMP: %s", device, use_amp)

    dist.barrier()

    raw_ds = load_image_dataset(data_cfg)
    label2id, id2label = get_label_mappings(raw_ds)
    if is_main:
        logger.info("Label mapping: %s", id2label)

    model, processor = load_model_and_processor(model_cfg, label2id, id2label)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    processed_ds = preprocess_dataset(raw_ds, processor, data_cfg)
    processed_ds.set_format("torch", columns=["pixel_values", "labels"])

    batch_size = training_cfg.get("per_device_train_batch_size", 16)
    eval_batch_size = training_cfg.get("per_device_eval_batch_size", 32)
    num_workers = training_cfg.get("dataloader_num_workers", 2)
    pin_memory = torch.cuda.is_available()

    train_sampler = DistributedSampler(
        processed_ds["train"],
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=base_cfg.get("seed", 42),
    )
    train_loader = DataLoader(
        processed_ds["train"],
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    val_sampler = DistributedSampler(
        processed_ds["validation"],
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    val_loader = DataLoader(
        processed_ds["validation"],
        batch_size=eval_batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_id: str | None = None
    if is_main:
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", base_cfg.get("mlflow_tracking_uri", "mlruns")
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(base_cfg.get("experiment_name", "image-classifier"))
        mlflow_run = mlflow.start_run()
        run_id = mlflow_run.info.run_id
        mlflow.set_tags(
            {
                "jenkins_build_number": os.getenv("JENKINS_BUILD_NUMBER", "local"),
                "docker_image": os.getenv("DOCKER_IMAGE_TAG", "local"),
                "git_commit": os.getenv("GIT_COMMIT_HASH", "unknown"),
                "training_strategy": f"DDP-{world_size}GPU+AMP",
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
                "global_batch_size": batch_size * world_size,
                "world_size": world_size,
                "amp_enabled": use_amp,
            }
        )
        (Path(training_cfg["output_dir"]) / "run_id.txt").write_text(run_id)
        logger.info("MLflow run ID: %s", run_id)

    dist.barrier()

    logging_steps = training_cfg.get("logging_steps", 10)
    best_val_acc = 0.0
    best_metrics: dict = {}

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_loss = torch.zeros(1, device=device)
        n_steps = torch.zeros(1, device=device)

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

            epoch_loss += loss.detach()
            n_steps += 1

            if is_main and step % logging_steps == 0:
                logger.info(
                    "Epoch %d/%d | step %d/%d | loss %.4f",
                    epoch,
                    num_epochs,
                    step,
                    len(train_loader),
                    loss.item(),
                )

        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_steps, op=dist.ReduceOp.SUM)
        avg_loss = (epoch_loss / n_steps).item()

        val_acc = evaluate_distributed(model, val_loader, device)

        if is_main:
            logger.info("Epoch %d | train_loss=%.4f | val_acc=%.4f", epoch, avg_loss, val_acc)
            mlflow.log_metrics({"train_loss": avg_loss, "eval_accuracy": val_acc}, step=epoch)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_metrics = {"train_loss": avg_loss, "eval_accuracy": val_acc}

    dist.barrier()
    if is_main:
        output_dir = Path(training_cfg["output_dir"])
        logger.info("Saving model to '%s' …", output_dir)
        unwrapped = model.module
        unwrapped.save_pretrained(str(output_dir))
        processor.save_pretrained(str(output_dir))

        mlflow.transformers.log_model(
            transformers_model={"model": unwrapped, "image_processor": processor},
            artifact_path="model",
            task="image-classification",
        )
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        card = _build_model_card(cfg, best_metrics, run_id, world_size)
        card_path = output_dir / "model_card.yaml"
        with open(card_path, "w") as fh:
            yaml.dump(card, fh, default_flow_style=False)
        mlflow.log_artifact(str(card_path), artifact_path="model_card")
        mlflow.end_run()
        logger.info("Done. Best val accuracy: %.4f", best_val_acc)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
