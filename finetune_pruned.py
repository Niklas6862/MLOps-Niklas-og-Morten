from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import mlflow
import torch
from src.compress import benchmark, evaluate_accuracy
from src.config import load_config
from src.data import collate_fn, load_image_dataset, preprocess_dataset
from src.train import compute_metrics
from src.utils import set_seed, setup_logging
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pruned model to recover accuracy")
    parser.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--model-dir",
        default="models/artifacts_pruned",
        help="Directory of the pruned model produced by pruning.py",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--benchmark-batches", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="models/artifacts/finetune_pruned_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    setup_logging(cfg.get("project", {}).get("log_level", "INFO"))
    set_seed(cfg.get("project", {}).get("seed", 42))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else model_dir.parent / (model_dir.name + "_finetuned")
    )

    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)

    raw = load_image_dataset(cfg["dataset"])
    processed = preprocess_dataset(raw, processor, cfg["dataset"])
    processed.set_format("torch", columns=["pixel_values", "labels"])

    batch_size = cfg.get("training", {}).get("per_device_eval_batch_size", 32)
    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        processed["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model.eval()
    logger.info("Evaluating pruned model before fine-tuning …")
    pre_acc = evaluate_accuracy(model, test_loader, device)
    pre_lat = benchmark(model, test_loader, device, n_batches=args.benchmark_batches)
    logger.info(
        "Pre-finetune: acc=%.4f  throughput=%.1f fps", pre_acc, pre_lat.get("throughput_fps", 0)
    )

    training_cfg = cfg.get("training", {})
    ft_args = TrainingArguments(
        output_dir=str(output_dir / "_ft_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 32),
        learning_rate=training_cfg.get("learning_rate", 2e-5),
        eval_strategy="epoch",
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        fp16=device == "cuda" and torch.cuda.is_available(),
    )

    logger.info("Fine-tuning for %d epoch(s) …", args.epochs)
    trainer = Trainer(
        model=model,
        args=ft_args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    trainer.train()

    post_acc = evaluate_accuracy(model, test_loader, device)
    post_lat = benchmark(model, test_loader, device, n_batches=args.benchmark_batches)
    logger.info(
        "Post-finetune: acc=%.4f (recovered %.4f)  throughput=%.1f fps",
        post_acc,
        post_acc - pre_acc,
        post_lat.get("throughput_fps", 0),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    run_id_file = model_dir / "run_id.txt"
    if run_id_file.exists():
        shutil.copy(run_id_file, output_dir / "run_id.txt")
    logger.info("Fine-tuned model saved to %s", output_dir)

    report = {
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "finetune_epochs": args.epochs,
        "pre_finetune": {"accuracy": round(pre_acc, 4), **pre_lat},
        "post_finetune": {
            "accuracy": round(post_acc, 4),
            "accuracy_recovered": round(post_acc - pre_acc, 4),
            **post_lat,
        },
    }

    # MLflow
    experiment_name = cfg.get("project", {}).get("experiment_name", "image-classifier")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    parent_run_id = run_id_file.read_text().strip() if run_id_file.exists() else None

    with mlflow.start_run(run_name="finetune-pruned"):
        mlflow.set_tags(
            {
                "run_type": "finetune_pruned",
                **({"training_run_id": parent_run_id} if parent_run_id else {}),
            }
        )
        mlflow.log_params({"finetune_epochs": args.epochs})
        for k, v in report["pre_finetune"].items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"pre_{k}", float(v))
        for k, v in report["post_finetune"].items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"post_{k}", float(v))

    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Fine-tune report saved to %s", report_path)


if __name__ == "__main__":
    main()
