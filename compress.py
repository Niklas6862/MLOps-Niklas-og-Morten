"""Model compression entry point.

Usage::

    # Dynamic INT8 quantization + benchmark
    python compress.py --method dynamic_quant

    # Prune 30% of weights, save pruned model
    python compress.py --method prune --prune-amount 0.3

    # Prune 50%, then fine-tune to recover accuracy
    python compress.py --method prune --prune-amount 0.5 --finetune

    # Sweep pruning amounts (reports accuracy at each level)
    python compress.py --method prune_sweep

    # Baseline benchmark only (no compression)
    python compress.py --method none
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments

from src.compress import (
    apply_dynamic_quantization,
    apply_pruning,
    benchmark,
    evaluate_accuracy,
    make_pruning_permanent,
)
from src.config import load_config
from src.data import collate_fn, load_image_dataset, preprocess_dataset
from src.train import compute_metrics
from src.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
]

PRUNE_SWEEP_AMOUNTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress and benchmark an image classifier")
    parser.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--model-dir", default="models/artifacts")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the compressed model (default: <model-dir>_compressed)",
    )
    parser.add_argument(
        "--method",
        choices=["none", "dynamic_quant", "prune", "prune_sweep"],
        default="dynamic_quant",
        help="Compression method (default: dynamic_quant)",
    )
    parser.add_argument(
        "--prune-amount",
        type=float,
        default=0.3,
        metavar="FRAC",
        help="Fraction of weights to prune, e.g. 0.3 = 30%% (default: 0.3)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune the pruned model for --finetune-epochs to recover accuracy",
    )
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--benchmark-batches", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        default="models/artifacts/compression_report.json",
        help="Path to write the JSON compression report",
    )
    return parser.parse_args()


def _build_loaders(
    cfg: dict, processor: AutoImageProcessor, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for fine-tuning and evaluation."""
    raw = load_image_dataset(cfg["dataset"])
    processed = preprocess_dataset(raw, processor, cfg["dataset"])
    processed.set_format("torch", columns=["pixel_values", "labels"])
    train_loader = DataLoader(
        processed["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        processed["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader


def _finetune(
    model: torch.nn.Module,
    processor: AutoImageProcessor,
    cfg: dict,
    epochs: int,
    output_dir: Path,
    device: str,
) -> None:
    """Fine-tune *model* in-place using HuggingFace Trainer."""
    logger.info("Fine-tuning pruned model for %d epoch(s) …", epochs)
    raw = load_image_dataset(cfg["dataset"])
    processed = preprocess_dataset(raw, processor, cfg["dataset"])
    processed.set_format("torch", columns=["pixel_values", "labels"])

    training_cfg = cfg.get("training", {})
    ft_args = TrainingArguments(
        output_dir=str(output_dir / "_ft_ckpt"),
        num_train_epochs=epochs,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 32),
        learning_rate=training_cfg.get("learning_rate", 2e-5),
        eval_strategy="epoch",
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        # fp16 only when training on CUDA
        fp16=device == "cuda" and torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=ft_args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    trainer.train()


def _save_model(model: torch.nn.Module, processor: AutoImageProcessor, src_dir: Path, dst_dir: Path) -> None:
    """Save model + processor to *dst_dir*, copying config files from *src_dir*."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst_dir)
    processor.save_pretrained(dst_dir)
    # Copy run_id.txt so downstream stages can log to the same MLflow run
    run_id_file = src_dir / "run_id.txt"
    if run_id_file.exists():
        shutil.copy(run_id_file, dst_dir / "run_id.txt")
    logger.info("Compressed model saved to %s", dst_dir)


def _log_to_mlflow(report: dict, model_dir: str) -> None:
    run_id_file = Path(model_dir) / "run_id.txt"
    if not run_id_file.exists():
        return
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    parent_run_id = run_id_file.read_text().strip()
    method = report.get("method", "unknown")

    # Nested child run — appears indented under the training run in the MLflow UI,
    # with its own metric columns so baseline vs. compressed are directly comparable.
    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name=f"compression-{method}", nested=True) as child:
            mlflow.set_tags({"compressed": "true", "compression_method": method})

            baseline = report.get("baseline", {})
            compressed = report.get("compressed", {})

            # Log baseline and compressed metrics with clean names (no section prefix)
            # so they appear as columns in the MLflow comparison view.
            for k, v in baseline.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"baseline_{k}", float(v))
            for k, v in compressed.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"compressed_{k}", float(v))

            if "finetuned" in report:
                for k, v in report["finetuned"].items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"finetuned_{k}", float(v))

            mlflow.log_params({
                "compression_method": method,
                "prune_amount": report.get("compressed", {}).get("pruned_amount", "n/a"),
            })

            logger.info("Compression results logged as child run %s under %s", child.info.run_id, parent_run_id)


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    setup_logging(cfg.get("project", {}).get("log_level", "INFO"))
    set_seed(cfg.get("project", {}).get("seed", 42))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir.parent / (model_dir.name + "_compressed")
    batch_size = cfg.get("training", {}).get("per_device_eval_batch_size", 32)

    logger.info("Loading model from '%s' …", model_dir)
    processor = AutoImageProcessor.from_pretrained(model_dir)

    def _fresh_model() -> AutoModelForImageClassification:
        m = AutoModelForImageClassification.from_pretrained(model_dir)
        m.eval()
        return m

    _, test_loader = _build_loaders(cfg, processor, batch_size)

    report: dict = {"method": args.method, "model_dir": str(model_dir)}

    # ── Baseline ──────────────────────────────────────────────────────────────
    logger.info("Benchmarking baseline …")
    base_model = _fresh_model()
    base_acc = evaluate_accuracy(base_model, test_loader, device)
    base_lat = benchmark(base_model, test_loader, device, n_batches=args.benchmark_batches)
    report["baseline"] = {"accuracy": round(base_acc, 4), **base_lat}
    logger.info(
        "Baseline: acc=%.4f  throughput=%.1f fps",
        base_acc,
        base_lat.get("throughput_fps", 0),
    )
    del base_model

    # ── Benchmark only ─────────────────────────────────────────────────────────
    if args.method == "none":
        _finalize(report, args.output, args.model_dir)
        return

    # ── Pruning sweep ──────────────────────────────────────────────────────────
    if args.method == "prune_sweep":
        sweep: list[dict] = []
        for amount in PRUNE_SWEEP_AMOUNTS:
            m = _fresh_model()
            stats = apply_pruning(m, amount)
            make_pruning_permanent(m)
            acc = evaluate_accuracy(m, test_loader, device)
            entry = {"prune_amount": amount, "accuracy": round(acc, 4), **stats}
            sweep.append(entry)
            logger.info("amount=%.0f%%  acc=%.4f  Δacc=%.4f", amount * 100, acc, acc - base_acc)
            del m
        report["prune_sweep"] = sweep
        _finalize(report, args.output, args.model_dir)
        return

    # ── Single compression run ─────────────────────────────────────────────────
    model = _fresh_model()
    prune_stats: dict = {}

    if args.method == "prune":
        prune_stats = apply_pruning(model, args.prune_amount)
        make_pruning_permanent(model)

        if args.finetune:
            _finetune(model, processor, cfg, args.finetune_epochs, output_dir, device)
            ft_acc = evaluate_accuracy(model, test_loader, device)
            ft_lat = benchmark(model, test_loader, device, n_batches=args.benchmark_batches)
            report["finetuned"] = {
                "accuracy": round(ft_acc, 4),
                "accuracy_recovered": round(ft_acc - base_acc, 4),
                "finetune_epochs": args.finetune_epochs,
                **ft_lat,
            }
            logger.info(
                "After fine-tuning: acc=%.4f (Δbaseline=%.4f)",
                ft_acc,
                ft_acc - base_acc,
            )

        _save_model(model, processor, model_dir, output_dir)

    elif args.method == "dynamic_quant":
        # Dynamic quantization is CPU-only — benchmark on CPU regardless of training device
        model = apply_dynamic_quantization(model)
        device = "cpu"

    # ── Compressed benchmark ───────────────────────────────────────────────────
    logger.info("Benchmarking compressed model …")
    comp_acc = evaluate_accuracy(model, test_loader, device)
    comp_lat = benchmark(model, test_loader, device, n_batches=args.benchmark_batches)

    speedup = (
        round(comp_lat["throughput_fps"] / base_lat["throughput_fps"], 2)
        if base_lat.get("throughput_fps") and comp_lat.get("throughput_fps")
        else None
    )
    report["compressed"] = {
        "accuracy": round(comp_acc, 4),
        "accuracy_drop": round(base_acc - comp_acc, 4),
        **comp_lat,
        **prune_stats,
        **({"speedup_x": speedup} if speedup else {}),
    }
    logger.info(
        "Compressed: acc=%.4f (drop=%.4f)  throughput=%.1f fps%s",
        comp_acc,
        base_acc - comp_acc,
        comp_lat.get("throughput_fps", 0),
        f"  speedup={speedup:.2f}x" if speedup else "",
    )

    _finalize(report, args.output, args.model_dir)


def _finalize(report: dict, output_path: str, model_dir: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Compression report saved to %s", path)
    _log_to_mlflow(report, model_dir)


if __name__ == "__main__":
    main()
