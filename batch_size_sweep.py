from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from src.compress import apply_dynamic_quantization
from src.config import load_config
from src.data import collate_fn, load_image_dataset, preprocess_dataset
from src.utils import setup_logging
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
]
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
_WARMUP = 2
_TIMED = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep batch sizes and record throughput/latency")
    parser.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--model-dir", default="models/artifacts")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--quantized", action="store_true", help="Apply dynamic INT8 quantization (forces CPU)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="models/artifacts/batch_sweep_report.json")
    parser.add_argument("--plot", default="models/artifacts/batch_sweep_plot.png")
    return parser.parse_args()


def _bench_one(model: torch.nn.Module, dataset, batch_size: int, device: str) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    latencies: list[float] = []
    n_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pv = batch["pixel_values"].to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(pixel_values=pv)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            if i >= _WARMUP:
                latencies.append(elapsed)
                n_samples += pv.shape[0]
            if i >= _WARMUP + _TIMED - 1:
                break

    if not latencies:
        return {}

    arr = np.array(latencies)
    return {
        "batch_size": batch_size,
        "throughput_fps": round(float(n_samples / arr.sum()), 1),
        "mean_latency_ms": round(float(arr.mean() * 1000), 2),
        "p95_latency_ms": round(float(np.percentile(arr, 95) * 1000), 2),
    }


def _save_plot(results: list[dict], path: str) -> None:
    plt.switch_backend("Agg")
    batch_sizes = [r["batch_size"] for r in results]
    throughputs = [r["throughput_fps"] for r in results]
    latencies = [r["mean_latency_ms"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(batch_sizes, throughputs, marker="o")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Throughput (images/s)")
    ax1.set_title("Throughput vs Batch Size")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)

    ax2.plot(batch_sizes, latencies, marker="o", color="orange")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Mean batch latency (ms)")
    ax2.set_title("Latency vs Batch Size")
    ax2.set_xscale("log", base=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Batch sweep plot saved to %s", path)


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(*args.config)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.quantized:
        device = "cpu"

    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir)
    model.eval()
    if args.quantized:
        model = apply_dynamic_quantization(model)
    model.to(device)

    raw = load_image_dataset(cfg["dataset"])
    processed = preprocess_dataset(raw, processor, cfg["dataset"])
    processed.set_format("torch", columns=["pixel_values", "labels"])
    test_dataset = processed["test"]

    results = []
    for bs in args.batch_sizes:
        entry = _bench_one(model, test_dataset, bs, device)
        if entry:
            results.append(entry)
            logger.info(
                "batch_size=%d  throughput=%.1f fps  latency=%.2f ms",
                bs,
                entry["throughput_fps"],
                entry["mean_latency_ms"],
            )

    report = {
        "model_dir": args.model_dir,
        "device": device,
        "quantized": args.quantized,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Batch sweep report saved to %s", args.output)

    _save_plot(results, args.plot)

    experiment_name = cfg.get("project", {}).get("experiment_name", "image-classifier")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_id_file = Path(args.model_dir) / "run_id.txt"
    parent_run_id = run_id_file.read_text().strip() if run_id_file.exists() else None

    with mlflow.start_run(run_name="batch-size-sweep") as run:
        mlflow.set_tags({
            "run_type": "batch_sweep",
            "device": device,
            "quantized": str(args.quantized),
            **({"training_run_id": parent_run_id} if parent_run_id else {}),
        })
        for entry in results:
            bs = entry["batch_size"]
            mlflow.log_metric("throughput_fps", entry["throughput_fps"], step=bs)
            mlflow.log_metric("mean_latency_ms", entry["mean_latency_ms"], step=bs)
            mlflow.log_metric("p95_latency_ms", entry["p95_latency_ms"], step=bs)
        mlflow.log_artifact(args.plot)
        mlflow.log_artifact(args.output)
        logger.info("Batch sweep logged as MLflow run %s", run.info.run_id)


if __name__ == "__main__":
    main()
