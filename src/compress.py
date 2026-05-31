"""Model compression utilities: dynamic quantization and magnitude pruning."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Return *model* with Linear layers quantized to INT8 (CPU inference only)."""
    logger.info("Applying dynamic INT8 quantization to Linear layers …")
    return torch.quantization.quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)


def apply_pruning(model: nn.Module, amount: float) -> dict[str, Any]:
    """Globally prune *amount* fraction of Linear weights by L1 magnitude.

    Masks are applied but not baked in — call make_pruning_permanent() before saving.
    Returns sparsity statistics.
    """
    logger.info("Applying %.0f%% global L1-unstructured pruning …", amount * 100)
    params = [
        (m, "weight")
        for _, m in model.named_modules()
        if isinstance(m, nn.Linear) and m.weight.requires_grad
    ]
    if not params:
        logger.warning("No prunable Linear layers found.")
        return {}

    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)

    total = sum(m.weight_mask.numel() for m, _ in params)
    zeroed = sum(int((m.weight_mask == 0).sum()) for m, _ in params)
    stats: dict[str, Any] = {
        "pruned_amount": amount,
        "total_weights": total,
        "zeroed_weights": zeroed,
        "actual_sparsity": round(zeroed / total, 4) if total else 0.0,
    }
    logger.info(
        "Sparsity: %.2f%% (%d/%d weights zeroed)",
        stats["actual_sparsity"] * 100,
        zeroed,
        total,
    )
    return stats


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """Bake pruning masks into weight tensors and remove reparametrisation hooks."""
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            try:
                prune.remove(m, "weight")
            except ValueError:
                pass
    return model


def benchmark(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    warmup: int = 2,
    n_batches: int = 30,
) -> dict[str, float]:
    """Measure per-batch latency and overall throughput."""
    model.eval()
    model.to(device)
    batches = list(dataloader)
    warmup = min(warmup, max(0, len(batches) - 1))
    latencies: list[float] = []
    n_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(batches):
            pv = batch["pixel_values"].to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(pixel_values=pv)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            if i >= warmup:
                latencies.append(elapsed)
                n_samples += pv.shape[0]
            if i >= warmup + n_batches - 1:
                break

    if not latencies:
        return {}

    arr = np.array(latencies)
    return {
        "mean_batch_ms": round(float(arr.mean() * 1000), 2),
        "p95_batch_ms": round(float(np.percentile(arr, 95) * 1000), 2),
        "throughput_fps": round(float(n_samples / arr.sum()), 1),
        "n_samples_timed": n_samples,
    }


def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> float:
    """Return classification accuracy (0–1) over all batches in *dataloader*."""
    model.eval()
    model.to(device)
    correct = total = 0

    with torch.no_grad():
        for batch in dataloader:
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            preds = model(pixel_values=pv).logits.argmax(-1)
            correct += int((preds == labels).sum())
            total += labels.size(0)

    return correct / total if total else 0.0
