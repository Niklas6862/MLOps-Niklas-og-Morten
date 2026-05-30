"""Batch inference entry point.

Usage::

    # Classify all images in a directory
    python batch_inference.py --image-dir path/to/images/

    # Run on the test split and measure accuracy
    python batch_inference.py --dataset-split test

    # Apply dynamic INT8 quantization before inference
    python batch_inference.py --dataset-split test --quantized

    # Save full predictions to JSON
    python batch_inference.py --dataset-split test --output predictions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.compress import apply_dynamic_quantization
from src.config import load_config
from src.data import collate_fn, load_image_dataset, preprocess_dataset
from src.utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class ImageDirDataset(Dataset):
    """Dataset that reads all images from a flat directory."""

    def __init__(self, image_dir: Path, processor: AutoImageProcessor) -> None:
        self.paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.paths[idx]).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "path": str(self.paths[idx])}


def _collate_dir(examples: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]),
        "paths": [ex["path"] for ex in examples],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference on an image classifier")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-dir", type=Path, help="Directory of images to classify")
    src.add_argument(
        "--dataset-split",
        choices=["train", "validation", "test"],
        help="Evaluate on a dataset split (measures accuracy)",
    )
    parser.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--model-dir", default="models/artifacts")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Apply dynamic INT8 quantization before inference (forces CPU)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--output", default=None, help="Write full predictions JSON to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(*args.config)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.quantized:
        device = "cpu"

    logger.info("Loading model from '%s' …", args.model_dir)
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir)
    model.eval()

    if args.quantized:
        logger.info("Applying dynamic INT8 quantization …")
        model = apply_dynamic_quantization(model)

    model.to(device)
    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}

    # ── Build dataloader ───────────────────────────────────────────────────────
    if args.image_dir:
        ds = ImageDirDataset(args.image_dir, processor)
        loader: DataLoader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_dir
        )
        has_labels = False
    else:
        raw = load_image_dataset(cfg["dataset"])
        processed = preprocess_dataset(raw, processor, cfg["dataset"])
        processed[args.dataset_split].set_format("torch", columns=["pixel_values", "labels"])
        loader = DataLoader(
            processed[args.dataset_split],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        has_labels = True

    # ── Run inference ─────────────────────────────────────────────────────────
    all_preds: list[dict] = []
    correct = total = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            pv = batch["pixel_values"].to(device)
            outputs = model(pixel_values=pv)
            probs = torch.softmax(outputs.logits, dim=-1)
            k = min(args.top_k, probs.shape[-1])
            top_probs, top_ids = torch.topk(probs, k, dim=-1)

            if has_labels:
                labels = batch["labels"].to(device)
                correct += int((top_ids[:, 0] == labels).sum())
                total += labels.size(0)

            for i in range(pv.shape[0]):
                entry: dict = {
                    "predictions": [
                        {
                            "label": id2label[int(top_ids[i, j])],
                            "score": round(float(top_probs[i, j]), 4),
                        }
                        for j in range(k)
                    ]
                }
                if not has_labels and "paths" in batch:
                    entry["path"] = batch["paths"][i]
                all_preds.append(entry)

    elapsed = time.perf_counter() - t_start
    n = len(all_preds)
    summary: dict = {
        "n_images": n,
        "total_time_s": round(elapsed, 3),
        "throughput_fps": round(n / elapsed, 1) if elapsed else 0,
        "mean_latency_ms": round(elapsed / n * 1000, 2) if n else 0,
        "quantized": args.quantized,
        "device": device,
    }
    if has_labels and total:
        summary["accuracy"] = round(correct / total, 4)

    logger.info(
        "Processed %d images in %.2fs — %.1f fps%s",
        n,
        elapsed,
        summary["throughput_fps"],
        f"  accuracy={summary['accuracy']:.4f}" if "accuracy" in summary else "",
    )

    result = {"summary": summary, "predictions": all_preds}
    print(json.dumps({"summary": summary}, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Full predictions saved to %s", args.output)


if __name__ == "__main__":
    main()
