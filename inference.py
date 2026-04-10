"""Inference entry point.

Usage::

    python inference.py path/to/image.jpg
    python inference.py path/to/image.jpg --model-dir models/artifacts --top-k 3

Loads a fine-tuned model from disk, runs it on a single image, and prints
top-k predictions as JSON.
"""
from __future__ import annotations

import argparse
import json
import logging

from src.infer import load_pipeline, predict_image
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("image_path", help="Path to the input image file")
    parser.add_argument(
        "--model-dir",
        default="models/artifacts",
        help="Directory containing the fine-tuned model (default: models/artifacts)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to return (default: 3)",
    )
    parser.add_argument("--device", default=None, help="Force device: 'cpu' or 'cuda'")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    model, processor = load_pipeline(args.model_dir)
    results = predict_image(
        image_path=args.image_path,
        model=model,
        processor=processor,
        top_k=args.top_k,
        device=args.device,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
