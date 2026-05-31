"""Exercise 1: Continual learning on SplitMNIST with replay + EWC.

The script uses Avalanche to create a two-experience SplitMNIST benchmark:
digits 0-4 followed by digits 5-9.  It first trains on digits 0-4, then
compares naive sequential fine-tuning against experience replay combined with
Elastic Weight Consolidation (EWC).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import random
import sys
import types
import warnings
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = PROJECT_DIR / ".cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(LOCAL_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)

# Keep generated cache files inside the project so the script works in
# restricted teaching or notebook environments.
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR / "xdg"))

# Avalanche imports TensorBoard at package import time; prefer its stub TF API
# in mixed environments.
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

try:
    (Path.home() / ".avalanche").mkdir(parents=True, exist_ok=True)
except OSError:
    local_home = LOCAL_CACHE_DIR / "home"
    local_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(local_home)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from avalanche.benchmarks.classic import SplitMNIST
except ImportError:  # Avalanche has exposed this in different modules.
    from avalanche.benchmarks import SplitMNIST

try:
    from avalanche.training.storage_policy import ReservoirSamplingBuffer
except ImportError:
    ReservoirSamplingBuffer = None


OLD_DIGITS = "0-4"
NEW_DIGITS = "5-9"


class ToTensorIfNeeded:
    """Convert PIL/ndarray inputs to tensors and normalize tensor-shaped MNIST."""

    def __call__(self, image) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            return transforms.functional.to_tensor(image)

        tensor = image.float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.max() > 1:
            tensor = tensor / 255.0
        return tensor


class MnistCNN(nn.Module):
    """Small CNN that predicts all ten MNIST classes throughout the stream."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class EWCRegularizer:
    """Diagonal Fisher EWC regularizer computed after Task 1."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        ewc_lambda: float,
        max_batches: int | None = None,
    ) -> None:
        self.ewc_lambda = ewc_lambda
        self.means = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.fisher = self._estimate_fisher(model, dataloader, device, max_batches)

    def _estimate_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        max_batches: int | None,
    ) -> dict[str, torch.Tensor]:
        model.eval()
        fisher = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        samples_seen = 0

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            data, target = unpack_batch(batch)
            data = data.to(device)
            target = target.to(device, dtype=torch.long)

            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(data), target)
            loss.backward()

            # EWC uses the diagonal Fisher approximation, estimated here as the
            # average squared gradient on Task 1 samples.
            batch_size = data.size(0)
            samples_seen += batch_size
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2) * batch_size

        if samples_seen == 0:
            raise RuntimeError("Cannot estimate EWC Fisher: no samples were seen.")

        for name in fisher:
            fisher[name] /= samples_seen
        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        penalty = torch.zeros((), device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                penalty = penalty + (self.fisher[name] * (param - self.means[name]).pow(2)).sum()
        return 0.5 * self.ewc_lambda * penalty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continual learning with naive fine-tuning vs replay + EWC on SplitMNIST."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="MNIST download/cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "continual_learning",
        help="Directory for plots and metric files.",
    )
    parser.add_argument("--task1-epochs", type=int, default=3, help="Epochs for digits 0-4.")
    parser.add_argument("--task2-epochs", type=int, default=5, help="Epochs for digits 5-9.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for current-task samples.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Evaluation batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=500,
        help="Replay samples from Task 1.",
    )
    parser.add_argument(
        "--replay-batch-size",
        type=int,
        default=None,
        help="Replay samples per Task 2 batch. Defaults to --batch-size for a 50/50 ratio.",
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=1000.0,
        help="EWC penalty strength. Larger values protect Task 1 more strongly.",
    )
    parser.add_argument(
        "--ewc-max-batches",
        type=int,
        default=None,
        help="Optional cap on batches used to estimate the Fisher matrix.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Training device.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and mps_available():
        torch.mps.manual_seed(seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if mps_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(device_arg)
    if device.type == "mps" and not mps_available():
        raise RuntimeError("MPS was requested, but it is not available.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")
    return device


def mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def make_split_mnist(args: argparse.Namespace):
    # Avalanche and torchvision MNIST wrappers differ by version: some return
    # PIL images, while others already return tensors.
    transform = transforms.Compose(
        [
            ToTensorIfNeeded(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    kwargs = {
        "n_experiences": 2,
        "return_task_id": False,
        "seed": args.seed,
        "fixed_class_order": list(range(10)),
        "shuffle": False,
        "train_transform": transform,
        "eval_transform": transform,
        "dataset_root": str(args.data_dir),
    }
    try:
        return SplitMNIST(**kwargs)
    except TypeError:
        kwargs.pop("dataset_root")
        return SplitMNIST(**kwargs)


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        drop_last=drop_last,
    )


def unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError(f"Expected a batch with at least data and target, got: {type(batch)!r}")
    return batch[0], batch[1]


def build_replay_dataset(task1_experience, buffer_size: int):
    if ReservoirSamplingBuffer is None:
        raise ImportError(
            "Avalanche ReservoirSamplingBuffer could not be imported. "
            "Install/update avalanche-lib to use this script."
        )

    # Let Avalanche own the replay-memory sampling policy while the training
    # loop below stays explicit for teaching and plotting.
    storage_policy = ReservoirSamplingBuffer(max_size=buffer_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stderr(io.StringIO()):
            storage_policy.update(SimpleNamespace(experience=task1_experience))
    return storage_policy.buffer


def train_task1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(model, loader, optimizer, device)
        history.append({"epoch": epoch, **metrics})
        print(f"Task 1 epoch {epoch:02d}/{epochs}: loss={metrics['loss']:.4f}")
    return history


def train_task2_naive(
    model: nn.Module,
    train_loader: DataLoader,
    old_test_loader: DataLoader,
    new_test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    initial_old_accuracy: float,
) -> list[dict[str, float]]:
    print("\nTask 2: naive sequential fine-tuning on digits 5-9")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = [
        evaluate_task2_epoch(
            0,
            model,
            old_test_loader,
            new_test_loader,
            device,
            initial_old_accuracy,
        )
    ]
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        eval_metrics = evaluate_task2_epoch(
            epoch,
            model,
            old_test_loader,
            new_test_loader,
            device,
            initial_old_accuracy,
        )
        history.append({**eval_metrics, "train_loss": train_metrics["loss"]})
        print_task2_metrics("Naive", epochs, history[-1])
    return history


def train_task2_replay_ewc(
    model: nn.Module,
    train_loader: DataLoader,
    replay_loader: DataLoader,
    old_test_loader: DataLoader,
    new_test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    ewc: EWCRegularizer,
    initial_old_accuracy: float,
) -> list[dict[str, float]]:
    print("\nTask 2: replay + EWC on digits 5-9 with replayed digits 0-4")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    replay_batches = cycle(replay_loader)
    history = [
        evaluate_task2_epoch(
            0,
            model,
            old_test_loader,
            new_test_loader,
            device,
            initial_old_accuracy,
        )
    ]
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            replay_batches=replay_batches,
            ewc=ewc,
        )
        eval_metrics = evaluate_task2_epoch(
            epoch,
            model,
            old_test_loader,
            new_test_loader,
            device,
            initial_old_accuracy,
        )
        history.append({**eval_metrics, **train_metrics})
        print_task2_metrics("Replay+EWC", epochs, history[-1])
    return history


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    replay_batches: Iterable | None = None,
    ewc: EWCRegularizer | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_ewc_loss = 0.0
    total_samples = 0

    for batch in loader:
        data, target = unpack_batch(batch)
        data = data.to(device)
        target = target.to(device, dtype=torch.long)

        if replay_batches is not None:
            # Concatenating current-task and replay samples gives the requested
            # 50/50 ratio when the two batch sizes match.
            replay_data, replay_target = unpack_batch(next(replay_batches))
            replay_data = replay_data.to(device)
            replay_target = replay_target.to(device, dtype=torch.long)
            data = torch.cat((data, replay_data), dim=0)
            target = torch.cat((target, replay_target), dim=0)

        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        ce_loss = F.cross_entropy(logits, target)
        ewc_loss = ewc.penalty(model) if ewc is not None else torch.zeros((), device=device)
        # Replay supplies old examples; EWC adds a parameter-space constraint
        # around the Task 1 solution.
        loss = ce_loss + ewc_loss
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        total_samples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_ce_loss += ce_loss.detach().item() * batch_size
        total_ewc_loss += ewc_loss.detach().item() * batch_size

    return {
        "loss": total_loss / total_samples,
        "ce_loss": total_ce_loss / total_samples,
        "ewc_loss": total_ewc_loss / total_samples,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        data, target = unpack_batch(batch)
        data = data.to(device)
        target = target.to(device, dtype=torch.long)
        logits = model(data)
        total_loss += F.cross_entropy(logits, target, reduction="sum").item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == target).sum().item()
        total += target.numel()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
    }


def evaluate_task2_epoch(
    epoch: int,
    model: nn.Module,
    old_test_loader: DataLoader,
    new_test_loader: DataLoader,
    device: torch.device,
    initial_old_accuracy: float,
) -> dict[str, float]:
    old_metrics = evaluate(model, old_test_loader, device)
    new_metrics = evaluate(model, new_test_loader, device)
    return {
        "epoch": epoch,
        "old_accuracy": old_metrics["accuracy"],
        "new_accuracy": new_metrics["accuracy"],
        "old_loss": old_metrics["loss"],
        "new_loss": new_metrics["loss"],
        "old_accuracy_drop": initial_old_accuracy - old_metrics["accuracy"],
    }


def print_task2_metrics(label: str, total_epochs: int, metrics: dict[str, float]) -> None:
    print(
        f"{label} epoch {int(metrics['epoch']):02d}/{total_epochs}: "
        f"old_acc={percent(metrics['old_accuracy'])} "
        f"(drop={percentage_points(metrics['old_accuracy_drop'])}), "
        f"new_acc={percent(metrics['new_accuracy'])}"
    )


def save_history_csv(path: Path, histories: dict[str, list[dict[str, float]]]) -> None:
    rows = []
    for method, history in histories.items():
        for item in history:
            rows.append({"method": method, **item})

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_task2_accuracy(path: Path, histories: dict[str, list[dict[str, float]]]) -> None:
    plt.figure(figsize=(8, 5))
    for label, history in histories.items():
        epochs = [item["epoch"] for item in history]
        old_acc = [item["old_accuracy"] * 100 for item in history]
        new_acc = [item["new_accuracy"] * 100 for item in history]
        plt.plot(epochs, old_acc, marker="o", label=f"{label}: old digits {OLD_DIGITS}")
        plt.plot(
            epochs,
            new_acc,
            marker="s",
            linestyle="--",
            label=f"{label}: new digits {NEW_DIGITS}",
        )

    plt.xlabel("Task 2 epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("SplitMNIST accuracy during Task 2")
    plt.xticks(range(max(item["epoch"] for history in histories.values() for item in history) + 1))
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_forgetting(path: Path, histories: dict[str, list[dict[str, float]]]) -> None:
    plt.figure(figsize=(8, 5))
    for label, history in histories.items():
        epochs = [item["epoch"] for item in history]
        drops = [item["old_accuracy_drop"] * 100 for item in history]
        plt.plot(epochs, drops, marker="o", label=label)

    plt.xlabel("Task 2 epoch")
    plt.ylabel("Accuracy drop on old digits 0-4 (percentage points)")
    plt.title("Catastrophic forgetting during Task 2")
    plt.xticks(range(max(item["epoch"] for history in histories.values() for item in history) + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_comparison_rows(
    initial_old: dict[str, float],
    initial_new: dict[str, float],
    naive_history: list[dict[str, float]],
    replay_ewc_history: list[dict[str, float]],
) -> list[dict[str, str | float]]:
    initial_old_acc = initial_old["accuracy"]
    rows = [
        {
            "stage": "After Task 1",
            "old_digits_accuracy": initial_old_acc,
            "new_digits_accuracy": initial_new["accuracy"],
            "old_digits_drop": 0.0,
        },
        {
            "stage": "Naive sequential after Task 2",
            "old_digits_accuracy": naive_history[-1]["old_accuracy"],
            "new_digits_accuracy": naive_history[-1]["new_accuracy"],
            "old_digits_drop": initial_old_acc - naive_history[-1]["old_accuracy"],
        },
        {
            "stage": "Replay + EWC after Task 2",
            "old_digits_accuracy": replay_ewc_history[-1]["old_accuracy"],
            "new_digits_accuracy": replay_ewc_history[-1]["new_accuracy"],
            "old_digits_drop": initial_old_acc - replay_ewc_history[-1]["old_accuracy"],
        },
    ]
    return rows


def print_comparison_table(rows: list[dict[str, str | float]]) -> None:
    print("\nFinal comparison")
    print("-" * 86)
    print(f"{'Stage':36} | {'Old digits 0-4':>14} | {'New digits 5-9':>14} | {'Old drop':>10}")
    print("-" * 86)
    for row in rows:
        print(
            f"{row['stage']:36} | "
            f"{percent(float(row['old_digits_accuracy'])):>14} | "
            f"{percent(float(row['new_digits_accuracy'])):>14} | "
            f"{percentage_points(float(row['old_digits_drop'])):>10}"
        )
    print("-" * 86)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def percentage_points(value: float) -> str:
    return f"{value * 100:.2f} pp"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    replay_batch_size = args.replay_batch_size or args.batch_size

    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    print(f"Replay ratio during Task 2: {args.batch_size}:{replay_batch_size} new:replay samples")

    benchmark = make_split_mnist(args)
    task1_train_exp = benchmark.train_stream[0]
    task2_train_exp = benchmark.train_stream[1]
    task1_test_exp = benchmark.test_stream[0]
    task2_test_exp = benchmark.test_stream[1]

    task1_train_loader = make_loader(
        task1_train_exp.dataset,
        args.batch_size,
        True,
        args.num_workers,
        make_generator(args.seed + 1),
    )
    task2_train_loader_naive = make_loader(
        task2_train_exp.dataset,
        args.batch_size,
        True,
        args.num_workers,
        make_generator(args.seed + 2),
    )
    task2_train_loader_replay = make_loader(
        task2_train_exp.dataset,
        args.batch_size,
        True,
        args.num_workers,
        make_generator(args.seed + 2),
    )
    fisher_loader = make_loader(
        task1_train_exp.dataset,
        args.batch_size,
        False,
        args.num_workers,
        make_generator(args.seed + 3),
    )
    task1_test_loader = make_loader(
        task1_test_exp.dataset,
        args.eval_batch_size,
        False,
        args.num_workers,
        make_generator(args.seed + 4),
    )
    task2_test_loader = make_loader(
        task2_test_exp.dataset,
        args.eval_batch_size,
        False,
        args.num_workers,
        make_generator(args.seed + 5),
    )

    print(
        "Avalanche SplitMNIST experiences: "
        f"Task 1 classes={task1_train_exp.classes_in_this_experience}, "
        f"Task 2 classes={task2_train_exp.classes_in_this_experience}"
    )

    base_model = MnistCNN().to(device)
    print(f"\nTask 1: initial training on digits {OLD_DIGITS}")
    task1_history = train_task1(
        base_model,
        task1_train_loader,
        device,
        args.task1_epochs,
        args.lr,
        args.weight_decay,
    )
    initial_old = evaluate(base_model, task1_test_loader, device)
    initial_new = evaluate(base_model, task2_test_loader, device)
    print(
        f"After Task 1: old digits {OLD_DIGITS} acc={percent(initial_old['accuracy'])}, "
        f"new digits {NEW_DIGITS} acc={percent(initial_new['accuracy'])}"
    )

    # Start both Task 2 variants from the exact same Task 1 checkpoint so the
    # comparison isolates the training method.
    task1_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in base_model.state_dict().items()
    }

    naive_model = MnistCNN().to(device)
    naive_model.load_state_dict(task1_state)
    naive_history = train_task2_naive(
        naive_model,
        task2_train_loader_naive,
        task1_test_loader,
        task2_test_loader,
        device,
        args.task2_epochs,
        args.lr,
        args.weight_decay,
        initial_old["accuracy"],
    )

    print("\nBuilding Avalanche replay buffer from Task 1 samples")
    replay_dataset = build_replay_dataset(task1_train_exp, args.replay_buffer_size)
    effective_replay_batch_size = min(replay_batch_size, len(replay_dataset))
    replay_loader = make_loader(
        replay_dataset,
        effective_replay_batch_size,
        True,
        args.num_workers,
        make_generator(args.seed + 6),
        drop_last=False,
    )
    print(
        f"Replay buffer size: {len(replay_dataset)} samples; "
        f"replay batch size: {effective_replay_batch_size}"
    )

    replay_ewc_model = MnistCNN().to(device)
    replay_ewc_model.load_state_dict(task1_state)
    print(f"Estimating EWC Fisher information from Task 1 with lambda={args.ewc_lambda:g}")
    ewc = EWCRegularizer(
        replay_ewc_model,
        fisher_loader,
        device,
        args.ewc_lambda,
        max_batches=args.ewc_max_batches,
    )
    replay_ewc_history = train_task2_replay_ewc(
        replay_ewc_model,
        task2_train_loader_replay,
        replay_loader,
        task1_test_loader,
        task2_test_loader,
        device,
        args.task2_epochs,
        args.lr,
        args.weight_decay,
        ewc,
        initial_old["accuracy"],
    )

    comparison_rows = make_comparison_rows(
        initial_old,
        initial_new,
        naive_history,
        replay_ewc_history,
    )
    print_comparison_table(comparison_rows)

    histories = {"Naive": naive_history, "Replay + EWC": replay_ewc_history}
    save_history_csv(args.output_dir / "task2_history.csv", histories)
    save_comparison_csv(args.output_dir / "comparison_table.csv", comparison_rows)
    plot_task2_accuracy(args.output_dir / "task2_accuracy.png", histories)
    plot_forgetting(args.output_dir / "old_digit_forgetting.png", histories)

    metrics = {
        "config": {
            "seed": args.seed,
            "task1_epochs": args.task1_epochs,
            "task2_epochs": args.task2_epochs,
            "batch_size": args.batch_size,
            "replay_batch_size": replay_batch_size,
            "replay_buffer_size": args.replay_buffer_size,
            "ewc_lambda": args.ewc_lambda,
            "device": str(device),
        },
        "task1_history": task1_history,
        "comparison": comparison_rows,
        "task2_history": histories,
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\nSaved plots and metrics to: {args.output_dir.resolve()}")
    print(f" - {args.output_dir / 'task2_accuracy.png'}")
    print(f" - {args.output_dir / 'old_digit_forgetting.png'}")
    print(f" - {args.output_dir / 'comparison_table.csv'}")
    print(f" - {args.output_dir / 'task2_history.csv'}")
    print(f" - {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
