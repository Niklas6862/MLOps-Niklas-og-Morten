"""Exercise 2: MNIST class unlearning with gradient ascent and confusion.

This script trains a normal MNIST classifier, copies the trained weights, and
then compares two targeted unlearning procedures for one digit:

1. Gradient ascent on the target-class loss.
2. Confuse unlearning, where target-class samples are trained toward wrong labels.

Both methods include an optional retain-class loss term so the model is nudged to
keep useful behavior on the non-target digits while forgetting the target digit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from itertools import cycle
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = PROJECT_DIR / ".cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(LOCAL_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)

# Keep generated cache files inside the project so the script works in
# restricted teaching or notebook environments.
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class MnistCNN(nn.Module):
    """Small CNN classifier for all ten MNIST classes."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Targeted MNIST class unlearning with gradient ascent and confuse unlearning."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--target-class",
        type=int,
        default=6,
        choices=range(10),
        help="Digit to forget.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="MNIST download/cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "unlearning_gradient_ascent",
        help="Directory for plots and metric files.",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=3,
        help="Baseline classifier training epochs.",
    )
    parser.add_argument("--unlearn-epochs", type=int, default=3, help="Unlearning epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Baseline and forget-class batch size.",
    )
    parser.add_argument(
        "--retain-batch-size",
        type=int,
        default=None,
        help="Retain-class batch size during unlearning. Defaults to --batch-size.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Evaluation batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Baseline Adam learning rate.")
    parser.add_argument(
        "--ascent-lr",
        type=float,
        default=2e-4,
        help="Gradient ascent Adam learning rate.",
    )
    parser.add_argument(
        "--confuse-lr",
        type=float,
        default=2e-4,
        help="Confuse unlearning Adam learning rate.",
    )
    parser.add_argument(
        "--retain-weight",
        type=float,
        default=3.0,
        help=(
            "Weight for normal retain-class loss during unlearning. "
            "Use 0 for target-only unlearning."
        ),
    )
    parser.add_argument(
        "--forget-weight",
        type=float,
        default=1.0,
        help="Weight for target-class ascent/confusion loss during unlearning.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help="Gradient clipping norm for unlearning. Use 0 to disable.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Training device.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests. Defaults to full training set.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests. Defaults to full test set.",
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


def mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


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


def make_mnist_datasets(data_dir: Path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def get_targets(dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        # Subset indices refer to the parent dataset, so targets must be
        # filtered recursively when smoke-test caps are active.
        parent_targets = get_targets(dataset.dataset)
        return parent_targets[torch.as_tensor(dataset.indices)]
    return torch.as_tensor(dataset.targets)


def subset_by_class(
    dataset,
    target_class: int,
    keep_target: bool,
    max_samples: int | None = None,
) -> Subset:
    targets = get_targets(dataset)
    if keep_target:
        indices = torch.nonzero(targets == target_class, as_tuple=False).flatten()
    else:
        indices = torch.nonzero(targets != target_class, as_tuple=False).flatten()

    if max_samples is not None:
        indices = indices[:max_samples]
    return Subset(dataset, indices.tolist())


def maybe_cap_dataset(dataset, max_samples: int | None) -> Subset | object:
    if max_samples is None:
        return dataset
    return Subset(dataset, list(range(min(max_samples, len(dataset)))))


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
    )


def train_classifier(
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
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for data, target in loader:
            data = data.to(device)
            target = target.to(device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * target.numel()
            total_correct += (logits.argmax(dim=1) == target).sum().item()
            total += target.numel()

        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_accuracy": total_correct / total,
        }
        history.append(metrics)
        print(
            f"Baseline epoch {epoch:02d}/{epochs}: "
            f"loss={metrics['train_loss']:.4f}, acc={percent(metrics['train_accuracy'])}"
        )
    return history


def run_unlearning(
    method: str,
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    eval_loaders: dict[str, DataLoader],
    device: torch.device,
    args: argparse.Namespace,
    lr: float,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    retain_batches = cycle(retain_loader)
    confuse_generator = make_generator(args.seed + 500)
    history = [
        evaluate_snapshot(0, method, model, eval_loaders, device, args.target_class)
    ]

    for epoch in range(1, args.unlearn_epochs + 1):
        model.train()
        running_loss = 0.0
        running_forget_loss = 0.0
        running_retain_loss = 0.0
        total = 0

        for forget_data, forget_target in forget_loader:
            forget_data = forget_data.to(device)
            forget_target = forget_target.to(device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            forget_logits = model(forget_data)
            true_forget_loss = F.cross_entropy(forget_logits, forget_target)

            if method == "Gradient Ascent":
                # Maximize the correct-label loss on the forgotten class.
                forget_loss = -args.forget_weight * true_forget_loss
            elif method == "Confuse":
                # Train the forgotten class toward deliberately wrong labels.
                wrong_target = make_wrong_labels(forget_target, confuse_generator, device)
                forget_loss = args.forget_weight * F.cross_entropy(forget_logits, wrong_target)
            else:
                raise ValueError(f"Unknown unlearning method: {method}")

            retain_loss = torch.zeros((), device=device)
            if args.retain_weight > 0:
                # The retain term keeps non-target classes anchored while the
                # target-class behavior is being degraded.
                retain_data, retain_target = next(retain_batches)
                retain_data = retain_data.to(device)
                retain_target = retain_target.to(device, dtype=torch.long)
                retain_loss = F.cross_entropy(model(retain_data), retain_target)

            loss = forget_loss + args.retain_weight * retain_loss
            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            batch_size = forget_target.numel()
            running_loss += loss.detach().item() * batch_size
            running_forget_loss += true_forget_loss.detach().item() * batch_size
            running_retain_loss += retain_loss.detach().item() * batch_size
            total += batch_size

        snapshot = evaluate_snapshot(
            epoch,
            method,
            model,
            eval_loaders,
            device,
            args.target_class,
        )
        snapshot.update(
            {
                "train_loss": running_loss / total,
                "forget_ce_loss": running_forget_loss / total,
                "retain_ce_loss": running_retain_loss / total,
            }
        )
        history.append(snapshot)
        print_unlearning_metrics(method, args.unlearn_epochs, snapshot)

    return history


def make_wrong_labels(
    target: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    shifts = torch.randint(1, 10, target.shape, generator=generator)
    wrong = (target.detach().cpu() + shifts) % 10
    return wrong.to(device=device, dtype=torch.long)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data = data.to(device)
        target = target.to(device, dtype=torch.long)
        logits = model(data)
        total_loss += F.cross_entropy(logits, target, reduction="sum").item()
        correct += (logits.argmax(dim=1) == target).sum().item()
        total += target.numel()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
    }


@torch.no_grad()
def per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[int, float]:
    model.eval()
    correct = torch.zeros(10, dtype=torch.long)
    total = torch.zeros(10, dtype=torch.long)

    for data, target in loader:
        data = data.to(device)
        predictions = model(data).argmax(dim=1).cpu()
        target_cpu = target.long()
        for class_idx in range(10):
            mask = target_cpu == class_idx
            total[class_idx] += mask.sum()
            correct[class_idx] += (predictions[mask] == target_cpu[mask]).sum()

    return {
        class_idx: (
            correct[class_idx].item() / total[class_idx].item()
            if total[class_idx]
            else 0.0
        )
        for class_idx in range(10)
    }


def evaluate_snapshot(
    epoch: int,
    method: str,
    model: nn.Module,
    eval_loaders: dict[str, DataLoader],
    device: torch.device,
    target_class: int,
) -> dict[str, float | str]:
    all_metrics = evaluate(model, eval_loaders["all"], device)
    target_metrics = evaluate(model, eval_loaders["forget"], device)
    retain_metrics = evaluate(model, eval_loaders["retain"], device)
    return {
        "method": method,
        "epoch": epoch,
        "target_class": target_class,
        "overall_accuracy": all_metrics["accuracy"],
        "target_accuracy": target_metrics["accuracy"],
        "retain_accuracy": retain_metrics["accuracy"],
        "overall_loss": all_metrics["loss"],
        "target_loss": target_metrics["loss"],
        "retain_loss": retain_metrics["loss"],
    }


def print_unlearning_metrics(
    method: str,
    total_epochs: int,
    metrics: dict[str, float | str],
) -> None:
    print(
        f"{method} epoch {int(metrics['epoch']):02d}/{total_epochs}: "
        f"target_acc={percent(float(metrics['target_accuracy']))}, "
        f"retain_acc={percent(float(metrics['retain_accuracy']))}, "
        f"overall_acc={percent(float(metrics['overall_accuracy']))}"
    )


def make_comparison_rows(
    baseline: dict[str, float | str],
    ascent_final: dict[str, float | str],
    confuse_final: dict[str, float | str],
) -> list[dict[str, str | float]]:
    return [
        {
            "stage": "Baseline",
            "overall_accuracy": float(baseline["overall_accuracy"]),
            "target_accuracy": float(baseline["target_accuracy"]),
            "retain_accuracy": float(baseline["retain_accuracy"]),
            "target_accuracy_drop": 0.0,
            "retain_accuracy_drop": 0.0,
        },
        {
            "stage": "Gradient ascent unlearning",
            "overall_accuracy": float(ascent_final["overall_accuracy"]),
            "target_accuracy": float(ascent_final["target_accuracy"]),
            "retain_accuracy": float(ascent_final["retain_accuracy"]),
            "target_accuracy_drop": (
                float(baseline["target_accuracy"])
                - float(ascent_final["target_accuracy"])
            ),
            "retain_accuracy_drop": (
                float(baseline["retain_accuracy"])
                - float(ascent_final["retain_accuracy"])
            ),
        },
        {
            "stage": "Confuse unlearning",
            "overall_accuracy": float(confuse_final["overall_accuracy"]),
            "target_accuracy": float(confuse_final["target_accuracy"]),
            "retain_accuracy": float(confuse_final["retain_accuracy"]),
            "target_accuracy_drop": (
                float(baseline["target_accuracy"])
                - float(confuse_final["target_accuracy"])
            ),
            "retain_accuracy_drop": (
                float(baseline["retain_accuracy"])
                - float(confuse_final["retain_accuracy"])
            ),
        },
    ]


def print_comparison_table(rows: list[dict[str, str | float]], target_class: int) -> None:
    print("\nFinal comparison")
    print("-" * 112)
    print(
        f"{'Stage':34} | {'Overall':>10} | "
        f"{'Target ' + str(target_class):>10} | {'Retain':>10} | "
        f"{'Target drop':>12} | {'Retain drop':>12}"
    )
    print("-" * 112)
    for row in rows:
        print(
            f"{row['stage']:34} | "
            f"{percent(float(row['overall_accuracy'])):>10} | "
            f"{percent(float(row['target_accuracy'])):>10} | "
            f"{percent(float(row['retain_accuracy'])):>10} | "
            f"{percentage_points(float(row['target_accuracy_drop'])):>12} | "
            f"{percentage_points(float(row['retain_accuracy_drop'])):>12}"
        )
    print("-" * 112)


def save_history_csv(path: Path, histories: dict[str, list[dict[str, float | str]]]) -> None:
    rows = []
    for method, history in histories.items():
        for item in history:
            rows.append({"method_name": method, **item})

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


def plot_unlearning_accuracy(
    path: Path,
    histories: dict[str, list[dict[str, float | str]]],
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8, 5))
    for label, history in histories.items():
        epochs = [int(item["epoch"]) for item in history]
        values = [float(item[metric_key]) * 100 for item in history]
        plt.plot(epochs, values, marker="o", label=label)

    max_epoch = max(int(item["epoch"]) for history in histories.values() for item in history)
    plt.xlabel("Unlearning epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(range(max_epoch + 1))
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_per_class_accuracy(
    path: Path,
    baseline: dict[int, float],
    ascent: dict[int, float],
    confuse: dict[int, float],
    target_class: int,
) -> None:
    classes = list(range(10))
    x = np.arange(len(classes))
    width = 0.26

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, [baseline[idx] * 100 for idx in classes], width, label="Baseline")
    plt.bar(x, [ascent[idx] * 100 for idx in classes], width, label="Gradient ascent")
    plt.bar(x + width, [confuse[idx] * 100 for idx in classes], width, label="Confuse")
    plt.axvline(target_class, color="black", linestyle=":", linewidth=1)
    plt.xlabel("MNIST class")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Per-class accuracy after unlearning target digit {target_class}")
    plt.xticks(x, classes)
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def percentage_points(value: float) -> str:
    return f"{value * 100:.2f} pp"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    retain_batch_size = args.retain_batch_size or args.batch_size

    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    print(f"Target class to forget: {args.target_class}")

    train_dataset, test_dataset = make_mnist_datasets(args.data_dir)
    train_dataset = maybe_cap_dataset(train_dataset, args.max_train_samples)
    test_dataset = maybe_cap_dataset(test_dataset, args.max_test_samples)

    train_forget_dataset = subset_by_class(train_dataset, args.target_class, keep_target=True)
    train_retain_dataset = subset_by_class(train_dataset, args.target_class, keep_target=False)
    test_forget_dataset = subset_by_class(test_dataset, args.target_class, keep_target=True)
    test_retain_dataset = subset_by_class(test_dataset, args.target_class, keep_target=False)
    if len(train_forget_dataset) == 0 or len(test_forget_dataset) == 0:
        raise RuntimeError(
            f"No samples for target class {args.target_class}. "
            "Increase or remove --max-train-samples/--max-test-samples."
        )

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        True,
        args.num_workers,
        make_generator(args.seed + 1),
    )
    forget_loader_ascent = make_loader(
        train_forget_dataset,
        args.batch_size,
        True,
        args.num_workers,
        make_generator(args.seed + 2),
    )
    retain_loader_ascent = make_loader(
        train_retain_dataset,
        retain_batch_size,
        True,
        args.num_workers,
        make_generator(args.seed + 3),
    )
    forget_loader_confuse = make_loader(
        train_forget_dataset,
        args.batch_size,
        True,
        args.num_workers,
        make_generator(args.seed + 2),
    )
    retain_loader_confuse = make_loader(
        train_retain_dataset,
        retain_batch_size,
        True,
        args.num_workers,
        make_generator(args.seed + 3),
    )

    eval_loaders = {
        "all": make_loader(
            test_dataset,
            args.eval_batch_size,
            False,
            args.num_workers,
            make_generator(args.seed + 4),
        ),
        "forget": make_loader(
            test_forget_dataset,
            args.eval_batch_size,
            False,
            args.num_workers,
            make_generator(args.seed + 5),
        ),
        "retain": make_loader(
            test_retain_dataset,
            args.eval_batch_size,
            False,
            args.num_workers,
            make_generator(args.seed + 6),
        ),
    }

    print(
        "Dataset sizes: "
        f"train={len(train_dataset)}, target_train={len(train_forget_dataset)}, "
        f"retain_train={len(train_retain_dataset)}, test={len(test_dataset)}"
    )

    baseline_model = MnistCNN().to(device)
    print("\nTask 1: train baseline classifier on full MNIST")
    baseline_train_history = train_classifier(
        baseline_model,
        train_loader,
        device,
        args.train_epochs,
        args.lr,
        args.weight_decay,
    )
    baseline_metrics = evaluate_snapshot(
        0,
        "Baseline",
        baseline_model,
        eval_loaders,
        device,
        args.target_class,
    )
    print(
        "Baseline evaluation: "
        f"overall_acc={percent(float(baseline_metrics['overall_accuracy']))}, "
        f"target_{args.target_class}_acc={percent(float(baseline_metrics['target_accuracy']))}, "
        f"retain_acc={percent(float(baseline_metrics['retain_accuracy']))}"
    )

    baseline_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in baseline_model.state_dict().items()
    }

    ascent_model = MnistCNN().to(device)
    ascent_model.load_state_dict(baseline_state)
    print("\nTask 2a: gradient ascent unlearning")
    ascent_history = run_unlearning(
        "Gradient Ascent",
        ascent_model,
        forget_loader_ascent,
        retain_loader_ascent,
        eval_loaders,
        device,
        args,
        args.ascent_lr,
    )

    confuse_model = MnistCNN().to(device)
    confuse_model.load_state_dict(baseline_state)
    print("\nTask 2b: confuse unlearning")
    confuse_history = run_unlearning(
        "Confuse",
        confuse_model,
        forget_loader_confuse,
        retain_loader_confuse,
        eval_loaders,
        device,
        args,
        args.confuse_lr,
    )

    comparison_rows = make_comparison_rows(
        baseline_metrics,
        ascent_history[-1],
        confuse_history[-1],
    )
    print_comparison_table(comparison_rows, args.target_class)

    full_test_loader = eval_loaders["all"]
    baseline_per_class = per_class_accuracy(baseline_model, full_test_loader, device)
    ascent_per_class = per_class_accuracy(ascent_model, full_test_loader, device)
    confuse_per_class = per_class_accuracy(confuse_model, full_test_loader, device)

    histories = {
        "Gradient ascent": ascent_history,
        "Confuse": confuse_history,
    }
    save_history_csv(args.output_dir / "unlearning_history.csv", histories)
    save_comparison_csv(args.output_dir / "comparison_table.csv", comparison_rows)
    plot_unlearning_accuracy(
        args.output_dir / "target_accuracy_unlearning.png",
        histories,
        "target_accuracy",
        f"Target digit {args.target_class} accuracy during unlearning",
        "Target accuracy (%)",
    )
    plot_unlearning_accuracy(
        args.output_dir / "retain_accuracy_unlearning.png",
        histories,
        "retain_accuracy",
        f"Retained digit accuracy during unlearning target {args.target_class}",
        "Retain accuracy (%)",
    )
    plot_per_class_accuracy(
        args.output_dir / "per_class_accuracy_comparison.png",
        baseline_per_class,
        ascent_per_class,
        confuse_per_class,
        args.target_class,
    )

    metrics = {
        "config": {
            "seed": args.seed,
            "target_class": args.target_class,
            "train_epochs": args.train_epochs,
            "unlearn_epochs": args.unlearn_epochs,
            "batch_size": args.batch_size,
            "retain_batch_size": retain_batch_size,
            "lr": args.lr,
            "ascent_lr": args.ascent_lr,
            "confuse_lr": args.confuse_lr,
            "retain_weight": args.retain_weight,
            "forget_weight": args.forget_weight,
            "max_grad_norm": args.max_grad_norm,
            "device": str(device),
        },
        "baseline_train_history": baseline_train_history,
        "comparison": comparison_rows,
        "histories": histories,
        "per_class_accuracy": {
            "baseline": baseline_per_class,
            "gradient_ascent": ascent_per_class,
            "confuse": confuse_per_class,
        },
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\nSaved plots and metrics to: {args.output_dir.resolve()}")
    print(f" - {args.output_dir / 'target_accuracy_unlearning.png'}")
    print(f" - {args.output_dir / 'retain_accuracy_unlearning.png'}")
    print(f" - {args.output_dir / 'per_class_accuracy_comparison.png'}")
    print(f" - {args.output_dir / 'comparison_table.csv'}")
    print(f" - {args.output_dir / 'unlearning_history.csv'}")
    print(f" - {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
