import random
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class UnlearningConfig:
    forget_digit: int = 7
    training_epochs: int = 5
    scrub_steps: int = 150
    batch_size: int = 512
    training_lr: float = 1e-3
    scrub_lr: float = 5e-5
    seed: int = 42
    data_dir: str = "~/.cache/mnist"
    plot_path: str = "mm7/unlearning_results.png"


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = choose_device()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DigitConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images))


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def classwise_accuracy(model: nn.Module, loader: DataLoader, num_classes: int = 10) -> list[float]:
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    totals = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            labels_cpu = labels.long()
            predictions = model(images.to(DEVICE)).argmax(dim=1).cpu()
            totals += torch.bincount(labels_cpu, minlength=num_classes)
            correct += torch.bincount(
                labels_cpu[predictions.eq(labels_cpu)],
                minlength=num_classes,
            )

    return [
        correct[digit].item() / totals[digit].item() if totals[digit] else 0.0
        for digit in range(num_classes)
    ]


def train_classifier(
    model: nn.Module,
    loader: DataLoader,
    eval_loader: DataLoader,
    cfg: UnlearningConfig,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training_lr)
    latest_scores: list[float] = []

    print("\n=== Fit classifier on all MNIST digits ===")
    for epoch in range(1, cfg.training_epochs + 1):
        model.train()
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            F.cross_entropy(model(images), labels).backward()
            optimizer.step()

        latest_scores = classwise_accuracy(model, eval_loader)
        mean_score = sum(latest_scores) / len(latest_scores)
        print(
            f"  Epoch {epoch}: mean class acc = {mean_score:.3f} | "
            f"class {cfg.forget_digit} = {latest_scores[cfg.forget_digit]:.3f}"
        )

    return latest_scores


def split_forgetting_targets(dataset, forget_digit: int) -> tuple[Subset, Subset]:
    forget_indices: list[int] = []
    retain_indices: list[int] = []

    for idx, (_, label) in enumerate(dataset):
        if label == forget_digit:
            forget_indices.append(idx)
        else:
            retain_indices.append(idx)

    return Subset(dataset, forget_indices), Subset(dataset, retain_indices)


def push_predictions_toward_uniform(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    cfg: UnlearningConfig,
    num_classes: int = 10,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scrub_lr)
    retain_batches = cycle(retain_loader)
    completed = 0
    model.train()

    print(f"\n=== Scrub class {cfg.forget_digit} for {cfg.scrub_steps} mini-batches ===")
    while completed < cfg.scrub_steps:
        for forget_images, _ in forget_loader:
            if completed >= cfg.scrub_steps:
                break

            forget_images = forget_images.to(DEVICE)
            target_distribution = torch.full(
                (forget_images.size(0), num_classes),
                1.0 / num_classes,
                device=DEVICE,
            )

            optimizer.zero_grad()
            forget_log_probs = F.log_softmax(model(forget_images), dim=1)
            F.kl_div(forget_log_probs, target_distribution, reduction="batchmean").backward()
            optimizer.step()

            retain_images, retain_labels = next(retain_batches)
            retain_images = retain_images.to(DEVICE)
            retain_labels = retain_labels.to(DEVICE)

            optimizer.zero_grad()
            F.cross_entropy(model(retain_images), retain_labels).backward()
            optimizer.step()

            completed += 1
            if completed % 50 == 0:
                print(f"  Completed {completed}/{cfg.scrub_steps} scrub steps")


def report_scores(before: list[float], after: list[float], forget_digit: int) -> None:
    print("\n" + "=" * 54)
    print(f"{'Digit':<8} {'Before':>9} {'After':>9} {'Change':>9}")
    print("-" * 54)
    for digit, (old_score, new_score) in enumerate(zip(before, after)):
        marker = "  <-- target" if digit == forget_digit else ""
        print(
            f"  {digit:<6} {old_score:>9.3f} {new_score:>9.3f} {new_score - old_score:>+9.3f}{marker}"
        )

    retained_before = sum(score for digit, score in enumerate(before) if digit != forget_digit) / 9
    retained_after = sum(score for digit, score in enumerate(after) if digit != forget_digit) / 9
    print("=" * 54)
    print(
        f"\nTarget digit {forget_digit}: {before[forget_digit]:.3f} -> "
        f"{after[forget_digit]:.3f} (drop {before[forget_digit] - after[forget_digit]:.3f})"
    )
    print(
        f"Retained digit average: {retained_before:.3f} -> "
        f"{retained_after:.3f} (drop {retained_before - retained_after:.3f})"
    )


def plot_before_after(before: list[float], after: list[float], cfg: UnlearningConfig) -> None:
    digits = list(range(10))
    positions = np.arange(len(digits))
    width = 0.35
    after_colors = ["tomato" if digit == cfg.forget_digit else "seagreen" for digit in digits]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        positions - width / 2,
        before,
        width,
        label="Before scrubbing",
        color="steelblue",
        alpha=0.85,
    )
    ax.bar(
        positions + width / 2,
        after,
        width,
        label="After scrubbing",
        color=after_colors,
        alpha=0.85,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{digit} (target)" if digit == cfg.forget_digit else str(digit) for digit in digits]
    )
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"MNIST class unlearning for digit {cfg.forget_digit}")
    ax.axhline(1.0, color="dimgray", linestyle=":", linewidth=0.8)
    ax.legend()

    fig.tight_layout()
    output = Path(cfg.plot_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {output}")


def main() -> None:
    cfg = UnlearningConfig()
    seed_everything(cfg.seed)
    print(f"Device: {DEVICE}")

    transform = transforms.ToTensor()
    train_data = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=transform)

    train_loader = make_loader(train_data, cfg.batch_size, shuffle=True)
    test_loader = make_loader(test_data, 256, shuffle=False)

    model = DigitConvNet().to(DEVICE)
    before = train_classifier(model, train_loader, test_loader, cfg)

    forget_set, retain_set = split_forgetting_targets(train_data, cfg.forget_digit)
    forget_loader = make_loader(forget_set, cfg.batch_size, shuffle=True)
    retain_loader = make_loader(retain_set, cfg.batch_size, shuffle=True)

    push_predictions_toward_uniform(model, forget_loader, retain_loader, cfg)
    after = classwise_accuracy(model, test_loader)

    report_scores(before, after, cfg.forget_digit)
    plot_before_after(before, after, cfg)


if __name__ == "__main__":
    main()
