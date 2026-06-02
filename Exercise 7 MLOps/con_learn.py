import copy
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ContinualConfig:
    first_task_digits: tuple[int, ...] = (0, 1, 2, 3, 4)
    second_task_digits: tuple[int, ...] = (5, 6, 7, 8, 9)
    first_task_epochs: int = 3
    second_task_epochs: int = 5
    batch_size: int = 512
    learning_rate: float = 1e-3
    replay_examples: int = 5000
    ewc_strength: float = 500.0
    seed: int = 42
    data_dir: str = "~/.cache/mnist"
    plot_path: str = "mm7/continual_learning_results.png"


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = choose_device()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DigitConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def keep_digits(dataset, digits: tuple[int, ...]) -> Subset:
    chosen = [idx for idx, (_, label) in enumerate(dataset) if label in digits]
    return Subset(dataset, chosen)


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    hits = 0
    seen = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            predictions = model(images).argmax(dim=1)
            hits += predictions.eq(labels).sum().item()
            seen += labels.numel()
    return hits / seen if seen else 0.0


def train_one_pass(model: nn.Module, loader: DataLoader, optimizer, regularizer=None) -> None:
    model.train()
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        if regularizer is not None:
            loss = loss + regularizer(model)
        loss.backward()
        optimizer.step()


class ElasticWeightAnchor:
    def __init__(self, model: nn.Module, dataset, strength: float, batch_size: int) -> None:
        self.strength = strength
        self.reference = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.importance = self._estimate_parameter_importance(model, dataset, batch_size)

    def _estimate_parameter_importance(
        self, model: nn.Module, dataset, batch_size: int
    ) -> dict[str, torch.Tensor]:
        model.eval()
        scores = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            model.zero_grad()
            F.cross_entropy(model(images), labels).backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    scores[name] += param.grad.detach().square() * labels.numel()

        for name in scores:
            scores[name] /= len(dataset)
        return scores

    def __call__(self, model: nn.Module) -> torch.Tensor:
        penalty = torch.zeros((), device=DEVICE)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.importance:
                drift = (param - self.reference[name]).square()
                penalty = penalty + (self.importance[name] * drift).sum()
        return 0.5 * self.strength * penalty


def sample_replay_set(dataset, sample_count: int) -> Subset:
    available = len(dataset)
    requested = min(sample_count, available)
    indices = torch.randperm(available)[:requested].tolist()
    return Subset(dataset, indices)


def plot_task_two_history(
    baseline: float,
    naive_old: list[float],
    naive_new: list[float],
    replay_old: list[float],
    replay_new: list[float],
    output_path: str,
) -> None:
    epochs = range(1, len(naive_old) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].axhline(
        baseline,
        color="dimgray",
        linestyle=":",
        linewidth=1.5,
        label=f"Initial task score ({baseline:.2f})",
    )
    axes[0].plot(epochs, naive_old, color="crimson", marker="o", label="Plain fine-tune")
    axes[0].plot(epochs, replay_old, color="royalblue", marker="o", label="Replay + EWC")
    axes[0].set_title("Retained accuracy on digits 0-4")
    axes[0].set_xlabel("Epoch while learning digits 5-9")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].plot(epochs, naive_new, color="crimson", marker="o", label="Plain fine-tune")
    axes[1].plot(epochs, replay_new, color="royalblue", marker="o", label="Replay + EWC")
    axes[1].set_title("New-task accuracy on digits 5-9")
    axes[1].set_xlabel("Epoch while learning digits 5-9")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    fig.suptitle("Sequential MNIST: forgetting versus replay regularization", fontsize=13)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {output}")


def print_summary(baseline: float, naive: tuple[float, float], replay: tuple[float, float]) -> None:
    print("\n" + "=" * 59)
    print(f"{'Method':<25} {'Digits 0-4':<15} {'Digits 5-9'}")
    print("-" * 59)
    print(f"{'After task 1':<25} {baseline:<15.3f} {'N/A'}")
    print(f"{'Plain fine-tuning':<25} {naive[0]:<15.3f} {naive[1]:.3f}")
    print(f"{'Replay + EWC':<25} {replay[0]:<15.3f} {replay[1]:.3f}")
    print("=" * 59)


def main() -> None:
    cfg = ContinualConfig()
    seed_everything(cfg.seed)
    print(f"Device: {DEVICE}")

    transform = transforms.ToTensor()
    train_full = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=transform)

    task1_train = keep_digits(train_full, cfg.first_task_digits)
    task2_train = keep_digits(train_full, cfg.second_task_digits)
    task1_test = keep_digits(test_full, cfg.first_task_digits)
    task2_test = keep_digits(test_full, cfg.second_task_digits)

    task1_loader = make_loader(task1_train, cfg.batch_size, shuffle=True)
    task2_loader = make_loader(task2_train, cfg.batch_size, shuffle=True)
    task1_eval_loader = make_loader(task1_test, 256, shuffle=False)
    task2_eval_loader = make_loader(task2_test, 256, shuffle=False)

    print("\n=== Stage 1: learn digits 0-4 ===")
    model = DigitConvNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    for epoch in range(1, cfg.first_task_epochs + 1):
        train_one_pass(model, task1_loader, optimizer)
        print(f"  Epoch {epoch}: acc(0-4) = {accuracy(model, task1_eval_loader):.3f}")

    first_task_score = accuracy(model, task1_eval_loader)
    checkpoint_after_task1 = copy.deepcopy(model.state_dict())

    print("\n=== Stage 2A: plain fine-tuning on digits 5-9 ===")
    model.load_state_dict(checkpoint_after_task1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    naive_old_scores: list[float] = []
    naive_new_scores: list[float] = []
    for epoch in range(1, cfg.second_task_epochs + 1):
        train_one_pass(model, task2_loader, optimizer)
        old_score = accuracy(model, task1_eval_loader)
        new_score = accuracy(model, task2_eval_loader)
        naive_old_scores.append(old_score)
        naive_new_scores.append(new_score)
        print(f"  Epoch {epoch}: acc(0-4) = {old_score:.3f} | acc(5-9) = {new_score:.3f}")

    print("\n=== Stage 2B: replay buffer plus EWC on digits 5-9 ===")
    model.load_state_dict(checkpoint_after_task1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    replay_memory = sample_replay_set(task1_train, cfg.replay_examples)
    mixed_training_data = ConcatDataset([task2_train, replay_memory])
    mixed_loader = make_loader(mixed_training_data, cfg.batch_size, shuffle=True)
    anchor = ElasticWeightAnchor(
        model,
        task1_train,
        strength=cfg.ewc_strength,
        batch_size=cfg.batch_size,
    )

    replay_old_scores: list[float] = []
    replay_new_scores: list[float] = []
    for epoch in range(1, cfg.second_task_epochs + 1):
        train_one_pass(model, mixed_loader, optimizer, regularizer=anchor)
        old_score = accuracy(model, task1_eval_loader)
        new_score = accuracy(model, task2_eval_loader)
        replay_old_scores.append(old_score)
        replay_new_scores.append(new_score)
        print(f"  Epoch {epoch}: acc(0-4) = {old_score:.3f} | acc(5-9) = {new_score:.3f}")

    print_summary(
        first_task_score,
        naive=(naive_old_scores[-1], naive_new_scores[-1]),
        replay=(replay_old_scores[-1], replay_new_scores[-1]),
    )
    plot_task_two_history(
        first_task_score,
        naive_old_scores,
        naive_new_scores,
        replay_old_scores,
        replay_new_scores,
        cfg.plot_path,
    )


if __name__ == "__main__":
    main()
