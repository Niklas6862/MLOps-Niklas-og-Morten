import copy

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK1 = [0, 1, 2, 3, 4]
TASK2 = [5, 6, 7, 8, 9]
EPOCHS_T1 = 3
EPOCHS_T2 = 5
BATCH = 512
LR = 1e-3
REPLAY_SIZE = 5000
EWC_LAMBDA = 500


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def filter_classes(dataset, classes):
    idx = [i for i, (_, y) in enumerate(dataset) if y in classes]
    return Subset(dataset, idx)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total if total else 0.0


def train_epoch(model, loader, optimizer, ewc=None):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        if ewc is not None:
            loss = loss + ewc.penalty(model)
        loss.backward()
        optimizer.step()


class EWC:
    def __init__(self, model, dataset, lam):
        self.lam = lam
        self.opt = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(model, dataset)

    def _compute_fisher(self, model, dataset):
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2).detach() * len(y)
        n_samples = len(dataset)
        for n in fisher:
            fisher[n] /= n_samples
        return fisher

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.opt[n]).pow(2)).sum()
        return 0.5 * self.lam * loss


def main():
    print(f"Device: {DEVICE}")
    transform = transforms.ToTensor()
    train_full = datasets.MNIST("~/.cache/mnist", train=True, download=True, transform=transform)
    test_full = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=transform)

    task1_train = filter_classes(train_full, TASK1)
    task2_train = filter_classes(train_full, TASK2)
    task1_test = filter_classes(test_full, TASK1)
    task2_test = filter_classes(test_full, TASK2)

    t1_loader = DataLoader(task1_train, batch_size=BATCH, shuffle=True)
    t2_loader = DataLoader(task2_train, batch_size=BATCH, shuffle=True)
    t1_test_loader = DataLoader(task1_test, batch_size=256)
    t2_test_loader = DataLoader(task2_test, batch_size=256)

    # ── Task 1 ───────────────────────────────────────────────────────────────
    print("\n=== Task 1: training on digits 0–4 ===")
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS_T1):
        train_epoch(model, t1_loader, opt)
        acc = evaluate(model, t1_test_loader)
        print(f"  Epoch {ep + 1}: acc(0-4) = {acc:.3f}")

    baseline_t1 = evaluate(model, t1_test_loader)
    task1_state = copy.deepcopy(model.state_dict())

    # ── Task 2: Naive (no replay) ─────────────────────────────────────────────
    print("\n=== Task 2 Naive: digits 5–9, no replay ===")
    model.load_state_dict(task1_state)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    naive_old, naive_new = [], []
    for ep in range(EPOCHS_T2):
        train_epoch(model, t2_loader, opt)
        acc_old = evaluate(model, t1_test_loader)
        acc_new = evaluate(model, t2_test_loader)
        naive_old.append(acc_old)
        naive_new.append(acc_new)
        print(f"  Epoch {ep + 1}: acc(0-4) = {acc_old:.3f} | acc(5-9) = {acc_new:.3f}")

    # ── Task 2: Replay + EWC ─────────────────────────────────────────────────
    print("\n=== Task 2 Replay+EWC: digits 5–9 with memory buffer ===")
    model.load_state_dict(task1_state)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    buffer_idx = torch.randperm(len(task1_train))[:REPLAY_SIZE].tolist()
    buffer = Subset(task1_train, buffer_idx)
    combined_loader = DataLoader(
        ConcatDataset([task2_train, buffer]), batch_size=BATCH, shuffle=True
    )

    ewc = EWC(model, task1_train, lam=EWC_LAMBDA)

    replay_old, replay_new = [], []
    for ep in range(EPOCHS_T2):
        train_epoch(model, combined_loader, opt, ewc=ewc)
        acc_old = evaluate(model, t1_test_loader)
        acc_new = evaluate(model, t2_test_loader)
        replay_old.append(acc_old)
        replay_new.append(acc_new)
        print(f"  Epoch {ep + 1}: acc(0-4) = {acc_old:.3f} | acc(5-9) = {acc_new:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Method':<25} {'Acc (0–4)':<14} {'Acc (5–9)'}")
    print("-" * 55)
    print(f"{'Task 1 baseline':<25} {baseline_t1:<14.3f} {'N/A'}")
    print(f"{'Naive sequential':<25} {naive_old[-1]:<14.3f} {naive_new[-1]:.3f}")
    print(f"{'Replay + EWC':<25} {replay_old[-1]:<14.3f} {replay_new[-1]:.3f}")
    print("=" * 55)

    # ── Plot ──────────────────────────────────────────────────────────────────
    epochs = list(range(1, EPOCHS_T2 + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].axhline(
        baseline_t1,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"Task 1 baseline ({baseline_t1:.2f})",
    )
    axes[0].plot(epochs, naive_old, "r-o", linewidth=2, label="Naive")
    axes[0].plot(epochs, replay_old, "b-o", linewidth=2, label="Replay + EWC")
    axes[0].set_title("Accuracy on old task (digits 0–4)")
    axes[0].set_xlabel("Epoch (Task 2 training)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].plot(epochs, naive_new, "r-o", linewidth=2, label="Naive")
    axes[1].plot(epochs, replay_new, "b-o", linewidth=2, label="Replay + EWC")
    axes[1].set_title("Accuracy on new task (digits 5–9)")
    axes[1].set_xlabel("Epoch (Task 2 training)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.suptitle("Catastrophic Forgetting: Naive vs Replay + EWC", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "mm7/continual_learning_results.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
