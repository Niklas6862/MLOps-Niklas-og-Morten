# Exercise 1: Continual Learning with Experience Replay

This exercise is implemented in `continual_learning_experience_replay.py` as a standalone Python script.

## What the Script Does

The experiment uses Avalanche SplitMNIST to create two sequential tasks:

- Task 1: MNIST digits 0-4.
- Task 2: MNIST digits 5-9.

The same CNN is first trained on Task 1. From that shared Task 1 checkpoint, the script then runs two Task 2 variants:

- Naive sequential learning: train only on digits 5-9. This usually learns the new digits well while damaging performance on digits 0-4.
- Replay + EWC: train on digits 5-9 while mixing in a 500-sample replay buffer from digits 0-4 and adding an Elastic Weight Consolidation penalty.

Avalanche is used for the SplitMNIST benchmark and for the replay memory buffer through `ReservoirSamplingBuffer`. The training loop is kept explicit in PyTorch so the per-epoch forgetting curve, replay ratio, and EWC penalty are easy to inspect.

## Default Configuration

- Random seed: `42`.
- Dataset: full MNIST, split into digits `0-4` then `5-9`.
- Device: `auto`, preferring Apple MPS on an M1/M2 Mac when available.
- Task 1 epochs: `3`.
- Task 2 epochs: `5`.
- Current-task batch size: `128`.
- Replay buffer size: `500`.
- Replay batch size: defaults to `128`, giving a 50/50 new-to-replay ratio during Task 2.
- EWC lambda: `1000.0`, configurable with `--ewc-lambda`.

The EWC implementation estimates a diagonal Fisher matrix after Task 1. During replay training, each Task 2 batch is concatenated with a replay batch from old digits, and the loss is:

```text
cross_entropy(new + replay samples) + EWC penalty
```

## How to Run

Install dependencies. A virtual environment is recommended because Avalanche pulls in several research-library dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-continual-learning.txt
```

Run the default experiment:

```bash
python3 continual_learning_experience_replay.py
```

Useful overrides:

```bash
python3 continual_learning_experience_replay.py \
  --task1-epochs 5 \
  --task2-epochs 10 \
  --replay-buffer-size 500 \
  --replay-batch-size 128 \
  --ewc-lambda 1000
```

If Avalanche installation has issues on Python 3.12, use a Python 3.10 or 3.11 virtual environment. Avalanche support can lag behind the newest Python releases.

## Outputs

The script prints:

- Task 1 accuracy on old digits after initial training.
- Per-epoch Task 2 accuracy on old digits and new digits.
- Per-epoch old-digit accuracy drop during Task 2.
- A final comparison table for initial training, naive sequential learning, and replay + EWC.

It also saves artifacts under `outputs/continual_learning/`:

- `task2_accuracy.png`: old and new digit accuracy throughout Task 2.
- `old_digit_forgetting.png`: old-digit performance drop throughout Task 2.
- `comparison_table.csv`: final comparison table.
- `task2_history.csv`: per-epoch metrics for both Task 2 methods.
- `metrics.json`: complete config and metrics.

## Thoughts

The naive model is intentionally allowed to forget. It starts Task 2 from the Task 1 weights but sees only digits 5-9 afterward, so gradients reshape the representation toward the new classes. The old output classes remain in the classifier, but the feature extractor no longer receives reminders of what made digits 0-4 separable.

Experience replay addresses that by reintroducing examples from the old task. With the default 50/50 ratio, every Task 2 optimization step still includes old-task supervision. This tends to reduce forgetting, although it can slightly slow learning on digits 5-9 because the model is solving a mixed objective.

EWC adds a complementary constraint. Instead of storing more data, it estimates which parameters mattered for Task 1 and penalizes moving them too far. Replay gives direct old examples; EWC adds a parameter-space memory of Task 1. The default lambda is deliberately configurable because the best value is empirical: too low behaves like replay alone, while too high can protect old digits at the cost of learning digits 5-9.

## References

- Avalanche replay documentation: https://avalanche.continualai.org/how-tos/dataloading_buffers_replay
- Avalanche training and plugin documentation: https://avalanche.continualai.org/avalanche-v0.4.0/from-zero-to-hero-tutorial/04_training
