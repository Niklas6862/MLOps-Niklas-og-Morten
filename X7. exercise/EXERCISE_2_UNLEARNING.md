# Exercise 2: Unlearning with Gradient Ascent

This exercise is implemented in `unlearning_gradient_ascent.py` as a standalone Python script.

## What the Script Does

The experiment trains a fresh CNN classifier on the full MNIST dataset, then tries to make the model forget one selected class without retraining from scratch.

Default target class:

- Digit `6`.

The script compares two unlearning methods:

- Gradient ascent: maximize the cross-entropy loss on digit `6`, which pushes the model away from correctly recognizing that class.
- Confuse unlearning: train digit `6` samples toward wrong random labels, which makes the target class representation intentionally unreliable.

Both methods also include a retain-class loss by default. This means each unlearning step can use digit `6` samples for forgetting and non-`6` samples for preservation. The retain term is controlled by `--retain-weight`; setting it to `0` makes the unlearning target-only.

## Default Configuration

- Random seed: `42`.
- Dataset: full MNIST digits `0-9`.
- Target class to forget: `6`.
- Device: `auto`, preferring Apple MPS on supported Macs when available.
- Baseline training epochs: `3`.
- Unlearning epochs: `3`.
- Baseline learning rate: `0.001`.
- Gradient ascent learning rate: `0.0002`.
- Confuse unlearning learning rate: `0.0002`.
- Retain loss weight: `3.0`.
- Forget loss weight: `1.0`.
- Gradient clipping during unlearning: `5.0`.

## How to Run

Install dependencies if needed:

```bash
python3 -m pip install -r requirements-unlearning.txt
```

Run the default experiment:

```bash
python3 unlearning_gradient_ascent.py
```

Useful overrides:

```bash
python3 unlearning_gradient_ascent.py \
  --target-class 6 \
  --train-epochs 3 \
  --unlearn-epochs 3 \
  --ascent-lr 0.0002 \
  --confuse-lr 0.0002 \
  --retain-weight 3.0
```

For pure target-only unlearning:

```bash
python3 unlearning_gradient_ascent.py --retain-weight 0
```

## Outputs

The script prints:

- Baseline training loss and accuracy.
- Baseline evaluation accuracy on all digits, target digit `6`, and retained digits.
- Per-epoch target, retained, and overall accuracy during both unlearning methods.
- A final comparison table for baseline, gradient ascent, and confuse unlearning.

It saves artifacts under `outputs/unlearning_gradient_ascent/`:

- `target_accuracy_unlearning.png`: target-class accuracy over unlearning epochs.
- `retain_accuracy_unlearning.png`: retained-class accuracy over unlearning epochs.
- `per_class_accuracy_comparison.png`: per-class baseline vs final unlearned accuracies.
- `comparison_table.csv`: final comparison table.
- `unlearning_history.csv`: per-epoch unlearning metrics.
- `metrics.json`: complete config and metrics.

## Thoughts

A normal classifier should reach high accuracy on every MNIST digit, including digit `6`. The unlearning phase tries to selectively damage the model's performance on digit `6` while preserving performance on the other nine digits.

Gradient ascent is direct: it takes the loss the model would normally minimize for digit `6` and climbs it instead. This can forget quickly, but it can also disturb shared features if the step size is too large.

Confuse unlearning is a softer alternative. Rather than only maximizing the correct-label loss, it supplies incorrect labels for the forgotten class. This teaches the model that digit `6` should not map consistently to class `6`.

The retain loss is included because MNIST classes share low-level visual features. Without any retain signal, unlearning can damage useful convolutional features and reduce accuracy on non-target classes. The best result is not necessarily the lowest possible target accuracy; it is the best tradeoff between target forgetting and retained-class preservation.
