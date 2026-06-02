from __future__ import annotations

import torch
import torch.nn as nn


def _small_model() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))


def test_apply_pruning_sparsity() -> None:
    from src.compress import apply_pruning

    model = _small_model()
    stats = apply_pruning(model, amount=0.5)

    assert stats["actual_sparsity"] >= 0.4  # allow small rounding
    assert stats["zeroed_weights"] > 0


def test_make_pruning_permanent() -> None:
    from src.compress import apply_pruning, make_pruning_permanent

    model = _small_model()
    apply_pruning(model, amount=0.3)
    model = make_pruning_permanent(model)

    # After baking in, weight_mask attribute should be gone
    for m in model.modules():
        if isinstance(m, nn.Linear):
            assert not hasattr(m, "weight_mask")


def test_apply_dynamic_quantization() -> None:
    from src.compress import apply_dynamic_quantization

    model = _small_model()
    qmodel = apply_dynamic_quantization(model)
    x = torch.randn(2, 16)
    out = qmodel(x)
    assert out.shape == (2, 4)
