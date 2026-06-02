from __future__ import annotations


def test_extrapolate_costs_basic() -> None:
    from src.carbon import extrapolate_costs

    result = extrapolate_costs(energy_kwh=1.0, co2_g=500.0, n_train_samples=1000)

    assert result["yearly_training_co2_kg"] == round(500.0 / 1000 * 12, 4)
    assert result["yearly_training_energy_kwh"] == round(1.0 * 12, 6)
    assert result["co2_per_request_g"] == round(500.0 / 1000, 8)
    assert result["assumptions"]["retrains_per_year"] == 12
    assert result["assumptions"]["n_train_samples"] == 1000


def test_extrapolate_costs_zero_samples() -> None:
    from src.carbon import extrapolate_costs

    result = extrapolate_costs(energy_kwh=1.0, co2_g=100.0, n_train_samples=0)
    # Should not divide by zero
    assert result["co2_per_request_g"] == round(100.0 / 1, 8)
