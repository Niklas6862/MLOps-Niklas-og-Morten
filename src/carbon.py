"""CarbonTracker integration: Trainer callback, log parsing, and cost extrapolation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class CarbonTrackerCallback(TrainerCallback):
    """HuggingFace Trainer callback that wraps carbontracker per-epoch tracking."""

    def __init__(self, num_epochs: int, log_dir: str = "models/artifacts/carbon"):
        from carbontracker.tracker import CarbonTracker

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        # components="gpu" skips Intel RAPL CPU registers (/sys/class/powercap/)
        # which are unavailable inside Docker. ignore_errors=True prevents the
        # background tracker thread from sending SIGTERM to the main process on failure.
        self.tracker = CarbonTracker(
            epochs=num_epochs,
            log_dir=log_dir,
            verbose=2,
            components="gpu",
            ignore_errors=True,
        )
        self.log_dir = Path(log_dir)
        self._active = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.tracker.epoch_start()
        self._active = True

    def on_epoch_end(self, args, state, control, **kwargs):
        self.tracker.epoch_end()

    def on_train_end(self, args, state, control, **kwargs):
        if self._active:
            self.tracker.stop()


def parse_carbon_log(log_dir: str | Path) -> dict[str, float] | None:
    """Parse carbontracker logs using the official parser and return energy + CO2.

    Returns None if no logs exist or actual consumption data is unavailable.
    """
    try:
        from carbontracker.parser import parse_all_logs

        logs = parse_all_logs(log_dir=str(log_dir))
        if not logs:
            logger.warning("No carbontracker logs found in '%s'.", log_dir)
            return None

        actual = logs[-1].get("actual")
        if actual is None:
            logger.warning("Carbontracker log has no actual consumption data.")
            return None

        energy = actual.get("energy (kWh)")
        co2 = actual.get("co2eq (g)")
        if energy is None or co2 is None:
            logger.warning("Missing energy/CO2 fields in carbontracker data: %s", actual)
            return None

        return {"energy_kwh": float(energy), "co2_g": float(co2)}
    except Exception as exc:
        logger.warning("Could not parse carbontracker logs: %s", exc)
        return None


def extrapolate_costs(
    energy_kwh: float,
    co2_g: float,
    n_train_samples: int,
    retrains_per_year: int = 12,
    requests_per_day: int = 1000,
) -> dict[str, Any]:
    """Derive yearly training and per-request inference cost estimates.

    Per-request CO2 is approximated by scaling the training energy by sample count,
    assuming inference energy per sample ≈ training energy per sample.
    """
    yearly_training_co2_kg = (co2_g / 1000) * retrains_per_year
    yearly_training_kwh = energy_kwh * retrains_per_year

    co2_per_request_g = co2_g / max(n_train_samples, 1)
    yearly_requests = requests_per_day * 365
    yearly_inference_co2_kg = (co2_per_request_g * yearly_requests) / 1000
    yearly_inference_kwh = (energy_kwh / max(n_train_samples, 1)) * yearly_requests

    return {
        "total_training_energy_kwh": round(energy_kwh, 6),
        "total_training_co2_g": round(co2_g, 4),
        "yearly_training_co2_kg": round(yearly_training_co2_kg, 4),
        "yearly_training_energy_kwh": round(yearly_training_kwh, 6),
        "co2_per_request_g": round(co2_per_request_g, 8),
        "yearly_inference_co2_kg": round(yearly_inference_co2_kg, 4),
        "yearly_inference_energy_kwh": round(yearly_inference_kwh, 6),
        "assumptions": {
            "retrains_per_year": retrains_per_year,
            "requests_per_day": requests_per_day,
            "n_train_samples": n_train_samples,
        },
    }
