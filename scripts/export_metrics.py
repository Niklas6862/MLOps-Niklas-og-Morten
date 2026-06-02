"""Export MLflow runs to SQLite so Grafana can visualise them."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import mlflow
import pandas as pd


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS training_metrics (
            run_id TEXT PRIMARY KEY,
            run_name TEXT,
            start_time INTEGER,
            model_name TEXT,
            dataset TEXT,
            num_epochs INTEGER,
            eval_accuracy REAL,
            model_size_mb REAL,
            carbon_energy_kwh REAL,
            carbon_co2_g REAL,
            carbon_yearly_training_co2_kg REAL,
            carbon_co2_per_request_g REAL,
            carbon_yearly_inference_co2_kg REAL
        );
        CREATE TABLE IF NOT EXISTS drift_metrics (
            run_id TEXT PRIMARY KEY,
            run_name TEXT,
            start_time INTEGER,
            artificial_shift TEXT,
            data_drift_fraction REAL,
            data_drift_detected INTEGER,
            data_drift_mean_p_value REAL,
            pred_drift_kl_divergence REAL,
            pred_drift_detected INTEGER,
            overall_drift_detected INTEGER
        );
        CREATE TABLE IF NOT EXISTS compression_metrics (
            run_id TEXT PRIMARY KEY,
            run_name TEXT,
            start_time INTEGER,
            compression_method TEXT,
            baseline_accuracy REAL,
            compressed_accuracy REAL,
            accuracy_drop REAL,
            baseline_throughput_fps REAL,
            compressed_throughput_fps REAL,
            speedup_x REAL,
            actual_sparsity REAL
        );
    """)
    conn.commit()


def _get(row: pd.Series, col: str, cast=None):
    v = row.get(col)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return cast(v) if cast else v


def _ts(row: pd.Series) -> int:
    v = row.get("start_time")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    return int(pd.Timestamp(v).timestamp() * 1000)


def export(mlflow_uri: str, db_path: str) -> None:
    mlflow.set_tracking_uri(mlflow_uri)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _create_tables(conn)

    for exp in mlflow.search_experiments():
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="pandas")
        if runs.empty:
            continue

        for _, row in runs.iterrows():
            run_id = row["run_id"]
            run_type = _get(row, "tags.run_type") or ""
            training_strategy = _get(row, "tags.training_strategy") or ""
            run_name = _get(row, "tags.mlflow.runName") or run_id[:8]
            start_time = _ts(row)

            if training_strategy:
                conn.execute(
                    "INSERT OR REPLACE INTO training_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        run_name,
                        start_time,
                        _get(row, "params.model_name"),
                        _get(row, "params.dataset"),
                        _get(row, "params.num_epochs", int),
                        _get(row, "metrics.eval_accuracy"),
                        _get(row, "metrics.model_size_mb"),
                        _get(row, "metrics.carbon_energy_kwh"),
                        _get(row, "metrics.carbon_co2_g"),
                        _get(row, "metrics.carbon_yearly_training_co2_kg"),
                        _get(row, "metrics.carbon_co2_per_request_g"),
                        _get(row, "metrics.carbon_yearly_inference_co2_kg"),
                    ),
                )

            elif run_type == "drift-check":
                conn.execute(
                    "INSERT OR REPLACE INTO drift_metrics VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        run_name,
                        start_time,
                        _get(row, "tags.artificial_shift"),
                        _get(row, "metrics.data_drift_fraction"),
                        _get(row, "metrics.data_drift_detected", int),
                        _get(row, "metrics.data_drift_mean_p_value"),
                        _get(row, "metrics.pred_drift_kl_divergence"),
                        _get(row, "metrics.pred_drift_detected", int),
                        _get(row, "metrics.overall_drift_detected", int),
                    ),
                )

            elif run_type in ("compression", "pruning", "finetune_pruned"):
                # pruning.py logs "pruned_*", compress.py logs "compressed_*"
                acc = _get(row, "metrics.compressed_accuracy") or _get(row, "metrics.post_accuracy")
                drop = _get(row, "metrics.compressed_accuracy_drop") or _get(
                    row, "metrics.pruned_accuracy_drop"
                )
                fps_base = _get(row, "metrics.baseline_throughput_fps") or _get(
                    row, "metrics.pre_throughput_fps"
                )
                fps_comp = (
                    _get(row, "metrics.compressed_throughput_fps")
                    or _get(row, "metrics.post_throughput_fps")
                    or _get(row, "metrics.pruned_throughput_fps")
                )
                speedup = _get(row, "metrics.compressed_speedup_x") or _get(
                    row, "metrics.pruned_speedup_x"
                )
                sparsity = _get(row, "metrics.compressed_actual_sparsity") or _get(
                    row, "metrics.pruned_actual_sparsity"
                )
                method = _get(row, "tags.compression_method") or run_type

                conn.execute(
                    "INSERT OR REPLACE INTO compression_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        run_name,
                        start_time,
                        method,
                        _get(row, "metrics.baseline_accuracy"),
                        acc,
                        drop,
                        fps_base,
                        fps_comp,
                        speedup,
                        sparsity,
                    ),
                )

    conn.commit()
    conn.close()
    print(f"Exported → {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MLflow metrics to SQLite for Grafana")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    parser.add_argument("--db", default="monitoring/metrics.db")
    args = parser.parse_args()
    export(args.mlflow_uri, args.db)


if __name__ == "__main__":
    main()
