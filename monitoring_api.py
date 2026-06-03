import os
import random
import time
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]

PREDICTIONS_TOTAL = Counter("predictions_total", "Number of predictions per class", ["class_name"])
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Model inference latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
CONFIDENCE_SCORE = Histogram(
    "confidence_score",
    "Predicted confidence score (softmax probability)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

TRAIN_EVAL_ACCURACY = Gauge(
    "mlops_training_eval_accuracy", "Eval accuracy", ["run_name", "model_name", "dataset"]
)
TRAIN_MODEL_SIZE = Gauge(
    "mlops_training_model_size_mb", "Model size MB", ["run_name", "model_name"]
)
TRAIN_NUM_EPOCHS = Gauge(
    "mlops_training_num_epochs", "Number of epochs", ["run_name", "model_name"]
)
TRAIN_CO2_G = Gauge("mlops_training_carbon_co2_g", "CO2 emitted (g)", ["run_name", "model_name"])
TRAIN_ENERGY_KWH = Gauge(
    "mlops_training_carbon_energy_kwh", "Energy (kWh)", ["run_name", "model_name"]
)
TRAIN_YEARLY_CO2_KG = Gauge(
    "mlops_training_carbon_yearly_training_co2_kg",
    "Yearly training CO2 (kg)",
    ["run_name", "model_name"],
)
TRAIN_CO2_PER_REQUEST = Gauge(
    "mlops_training_carbon_co2_per_request_g", "CO2 per request (g)", ["run_name", "model_name"]
)
TRAIN_YEARLY_INFERENCE_CO2 = Gauge(
    "mlops_training_carbon_yearly_inference_co2_kg",
    "Yearly inference CO2 (kg)",
    ["run_name", "model_name"],
)

DRIFT_OVERALL = Gauge("mlops_drift_overall_detected", "Overall drift detected (0/1)", ["run_name"])
DRIFT_DATA_FRACTION = Gauge(
    "mlops_drift_data_fraction", "Data drift fraction (KS test)", ["run_name"]
)
DRIFT_KL_DIVERGENCE = Gauge(
    "mlops_drift_pred_kl_divergence", "Prediction drift KL divergence", ["run_name"]
)

COMP_BASELINE_ACC = Gauge(
    "mlops_compression_baseline_accuracy", "Baseline accuracy", ["run_name", "method"]
)
COMP_COMPRESSED_ACC = Gauge(
    "mlops_compression_compressed_accuracy", "Compressed accuracy", ["run_name", "method"]
)
COMP_ACCURACY_DROP = Gauge(
    "mlops_compression_accuracy_drop", "Accuracy drop", ["run_name", "method"]
)
COMP_SPEEDUP = Gauge("mlops_compression_speedup_x", "Compression speedup", ["run_name", "method"])


def _get(row, col, cast=None):
    v = row.get(col)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return cast(v) if cast else v


def load_mlflow_metrics():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    try:
        for exp in mlflow.search_experiments():
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="pandas")
            if runs.empty:
                continue
            for _, row in runs.iterrows():
                run_name = _get(row, "tags.mlflow.runName") or row["run_id"][:8]
                training_strategy = _get(row, "tags.training_strategy") or ""
                run_type = _get(row, "tags.run_type") or ""

                if training_strategy:
                    model_name = _get(row, "params.model_name") or ""
                    dataset = _get(row, "params.dataset") or ""
                    lmap = {"run_name": run_name, "model_name": model_name}

                    if (v := _get(row, "metrics.eval_accuracy")) is not None:
                        TRAIN_EVAL_ACCURACY.labels(
                            run_name=run_name, model_name=model_name, dataset=dataset
                        ).set(v)
                    if (v := _get(row, "metrics.model_size_mb")) is not None:
                        TRAIN_MODEL_SIZE.labels(**lmap).set(v)
                    if (v := _get(row, "params.num_epochs", int)) is not None:
                        TRAIN_NUM_EPOCHS.labels(**lmap).set(v)
                    if (v := _get(row, "metrics.carbon_co2_g")) is not None:
                        TRAIN_CO2_G.labels(**lmap).set(v)
                    if (v := _get(row, "metrics.carbon_energy_kwh")) is not None:
                        TRAIN_ENERGY_KWH.labels(**lmap).set(v)
                    if (v := _get(row, "metrics.carbon_yearly_training_co2_kg")) is not None:
                        TRAIN_YEARLY_CO2_KG.labels(**lmap).set(v)
                    if (v := _get(row, "metrics.carbon_co2_per_request_g")) is not None:
                        TRAIN_CO2_PER_REQUEST.labels(**lmap).set(v)
                    if (v := _get(row, "metrics.carbon_yearly_inference_co2_kg")) is not None:
                        TRAIN_YEARLY_INFERENCE_CO2.labels(**lmap).set(v)

                elif run_type == "drift-check":
                    if (v := _get(row, "metrics.overall_drift_detected")) is not None:
                        DRIFT_OVERALL.labels(run_name=run_name).set(v)
                    if (v := _get(row, "metrics.data_drift_fraction")) is not None:
                        DRIFT_DATA_FRACTION.labels(run_name=run_name).set(v)
                    if (v := _get(row, "metrics.pred_drift_kl_divergence")) is not None:
                        DRIFT_KL_DIVERGENCE.labels(run_name=run_name).set(v)

                elif run_type in ("compression", "pruning", "finetune_pruned"):
                    method = _get(row, "tags.compression_method") or run_type
                    lmap = {"run_name": run_name, "method": method}
                    if (v := _get(row, "metrics.baseline_accuracy")) is not None:
                        COMP_BASELINE_ACC.labels(**lmap).set(v)
                    acc = _get(row, "metrics.compressed_accuracy") or _get(
                        row, "metrics.post_accuracy"
                    )
                    if acc is not None:
                        COMP_COMPRESSED_ACC.labels(**lmap).set(acc)
                    drop = _get(row, "metrics.compressed_accuracy_drop") or _get(
                        row, "metrics.pruned_accuracy_drop"
                    )
                    if drop is not None:
                        COMP_ACCURACY_DROP.labels(**lmap).set(drop)
                    speedup = _get(row, "metrics.compressed_speedup_x") or _get(
                        row, "metrics.pruned_speedup_x"
                    )
                    if speedup is not None:
                        COMP_SPEEDUP.labels(**lmap).set(speedup)
    except Exception as e:
        print(f"Warning: could not load MLflow metrics: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_mlflow_metrics()
    yield


app = FastAPI(title="Beans Classifier Monitoring", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)
app.mount("/metrics-raw", make_asgi_app())


class PredictRequest(BaseModel):
    image_id: str = "sample"


class PredictResponse(BaseModel):
    class_name: str
    class_index: int
    confidence: float
    inference_time_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.perf_counter()
    time.sleep(random.uniform(0.002, 0.015))
    class_idx = random.randint(0, len(CLASS_NAMES) - 1)
    class_name = CLASS_NAMES[class_idx]
    confidence = random.betavariate(5, 2)
    elapsed = time.perf_counter() - start
    PREDICTIONS_TOTAL.labels(class_name=class_name).inc()
    INFERENCE_LATENCY.observe(elapsed)
    CONFIDENCE_SCORE.observe(confidence)
    return PredictResponse(
        class_name=class_name,
        class_index=class_idx,
        confidence=round(confidence, 4),
        inference_time_ms=round(elapsed * 1000, 2),
    )
