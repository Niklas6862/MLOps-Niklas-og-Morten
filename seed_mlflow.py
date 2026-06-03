"""Seed mlruns with realistic fake data so the Grafana dashboard panels populate."""
import mlflow

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("beans-classifier")

# --- Training run (FP32) ---
with mlflow.start_run(run_name="vit-base-fp32"):
    mlflow.set_tag("training_strategy", "FP32")
    mlflow.log_param("model_name", "google/vit-base-patch16-224")
    mlflow.log_param("dataset", "beans")
    mlflow.log_param("num_epochs", 3)
    mlflow.log_metric("eval_accuracy", 0.9312)
    mlflow.log_metric("model_size_mb", 327.4)
    mlflow.log_metric("carbon_co2_g", 42.7)
    mlflow.log_metric("carbon_energy_kwh", 0.118)
    mlflow.log_metric("carbon_yearly_training_co2_kg", 15.6)
    mlflow.log_metric("carbon_co2_per_request_g", 0.00031)
    mlflow.log_metric("carbon_yearly_inference_co2_kg", 0.97)

# --- Drift check run ---
with mlflow.start_run(run_name="drift-check-v1"):
    mlflow.set_tag("run_type", "drift-check")
    mlflow.set_tag("reference", "train")
    mlflow.set_tag("current_split", "validation")
    mlflow.log_metric("overall_drift_detected", 0)
    mlflow.log_metric("data_drift_fraction", 0.04)
    mlflow.log_metric("pred_drift_kl_divergence", 0.012)

# --- Compression run ---
with mlflow.start_run(run_name="dynamic-quant-v1"):
    mlflow.set_tag("run_type", "compression")
    mlflow.set_tag("compression_method", "dynamic_quant")
    mlflow.log_metric("baseline_accuracy", 0.9312)
    mlflow.log_metric("compressed_accuracy", 0.9187)
    mlflow.log_metric("compressed_accuracy_drop", 0.0125)
    mlflow.log_metric("compressed_speedup_x", 2.3)

print("Done — mlruns/ is ready. Restart the monitoring stack or POST /reload-metrics.")
