"""Smoke test: MLflow is installed and basic tracking API works."""
from __future__ import annotations


def test_mlflow_importable():
    import mlflow
    assert hasattr(mlflow, "start_run")
    assert hasattr(mlflow, "log_params")
    assert hasattr(mlflow, "log_metrics")


def test_mlflow_file_backend_works(tmp_path):
    """Verify the file-based backend can store a run end-to-end."""
    import mlflow

    mlflow.set_tracking_uri(f"file:{tmp_path}")
    mlflow.set_experiment("test_smoke")

    with mlflow.start_run(run_name="smoke-run") as run:
        mlflow.log_param("a", 1)
        mlflow.log_metric("b", 2.5)

    # Confirm the run is retrievable
    client = mlflow.MlflowClient(tracking_uri=f"file:{tmp_path}")
    fetched = client.get_run(run.info.run_id)
    assert fetched.data.params["a"] == "1"
    assert fetched.data.metrics["b"] == 2.5
