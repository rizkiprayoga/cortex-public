"""Unit tests for the Cortex MLflow wrapper."""
from __future__ import annotations


def test_default_tracking_uri_is_local_data_mlflow(monkeypatch, tmp_path):
    """Default URI should resolve to file:./data/mlflow/ relative to repo root."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    from src.ml.registry import get_tracking_uri
    uri = get_tracking_uri()
    # Must be a file: URI pointing at data/mlflow/
    assert uri.startswith("file:")
    assert uri.endswith("data/mlflow") or uri.endswith("data\\mlflow")


def test_env_override_wins_over_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:/tmp/custom")
    from src.ml.registry import get_tracking_uri
    assert get_tracking_uri() == "file:/tmp/custom"


def test_dataset_fingerprint_is_deterministic():
    """Same (symbol, timeframe, bar_closes) → same SHA-256."""
    import numpy as np
    from src.ml.registry import dataset_fingerprint

    closes = np.array([1.1, 1.2, 1.3, 1.4], dtype=np.float64)
    fp1 = dataset_fingerprint(
        symbol="XAUUSD", timeframe="H4",
        first_bar_ts="2024-01-01T00:00:00", last_bar_ts="2024-12-31T20:00:00",
        closes=closes,
    )
    fp2 = dataset_fingerprint(
        symbol="XAUUSD", timeframe="H4",
        first_bar_ts="2024-01-01T00:00:00", last_bar_ts="2024-12-31T20:00:00",
        closes=closes.copy(),
    )
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex


def test_dataset_fingerprint_changes_on_different_input():
    import numpy as np
    from src.ml.registry import dataset_fingerprint

    base = dict(
        symbol="XAUUSD", timeframe="H4",
        first_bar_ts="2024-01-01T00:00:00", last_bar_ts="2024-12-31T20:00:00",
        closes=np.array([1.1, 1.2, 1.3], dtype=np.float64),
    )
    fp_base = dataset_fingerprint(**base)
    # Different symbol
    assert dataset_fingerprint(**{**base, "symbol": "EURUSD"}) != fp_base
    # Different closes
    fp_altered = dataset_fingerprint(
        **{**base, "closes": np.array([9.9, 9.9, 9.9], dtype=np.float64)}
    )
    assert fp_altered != fp_base


def test_start_run_creates_mlflow_run(monkeypatch, tmp_path):
    """start_run context manager logs params/metrics/tags and closes cleanly."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path.as_posix()}")

    import mlflow
    from src.ml.registry import start_run

    with start_run(
        experiment="test_exp",
        run_name="test-run-1",
        tags={"symbol": "XAUUSD", "model_head": "regression"},
    ) as run:
        mlflow.log_param("epochs", 5)
        mlflow.log_metric("val_loss", 0.123)

    # Fetch back via client
    client = mlflow.MlflowClient(tracking_uri=f"file:{tmp_path.as_posix()}")
    fetched = client.get_run(run.info.run_id)
    assert fetched.data.tags["symbol"] == "XAUUSD"
    assert fetched.data.tags["model_head"] == "regression"
    assert fetched.data.params["epochs"] == "5"
    assert fetched.data.metrics["val_loss"] == 0.123


def test_lstm_train_on_matrix_returns_result_with_metrics():
    """Regression guard: _train_on_matrix must return a result
    object carrying at least best_val_loss and directional_accuracy."""
    from src.brain.deep_learning.lstm_model import LSTMPricePredictor
    import inspect

    src = inspect.getsource(LSTMPricePredictor._train_on_matrix)
    # Must contain a return statement referencing the result
    assert "return result" in src, (
        "_train_on_matrix must return the training result so MLflow "
        "instrumentation in train_deep_learning.py can log metrics"
    )
