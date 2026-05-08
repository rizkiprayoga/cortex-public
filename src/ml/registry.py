"""
Cortex MLflow wrapper — thin layer of project conventions around mlflow.

See docs/operations/model_registry.md for operator usage.

Conventions enforced here:
- Tracking URI defaults to file:./data/mlflow/ (git-ignored).
- Experiments are named per model family (e.g. "hmm_regime", "lstm_price").
- Each run carries tags: symbol, model_head, training_script.
- Dataset fingerprint is logged as a param + as an artifact JSON file.
"""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_DIR = REPO_ROOT / "data" / "mlflow"


def get_tracking_uri() -> str:
    """
    Resolve the MLflow tracking URI.

    Env var MLFLOW_TRACKING_URI wins if set. Otherwise defaults to the
    file-based backend at REPO_ROOT/data/mlflow/.
    """
    override = os.environ.get("MLFLOW_TRACKING_URI")
    if override:
        return override
    return f"file:{DEFAULT_TRACKING_DIR.as_posix()}"


def dataset_fingerprint(
    symbol: str,
    timeframe: str,
    first_bar_ts: str,
    last_bar_ts: str,
    closes: np.ndarray,
) -> str:
    """
    Compute a deterministic SHA-256 over the training-data characteristics.

    Two training runs with the same fingerprint saw the same data. Changes
    to symbol, timeframe, window bounds, OR the close-price series
    produce a different hash.
    """
    h = hashlib.sha256()
    h.update(symbol.encode("utf-8"))
    h.update(b"|")
    h.update(timeframe.encode("utf-8"))
    h.update(b"|")
    h.update(first_bar_ts.encode("utf-8"))
    h.update(b"|")
    h.update(last_bar_ts.encode("utf-8"))
    h.update(b"|")
    h.update(closes.astype(np.float64).tobytes())
    return h.hexdigest()


@contextmanager
def start_run(
    experiment: str,
    run_name: str,
    tags: Optional[dict] = None,
) -> Iterator:
    """
    Context manager wrapping mlflow.start_run() with Cortex conventions.

    - Sets tracking URI from get_tracking_uri() if not already set on the
      active mlflow module.
    - Ensures the experiment exists (creates if needed).
    - Applies tags immediately after run creation so they're attached even
      if the body raises.

    Yields the mlflow.ActiveRun so callers can access run.info.run_id etc.
    """
    import mlflow

    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run
