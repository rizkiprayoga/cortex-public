"""
Feature-distribution drift detection (A-8).

Used by the daily cron in main.py to monitor whether the features the bot
sees today still look like the features the LSTM was trained on.

Two complementary metrics:

- **PSI (Population Stability Index)**: binned KL-style divergence. Rule of
  thumb: <0.10 "no shift", 0.10-0.25 "slight", >0.25 "significant", >0.35
  "major". Works well on continuous features and is the industry standard
  for credit-risk / tabular-ML drift monitoring.

- **KS (Kolmogorov-Smirnov statistic)**: max absolute difference between
  empirical CDFs. Non-parametric, complements PSI by detecting tail shifts
  PSI may miss when the mean is unchanged.

Also provides persistence of per-symbol training distributions alongside
the saved model, so the comparison has something to reference post-retrain.

### Schema versions

- **v1** (deprecated): samples stored post-``FeatureEngineer.to_matrix()``
  z-score normalization. Broken: the z-score scale is batch-local, so a
  30-day drift window uses a different reference than the years-long
  training batch. ``load_training_distribution`` now refuses v1 files and
  logs a bootstrap instruction.
- **v2** (current): samples stored raw (pre-normalization). Drift is
  computed on raw values; low-cardinality / categorical columns
  (one-hots, cyclic time encodings) are auto-skipped because PSI/KS
  are undefined on them when the current window collapses into a
  single category.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 2
# Features with ≤ this many unique training values are treated as
# categorical / one-hot / cyclic and skipped during drift computation.
# One-hot indicators have 2, weekday/hour encodings have 5-24, month
# encodings have 12. 20 is the conservative cut-off that catches
# weekday/hour/month encodings while leaving genuine low-cardinality
# continuous signals alone.
CATEGORICAL_UNIQUE_THRESHOLD = 20


# ───── PSI ────────────────────────────────────────────────────────────────

def psi(
    training: np.ndarray, current: np.ndarray, bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Population Stability Index between two 1-D samples.

    Uses quantile-based bins derived from ``training`` so that each bin
    originally has ≈ 1/bins of the training mass. Returns 0.0 for
    identical samples, higher values for more drift.

    ``epsilon`` stabilizes log() when a bin is empty in one sample.
    """
    training = np.asarray(training, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()
    if training.size == 0 or current.size == 0:
        return 0.0

    # Quantile bin edges from training
    q = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(training, q))
    if edges.size < 2:
        # Degenerate: all-equal training data. Can't bin.
        return 0.0

    # Extend the outer edges to +/- infinity so current values outside the
    # training range still fall into the outermost bins.
    edges = edges.astype(float).copy()
    edges[0] = -np.inf
    edges[-1] = np.inf

    tr_hist, _ = np.histogram(training, bins=edges)
    cu_hist, _ = np.histogram(current, bins=edges)

    tr_frac = tr_hist / max(tr_hist.sum(), 1)
    cu_frac = cu_hist / max(cu_hist.sum(), 1)

    # Replace zeros with epsilon so log is finite
    tr_frac = np.where(tr_frac == 0, epsilon, tr_frac)
    cu_frac = np.where(cu_frac == 0, epsilon, cu_frac)

    psi_value = float(np.sum((cu_frac - tr_frac) * np.log(cu_frac / tr_frac)))
    return psi_value


# ───── KS ─────────────────────────────────────────────────────────────────

def ks_statistic(training: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (D value, not the p-value).

    D = sup |CDF_train(x) - CDF_current(x)|. Range [0, 1].

    Uses scipy.stats.ks_2samp if available; falls back to a pure-numpy
    implementation so tests still pass in a minimal env.
    """
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(training, current).statistic)
    except Exception:
        # Fallback: manual empirical-CDF computation
        t = np.sort(np.asarray(training, dtype=float).ravel())
        c = np.sort(np.asarray(current, dtype=float).ravel())
        all_values = np.concatenate([t, c])
        cdf_t = np.searchsorted(t, all_values, side="right") / max(len(t), 1)
        cdf_c = np.searchsorted(c, all_values, side="right") / max(len(c), 1)
        return float(np.max(np.abs(cdf_t - cdf_c)))


# ───── Training-distribution snapshot ─────────────────────────────────────

def save_training_distribution(
    path: Path | str,
    symbol: str, timeframe: str,
    feature_matrix: np.ndarray, feature_names: tuple[str, ...] | list[str],
    n_samples_keep: int = 2000,
) -> Path:
    """Serialize per-feature training summary + a downsampled reference.

    Saves each feature's mean/std/quantiles and a random sample of up to
    ``n_samples_keep`` raw values — the sample is what PSI / KS will
    compare against when drift is computed later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"feature_matrix must be 2-D, got shape {matrix.shape}")
    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_matrix has {matrix.shape[1]} cols but "
            f"{len(feature_names)} feature_names given"
        )

    rng = np.random.default_rng(42)
    n_rows = matrix.shape[0]
    if n_rows > n_samples_keep:
        idx = rng.choice(n_rows, size=n_samples_keep, replace=False)
        sampled = matrix[idx]
    else:
        sampled = matrix

    features: dict[str, dict] = {}
    for i, fname in enumerate(feature_names):
        col = matrix[:, i]
        sample_col = sampled[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            col = np.array([0.0])
        features[str(fname)] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "q10": float(np.quantile(col, 0.10)),
            "q50": float(np.quantile(col, 0.50)),
            "q90": float(np.quantile(col, 0.90)),
            # Retain the sampled reference for PSI/KS at drift time
            "samples": [float(v) for v in sample_col[np.isfinite(sample_col)]],
        }

    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "symbol": symbol,
        "timeframe": timeframe,
        "n_samples": int(n_rows),
        "feature_names": list(feature_names),
        "features": features,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    logger.info("Training distribution saved for %s -> %s", symbol, path)
    return path


def load_training_distribution(path: Path | str) -> Optional[dict]:
    """Load the JSON produced by ``save_training_distribution``.

    Returns None when the file is missing or uses a deprecated schema so
    callers can degrade gracefully. v1 files stored z-scored samples whose
    scale reference was batch-local — drift metrics against them are
    unreliable. Operators must re-run the training-dist bootstrap after
    upgrade.
    """
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    schema = payload.get("schema_version", 1)
    if schema < CURRENT_SCHEMA_VERSION:
        logger.warning(
            "Training distribution at %s uses schema v%s (current is v%s) — "
            "samples were z-scored with a batch-local scale and produce "
            "unreliable drift metrics. Re-run "
            "`python scripts/bootstrap_training_distributions.py` to upgrade.",
            path, schema, CURRENT_SCHEMA_VERSION,
        )
        return None
    return payload


# ───── Drift computation ──────────────────────────────────────────────────

def compute_drift(
    training_dist: dict,
    current_feature_matrix: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    categorical_threshold: int = CATEGORICAL_UNIQUE_THRESHOLD,
) -> dict:
    """Compute per-feature PSI + KS against a saved training distribution.

    Args:
        training_dist: payload from ``load_training_distribution``
        current_feature_matrix: 2-D array [n_rows, n_features] of recent data
        feature_names: names matching columns of ``current_feature_matrix``
        categorical_threshold: features with ≤ this many unique training
            values are skipped (PSI/KS are undefined on one-hot / cyclic
            encodings when the current window collapses into a single bin).

    Returns a dict with ``per_feature`` (dict of name -> {psi, ks}),
    ``psi_max``, ``ks_max``, ``n_current_samples``, and ``skipped`` (list
    of (name, reason) tuples) for operator visibility.
    """
    matrix = np.asarray(current_feature_matrix, dtype=float)
    per_feature: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    for i, fname in enumerate(feature_names):
        tr_samples = training_dist.get("features", {}).get(fname, {}).get("samples", [])
        if not tr_samples:
            skipped.append((str(fname), "no_training_samples"))
            continue
        tr = np.asarray(tr_samples, dtype=float)
        tr = tr[np.isfinite(tr)]
        if tr.size == 0:
            skipped.append((str(fname), "no_finite_training_samples"))
            continue
        # Skip low-cardinality features (one-hots, cyclic encodings). PSI/KS
        # on them blow up the instant the current window lands in a single
        # category, which is the norm for month/weekday/regime features.
        tr_unique = len(np.unique(tr))
        if tr_unique <= categorical_threshold:
            skipped.append((str(fname), f"categorical_unique={tr_unique}"))
            continue
        cu = matrix[:, i]
        cu = cu[np.isfinite(cu)]
        if cu.size == 0:
            skipped.append((str(fname), "no_finite_current_samples"))
            continue
        per_feature[str(fname)] = {
            "psi": psi(tr, cu),
            "ks": ks_statistic(tr, cu),
        }
    psi_max = max((v["psi"] for v in per_feature.values()), default=0.0)
    ks_max = max((v["ks"] for v in per_feature.values()), default=0.0)
    return {
        "per_feature": per_feature,
        "psi_max": psi_max,
        "ks_max": ks_max,
        "n_current_samples": int(matrix.shape[0]),
        "skipped": skipped,
    }
