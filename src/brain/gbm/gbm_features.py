"""
GBM flat-row feature builder — the model bake-off (spec §3 invariants #6, #7, #8).

Single source of truth for the feature set used by BOTH the training path
(vectorized over a DataFrame) and the serving path (single-row computation
on the live tick window). Train/serve parity is guaranteed by construction:
``build_feature_row`` simply delegates to ``build_features`` and returns the
last row.

Inputs:
  - OHLCV DataFrame (UTC index, naive ok) with 'open', 'high', 'low',
    'close', 'volume' columns
  - Regime features already injected (regime_0..4 one-hot + regime_probability),
    typically via FeatureEngineer.inject_regime_features upstream

Output: flat DataFrame of features, NaN-free after the 60-bar warmup
(longest lag horizon). The first 60 rows may carry NaN — drop them in
the caller before fitting.

Lookahead safety (invariant #6) is enforced structurally:
  - Lag features use ``df.shift(k)`` with k >= 1
  - Rolling stats use ``.rolling(W).agg().shift(1)`` so the current bar
    is NEVER part of its own window
"""
from __future__ import annotations

import pandas as pd

# Lag horizons in H4 bars (4h, 20h, 80h ≈ 3.3 days, 240h = 10 days)
LAG_HORIZONS = (1, 5, 20, 60)

# Rolling-stat windows in H4 bars (10 = 40h, 20 = 80h, 60 = 240h)
ROLLING_WINDOWS = (10, 20, 60)

# Regime feature column names — must already be present (injected upstream
# by FeatureEngineer.inject_regime_features). Pass-through, never modified.
REGIME_COLUMNS = ("regime_0", "regime_1", "regime_2", "regime_3", "regime_4",
                  "regime_probability")

# Technical features that get crossed with regime indicators. Trees benefit
# from explicit regime × signal interactions (the model can split on regime
# before splitting on the technical, but precomputing the cross gives a
# direct numeric handle that's cheaper to learn than a deep nested split).
CROSS_BASE_FEATURES = ("close_lag_1", "close_mean_20")


def _add_lag_features(df: pd.DataFrame, out: pd.DataFrame) -> None:
    for col in ("close", "volume"):
        for k in LAG_HORIZONS:
            out[f"{col}_lag_{k}"] = df[col].shift(k)


def _add_rolling_stats(df: pd.DataFrame, out: pd.DataFrame) -> None:
    for col in ("close", "volume"):
        for w in ROLLING_WINDOWS:
            # .shift(1) excludes current bar from its own window — invariant #6
            out[f"{col}_mean_{w}"] = df[col].rolling(window=w).mean().shift(1)
            out[f"{col}_std_{w}"]  = df[col].rolling(window=w).std().shift(1)


def _passthrough_regime(df: pd.DataFrame, out: pd.DataFrame) -> None:
    for r in REGIME_COLUMNS:
        if r in df.columns:
            out[r] = df[r].values


def _add_regime_crosses(df: pd.DataFrame, out: pd.DataFrame) -> None:
    """Cross top-K technicals with each one-hot regime indicator."""
    for base in CROSS_BASE_FEATURES:
        if base not in out.columns:
            continue
        for r in ("regime_0", "regime_1", "regime_2", "regime_3", "regime_4"):
            if r not in df.columns:
                continue
            out[f"{base}_x_{r}"] = out[base] * df[r].values


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full GBM feature DataFrame (training path — vectorized).

    Args:
        df: OHLCV DataFrame with 6 regime features already injected
            (open/high/low/close/volume + regime_0..4 + regime_probability).

    Returns:
        Feature DataFrame indexed identically to ``df``. NaN-free after
        the 60-bar warmup; warmup rows may carry NaN in lag/rolling cols.
        Column order is deterministic (lag → rolling → regime → crosses).
    """
    out = pd.DataFrame(index=df.index)
    _add_lag_features(df, out)
    _add_rolling_stats(df, out)
    _passthrough_regime(df, out)
    _add_regime_crosses(df, out)
    return out


def build_feature_row(history_df: pd.DataFrame) -> dict[str, float]:
    """
    Build a single feature row for the LATEST bar in ``history_df`` (serving path).

    The contract: ``build_feature_row(history.iloc[:t+1])`` must equal
    ``build_features(history).iloc[t]`` bit-for-bit (spec §3 invariant #8).
    Implementation guarantees this by delegating — same code path, same
    pandas semantics, no chance of train/serve drift.

    Args:
        history_df: OHLCV+regime DataFrame ending at the bar to be scored.
            Must contain enough prior bars for the longest lag/rolling
            window (60 bars) to avoid NaN in the returned dict.

    Returns:
        Dict mapping feature name → float for the last row of history_df.
    """
    full = build_features(history_df)
    return full.iloc[-1].to_dict()


# ---------------------------------------------------------------------------
# Phase 2B Q1 (2026-04-27) — rich feature surface for the GBM track.
#
# The thin builder above gives the GBM only 36 features (lag + rolling +
# regime + crosses on close stats). Meanwhile, FeatureEngineer.transform()
# already produces ~56 indicator-rich technical features per timeframe
# (ADX, Kaufman ER, MACD, RSI, MFI, OBV, BB, ATR, Hurst, skew/kurt, etc.).
# Phase A's GBM was trained blind to those — the rich builder closes that
# information gap so we can A/B "GBM-thin vs GBM-rich" cleanly.
#
# Input contract:
#   ``rich_feature_df`` is the output of ``FeatureEngineer.transform()`` PLUS
#   regime columns from ``inject_regime_features()``. It does NOT contain
#   raw OHLCV (transform drops those). Index is the H4 bar grid.
#
# Output: rich_feature_df + GBM-specific augmentations (lagged log_return +
# regime crosses on top indicators). Train/serve parity is guaranteed by
# the same vectorized-then-take-last pattern as the thin builder.
# ---------------------------------------------------------------------------

# Indicators chosen as cross bases — strong tree-split candidates that
# benefit from explicit regime conditioning. Skip if absent in input.
RICH_CROSS_BASES = (
    "adx", "efficiency_ratio", "rsi_14",
    "atr_14", "macd_histogram", "rolling_skewness_20",
)

# Lag horizons applied to log returns (the rich set has log_return at
# 1/5/10/20 already, but explicit shifts of log_return give the model a
# direct handle on "what was the 1-bar return N bars ago" without needing
# to learn the chain through autocorr features).
RICH_LAG_HORIZONS = (1, 5, 20)
RICH_LAG_BASES = ("log_return",)


def build_features_rich(rich_feature_df: pd.DataFrame) -> pd.DataFrame:
    """Build the GBM rich-feature DataFrame.

    Augmentations on top of the rich technical set:
      - Lagged log returns at 1/5/20 bars — direct multi-horizon momentum
      - Regime crosses on 6 top technicals — explicit regime conditioning
        (the tree can split on regime first then split on the technical,
        but precomputing the cross gives a cheaper-to-learn handle).
    """
    out = rich_feature_df.copy()

    # Lagged log returns
    for col in RICH_LAG_BASES:
        if col not in out.columns:
            continue
        for k in RICH_LAG_HORIZONS:
            out[f"{col}_lag_{k}"] = out[col].shift(k)

    # Regime crosses on top technicals
    regime_cols = ("regime_0", "regime_1", "regime_2", "regime_3", "regime_4")
    for base in RICH_CROSS_BASES:
        if base not in out.columns:
            continue
        for r in regime_cols:
            if r not in out.columns:
                continue
            out[f"{base}_x_{r}"] = out[base] * out[r]

    return out


def build_feature_row_rich(rich_history_df: pd.DataFrame) -> dict[str, float]:
    """Single-row companion to ``build_features_rich`` (serving path).

    Same contract as ``build_feature_row``: vectorize then take the last
    row, so train/serve are bit-for-bit identical.
    """
    full = build_features_rich(rich_history_df)
    return full.iloc[-1].to_dict()
