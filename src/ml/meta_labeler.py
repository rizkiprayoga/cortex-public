"""
Meta-labeler for Cortex signals (M-1 + Phase A Sprint 4 extension).

López-de-Prado meta-labeling: the primary model (HMM + LSTM/GBM) decides
the SIDE (buy / sell). A secondary model (LightGBM here) decides the
SIZE — in practice that means a binary decision to take the trade or
skip it.

Training:
    Given a historical trade (from ``backtest_trades``), we know its
    outcome (win / loss via ``pnl > 0``). We train the secondary model
    on the *features visible at entry time*:
      - Base (5): combined_score, regime, direction, hour, dow.
      - Fundamentals (17): FRED macro / Stooq yields / ECB curve / COT
        positioning / yfinance cross-asset, joined onto each trade via
        ``read_feature_store_safe(as_of=entry_ts)`` (lookahead-safe per
        spec invariant #11).

Inference:
    At signal time, compute the same features and ask the labeler for
    ``P(win)``. If the probability clears a per-symbol threshold, the
    trade proceeds; otherwise it is skipped.

    Live signal_combiner currently passes ``fundamentals=None`` because
    the bot's hot path doesn't yet wire fundamentals through. The
    enriched-fundamentals slots are NaN-filled at inference; LightGBM
    handles NaN natively, but predictions are degraded vs training.
    Operator can ship live-fundamentals plumbing in a follow-up to
    restore train/serve parity.

Overfitting safety:
    The labeler never decides direction, so it cannot cascade the
    primary model's errors. Worst case it filters away ~all signals
    (zero PF instead of negative PF).

Why LightGBM:
    Tabular financial data — gradient boosting dominates (Numerai,
    Kaggle, recent 2025 surveys). Handles NaN natively, trains in
    seconds, saves cleanly as a pickle.

Spec invariants enforced here:
  - #11 lookahead-safe fundamentals via read_feature_store_safe (the
    helper ``_enrich_with_fundamentals`` is the only call site)
  - #12 schema hash baked into bundle on save; signal_combiner
    refuses to load on mismatch
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.brain.meta_labeler_features import (
    COT_GROUP_PER_SYMBOL,
    COT_TFF_PREFIX_PER_SYMBOL,
    EXEC_FEATURE_NAMES,
    EXPECTED_FEATURE_NAMES,
    EXPECTED_SCHEMA_HASH,
    FUNDAMENTAL_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)


# Path templates — primary-suffixed canonical, plus the legacy
# unsuffixed name for back-compat with M-1 bundles still on disk.
MODEL_PATH_TEMPLATE = "data/models/meta_labeler_{symbol}_{primary}.pkl"
LEGACY_MODEL_PATH_TEMPLATE = "data/models/meta_labeler_{symbol}.pkl"

# Categorical encodings — training and inference must agree on ordinals.
REGIME_VALUES = ("Crash", "Bear", "Neutral", "Bull", "Euphoria")
DIRECTION_VALUES = ("buy", "sell")

# Re-export EXPECTED_FEATURE_NAMES under FEATURE_NAMES for back-compat
# with imports elsewhere. The constant is the source of truth.
FEATURE_NAMES = EXPECTED_FEATURE_NAMES


@dataclass(frozen=True)
class MetaLabelerTrainingResult:
    symbol: str
    n_train: int
    n_val: int
    val_accuracy: float
    val_precision: float
    val_recall: float
    coverage_at_default_threshold: float
    pf_without_gate: float
    pf_with_gate: float
    threshold: float


def _encode_regime(label: str) -> int:
    try:
        return REGIME_VALUES.index(label)
    except (ValueError, TypeError):
        return -1


def _encode_direction(label: str) -> int:
    try:
        return DIRECTION_VALUES.index((label or "").lower())
    except (ValueError, TypeError):
        return -1


# ---------------------------------------------------------------------------
# Fundamental enrichment — invariant #11 (lookahead-safe via _safe wrapper)
# ---------------------------------------------------------------------------

# Per-feature_group: list of (db_column, schema_column) pairs to extract
# from the latest row at as_of. db_column is the raw column name in
# feature_store JSONB; schema_column is the unified output name (already
# in EXPECTED_FEATURE_NAMES).
_FUNDAMENTAL_COLUMN_MAP: dict[str, list[tuple[str, str]]] = {
    "fred_macro": [
        ("dxy",        "fred_macro__dxy"),
        ("vix",        "fred_macro__vix"),
        ("t10y",       "fred_macro__t10y"),
        ("t2y",        "fred_macro__t2y"),
        ("fed_funds",  "fred_macro__fed_funds"),
    ],
    "stooq_yields": [
        ("us_10y_daily",   "stooq_yields__us_10y_daily"),
        ("us_2y_daily",    "stooq_yields__us_2y_daily"),
        ("us_slope_daily", "stooq_yields__us_slope_daily"),
    ],
    "ecb_yield_curve": [
        ("eu_aaa_10y_daily", "ecb_yield_curve__eu_aaa_10y_daily"),
        ("eu_aaa_2y_daily",  "ecb_yield_curve__eu_aaa_2y_daily"),
    ],
    "cot_disagg": [
        ("net_spec",      "cot_disagg__net_spec"),
        ("open_interest", "cot_disagg__open_interest"),
    ],
    "cot_tff": [
        # db_column is currency-templated — resolved per-symbol at read time.
        ("{cur}_net_spec", "cot_tff__net_spec"),
        ("{cur}_lev_long", "cot_tff__lev_long"),
    ],
    "yfinance_cross_asset": [
        ("vix_level",  "yfinance_cross_asset__vix_level"),
        ("dxy_zscore", "yfinance_cross_asset__dxy_zscore"),
        ("spx_zscore", "yfinance_cross_asset__spx_zscore"),
    ],
}


async def _read_one_fundamental_row(
    store, symbol: str, feature_group: str, as_of,
) -> dict:
    """Read the most-recent feature_store row for (symbol, feature_group)
    at-or-before ``as_of`` (lookahead-safe).

    Returns the JSONB ``values`` dict, or empty dict if no row available.
    Symbol routing: ``ecb_yield_curve`` is global; ``cot_disagg`` is
    XAUUSD-only; everything else uses the trade's symbol.
    """
    from src.data_pipeline.feature_engineering import read_feature_store_safe

    if feature_group == "ecb_yield_curve":
        read_symbol = "_GLOBAL"
    else:
        read_symbol = symbol

    try:
        df = await read_feature_store_safe(
            store=store, symbol=read_symbol,
            feature_group=feature_group, as_of=as_of,
        )
    except Exception as exc:
        logger.debug(
            "[%s/%s] read_feature_store_safe failed at %s: %s",
            symbol, feature_group, as_of, exc,
        )
        return {}

    if df is None or df.empty:
        return {}
    # Take the latest row at-or-before as_of (df is sorted ascending).
    return df.iloc[-1].to_dict()


async def _enrich_with_fundamentals(
    trades_df: pd.DataFrame, symbol: str, store,
) -> pd.DataFrame:
    """Join 6 feature_store groups onto each trade via lookahead-safe reads.

    Spec invariant #11 — uses ``read_feature_store_safe`` exclusively
    (subtracts each source's release_lag_hours so the join can't pull
    rows from after the trade's entry_ts).

    For each row in ``trades_df`` (must contain ``entry_time``), looks
    up the latest feature_store row in each applicable group at or
    before the trade's entry timestamp. Per-symbol routing:
      - XAUUSD reads ``cot_disagg``; FX reads ``cot_tff`` with currency
        prefix from ``COT_TFF_PREFIX_PER_SYMBOL``.
      - The COT group not applicable to a symbol stays NaN (uniform
        schema across all symbols per invariant #12).

    Returns ``trades_df`` with 17 new columns matching
    ``FUNDAMENTAL_FEATURE_NAMES`` (NaN where the source was unavailable).
    """
    out = trades_df.copy()
    # Initialize all fundamental cols to NaN — uniform schema; whatever
    # we don't fill stays NaN and LightGBM handles it natively.
    for col in FUNDAMENTAL_FEATURE_NAMES:
        out[col] = np.nan

    if len(trades_df) == 0:
        return out

    entry_ts = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
    # read_feature_store_safe expects naive UTC datetime (matches the
    # convention used everywhere else — DB rows are naive UTC).
    entry_ts_naive = entry_ts.dt.tz_convert(None) if entry_ts.dt.tz is not None else entry_ts

    # Resolve which COT group applies to this symbol; skip the other.
    cot_group_for_symbol = COT_GROUP_PER_SYMBOL.get(symbol)
    cot_prefix = COT_TFF_PREFIX_PER_SYMBOL.get(symbol)

    applicable_groups = []
    for fg in ("fred_macro", "stooq_yields", "ecb_yield_curve",
               "cot_disagg", "cot_tff", "yfinance_cross_asset"):
        if fg in ("cot_disagg", "cot_tff") and fg != cot_group_for_symbol:
            # Skip the non-applicable COT group; its cols stay NaN.
            continue
        applicable_groups.append(fg)

    for i, ts in enumerate(entry_ts_naive):
        if pd.isna(ts):
            continue
        # ts is a Timestamp; read_feature_store_safe wants datetime
        as_of = ts.to_pydatetime()
        for fg in applicable_groups:
            row = await _read_one_fundamental_row(store, symbol, fg, as_of)
            if not row:
                continue
            for db_col, schema_col in _FUNDAMENTAL_COLUMN_MAP[fg]:
                # cot_tff db_col is currency-templated — substitute now.
                if fg == "cot_tff" and "{cur}" in db_col:
                    if cot_prefix is None:
                        continue
                    db_col_resolved = db_col.format(cur=cot_prefix)
                else:
                    db_col_resolved = db_col
                val = row.get(db_col_resolved)
                if val is not None:
                    out.iat[i, out.columns.get_loc(schema_col)] = float(val)
    return out


# ---------------------------------------------------------------------------
# Execution-conditional feature enrichment (Phase 2B Option 2 — 2026-04-27).
#
# 4 features that change every H4 bar — gives the meta-labeler something
# fast-moving to discriminate against, since fundamentals barely budge over
# a 22-month OOS window. Lookahead-safe by construction:
#   - RV uses H4 bars with timestamp STRICTLY BEFORE entry_time.
#   - score_avg_20 uses .shift(1) so the trade's own score isn't included
#     in its own rolling window.
# ---------------------------------------------------------------------------

async def _enrich_with_exec_features(
    trades_df: pd.DataFrame, symbol: str, store,
) -> pd.DataFrame:
    """Join 4 execution-conditional features onto each trade.

    Loads the H4 OHLCV slice for the window covering all trade entries
    (with 60-bar warmup) once via ``DataStore.get_ohlcv_range``, computes
    rolling 20/60-bar log-return std, then for each trade looks up the
    latest bar with ``bar_time < entry_time``. Avoids per-trade DB hits.

    score_avg_20 is computed inline from ``combined_score.shift(1).rolling(20)``
    on the (already chronologically sorted) trades_df.
    """
    out = trades_df.copy()
    for col in EXEC_FEATURE_NAMES:
        out[col] = np.nan

    if len(trades_df) == 0:
        return out

    entry_ts = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
    entry_naive = (
        entry_ts.dt.tz_convert(None) if entry_ts.dt.tz is not None else entry_ts
    )
    earliest = entry_naive.min()
    latest = entry_naive.max()
    if pd.isna(earliest) or pd.isna(latest):
        return out

    # 60 H4 bars = 240h = 10 days. Pad to 30 days for safe rolling warmup
    # (covers weekends + any gaps in the OHLCV history).
    from datetime import timedelta as _td
    fetch_start = (earliest - _td(days=30)).to_pydatetime()
    fetch_end = (latest + _td(hours=4)).to_pydatetime()

    try:
        bars = await store.get_ohlcv_range(symbol, "H4", fetch_start, fetch_end)
    except Exception as exc:
        logger.warning(
            "[%s] _enrich_with_exec_features: get_ohlcv_range failed: %s",
            symbol, exc,
        )
        bars = None

    if bars is None or len(bars) == 0:
        # No bars → leave RV cols as NaN. score_avg_20 can still be computed.
        scores = trades_df["combined_score"].astype(float)
        score_avg_20 = scores.shift(1).rolling(20, min_periods=5).mean()
        out["exec__score_avg_20"] = score_avg_20.values
        return out

    # Normalize to a DataFrame with naive UTC index + 'close' col.
    bars = bars.copy()
    if not isinstance(bars.index, pd.DatetimeIndex):
        # Some DataStore impls return a 'bar_timestamp' column.
        ts_col = (
            "bar_timestamp" if "bar_timestamp" in bars.columns
            else "timestamp" if "timestamp" in bars.columns
            else None
        )
        if ts_col is None:
            logger.warning(
                "[%s] _enrich_with_exec_features: bars df has no timestamp col",
                symbol,
            )
            return out
        bars.index = pd.to_datetime(bars[ts_col], utc=True, errors="coerce")
    if bars.index.tz is not None:
        bars.index = bars.index.tz_convert(None)
    bars = bars.sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]

    if "close" not in bars.columns:
        logger.warning("[%s] _enrich_with_exec_features: no 'close' col", symbol)
        return out

    # Log returns + rolling std. min_periods set so partial windows yield NaN
    # (LightGBM handles NaN; better than emitting tiny-sample stds).
    log_ret = np.log(bars["close"].astype(float) / bars["close"].shift(1).astype(float))
    rv_short = log_ret.rolling(20, min_periods=20).std()
    rv_long = log_ret.rolling(60, min_periods=60).std()

    bar_index = bars.index.values  # numpy datetime64[ns]
    rv_short_arr = rv_short.values
    rv_long_arr = rv_long.values

    rv_short_col = out.columns.get_loc("exec__rv_short_20")
    rv_long_col = out.columns.get_loc("exec__rv_long_60")
    rv_ratio_col = out.columns.get_loc("exec__rv_ratio")

    for i, ts in enumerate(entry_naive):
        if pd.isna(ts):
            continue
        ts_np = np.datetime64(pd.Timestamp(ts).to_datetime64(), "ns")
        # side="left" → idx points to first bar with bar_ts >= ts. We want
        # bar_ts STRICTLY < ts, so step back one. (No off-by-one even when
        # ts exactly matches a bar boundary — that bar's close would be
        # synchronous with entry, which is a leak we don't want.)
        idx = np.searchsorted(bar_index, ts_np, side="left") - 1
        if idx < 0:
            continue
        rs = rv_short_arr[idx]
        rl = rv_long_arr[idx]
        if not (np.isfinite(rs) and np.isfinite(rl)) or rl <= 0.0:
            continue
        out.iat[i, rv_short_col] = float(rs)
        out.iat[i, rv_long_col] = float(rl)
        out.iat[i, rv_ratio_col] = float(rs / rl)

    # score_avg_20: rolling mean of prior trades' combined_score (shift(1)
    # so the trade's own score isn't in its own window). min_periods=5 to
    # avoid noisy estimates over the first 5 trades — those rows get NaN.
    scores = trades_df["combined_score"].astype(float)
    score_avg_20 = scores.shift(1).rolling(20, min_periods=5).mean()
    out["exec__score_avg_20"] = score_avg_20.values

    return out


# ---------------------------------------------------------------------------
# Fundamentals prefetch + point-in-time lookup (Option B — full-fidelity
# meta-labeler inference). Lets the backtest hot path read fundamentals
# from feature_store once per (symbol, window) at start, then index into
# the cache by bar timestamp for each signal — much faster than the per-
# trade read pattern used at training time. Same lookahead-safe contract.
# ---------------------------------------------------------------------------

async def prefetch_fundamentals_for_window(
    store, symbol: str, start_ts, end_ts,
) -> "FundamentalsLookup":
    """Pre-fetch all 6 feature_groups for ``symbol`` over [start_ts, end_ts].

    Calls ``read_feature_store_safe`` once per applicable group with
    ``as_of=end_ts`` and ``start=start_ts``. The returned ``FundamentalsLookup``
    object lets the inference path resolve fundamentals at any bar
    timestamp within the window via ``.get(bar_ts)``.

    LOOKAHEAD-SAFETY (CRITICAL — fixed 2026-04-27):

    ``read_feature_store_safe`` subtracts ``release_lag_hours`` from the
    upper bound (``as_of=end_ts``) of the prefetch window — but the
    returned df is indexed by RAW observation timestamps, not by their
    release-time. A naive ``searchsorted(bar_ts)`` at lookup time would
    return rows with raw_timestamp <= bar_ts, which CAN INCLUDE ROWS THAT
    WEREN'T YET RELEASED at bar_ts (raw_timestamp > bar_ts - release_lag).
    That's a lookahead leak.

    Fix: at prefetch time we record each group's ``release_lag_hours``.
    At lookup time we shift the bar's timestamp BACKWARD by release_lag
    before searchsorting — so the row we pick is the latest one whose
    raw_timestamp + release_lag <= bar_ts (i.e., truly released by bar_ts).
    """
    from src.data_pipeline.feature_engineering import (
        read_feature_store_safe, _load_data_feeds_yaml,
    )

    cot_group_for_symbol = COT_GROUP_PER_SYMBOL.get(symbol)
    cot_prefix = COT_TFF_PREFIX_PER_SYMBOL.get(symbol)
    feeds_cfg = _load_data_feeds_yaml()

    # Pull data from BEFORE start_ts too — lookup at bar_ts uses
    # ``bar_ts - release_lag``, which for early bars in the window can be
    # weeks before start_ts (fred_macro release_lag is 504h = 21 days).
    # 400 days of lookback covers any current source's release_lag plus
    # generous warmup for rolling-z-score features. Without this, early
    # bars get NaN-filled fundamentals and inference silently degrades.
    from datetime import timedelta as _td
    fetch_start = start_ts - _td(days=400)

    cache: dict[str, pd.DataFrame] = {}
    release_lag_hours: dict[str, float] = {}
    for fg in ("fred_macro", "stooq_yields", "ecb_yield_curve",
               "cot_disagg", "cot_tff", "yfinance_cross_asset"):
        if fg in ("cot_disagg", "cot_tff") and fg != cot_group_for_symbol:
            continue
        read_symbol = "_GLOBAL" if fg == "ecb_yield_curve" else symbol
        try:
            df = await read_feature_store_safe(
                store=store, symbol=read_symbol, feature_group=fg,
                as_of=end_ts, start=fetch_start,
            )
        except Exception as exc:
            logger.debug(
                "[%s/%s] prefetch_fundamentals_for_window failed: %s",
                symbol, fg, exc,
            )
            df = pd.DataFrame()
        cache[fg] = df
        # Store release_lag_hours per group for lookahead-safe lookup.
        src = (feeds_cfg.get("sources", {}) or {}).get(fg, {}) or {}
        release_lag_hours[fg] = float(src.get("release_lag_hours", 0.0))

    return FundamentalsLookup(
        symbol=symbol, cache=cache, cot_prefix=cot_prefix,
        release_lag_hours=release_lag_hours,
    )


@dataclass
class FundamentalsLookup:
    """Forward-fill point-in-time lookup over a prefetched feature_store window.

    Keys returned by ``.get(bar_ts)`` match ``FUNDAMENTAL_FEATURE_NAMES``,
    so the dict can be passed straight into ``predict_proba(..., fundamentals=)``.

    Lookahead-safe: `.get(bar_ts)` shifts ``bar_ts`` back by each group's
    ``release_lag_hours`` before searchsorting on the cached df. This
    matches the training-side semantics in ``_enrich_with_fundamentals``
    (which calls ``read_feature_store_safe(as_of=bar_ts)`` per row,
    where ``read_feature_store_safe`` itself subtracts the lag).
    """
    symbol: str
    cache: dict
    cot_prefix: Optional[str]
    release_lag_hours: dict

    def get(self, bar_ts) -> dict:
        """Return a dict of {schema_col: value} for bar_ts via release-lag-safe forward-fill."""
        from datetime import timedelta as _td
        # Normalize to naive UTC Timestamp (matches feature_store index).
        ts = pd.Timestamp(bar_ts)
        if ts.tz is not None:
            ts = ts.tz_convert(None)

        out: dict = {}
        for fg, df in self.cache.items():
            if df is None or df.empty:
                continue
            # CRITICAL: shift bar_ts back by release_lag so we only pick rows
            # whose raw timestamp is <= bar_ts - release_lag (truly released).
            lag_h = self.release_lag_hours.get(fg, 0.0)
            effective_ts = ts - _td(hours=lag_h)
            idx = df.index.searchsorted(effective_ts, side="right") - 1
            if idx < 0:
                continue
            row = df.iloc[idx]
            for db_col, schema_col in _FUNDAMENTAL_COLUMN_MAP[fg]:
                if fg == "cot_tff" and "{cur}" in db_col:
                    if self.cot_prefix is None:
                        continue
                    db_col_resolved = db_col.format(cur=self.cot_prefix)
                else:
                    db_col_resolved = db_col
                val = row.get(db_col_resolved)
                if val is not None and not pd.isna(val):
                    out[schema_col] = float(val)
        return out


# ---------------------------------------------------------------------------
# Feature matrix + label assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert a (fundamentals + exec-enriched) trades DataFrame into
    the feature matrix expected by the LightGBM classifier.

    Expected columns on ``df``:
      - Base (always required): ``combined_score``, ``regime_label``,
        ``direction``, ``entry_time``.
      - Fundamentals (optional): the 17 columns in ``FUNDAMENTAL_FEATURE_NAMES``.
      - Exec (optional): the 4 columns in ``EXEC_FEATURE_NAMES``.

    Missing optional columns are NaN-filled; LightGBM handles NaN natively
    and the model learns to discount missing data.

    Returns a (N, len(EXPECTED_FEATURE_NAMES)) float64 array in canonical
    EXPECTED_FEATURE_NAMES order.
    """
    entry_ts = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")

    # Base features (5)
    cols: list[np.ndarray] = [
        df["combined_score"].astype(float).fillna(0.0).values,
        df["regime_label"].map(_encode_regime).values,
        df["direction"].map(_encode_direction).values,
        entry_ts.dt.hour.fillna(-1).astype(int).values,
        entry_ts.dt.dayofweek.fillna(-1).astype(int).values,
    ]
    # Fundamentals (17) + Exec (4) — pull from df if present, else NaN-fill.
    n = len(df)
    nan_col = np.full(n, np.nan, dtype=float)
    for fname in FUNDAMENTAL_FEATURE_NAMES + EXEC_FEATURE_NAMES:
        if fname in df.columns:
            cols.append(df[fname].astype(float).values)
        else:
            cols.append(nan_col)
    return np.column_stack(cols).astype(float)


def build_labels(df: pd.DataFrame) -> np.ndarray:
    """Binary label: 1 if the trade was profitable, else 0."""
    return (df["pnl"].astype(float) > 0).astype(int).values


def _compute_pf(pnls: np.ndarray) -> float:
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def train_meta_labeler(
    symbol: str, trades: pd.DataFrame,
    val_fraction: float = 0.2, threshold: float = 0.5, random_state: int = 42,
) -> tuple[LGBMClassifier, MetaLabelerTrainingResult]:
    """Train a LightGBM meta-labeler for one symbol.

    ``trades`` must be ordered chronologically by entry_time ascending; the
    validation split is taken from the tail (no look-ahead leak).

    If ``trades`` already contains the 17 fundamental columns (pre-enriched
    via ``_enrich_with_fundamentals``), they're used directly. Otherwise
    the fundamentals slots are NaN-filled.
    """
    if len(trades) < 50:
        raise ValueError(
            f"[{symbol}] too few trades ({len(trades)}) to train a meta-labeler; "
            "need at least 50 for a meaningful train/val split"
        )

    # Sort chronologically to avoid leak
    trades = trades.sort_values("entry_time").reset_index(drop=True)

    X = build_feature_matrix(trades)
    y = build_labels(trades)
    pnls = trades["pnl"].astype(float).values

    n = len(trades)
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val
    X_tr, X_va = X[:n_train], X[n_train:]
    y_tr, y_va = y[:n_train], y[n_train:]
    pnls_va = pnls[n_train:]

    clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        verbosity=-1,
    )
    clf.fit(X_tr, y_tr, feature_name=list(EXPECTED_FEATURE_NAMES))

    # Validation scoring
    proba_va = clf.predict_proba(X_va)[:, 1]
    preds_va = (proba_va >= threshold).astype(int)

    val_acc = float((preds_va == y_va).mean())
    tp = int(((preds_va == 1) & (y_va == 1)).sum())
    fp = int(((preds_va == 1) & (y_va == 0)).sum())
    fn = int(((preds_va == 0) & (y_va == 1)).sum())
    val_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    val_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    coverage = float(preds_va.mean())

    pf_no_gate = _compute_pf(pnls_va)
    pf_gated = _compute_pf(pnls_va[preds_va == 1]) if preds_va.sum() else 0.0

    result = MetaLabelerTrainingResult(
        symbol=symbol, n_train=n_train, n_val=n_val,
        val_accuracy=val_acc, val_precision=val_precision,
        val_recall=val_recall,
        coverage_at_default_threshold=coverage,
        pf_without_gate=pf_no_gate, pf_with_gate=pf_gated,
        threshold=threshold,
    )
    logger.info(
        "[%s] meta-labeler trained: n_train=%d n_val=%d "
        "val_acc=%.3f precision=%.3f recall=%.3f coverage=%.1f%% "
        "PF_no_gate=%.2f PF_gated=%.2f",
        symbol, n_train, n_val, val_acc, val_precision, val_recall,
        coverage * 100, pf_no_gate, pf_gated,
    )
    return clf, result


def save_meta_labeler(
    clf: LGBMClassifier, symbol: str,
    threshold: float = 0.5, primary: str = "lstm",
) -> Path:
    """Persist the classifier + threshold + schema_hash.

    Per spec §3 invariant #12, every artifact carries a
    ``feature_schema_hash`` so signal_combiner can refuse to load on
    mismatch with the runtime contract.

    ``primary`` is one of {"lstm", "gbm"} per spec §1 anchor 4 — each
    primary's meta-labeler is trained against its own backtest_trades
    pool. The artifact path is suffixed accordingly.
    """
    path = Path(MODEL_PATH_TEMPLATE.format(symbol=symbol, primary=primary))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": clf,
        "threshold": threshold,
        "feature_names": EXPECTED_FEATURE_NAMES,
        "feature_schema_hash": EXPECTED_SCHEMA_HASH,
        "primary_kind": primary,
        "phase": "phase_a",
    }
    joblib.dump(payload, path)
    logger.info(
        "[%s] meta-labeler saved to %s (primary=%s, schema_hash=%s)",
        symbol, path, primary, EXPECTED_SCHEMA_HASH,
    )
    return path


def load_meta_labeler(symbol: str, primary: str = "lstm") -> Optional[dict]:
    """Load a persisted meta-labeler bundle.

    Tries the primary-suffixed canonical path first, then falls back to
    the legacy unsuffixed name for back-compat with pre-Phase-A
    bundles (which lack ``feature_schema_hash`` and will be rejected
    by signal_combiner's load-time validation).

    Returns the joblib-loaded dict or None if no file exists.
    """
    suffixed = Path(MODEL_PATH_TEMPLATE.format(symbol=symbol, primary=primary))
    legacy = Path(LEGACY_MODEL_PATH_TEMPLATE.format(symbol=symbol))
    if suffixed.exists():
        return joblib.load(suffixed)
    if legacy.exists():
        return joblib.load(legacy)
    return None


def predict_proba(
    bundle: dict, combined_score: float, regime_label: str,
    direction: str, hour_of_day: int, day_of_week: int,
    *, fundamentals: Optional[dict] = None,
    exec_features: Optional[dict] = None,
) -> float:
    """Score a single signal with a loaded meta-labeler bundle.

    Schema-aware: reads ``bundle["feature_names"]`` to determine what shape
    the loaded model expects (legacy 22-feature bundles still load and
    score correctly even with the new exec_features kwarg present — the
    extra exec values are simply ignored when not in feature_names).

    Args:
      fundamentals: dict mapping ``FUNDAMENTAL_FEATURE_NAMES`` keys to floats.
      exec_features: dict mapping ``EXEC_FEATURE_NAMES`` keys to floats.

    Missing keys default to NaN — LightGBM handles NaN natively but
    predictions are degraded vs training when a non-trivial fraction of
    features are missing.

    Returns ``P(trade wins)`` ∈ [0, 1].
    """
    feature_names = bundle.get("feature_names") or EXPECTED_FEATURE_NAMES
    base_map = {
        "combined_score": float(combined_score),
        "regime_index": float(_encode_regime(regime_label)),
        "direction_index": float(_encode_direction(direction)),
        "hour_of_day": float(int(hour_of_day)),
        "day_of_week": float(int(day_of_week)),
    }
    fund = fundamentals or {}
    exec_ = exec_features or {}

    feats = []
    for name in feature_names:
        if name in base_map:
            feats.append(base_map[name])
        elif name in fund:
            feats.append(float(fund[name]) if fund[name] is not None else math.nan)
        elif name in exec_:
            feats.append(float(exec_[name]) if exec_[name] is not None else math.nan)
        else:
            feats.append(math.nan)

    proba = bundle["model"].predict_proba(np.array([feats], dtype=float))[0, 1]
    return float(proba)
