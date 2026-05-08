"""
backtest_full.py — Full-Strategy Backtest Engine

Runs the complete production pipeline on historical data:
    HMM regime → LSTM prediction → SignalCombiner (6 gates)
    → StrategyOrchestrator → 3-tier exit ladder → CircuitBreaker

Unlike the simple mode (20-bar MA crossover), this exercises the real
models and signal gates that run in live trading.

Usage:
    Called from scripts/backtest.py with --mode full.
    Not intended to be run directly — use backtest.py as the entry point.

Walk-forward methodology:
    - Train models on data BEFORE the backtest window (no look-ahead)
    - Features are pre-computed via FeatureEngineer.transform() + to_matrix()
      (all features are backward-looking rolling windows — safe)
    - Entry at next bar's open price (no look-ahead bias)
    - Deterministic: same input → same output
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.backtest import _compute_atr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ATR_PERIOD = 14
EMA_PERIOD = 50
WARMUP_BARS = 200        # deepest lookback window (price_percentile_200)
SIGNAL_WINDOW = 60       # bars fed to SignalCombiner.get_signal()
H1_PER_H4 = 4            # H1 bars per H4 bar

# Per-symbol strategy parameters — different assets need different exits
# These are defaults; overridden by config/settings.yaml in production
# USDJPY grid-test overrides (set by scripts/test_usdjpy_variants.py).
# Read at module load so the test harness can re-invoke this script with
# different parameter combinations without code edits.
#   CORTEX_USDJPY_SIGNAL_THRESHOLD  (default: backtest's 0.45)
#   CORTEX_USDJPY_TIME_EXIT_H1      (default: 60)
#   CORTEX_USDJPY_RISK_PCT          (default: 1.25)
#   CORTEX_USDJPY_EUPHORIA_MULT     (default: 0.0 = block; 0.5 = half-size buy)
_JPY_TIME_H1 = int(os.environ.get("CORTEX_USDJPY_TIME_EXIT_H1", "60"))
_JPY_RISK = float(os.environ.get("CORTEX_USDJPY_RISK_PCT", "1.25"))

SYMBOL_PARAMS = {
    # Phase A.2: increased risk_pct 1.0→1.5, 0.75→1.25 for industry-standard sizing
    # Industry standard for systems with PF≥1.5 is 2% risk per trade.
    # We use 1.25-1.5% to leave safety margin for live underperformance.
    "XAUUSD": {"atr_sl_mult": 2.0, "tp_r": 2.5, "be_trigger_r": 1.0,
               "time_exit_h1": 80, "risk_pct": 1.5},
    "EURUSD": {"atr_sl_mult": 1.5, "tp_r": 2.0, "be_trigger_r": 1.0,
               "time_exit_h1": 60, "risk_pct": 1.25},
    "USDJPY": {"atr_sl_mult": 2.0, "tp_r": 2.0, "be_trigger_r": 1.0,
               "time_exit_h1": _JPY_TIME_H1, "risk_pct": _JPY_RISK},
    "USDCAD": {"atr_sl_mult": 1.8, "tp_r": 2.0, "be_trigger_r": 1.0,
               "time_exit_h1": 60, "risk_pct": 1.25},
    # Phase 2B (2026-04-27): 6 expansion pairs activated. Risk params
    # mirror config/settings.yaml::strategy.per_symbol_params.
    "GBPUSD": {"atr_sl_mult": 1.5, "tp_r": 2.0, "be_trigger_r": 1.0,
               "time_exit_h1": 60, "risk_pct": 1.25},
    "AUDUSD": {"atr_sl_mult": 1.8, "tp_r": 2.0, "be_trigger_r": 1.0,
               "time_exit_h1": 60, "risk_pct": 1.25},
    "EURGBP": {"atr_sl_mult": 1.2, "tp_r": 1.8, "be_trigger_r": 1.0,
               "time_exit_h1": 60, "risk_pct": 1.25},
    "EURJPY": {"atr_sl_mult": 1.8, "tp_r": 2.0, "be_trigger_r": 1.0,
               "time_exit_h1": 60, "risk_pct": 1.25},
    "GBPJPY": {"atr_sl_mult": 2.0, "tp_r": 2.0, "be_trigger_r": 1.0,
               "time_exit_h1": 60, "risk_pct": 1.25},
    "AUDNZD": {"atr_sl_mult": 1.2, "tp_r": 1.5, "be_trigger_r": 1.0,
               "time_exit_h1": 72, "risk_pct": 1.25},
}
DEFAULT_PARAMS = {"atr_sl_mult": 2.0, "tp_r": 2.5, "be_trigger_r": 1.0,
                  "time_exit_h1": 80, "risk_pct": 1.0}

# Per-trade friction — applied to make backtest PF match live expectations.
# Prior to R-1 (2026-04-18) the backtest assumed frictionless fills at exact
# bar open/close, inflating PF by ~5-15% (1-yr XAU: 13.05 → 3.87). Every
# downstream decision gate (M-*, E-*) compares candidate PF vs backtest PF,
# so an honest baseline is a prereq for credible strategy research.
#
#   slippage_price: native price units added against the trader on entry
#                   and exit (buy pays more, sell receives less). Symmetric.
#   commission_per_lot_per_side: USD charged per lot on entry AND exit (so
#                   round-trip cost = 2× this × volume).
#
# ⚠ VALUES ARE ASSUMPTIONS, NOT CALIBRATED TO LIVE DATA.
# The numbers below are educated guesses from the broker ECN spread norms:
#   - XAU:    0.15/oz  ≈ 1.5 gold-pips (each gold-pip = $0.10)
#   - forex:  0.00005  = 0.5 pip on 5-digit; JPY 0.005 = 0.5 pip on 3-digit
#   - ETH:    2.0/ETH  ≈ crypto spreads typical of the broker crypto tier
# They are symmetric and deterministic — no regime, no time-of-day, no
# heavy-tail variance. This underrepresents news-window and 22:00-UTC
# rollover blowups. A proper per-symbol distribution needs live fill data:
# see E-3 Phase 1 (Execution quality audit) in docs/BACKLOG.md, which
# feeds into R-1b (replace assumptions with empirical slippage). Until
# that ships, treat backtest PF as an upper bound and add a safety
# margin to any decision gate built on these numbers.
# Override per run via ``friction_override`` kwarg.
#   units_per_lot: conversion from the backtest's internal ``volume``
#                  (base-currency or underlying units) to broker "lots"
#                  for commission accounting. XAU: 100 oz/lot. Forex:
#                  100,000 base units/lot. Crypto ETH: 1 ETH/lot on
#                  the broker's crypto tier. Without this, commission
#                  would be charged per UNIT not per LOT, blowing up
#                  by 3-5 orders of magnitude (the 2026-04-18 5-yr
#                  backtest bug: EURUSD single-trade lost $72K to
#                  commission alone).
DEFAULT_FRICTION: dict[str, dict[str, float]] = {
    "XAUUSD": {"slippage_price": 0.15,    "commission_per_lot_per_side": 5.0, "units_per_lot": 100.0},
    "EURUSD": {"slippage_price": 0.00005, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "USDJPY": {"slippage_price": 0.005,   "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "USDCAD": {"slippage_price": 0.00005, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "ETHUSD": {"slippage_price": 2.0,     "commission_per_lot_per_side": 0.0, "units_per_lot": 1.0},
    # Phase 2B (2026-04-27): 6 expansion pairs. Slippage estimates from
    # the broker ECN spread norms; same caveat as the existing 5 — uncalibrated
    # to live fills. R-1b (live execution audit) covers calibration.
    "GBPUSD": {"slippage_price": 0.00005, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "AUDUSD": {"slippage_price": 0.00005, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "EURGBP": {"slippage_price": 0.00007, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "EURJPY": {"slippage_price": 0.005,   "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "GBPJPY": {"slippage_price": 0.007,   "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "AUDNZD": {"slippage_price": 0.00010, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    # an earlier sprint Phase B (2026-04-29): 10 new pairs for the universe sweep.
    # Same the broker-ECN-norm estimates; R-1b calibration still pending.
    # USD-paired majors get 0.00005 (same as GBPUSD/AUDUSD/USDCAD).
    # 5-digit cross-currency pairs get 0.00007-0.00010 by liquidity tier.
    # 3-digit JPY-crosses get 0.005-0.010 (USDJPY/EURJPY=0.005, illiquid=0.010).
    "USDCHF": {"slippage_price": 0.00005, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "NZDUSD": {"slippage_price": 0.00005, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "EURCHF": {"slippage_price": 0.00007, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "GBPCHF": {"slippage_price": 0.00010, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "EURAUD": {"slippage_price": 0.00010, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "GBPAUD": {"slippage_price": 0.00015, "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "AUDJPY": {"slippage_price": 0.007,   "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "NZDJPY": {"slippage_price": 0.010,   "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "CADJPY": {"slippage_price": 0.007,   "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
    "CHFJPY": {"slippage_price": 0.010,   "commission_per_lot_per_side": 2.0, "units_per_lot": 100_000.0},
}
_NO_FRICTION = {
    "slippage_price": 0.0,
    "commission_per_lot_per_side": 0.0,
    "units_per_lot": 1.0,
}


def _resolve_friction(
    symbol: str, override: dict | None
) -> tuple[float, float, float]:
    """Return (slippage_price, commission_per_lot_per_side, units_per_lot).

    Resolution order: explicit override → DEFAULT_FRICTION[symbol] →
    zero friction (silent-but-logged fallback so backtests for novel
    symbols still run).
    """
    src = None
    if override is not None and symbol in override:
        src = override[symbol]
    elif override is not None and "*" in override:
        src = override["*"]
    else:
        src = DEFAULT_FRICTION.get(symbol)
    if src is None:
        logger.warning(
            "No DEFAULT_FRICTION entry for %s — running frictionless. "
            "Add an entry in scripts/backtest_full.py or pass "
            "friction_override={%r: {...}}.",
            symbol, symbol,
        )
        src = _NO_FRICTION
    return (
        float(src.get("slippage_price", 0.0)),
        float(src.get("commission_per_lot_per_side", 0.0)),
        float(src.get("units_per_lot", 1.0)),
    )


def _apply_entry_slippage(
    nominal_price: float, direction: str, slippage: float
) -> float:
    """Trader pays slippage on entry: buy fills above, sell fills below."""
    if direction == "buy":
        return nominal_price + slippage
    return nominal_price - slippage


def _apply_exit_slippage(
    nominal_price: float, direction: str, slippage: float
) -> float:
    """Trader pays slippage on exit: closing buy (sell) fills lower;
    closing sell (buy) fills higher."""
    if direction == "buy":
        return nominal_price - slippage
    return nominal_price + slippage


# Circuit breaker thresholds (mirror production defaults)
CB_DAILY_SOFT_PCT = 2.0
CB_DAILY_HARD_PCT = 3.0
CB_WEEKLY_SOFT_PCT = 5.0
CB_WEEKLY_HARD_PCT = 7.0
CB_PEAK_DD_PCT = 10.0


@dataclass
class _FullOpenTrade:
    """In-flight trade during full-strategy simulation (Triple Barrier)."""
    symbol: str
    direction: str
    entry_bar: int
    entry_time: str
    entry_price: float
    stop_loss: float
    atr: float
    atr_trail_mult: float
    volume: float
    initial_volume: float
    initial_r_dist: float = 0.0  # original risk distance (entry - initial SL)
    strategy_name: str = ""
    regime_label: str = ""
    combined_score: float = 0.0
    be_locked: bool = False      # breakeven stop set
    bars_held: int = 0           # H1 bars since entry (for time exit)
    tp_price: float = 0.0        # take-profit price
    # E-7 trend-mode fields (default off; only set by trend-mode path)
    time_exit_disabled: bool = False
    was_in_trend_mode_at_close: bool = False


def _compute_ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. NaN before `period` bars."""
    ema = np.full(len(series), np.nan)
    if len(series) < period:
        return ema
    ema[period - 1] = np.mean(series[:period])
    mult = 2.0 / (period + 1)
    for i in range(period, len(series)):
        ema[i] = series[i] * mult + ema[i - 1] * (1 - mult)
    return ema


def run_backtest_full(
    symbol: str,
    ohlcv: pd.DataFrame,
    initial_equity: float = 10000.0,
    d1_ohlcv: pd.DataFrame = None,
    w1_ohlcv: pd.DataFrame = None,
    h1_ohlcv: pd.DataFrame = None,
    # --- Parameter sweep overrides (all optional) ---
    hmm_weight_override: float | None = None,
    signal_threshold_override: float | None = None,
    long_only_symbols_override: set[str] | None = None,
    friction_override: dict | None = None,
    # --- the model bake-off cell selection ---
    primary: str = "lstm",
    variant: str = "prod",
    trend_mode: bool = False,                     # E-7 trend-mode A/B flag
    rich_features: bool = False,                  # Phase 2B Q1: GBM rich-feature surface (DEPRECATED — use feature_mode)
    feature_mode: str = "thin",                   # Phase 2B Q1.5: GBM feature surface — "thin" / "rich" / "parity"
) -> tuple[list[dict], list[dict]]:
    """
    Walk-forward simulation using the full production pipeline.

    Loads pre-trained HMM + primary predictor (LSTM or GBM), instantiates
    SignalCombiner, StrategyOrchestrator, and CircuitBreaker. Pre-computes
    features once (safe — all backward-looking), then walks bar-by-bar.

    The HMM and the primary predictor receive DIFFERENT feature matrices:
    - HMM: 56 technical features (same as it was trained on)
    - LSTM primary: ~87+ features (multi-TF + regime + calendar + zero-fill),
      windowed (SIGNAL_WINDOW × n_features) per bar.
    - GBM primary: 36 features (lag/rolling/regime/cross via
      ``src.brain.gbm.gbm_features.build_features``), single flat row per bar.

    Args:
        symbol:         Trading symbol (e.g. "XAUUSD")
        ohlcv:          H4 OHLCV DataFrame indexed by datetime
        initial_equity: Starting account balance
        d1_ohlcv:       Optional D1 OHLCV for regime injection + multi-TF
        w1_ohlcv:       Optional W1 OHLCV for multi-TF features
        primary:        "lstm" (default — production path) or "gbm" (Phase A
                        bake-off GBM-as-primary cell). Routes both model
                        load and feature pipeline.
        variant:        "prod" (default — load the unsuffixed live artifact)
                        or "default" / "tuned" (load the suffixed Phase A
                        bake-off artifact ``{kind}_{symbol}_{variant}.{ext}``).

    Returns:
        (equity_curve, trades) — both as lists of dicts, enriched with
        strategy_name, regime_label, combined_score, exit_reason.
    """
    from src.brain.hmm_regime import HMMRegimeClassifier
    from src.brain.deep_learning.lstm_model import LSTMPricePredictor
    from src.brain.signal_combiner import SignalCombiner
    from src.data_pipeline.feature_engineering import FeatureEngineer
    from src.strategy.orchestrator import StrategyOrchestrator
    from src.strategy.base import MarketContext
    from src.safety.circuit_breaker import CircuitBreaker

    if primary not in ("lstm", "gbm"):
        raise ValueError(f"primary must be 'lstm' or 'gbm', got {primary!r}")
    if variant not in ("prod", "default", "tuned"):
        raise ValueError(
            f"variant must be 'prod', 'default', or 'tuned', got {variant!r}"
        )
    if feature_mode not in ("thin", "rich", "parity"):
        raise ValueError(
            f"feature_mode must be 'thin', 'rich', or 'parity', "
            f"got {feature_mode!r}"
        )
    # Phase 2B Q1/Q1.5 (2026-04-27): feature_mode flips the GBM artifact
    # path AND the serving feature pipeline. No-op for LSTM (LSTM is
    # always rich-multi-TF via transform_multi_timeframe).
    # `rich_features=True` is the deprecated alias for feature_mode="rich".
    if rich_features and feature_mode == "thin":
        feature_mode = "rich"
    if feature_mode != "thin" and primary != "gbm":
        logger.info(
            "feature_mode=%r is no-op for primary=%r (LSTM uses its own "
            "fixed multi-TF pipeline regardless).", feature_mode, primary,
        )
    artifact_suffix = "" if variant == "prod" else f"_{variant}"
    if primary == "gbm" and feature_mode == "rich":
        artifact_suffix = f"_rich{artifact_suffix}"
    elif primary == "gbm" and feature_mode == "parity":
        artifact_suffix = f"_parity{artifact_suffix}"

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    hmm = HMMRegimeClassifier()
    if not hmm.load(symbol):
        raise FileNotFoundError(
            f"HMM model not found for {symbol}. "
            f"Run: python scripts/train_hmm.py --symbols {symbol}"
        )

    if primary == "lstm":
        lstm = LSTMPricePredictor()
        if not lstm.load(symbol, suffix=artifact_suffix):
            label = f"{symbol}{artifact_suffix or ' (prod)'}"
            raise FileNotFoundError(
                f"LSTM model not found for {label}. "
                f"Run: python scripts/train_deep_learning.py "
                f"--symbols {symbol}"
                + (" --tune" if artifact_suffix else "")
            )
        primary_predictor = lstm
    else:  # primary == "gbm"
        from src.brain.gbm.gbm_model import GBMPredictor
        gbm_path = Path("data/models") / f"gbm_{symbol}{artifact_suffix}.pkl"
        if not gbm_path.exists():
            raise FileNotFoundError(
                f"GBM model not found at {gbm_path}. "
                f"Run: python scripts/train_gbm.py --symbols {symbol}"
                + (" --tune" if artifact_suffix else "")
            )
        primary_predictor = GBMPredictor.load(gbm_path)

    # ------------------------------------------------------------------
    # 1b. Clean up any persistent halt flag from previous runs
    # ------------------------------------------------------------------
    halt_flag = Path("data/logs/TRADING_HALTED.flag")
    if halt_flag.exists():
        halt_flag.unlink()
        logger.info("Removed stale TRADING_HALTED.flag from previous run")

    # ------------------------------------------------------------------
    # 2. Pre-compute features
    # ------------------------------------------------------------------
    fe = FeatureEngineer()

    # Normalize column names — FeatureEngineer expects tick_volume
    ohlcv_clean = ohlcv.copy()
    if "volume" in ohlcv_clean.columns and "tick_volume" not in ohlcv_clean.columns:
        ohlcv_clean.rename(columns={"volume": "tick_volume"}, inplace=True)
    if "tick_volume" not in ohlcv_clean.columns:
        ohlcv_clean["tick_volume"] = 0

    # HMM feature matrix: D1 features (matches HMM training on D1 bars).
    # predict() normalizes internally using saved training stats, so we
    # pass raw (un-normalized) values here. Fall back to H4 if no D1.
    if d1_ohlcv is not None and len(d1_ohlcv) > 50:
        d1_hmm = d1_ohlcv.copy()
        if "volume" in d1_hmm.columns and "tick_volume" not in d1_hmm.columns:
            d1_hmm.rename(columns={"volume": "tick_volume"}, inplace=True)
        if "tick_volume" not in d1_hmm.columns:
            d1_hmm["tick_volume"] = 0
        hmm_d1_feature_df = fe.transform(d1_hmm)
        # Align to HMM manifest and pass raw values (predict() z-scores)
        hmm_manifest = hmm._feature_manifests.get(symbol, [])
        if hmm_manifest:
            hmm_d1_aligned = fe.align_to_manifest(hmm_d1_feature_df.copy(), hmm_manifest)
        else:
            hmm_d1_aligned = hmm_d1_feature_df[sorted(hmm_d1_feature_df.columns)]
        hmm_d1_matrix = np.nan_to_num(
            hmm_d1_aligned.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0,
        )
        _use_d1_hmm = True
    else:
        _use_d1_hmm = False

    # H4 features still needed for ATR/EMA computations + LSTM
    hmm_feature_df = fe.transform(ohlcv_clean)
    hmm_matrix = fe.to_matrix(hmm_feature_df)

    # ------------------------------------------------------------------
    # 2b. Primary-predictor feature matrix
    # ------------------------------------------------------------------
    # The shape of the matrix and the per-bar slicing strategy diverge
    # between primaries:
    # - LSTM: 2D ``(n_bars, n_features)`` matrix. Per-bar input is a
    #   SIGNAL_WINDOW × n_features 2D window slice.
    # - GBM:  2D ``(n_bars, n_features)`` matrix likewise, but per-bar
    #   input is a 1D ``(n_features,)`` flat row.
    # Variable names retain the ``lstm_*`` prefix to keep diffs vs the
    # production hot path small; ``primary == "gbm"`` rebinds them to
    # the GBM-flavored equivalents and the per-bar branch below picks
    # the right slice shape.
    if primary == "lstm":
        if d1_ohlcv is not None or w1_ohlcv is not None:
            ohlcv_by_tf = {"H4": ohlcv_clean}
            if d1_ohlcv is not None and len(d1_ohlcv) > 50:
                d1_clean = d1_ohlcv.copy()
                if "volume" in d1_clean.columns and "tick_volume" not in d1_clean.columns:
                    d1_clean.rename(columns={"volume": "tick_volume"}, inplace=True)
                ohlcv_by_tf["D1"] = d1_clean
            if w1_ohlcv is not None and len(w1_ohlcv) > 10:
                w1_clean = w1_ohlcv.copy()
                if "volume" in w1_clean.columns and "tick_volume" not in w1_clean.columns:
                    w1_clean.rename(columns={"volume": "tick_volume"}, inplace=True)
                ohlcv_by_tf["W1"] = w1_clean
            lstm_feature_df = fe.transform_multi_timeframe(ohlcv_by_tf, primary_tf="H4")
        else:
            lstm_feature_df = hmm_feature_df.copy()

        # Add calendar features
        try:
            from src.data_pipeline.market.calendar_features import CalendarFeatureBuilder
            cal = CalendarFeatureBuilder()
            cal_df = cal.get_historical_calendar_features(lstm_feature_df.index)
            lstm_feature_df = lstm_feature_df.join(cal_df, how="left")
        except Exception:
            pass

        # Zero-fill fundamental placeholders (per-symbol)
        for col in fe.get_zero_fill_feature_names(symbol):
            if col not in lstm_feature_df.columns:
                lstm_feature_df[col] = 0.0

        # Inject HMM regime features (one-hot + probability)
        if d1_ohlcv is not None and len(d1_ohlcv) > 50:
            lstm_feature_df = fe.inject_regime_features(
                lstm_feature_df, hmm, symbol, d1_ohlcv,
            )
        else:
            # Neutral regime placeholders
            for i in range(5):
                lstm_feature_df[f"regime_{i}"] = 0.2
            lstm_feature_df["regime_probability"] = 0.2

        lstm_feature_df = lstm_feature_df.fillna(0.0)

        # Align to LSTM manifest — ensures exact feature columns match training
        lstm_manifest = lstm._feature_manifests.get(symbol, [])
        if lstm_manifest:
            lstm_feature_df = fe.align_to_manifest(lstm_feature_df, lstm_manifest)
    else:  # primary == "gbm"
        # GBM training flow (scripts/train_gbm.py): inject regime onto a
        # ``volume``-renamed H4 frame, then ``gbm_features.build_features``.
        # Reproducing that here verbatim is the train/serve parity contract
        # (spec §3 invariant #8 — same helper, same column order).
        # Phase 2B Q1 (2026-04-27): if rich_features=True, mirror the
        # rich training pipeline instead — transform() then inject regime
        # then build_features_rich. Train/serve parity preserved by
        # delegating to the same helpers train_gbm.py uses.
        from src.brain.gbm.gbm_features import (
            build_features as gbm_build_features,
            build_features_rich as gbm_build_features_rich,
        )

        h4_for_gbm = ohlcv_clean.copy()
        if "tick_volume" in h4_for_gbm.columns and "volume" not in h4_for_gbm.columns:
            h4_for_gbm = h4_for_gbm.rename(columns={"tick_volume": "volume"})

        if feature_mode == "rich":
            # Rich path: single-TF transform() (drops OHLCV, ~56 technical
            # features), inject regime, then GBM augmentations.
            h4_rich = fe.transform(h4_for_gbm)
            if d1_ohlcv is not None and len(d1_ohlcv) > 50:
                d1_for_gbm = d1_ohlcv.copy()
                if "tick_volume" in d1_for_gbm.columns and "volume" not in d1_for_gbm.columns:
                    d1_for_gbm = d1_for_gbm.rename(columns={"tick_volume": "volume"})
                h4_with_regime = fe.inject_regime_features(
                    h4_rich, hmm, symbol, d1_for_gbm,
                )
            else:
                h4_with_regime = h4_rich.copy()
                for i in range(5):
                    h4_with_regime[f"regime_{i}"] = 0.2
                h4_with_regime["regime_probability"] = 0.2
            lstm_feature_df = gbm_build_features_rich(h4_with_regime)
        elif feature_mode == "parity":
            # Phase 2B Q1.5 — INFORMATION-EQUAL serving pipeline. Mirrors
            # the LSTM serving frame: multi-TF transform + calendar +
            # zero-fill cross-asset placeholders + regime injection. NO
            # GBM-specific augmentations. Train/serve parity is enforced
            # because train_gbm.py (--features parity) builds features
            # the exact same way.
            ohlcv_by_tf = {"H4": h4_for_gbm}
            if d1_ohlcv is not None and len(d1_ohlcv) > 50:
                d1_for_gbm = d1_ohlcv.copy()
                if "tick_volume" in d1_for_gbm.columns and "volume" not in d1_for_gbm.columns:
                    d1_for_gbm = d1_for_gbm.rename(columns={"tick_volume": "volume"})
                ohlcv_by_tf["D1"] = d1_for_gbm
            if w1_ohlcv is not None and len(w1_ohlcv) > 10:
                w1_for_gbm = w1_ohlcv.copy()
                if "tick_volume" in w1_for_gbm.columns and "volume" not in w1_for_gbm.columns:
                    w1_for_gbm = w1_for_gbm.rename(columns={"tick_volume": "volume"})
                ohlcv_by_tf["W1"] = w1_for_gbm
            feat = fe.transform_multi_timeframe(ohlcv_by_tf, primary_tf="H4")
            try:
                from src.data_pipeline.market.calendar_features import (
                    CalendarFeatureBuilder,
                )
                cal = CalendarFeatureBuilder()
                cal_df = cal.get_historical_calendar_features(feat.index)
                feat = feat.join(cal_df, how="left")
            except Exception:
                pass
            for col in fe.get_zero_fill_feature_names(symbol):
                if col not in feat.columns:
                    feat[col] = 0.0
            if d1_ohlcv is not None and len(d1_ohlcv) > 50:
                feat = fe.inject_regime_features(feat, hmm, symbol, d1_ohlcv)
            else:
                for i in range(5):
                    feat[f"regime_{i}"] = 0.2
                feat["regime_probability"] = 0.2
            lstm_feature_df = feat.fillna(0.0)
        else:  # thin
            if d1_ohlcv is not None and len(d1_ohlcv) > 50:
                d1_for_gbm = d1_ohlcv.copy()
                if "tick_volume" in d1_for_gbm.columns and "volume" not in d1_for_gbm.columns:
                    d1_for_gbm = d1_for_gbm.rename(columns={"tick_volume": "volume"})
                h4_with_regime = fe.inject_regime_features(
                    h4_for_gbm, hmm, symbol, d1_for_gbm,
                )
            else:
                # Neutral regime placeholders — same fallback shape as LSTM path.
                h4_with_regime = h4_for_gbm.copy()
                for i in range(5):
                    h4_with_regime[f"regime_{i}"] = 0.2
                h4_with_regime["regime_probability"] = 0.2
            lstm_feature_df = gbm_build_features(h4_with_regime)

        # Align to predictor's saved feature_names — same role as the
        # LSTM manifest alignment.
        lstm_feature_df = lstm_feature_df[primary_predictor.feature_names]
        # build_features leaves NaN in the warmup head; drop those rows
        # so subsequent index intersection with hmm_feature_df is clean.
        lstm_feature_df = lstm_feature_df.dropna(how="any")

    # Align both matrices to the same row indices
    # hmm_feature_df might have different NaN-dropped rows than lstm_feature_df
    common_index = hmm_feature_df.index.intersection(lstm_feature_df.index)
    hmm_feature_df = hmm_feature_df.loc[common_index]
    lstm_feature_df = lstm_feature_df.loc[common_index]
    hmm_matrix = fe.to_matrix(hmm_feature_df)
    lstm_matrix = fe.to_matrix(lstm_feature_df)
    feature_matrix = hmm_matrix  # for ATR/EMA computations

    # The transform() drops NaN rows (leading warmup). Map feature rows
    # back to the original OHLCV indices.
    feature_index = hmm_feature_df.index
    ohlcv_aligned = ohlcv_clean.loc[feature_index]

    n = len(feature_matrix)
    if n < WARMUP_BARS + SIGNAL_WINDOW:
        logger.warning(
            "Insufficient bars after feature engineering: %d (need %d+). "
            "Returning empty results.",
            n, WARMUP_BARS + SIGNAL_WINDOW,
        )
        return [], []

    logger.info(
        "Full backtest %s: %d bars after feature engineering, "
        "warmup=%d, starting simulation...",
        symbol, n, WARMUP_BARS,
    )

    # ------------------------------------------------------------------
    # 3. Pre-compute raw ATR + EMA50 for MarketContext
    # ------------------------------------------------------------------
    opens = ohlcv_aligned["open"].values.astype(float)
    highs = ohlcv_aligned["high"].values.astype(float)
    lows = ohlcv_aligned["low"].values.astype(float)
    closes = ohlcv_aligned["close"].values.astype(float)
    timestamps = [str(ts) for ts in ohlcv_aligned.index]

    atr_series = _compute_atr(highs, lows, closes, ATR_PERIOD)
    ema50_series = _compute_ema(closes, EMA_PERIOD)

    # ------------------------------------------------------------------
    # 4. Instantiate pipeline components
    # ------------------------------------------------------------------
    # Default long_only_symbols matches settings.yaml::strategy.long_only_symbols
    # post-2026-04-27 XAU bidirectional flip. Callers (parameter sweeps,
    # A/B harnesses) override via long_only_symbols_override. Pre-2026-04-27
    # this defaulted to {"XAUUSD", "ETHUSD"} — the stale default silently
    # ran the canonical 5y baseline with XAU as long-only, which explained
    # the 3.4% delta against CLAUDE.md's "verified 2026-04-27" baseline
    # (incoherent-rules audit 2026-04-30, NEW finding paired with C2+C3).
    long_only_symbols = {"ETHUSD"}
    if long_only_symbols_override is not None:
        long_only_symbols = long_only_symbols_override
    # Load per-symbol threshold overrides from config/model_config.yaml so the
    # backtest matches production. Env var still wins when present (grid-test
    # override).
    _default_threshold = 0.45
    _per_sym_threshold = {"USDJPY": 0.55}  # from config/model_config.yaml
    try:
        import yaml
        _mc = yaml.safe_load(
            Path("config/model_config.yaml").read_text(encoding="utf-8")
        )
        _sc = (_mc or {}).get("signal_combiner", {})
        _default_threshold = float(_sc.get("signal_threshold", _default_threshold))
        _per_sym_threshold = {
            str(k).upper(): float(v)
            for k, v in (_sc.get("signal_threshold_per_symbol") or {}).items()
        }
    except Exception:
        pass
    # Parameter sweep overrides
    if signal_threshold_override is not None:
        _default_threshold = signal_threshold_override
        _per_sym_threshold = {}  # uniform threshold when overriding

    _hmm_w = hmm_weight_override if hmm_weight_override is not None else 0.3
    _lstm_w = 1.0 - _hmm_w

    # Flicker override: env var CORTEX_FLICKER_BARS_REQUIRED drives sweep
    # experiments without code edits. Default 2 matches production.
    _flicker_env = os.environ.get("CORTEX_FLICKER_BARS_REQUIRED")
    _flicker_bars = int(_flicker_env) if _flicker_env else 2

    # Option B (2026-04-27): if meta-labeler is enabled, prefetch
    # fundamentals for the entire backtest window once per symbol so the
    # signal-time inference can use the full 22-feature schema (closes
    # the train/serve mismatch that previously NaN-filled inference).
    # Skip when meta-labeler is OFF — saves ~6 DB calls per pair.
    _meta_active = bool(os.environ.get("CORTEX_META_LABELER", "").strip()
                        in ("1", "true", "True"))
    _meta_shadow = bool(os.environ.get("CORTEX_META_LABELER_SHADOW", "").strip()
                        in ("1", "true", "True"))
    fundamentals_lookup = None
    if _meta_active or _meta_shadow:
        try:
            import asyncio
            import concurrent.futures
            from src.data_pipeline.data_store import DataStore
            from src.ml.meta_labeler import prefetch_fundamentals_for_window
            _start_ts = pd.Timestamp(timestamps[0]).to_pydatetime()
            _end_ts = pd.Timestamp(timestamps[-1]).to_pydatetime()

            async def _prefetch():
                store = DataStore()
                await store.connect()
                try:
                    return await prefetch_fundamentals_for_window(
                        store, symbol, _start_ts, _end_ts,
                    )
                finally:
                    await store.close()

            # backtest is already inside an asyncio event loop (called from
            # _main_async), so asyncio.run() raises. Run the prefetch in a
            # fresh thread that has no running loop.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fundamentals_lookup = pool.submit(
                    lambda: asyncio.run(_prefetch())
                ).result()
            logger.info(
                "[%s] prefetched fundamentals for meta-labeler inference "
                "(window %s → %s, %d feature_groups cached)",
                symbol, _start_ts.date(), _end_ts.date(),
                len(fundamentals_lookup.cache),
            )
        except Exception as exc:
            logger.warning(
                "[%s] fundamentals prefetch failed: %s — falling back to "
                "NaN-filled inference (Option A behavior)",
                symbol, exc,
            )
            fundamentals_lookup = None

    def _fund_fetcher(_sym: str, bar_ts) -> dict:
        if fundamentals_lookup is None:
            return {}
        return fundamentals_lookup.get(bar_ts)

    # Phase 2B Option 2 (2026-04-27): precompute rolling realized vol on
    # the same H4 OHLCV slice that drives the simulation. RV uses log
    # returns over closes; rolling 20- and 60-bar std with min_periods set
    # so warmup bars yield NaN (LightGBM handles NaN). Looked up by bar_ts
    # via searchsorted — strictly < ts (lookahead-safe since the current
    # bar's close is the entry signal's reference, not yet history).
    exec_fetcher = None
    if _meta_active or _meta_shadow:
        try:
            _close_series = pd.Series(closes.astype(float), index=ohlcv_aligned.index)
            _log_ret = np.log(_close_series / _close_series.shift(1))
            _rv_short = _log_ret.rolling(20, min_periods=20).std()
            _rv_long = _log_ret.rolling(60, min_periods=60).std()
            _bar_index = np.array(
                [pd.Timestamp(ts).to_datetime64() for ts in ohlcv_aligned.index],
                dtype="datetime64[ns]",
            )
            _rv_short_arr = _rv_short.values
            _rv_long_arr = _rv_long.values

            def _exec_fetcher(_sym: str, bar_ts) -> dict:
                if bar_ts is None:
                    return {}
                try:
                    ts = pd.Timestamp(bar_ts)
                    if ts.tz is not None:
                        ts = ts.tz_convert(None)
                    ts_np = np.datetime64(ts.to_datetime64(), "ns")
                except Exception:
                    return {}
                # Pick latest bar STRICTLY before ts (avoid using the bar
                # whose close is synchronous with signal generation).
                idx = np.searchsorted(_bar_index, ts_np, side="left") - 1
                if idx < 0:
                    return {}
                rs = _rv_short_arr[idx]
                rl = _rv_long_arr[idx]
                if not (np.isfinite(rs) and np.isfinite(rl)) or rl <= 0.0:
                    return {}
                return {
                    "exec__rv_short_20": float(rs),
                    "exec__rv_long_60": float(rl),
                    "exec__rv_ratio": float(rs / rl),
                }
            exec_fetcher = _exec_fetcher
            logger.info(
                "[%s] precomputed RV exec features over %d H4 bars",
                symbol, len(ohlcv_aligned),
            )
        except Exception as exc:
            logger.warning(
                "[%s] exec feature precompute failed: %s — meta-labeler "
                "will see NaN-filled exec features (degraded but functional)",
                symbol, exc,
            )
            exec_fetcher = None

    combiner = SignalCombiner(
        hmm=hmm,
        lstm=primary_predictor,  # duck-typed — LSTMPricePredictor or GBMPredictor
        hmm_weight=_hmm_w,
        lstm_weight=_lstm_w,
        long_only_mode=False,
        long_only_symbols=long_only_symbols,
        signal_threshold=_default_threshold,
        min_confidence=0.55,
        flicker_bars_required=_flicker_bars,
        fundamentals_fetcher=_fund_fetcher if fundamentals_lookup is not None else None,
        exec_features_fetcher=exec_fetcher,
    )
    # Attach per-symbol threshold map (read inside _fuse_signals)
    combiner.per_symbol_threshold = _per_sym_threshold
    orchestrator = StrategyOrchestrator()
    circuit_breaker = CircuitBreaker(
        max_daily_loss_soft_pct=CB_DAILY_SOFT_PCT,
        max_daily_loss_hard_pct=CB_DAILY_HARD_PCT,
        max_weekly_loss_soft_pct=CB_WEEKLY_SOFT_PCT,
        max_weekly_loss_hard_pct=CB_WEEKLY_HARD_PCT,
        max_peak_drawdown_pct=CB_PEAK_DD_PCT,
        # Disable consecutive loss breaker in backtests — it uses
        # wall clock time which doesn't advance during simulation
        consecutive_loss_limit=9999,
    )

    # ------------------------------------------------------------------
    # 5. Prepare H1 data for execution (if available)
    # ------------------------------------------------------------------
    # If H1 data is provided, we walk H1 bars for finer execution.
    # Signals still generate on H4 boundaries; entries/exits on H1.
    if h1_ohlcv is not None and len(h1_ohlcv) > 100:
        h1_clean = h1_ohlcv.copy()
        if "volume" in h1_clean.columns and "tick_volume" not in h1_clean.columns:
            h1_clean.rename(columns={"volume": "tick_volume"}, inplace=True)
        if "tick_volume" not in h1_clean.columns:
            h1_clean["tick_volume"] = 0
        # Align H1 to the H4 backtest period
        h1_mask = (h1_clean.index >= ohlcv_aligned.index[0]) & (
            h1_clean.index <= ohlcv_aligned.index[-1] + pd.Timedelta(hours=4)
        )
        h1_clean = h1_clean.loc[h1_mask]
        h1_opens = h1_clean["open"].values.astype(float)
        h1_highs = h1_clean["high"].values.astype(float)
        h1_lows = h1_clean["low"].values.astype(float)
        h1_closes = h1_clean["close"].values.astype(float)
        h1_timestamps = [str(ts) for ts in h1_clean.index]
        h1_atr = _compute_atr(h1_highs, h1_lows, h1_closes, ATR_PERIOD)
        use_h1 = len(h1_opens) > WARMUP_BARS * H1_PER_H4
    else:
        use_h1 = False

    # Build H4→index mapping for signal lookup
    h4_bar_timestamps = set(str(ts) for ts in ohlcv_aligned.index)

    # ------------------------------------------------------------------
    # 6. Walk-forward simulation
    # ------------------------------------------------------------------
    # Load per-symbol strategy parameters.
    # Sprint 3 architectural fix (2026-05-01): time_exit_h1 / tp_r /
    # be_trigger_r were previously hardcoded in SYMBOL_PARAMS and silently
    # diverged from settings.yaml for 5 of the 10 the trading universe pairs. yaml is
    # now the source of truth for ALL three Triple Barrier params; the
    # SYMBOL_PARAMS / DEFAULT_PARAMS values only apply when the yaml block
    # is absent.
    #
    # atr_sl_mult is INTENTIONALLY left as SYMBOL_PARAMS-only because live
    # strategies (MidVolCautious / HighVolDefensive / LowVolAggressive) own
    # SL computation via their own EMA-anchored or ATR-anchored formulas.
    # Backtest only falls back to atr_sl_mult * ATR when the strategy's
    # recommended stop is 0 — in practice never used. Yaml's atr_sl_mult
    # is informational documentation about per-pair vol-tiering, not config.
    sp = SYMBOL_PARAMS.get(symbol, DEFAULT_PARAMS)
    ATR_SL_MULT = sp["atr_sl_mult"]
    TP_R = sp["tp_r"]
    BE_TRIGGER_R = sp["be_trigger_r"]
    TIME_EXIT_BARS = sp["time_exit_h1"]
    RISK_PER_TRADE_PCT = sp["risk_pct"]
    # Override Triple Barrier params from settings.yaml when present
    try:
        import yaml as _yaml
        _settings = _yaml.safe_load(
            (Path(__file__).parent.parent / "config" / "settings.yaml").read_text(encoding="utf-8")
        )
        _yaml_psp = (
            ((_settings or {}).get("strategy", {}) or {})
            .get("per_symbol_params", {})
            .get(symbol, {})
        )
        _orig_te, _orig_tp, _orig_be = TIME_EXIT_BARS, TP_R, BE_TRIGGER_R
        if _yaml_psp.get("time_exit_h1_bars") is not None:
            TIME_EXIT_BARS = int(_yaml_psp["time_exit_h1_bars"])
        if _yaml_psp.get("tp_r_multiple") is not None:
            TP_R = float(_yaml_psp["tp_r_multiple"])
        if _yaml_psp.get("be_trigger_r") is not None:
            BE_TRIGGER_R = float(_yaml_psp["be_trigger_r"])
        if (TIME_EXIT_BARS, TP_R, BE_TRIGGER_R) != (_orig_te, _orig_tp, _orig_be):
            logger.info(
                "[%s] yaml override: time_exit=%d (was %d), tp_r=%.2f (was %.2f), "
                "be_trigger_r=%.2f (was %.2f)",
                symbol, TIME_EXIT_BARS, _orig_te, TP_R, _orig_tp, BE_TRIGGER_R, _orig_be,
            )
    except Exception as _exc:
        logger.warning("[%s] settings.yaml override failed: %s", symbol, _exc)

    equity = initial_equity
    peak_equity = initial_equity
    daily_start_equity = initial_equity
    weekly_start_equity = initial_equity

    # Resolve once — symbol is fixed for the whole run.
    slippage_price, commission_per_lot_per_side, units_per_lot = _resolve_friction(
        symbol, friction_override,
    )
    logger.info(
        "[%s] friction: slippage_price=%g, commission_per_lot_per_side=$%g, "
        "units_per_lot=%g",
        symbol, slippage_price, commission_per_lot_per_side, units_per_lot,
    )

    open_trade: Optional[_FullOpenTrade] = None
    pending_entry: Optional[dict] = None
    active_signal: Optional[dict] = None  # H4 signal valid for H1_PER_H4 bars
    signal_bars_remaining: int = 0

    equity_curve: list[dict] = []
    completed_trades: list[dict] = []

    last_day = None
    last_week = None
    last_month = None

    signal_attempts = 0
    gate_rejections = {"no_signal": 0, "below_threshold": 0, "confluence": 0,
                       "flicker": 0, "long_only": 0, "cb_blocked": 0,
                       "no_atr_ema": 0, "orchestrator_fail": 0}

    # Choose execution arrays based on H1 availability
    if use_h1:
        exec_opens = h1_opens
        exec_highs = h1_highs
        exec_lows = h1_lows
        exec_closes = h1_closes
        exec_timestamps = h1_timestamps
        exec_atr = h1_atr
        exec_n = len(exec_opens)
        exec_warmup = WARMUP_BARS * H1_PER_H4
    else:
        exec_opens = opens
        exec_highs = highs
        exec_lows = lows
        exec_closes = closes
        exec_timestamps = timestamps
        exec_atr = atr_series
        exec_n = n
        exec_warmup = WARMUP_BARS

    # E-7 trend-mode setup (no-op if trend_mode=False)
    trend_detector = None
    regime_tracker = None
    baseline_tp_r = None  # set below if trend_mode is on
    trend_tp_r = None
    # E-7 Task 14: initialize to None so the no-op path doesn't crash the
    # JSON-write block at the function tail. Reassigned to {} inside the
    # `if trend_mode:` block below.
    trend_mode_diag = None
    if trend_mode:
        from src.strategy.trend_mode import (
            TrendModeDetector,
            RegimeBarTracker,
            load_config_from_settings,
        )
        _, trend_cfg = load_config_from_settings("config/settings.yaml")
        trend_detector = TrendModeDetector(trend_cfg)
        regime_tracker = RegimeBarTracker()
        # Reuse TP_R already loaded from SYMBOL_PARAMS above. SYMBOL_PARAMS
        # is the in-file source of truth for tp_r in this script; settings.yaml
        # carries the same values for the live path. Two-source-of-truth was
        # caught in Task 10 review.
        baseline_tp_r = TP_R
        trend_tp_r = baseline_tp_r * trend_cfg.tp_r_multiplier
        # E-7 Task 14: per-(year_month, regime_label) counter — incremented
        # on each H4 bar where trend-mode is active. Written to JSON at run end.
        trend_mode_diag: dict[tuple[str, str], int] = {}

    for i in range(exec_warmup, exec_n):
        ts = exec_timestamps[i]

        # E-7: cache regime per-bar so trend-mode block can reuse the result
        # of signal-gen's HMM call (avoids ~2400 redundant hmm.predict() calls
        # over a 22mo backtest when no trade is open).
        _h4_regime_cache = None

        # Day/week/month boundary tracking
        try:
            bar_dt = pd.Timestamp(ts)
            # CB internals (period reset, consecutive-halt expiry) compare
            # against UTC-aware datetimes. Backtest timestamps are naive
            # UTC by convention (see CLAUDE.md "True UTC everywhere"); we
            # localize so the simulated clock — not wall-clock — drives
            # CB resets. Without this, daily/weekly soft breakers latch on
            # the first trip and never clear (the 5-yr run completes in
            # seconds of wall-time, so wall-clock day never advances).
            bar_dt_utc = bar_dt if bar_dt.tz is not None else bar_dt.tz_localize("UTC")
            current_day = bar_dt.date()
            current_week = bar_dt.isocalendar()[1]
            current_month = bar_dt.month
        except Exception:
            current_day = current_week = current_month = None
            bar_dt_utc = None

        if last_day is not None and current_day != last_day:
            daily_start_equity = equity
        if last_week is not None and current_week != last_week:
            weekly_start_equity = equity
        if last_month is not None and current_month != last_month:
            if circuit_breaker.is_halted():
                circuit_breaker.manual_reset()
                peak_equity = equity
                daily_start_equity = equity
                weekly_start_equity = equity
                logger.info("CB auto-reset at month boundary %s", bar_dt)
        last_day = current_day
        last_week = current_week
        last_month = current_month

        # --- Circuit breaker ---
        cb_snap = circuit_breaker.check_and_update(
            current_equity=equity,
            daily_start_equity=daily_start_equity,
            weekly_start_equity=weekly_start_equity,
            peak_equity=peak_equity,
            now=bar_dt_utc.to_pydatetime() if bar_dt_utc is not None else None,
        )

        # Force-close on CB halt
        if cb_snap.requires_flat and open_trade is not None:
            t = open_trade
            exit_price = _apply_exit_slippage(
                exec_closes[i], t.direction, slippage_price,
            )
            gross_pnl = ((exit_price - t.entry_price) if t.direction == "buy"
                    else (t.entry_price - exit_price)) * t.volume
            commission = (
                2.0 * commission_per_lot_per_side * (t.volume / units_per_lot)
            )
            pnl = gross_pnl - commission
            r_dist = t.initial_r_dist
            risk_amount = r_dist * t.initial_volume if r_dist > 0 else 1.0
            equity += pnl
            # E-7: stamp trend-mode state at close (cb-flat path)
            if trend_detector is not None:
                _t_dir_int = +1 if t.direction == "buy" else -1
                if trend_detector.is_active(symbol, _t_dir_int):
                    t.was_in_trend_mode_at_close = True
            completed_trades.append({
                "symbol": symbol, "direction": t.direction,
                "entry_time": t.entry_time, "exit_time": ts,
                "entry_price": round(t.entry_price, 5),
                "exit_price": round(exit_price, 5),
                "pnl": round(pnl, 2),
                "commission": round(commission, 2),
                "r_multiple": round(pnl / risk_amount, 4) if risk_amount > 0 else 0.0,
                "trend_pnl_delta": 0.0,
                "exit_reason": "circuit_breaker",
                "strategy_name": t.strategy_name,
                "regime_label": t.regime_label,
                "combined_score": round(t.combined_score, 4),
                "was_in_trend_mode_at_close": t.was_in_trend_mode_at_close,
            })
            open_trade = None
            pending_entry = None
            active_signal = None
            combiner.reset_state()

        # --- Execute pending entry at this bar's open ---
        # Smart news blackout (Phase C.1+C.2): block entries in the
        # pre-news + spike zone (T-24h to T+2h) but ALLOW the post-news
        # continuation window (T+2h to T+48h) where the retail edge is.
        # Per-symbol routing: FOMC hits all USD pairs; ECB→EUR; BoJ→JPY;
        # BoC→CAD. XAUUSD is exempt (gold often benefits from either
        # direction of rate decisions).
        news_blackout = False
        try:
            from src.data_pipeline.market.calendar_features import (
                is_in_news_blackout,
            )
            bar_dt_check = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)
            news_blackout = is_in_news_blackout(symbol, bar_dt_check)
        except Exception:
            news_blackout = False

        if (pending_entry is not None and open_trade is None
                and cb_snap.multiplier > 0 and not news_blackout):
            pe = pending_entry
            atr_val = exec_atr[i - 1] if i > 0 and not np.isnan(exec_atr[i - 1]) else None

            if atr_val is not None and atr_val > 0:
                # Sizing uses the NOMINAL (pre-slip) entry price — matches
                # live behavior where volume is computed at order-send time
                # before the fill price is known. Post-slip entry price is
                # used below for PnL only.
                #
                # Using post-slip for sizing caused a catastrophic bug
                # (2026-04-18): when pe["stop_price"] happened to equal
                # the bar open, post-slip entry produced sl_dist == slippage
                # (e.g. 0.00005 for EURUSD) → volume blew up to millions,
                # one unfavorable bar wiped the account (5-yr EUR/CAD
                # backtest: 1 trade, ~$80K loss).
                nominal_entry = exec_opens[i]
                sl_dist = abs(nominal_entry - pe["stop_price"]) if pe["stop_price"] > 0 else atr_val * ATR_SL_MULT
                if sl_dist <= 0:
                    sl_dist = atr_val * ATR_SL_MULT

                stop_loss = (nominal_entry - sl_dist if pe["direction"] == "buy"
                             else nominal_entry + sl_dist)

                # E-7: if trend-mode active for this signal direction, use
                # the widened TP multiplier (baseline tp_r * trend cfg multiplier)
                # AND mark the trade time-exit-disabled so Task 11's exit barrier
                # skips the time-exit branch.
                if trend_detector is not None:
                    _sig_dir_int = +1 if pe["direction"] == "buy" else -1
                    if trend_detector.is_active(symbol, _sig_dir_int):
                        _effective_tp_r = trend_tp_r
                        _time_exit_disabled_at_entry = True
                    else:
                        _effective_tp_r = TP_R
                        _time_exit_disabled_at_entry = False
                else:
                    _effective_tp_r = TP_R
                    _time_exit_disabled_at_entry = False

                # Triple Barrier: compute TP price at entry (from nominal)
                if pe["direction"] == "buy":
                    tp_price = nominal_entry + sl_dist * _effective_tp_r
                else:
                    tp_price = nominal_entry - sl_dist * _effective_tp_r

                # Effective entry price paid by the trader, for PnL only.
                # All SL/TP comparisons below still use `nominal_entry`
                # via t.entry_price — we overwrite it to the effective
                # value only for PnL accounting at close.
                entry_price = _apply_entry_slippage(
                    nominal_entry, pe["direction"], slippage_price,
                )

                risk_amount = equity * RISK_PER_TRADE_PCT / 100.0
                raw_volume = risk_amount / sl_dist if sl_dist > 0 else 0.0
                volume = (raw_volume * pe["allocation_pct"]
                          * cb_snap.multiplier * pe.get("size_discount", 1.0))

                if volume > 0:
                    open_trade = _FullOpenTrade(
                        symbol=symbol,
                        direction=pe["direction"],
                        entry_bar=i,
                        entry_time=ts,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        atr=atr_val,
                        atr_trail_mult=pe["atr_trail_mult"],
                        volume=volume,
                        initial_volume=volume,
                        initial_r_dist=sl_dist,
                        strategy_name=pe["strategy_name"],
                        regime_label=pe["regime_label"],
                        combined_score=pe["combined_score"],
                        tp_price=tp_price,
                        time_exit_disabled=_time_exit_disabled_at_entry,
                    )
            pending_entry = None

        # --- Triple Barrier exit checks ---
        if open_trade is not None:
            t = open_trade
            is_buy = t.direction == "buy"
            bar_high = exec_highs[i]
            bar_low = exec_lows[i]
            exit_price = None
            exit_reason = None
            r_dist = t.initial_r_dist

            t.bars_held += 1

            # 1. Stop loss hit
            if is_buy and bar_low <= t.stop_loss:
                exit_price = t.stop_loss
                exit_reason = "sl"
            elif not is_buy and bar_high >= t.stop_loss:
                exit_price = t.stop_loss
                exit_reason = "sl"

            # 2. Time exit (skipped when E-7 trend-mode disables it on this trade)
            if (
                exit_reason is None
                and not t.time_exit_disabled
                and t.bars_held >= TIME_EXIT_BARS
            ):
                exit_price = exec_closes[i]
                exit_reason = "time_exit"

            # 3. Take-profit
            if exit_reason is None:
                if is_buy and bar_high >= t.tp_price:
                    exit_price = t.tp_price
                    exit_reason = "take_profit"
                elif not is_buy and bar_low <= t.tp_price:
                    exit_price = t.tp_price
                    exit_reason = "take_profit"

            # 4. Breakeven lock at +1R
            if exit_reason is None and not t.be_locked and r_dist > 0:
                if is_buy:
                    current_r = (bar_high - t.entry_price) / r_dist
                else:
                    current_r = (t.entry_price - bar_low) / r_dist
                if current_r >= BE_TRIGGER_R:
                    t.be_locked = True
                    t.stop_loss = t.entry_price

            # Execute exit
            if exit_price is not None:
                # Slippage applied to the nominal exit price. SL/TP exits
                # use their trigger price as nominal; time/CB exits use bar
                # close. Slippage always costs the trader.
                exit_price = _apply_exit_slippage(
                    exit_price, t.direction, slippage_price,
                )
                gross_pnl = ((exit_price - t.entry_price) if is_buy
                        else (t.entry_price - exit_price)) * t.volume
                commission = (
                2.0 * commission_per_lot_per_side * (t.volume / units_per_lot)
            )
                pnl = gross_pnl - commission
                risk_amount = r_dist * t.initial_volume if r_dist > 0 else 1.0
                r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0
                equity += pnl
                # Record consecutive loss for circuit breaker
                if pnl < 0:
                    circuit_breaker.record_trade_result(is_loss=True)
                else:
                    circuit_breaker.record_trade_result(is_loss=False)

                # E-7: stamp whether trend-mode was active for this position
                # in this direction when it closed. Used by Task 14 diagnostics.
                if trend_detector is not None:
                    _t_dir_int = +1 if t.direction == "buy" else -1
                    if trend_detector.is_active(symbol, _t_dir_int):
                        t.was_in_trend_mode_at_close = True

                # E-7 Task 15: counterfactual PnL attribution.
                # When this trade was in trend-mode at close AND exited at
                # TP, the (widened) TP price differs from the baseline TP
                # by (trend_tp_r - baseline_tp_r) * initial_r_dist. The
                # attribution = (actual exit pnl) - (what the trade would
                # have earned exiting at baseline_tp_r).
                #
                # Limitation: this assumes the trade WOULD have hit the
                # baseline TP first under baseline rules. True for nearly
                # all real cases since baseline_tp_r < trend_tp_r and the
                # price MUST have crossed both to reach the widened TP.
                # Edge case ignored: trades that re-touched baseline TP
                # after the widened TP fired (rare and small in practice).
                # For SL/time-exit/reversal/breaker exits, delta = 0.
                trend_pnl_delta = 0.0
                if (
                    trend_detector is not None
                    and t.was_in_trend_mode_at_close
                    and exit_reason == "take_profit"
                    and t.initial_r_dist > 0
                ):
                    # Counterfactual baseline-TP price for this trade
                    if t.direction == "buy":
                        cf_tp_price = t.entry_price + (baseline_tp_r * t.initial_r_dist)
                    else:
                        cf_tp_price = t.entry_price - (baseline_tp_r * t.initial_r_dist)
                    cf_exit_price = _apply_exit_slippage(
                        cf_tp_price, t.direction, slippage_price,
                    )
                    if t.direction == "buy":
                        cf_gross = (cf_exit_price - t.entry_price) * t.volume
                    else:
                        cf_gross = (t.entry_price - cf_exit_price) * t.volume
                    cf_commission = (
                        2.0 * commission_per_lot_per_side * (t.volume / units_per_lot)
                    )
                    cf_pnl = cf_gross - cf_commission
                    trend_pnl_delta = pnl - cf_pnl

                completed_trades.append({
                    "symbol": symbol, "direction": t.direction,
                    "entry_time": t.entry_time, "exit_time": ts,
                    "entry_price": round(t.entry_price, 5),
                    "exit_price": round(exit_price, 5),
                    "pnl": round(pnl, 2),
                    "commission": round(commission, 2),
                    "r_multiple": round(r_multiple, 4),
                    "trend_pnl_delta": round(trend_pnl_delta, 2),
                    "exit_reason": exit_reason,
                    "strategy_name": t.strategy_name,
                    "regime_label": t.regime_label,
                    "combined_score": round(t.combined_score, 4),
                    "was_in_trend_mode_at_close": t.was_in_trend_mode_at_close,
                })
                open_trade = None

        # --- Generate signal on H4 boundaries ---
        is_h4_boundary = (not use_h1) or (ts in h4_bar_timestamps)

        if is_h4_boundary and open_trade is None:
            # Map to H4 feature index
            if use_h1:
                # Find closest H4 bar index for this timestamp
                h4_idx = None
                for j in range(n - 1, -1, -1):
                    if timestamps[j] <= ts:
                        h4_idx = j
                        break
                if h4_idx is None or h4_idx < WARMUP_BARS + SIGNAL_WINDOW:
                    h4_idx = None
            else:
                h4_idx = i if i >= WARMUP_BARS + SIGNAL_WINDOW else None

            if h4_idx is not None:
                if cb_snap.requires_flat or cb_snap.multiplier <= 0:
                    gate_rejections["cb_blocked"] += 1
                elif np.isnan(atr_series[h4_idx]) or np.isnan(ema50_series[h4_idx]):
                    gate_rejections["no_atr_ema"] += 1
                else:
                    signal_attempts += 1
                    # Use D1 features for HMM if available (matches training
                    # timeframe). Map H4 timestamp to nearest D1 bar index.
                    if _use_d1_hmm:
                        _h4_ts = pd.Timestamp(timestamps[h4_idx])
                        # Find closest D1 bar <= this H4 bar's timestamp (O(log n))
                        _d1_idx_raw = hmm_d1_aligned.index.searchsorted(_h4_ts, side="right") - 1
                        _d1_idx = int(_d1_idx_raw) if _d1_idx_raw >= 0 else None
                        if _d1_idx is not None and _d1_idx >= SIGNAL_WINDOW - 1:
                            hmm_window = hmm_d1_matrix[_d1_idx - SIGNAL_WINDOW + 1: _d1_idx + 1]
                        else:
                            hmm_window = hmm_matrix[h4_idx - SIGNAL_WINDOW + 1: h4_idx + 1]
                    else:
                        hmm_window = hmm_matrix[h4_idx - SIGNAL_WINDOW + 1: h4_idx + 1]
                    if primary == "lstm":
                        primary_input = lstm_matrix[h4_idx - SIGNAL_WINDOW + 1: h4_idx + 1]
                    else:  # primary == "gbm" — flat row, no temporal window
                        primary_input = lstm_matrix[h4_idx]

                    try:
                        # Pass bar timestamp so the meta-labeler's fundamentals
                        # fetcher (Option B inference path) can look up
                        # release-lag-safe fundamentals at this bar's time.
                        # Without this, the fetcher gets None → empty dict →
                        # NaN-filled predictions (the v2 silent-degradation bug).
                        _signal_bar_ts = pd.Timestamp(timestamps[h4_idx])
                        signal = combiner.get_signal(
                            symbol, hmm_window,
                            lstm_sequence=primary_input,
                            bar_timestamp=_signal_bar_ts,
                        )
                    except Exception as exc:
                        logger.debug("Signal error at H4 bar %d: %s", h4_idx, exc)
                        signal = None

                    # E-7 Fix 2: cache regime so trend-mode block can reuse
                    if signal is not None:
                        _h4_regime_cache = signal.regime

                    if signal is None:
                        gate_rejections["no_signal"] += 1
                    elif not signal.should_trade or not signal.direction:
                        reasons = " ".join(signal.reasoning) if signal.reasoning else ""
                        if "below_threshold" in reasons:
                            gate_rejections["below_threshold"] += 1
                        elif "confluence_fail" in reasons:
                            gate_rejections["confluence"] += 1
                        elif "flickering" in reasons:
                            gate_rejections["flicker"] += 1
                        elif "long_only_mode" in reasons:
                            gate_rejections["long_only"] += 1
                        else:
                            gate_rejections["no_signal"] += 1
                    elif signal.should_trade and signal.direction:
                        ctx = MarketContext(
                            symbol=symbol,
                            price=closes[h4_idx],
                            atr=atr_series[h4_idx],
                            ema50=ema50_series[h4_idx],
                        )
                        try:
                            decision = orchestrator.select(
                                signal, ctx,
                                current_equity=equity,
                                peak_equity=peak_equity,
                            )
                        except Exception:
                            decision = None

                        if decision is not None:
                            # Grid-test hook: if this is a USDJPY buy in
                            # Euphoria regime and the soft-guard is enabled
                            # (multiplier > 0), scale the entry size down.
                            _size_discount = signal.size_discount
                            if ("JPY" in symbol.upper()
                                    and signal.direction == "buy"
                                    and signal.regime.regime_index == 4):
                                _euph_mult = float(os.environ.get(
                                    "CORTEX_USDJPY_EUPHORIA_MULT", "0.0"
                                ))
                                if _euph_mult > 0.0:
                                    _size_discount *= _euph_mult
                            pending_entry = {
                                "direction": signal.direction,
                                "stop_price": decision.initial_stop_price,
                                "allocation_pct": decision.allocation_pct,
                                "atr_trail_mult": decision.atr_trail_mult,
                                "strategy_name": decision.strategy_name,
                                "regime_label": signal.regime.regime_label,
                                "combined_score": signal.combined_score,
                                "size_discount": _size_discount,
                            }
                            signal_bars_remaining = H1_PER_H4 if use_h1 else 1
        elif use_h1 and active_signal is not None and signal_bars_remaining > 0:
            # H1 bar within active H4 signal window — attempt entry
            signal_bars_remaining -= 1
            if open_trade is None and pending_entry is None:
                pending_entry = active_signal

        # --- E-7 trend-mode: per-H4-bar detector update + retroactive sweep ---
        # Fires on every H4 boundary regardless of open_trade state, so
        # activation/deactivation tracks regime even during open positions and
        # retroactive sweeps on flip-ON can rescue in-flight winners.
        if trend_detector is not None and is_h4_boundary:
            # Resolve h4_idx (mirror the existing pattern at lines 918-928)
            if use_h1:
                _tm_h4_idx = None
                for j in range(n - 1, -1, -1):
                    if timestamps[j] <= ts:
                        _tm_h4_idx = j
                        break
                if _tm_h4_idx is not None and _tm_h4_idx < WARMUP_BARS + SIGNAL_WINDOW:
                    _tm_h4_idx = None
            else:
                _tm_h4_idx = i if i >= WARMUP_BARS + SIGNAL_WINDOW else None

            if _tm_h4_idx is not None:
                # Reuse the same hmm_window construction the signal-gen block
                # uses (lines 939-949) so we feed the HMM the same input it'd
                # see during normal signal generation.
                if _use_d1_hmm:
                    _h4_ts_tm = pd.Timestamp(timestamps[_tm_h4_idx])
                    _d1_idx_tm_raw = hmm_d1_aligned.index.searchsorted(_h4_ts_tm, side="right") - 1
                    _d1_idx_tm = int(_d1_idx_tm_raw) if _d1_idx_tm_raw >= 0 else None
                    if _d1_idx_tm is not None and _d1_idx_tm >= SIGNAL_WINDOW - 1:
                        _hmm_window_tm = hmm_d1_matrix[_d1_idx_tm - SIGNAL_WINDOW + 1: _d1_idx_tm + 1]
                    else:
                        _hmm_window_tm = hmm_matrix[_tm_h4_idx - SIGNAL_WINDOW + 1: _tm_h4_idx + 1]
                else:
                    _hmm_window_tm = hmm_matrix[_tm_h4_idx - SIGNAL_WINDOW + 1: _tm_h4_idx + 1]

                # Reuse the regime computed by signal-gen on the same bar if
                # available (open_trade was None this iter); otherwise call
                # hmm.predict() directly (open_trade is not None — signal-gen
                # was skipped). Fix 2 from review: avoids redundant HMM call.
                if _h4_regime_cache is not None:
                    _tm_regime = _h4_regime_cache
                else:
                    try:
                        _tm_regime = hmm.predict(symbol, _hmm_window_tm)
                    except Exception as exc:
                        logger.debug(
                            "E-7 HMM predict error at H4 bar %d: %s",
                            _tm_h4_idx, exc,
                        )
                        _tm_regime = None

                if _tm_regime is not None:
                    _bars_in_regime = regime_tracker.update(symbol, _tm_regime.regime_index)
                    _row = hmm_feature_df.iloc[_tm_h4_idx]
                    _tm_adx = float(_row.get("adx", float("nan")))
                    _tm_plus_di = float(_row.get("plus_di", float("nan")))
                    _tm_minus_di = float(_row.get("minus_di", float("nan")))
                    _tm_er = float(_row.get("efficiency_ratio", float("nan")))

                    # NaN guard: early bars (warmup) won't have ADX yet
                    if not (
                        np.isnan(_tm_adx) or np.isnan(_tm_plus_di)
                        or np.isnan(_tm_minus_di) or np.isnan(_tm_er)
                    ):
                        _tm_state = trend_detector.update(
                            symbol=symbol,
                            bar_idx=_tm_h4_idx,
                            regime_index=_tm_regime.regime_index,
                            bars_in_regime=_bars_in_regime,
                            adx=_tm_adx,
                            plus_di=_tm_plus_di,
                            minus_di=_tm_minus_di,
                            er=_tm_er,
                        )

                        # E-7 Task 14: aggregate per-(year_month, regime_label).
                        # Only increment when trend-mode is currently active
                        # (don't count just_deactivated bars — they are not
                        # active for the position-sizing/exit consequence).
                        if _tm_state.active:
                            _bar_ts = pd.Timestamp(timestamps[_tm_h4_idx])
                            _ym = f"{_bar_ts.year:04d}-{_bar_ts.month:02d}"
                            _key = (_ym, _tm_regime.regime_label)
                            trend_mode_diag[_key] = trend_mode_diag.get(_key, 0) + 1

                        # E-7 v2 gate (2026-04-28): action paths only fire for symbols
                        # in enabled_symbols. Diagnostic state tracking above is
                        # universal so we can analyze "would trend-mode help this
                        # symbol?" post-hoc without committing to the action.
                        _v2_gate_open = (
                            symbol.upper() in trend_detector.config.enabled_symbols
                        )

                        # Retroactive sweep: on flip-ON, widen the open trade's
                        # tp_price + disable time-exit IF direction matches the
                        # newly-active trend direction. open_trade is SINGULAR
                        # in this backtest (max 1 trade at a time per symbol).
                        if (
                            _v2_gate_open
                            and _tm_state.just_activated
                            and open_trade is not None
                            and open_trade.initial_r_dist > 0
                        ):
                            _open_dir = +1 if open_trade.direction == "buy" else -1
                            if _open_dir == _tm_state.direction:
                                if open_trade.direction == "buy":
                                    open_trade.tp_price = (
                                        open_trade.entry_price
                                        + (trend_tp_r * open_trade.initial_r_dist)
                                    )
                                else:
                                    open_trade.tp_price = (
                                        open_trade.entry_price
                                        - (trend_tp_r * open_trade.initial_r_dist)
                                    )
                                open_trade.time_exit_disabled = True

                        # Soft revert on flip-OFF (spec §4.2 amendment, 2026-04-27):
                        # Re-enable time-exit on the open trade so it can close on
                        # the next time-exit cycle. Widened TP is preserved (no
                        # rug-pull on the upside). Without this, a stale position
                        # can stay open indefinitely after the regime that justified
                        # trend-mode is gone (USDJPY 2025: 286-day lockup observed).
                        # Gated by v2 enabled_symbols so a v1-style action can't
                        # fire on a v2-disabled symbol.
                        if (
                            _v2_gate_open
                            and _tm_state.just_deactivated
                            and open_trade is not None
                            and open_trade.time_exit_disabled
                        ):
                            open_trade.time_exit_disabled = False

        # Decay signal window
        if signal_bars_remaining > 0:
            active_signal = pending_entry
            signal_bars_remaining -= 1
        else:
            active_signal = None

        # --- Record equity ---
        peak_equity = max(peak_equity, equity)
        dd_pct = ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0.0
        equity_curve.append({
            "bar_timestamp": ts,
            "equity": round(equity, 2),
            "drawdown_pct": round(dd_pct, 4),
        })

    logger.info(
        "Full backtest %s complete: %d trades, final equity $%.2f",
        symbol, len(completed_trades), equity,
    )

    # Signal diagnostics
    logger.info(
        "Signal diagnostics %s: %d attempts, %d trades. "
        "Rejections: threshold=%d, confluence=%d, flicker=%d, "
        "long_only=%d, cb_blocked=%d, no_atr_ema=%d, no_signal=%d",
        symbol, signal_attempts, len(completed_trades),
        gate_rejections["below_threshold"],
        gate_rejections["confluence"],
        gate_rejections["flicker"],
        gate_rejections["long_only"],
        gate_rejections["cb_blocked"],
        gate_rejections["no_atr_ema"],
        gate_rejections["no_signal"],
    )

    # E-7 Task 14: write trend-mode diagnostic JSON if trend_mode was on.
    # NOTE: do NOT re-import Path here — it's already imported at module top
    # (line 26). A local re-import would shadow it as a function-local and
    # break the earlier `Path("data/logs/TRADING_HALTED.flag")` reference at
    # line ~314 with UnboundLocalError when trend_mode=False (caught by
    # tests/test_backtest/test_engine.py after Task 14 landed).
    if trend_mode_diag is not None and trend_detector is not None:
        import json
        out_dir = Path("data/logs/trend_mode_ab")
        out_dir.mkdir(parents=True, exist_ok=True)
        diag_path = out_dir / f"{symbol}_trend_mode_diag.json"
        diag_payload = {
            "symbol": symbol,
            "by_month_and_regime": {
                f"{ym}|{regime}": count
                for (ym, regime), count in trend_mode_diag.items()
            },
            "final_state_snapshot": trend_detector.snapshot(),
        }
        diag_path.write_text(json.dumps(diag_payload, indent=2))
        logger.info("E-7 trend-mode diagnostic written to %s", diag_path)

    return equity_curve, completed_trades
