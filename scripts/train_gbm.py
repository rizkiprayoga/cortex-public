"""
train_gbm.py — the model bake-off GBM-primary training script.

Mirrors ``scripts/train_deep_learning.py``'s CLI surface but trains a
LightGBM 3-class classifier on Triple-Barrier labels {-1, 0, +1} mapped
to {0, 1, 2}. MT5-free (DB-direct OHLCV reads via
``MT5DataFeed.get_historical_db_only`` — see T-9 / commit d622dd7).
Strict TVT split per spec §3 invariant #14: training/val data is clipped
to ``[train_start, val_end]`` upstream so the test window is NEVER
loaded into memory.

Pipeline per symbol:
  1. Fetch H4 OHLCV from DB, clip to [train_start, val_end_exclusive].
  2. Fetch D1 OHLCV (same clip) for HMM regime inference.
  3. Inject 6 regime columns via FeatureEngineer.inject_regime_features.
  4. Compute TB labels on H4 OHLCV via
     FeatureEngineer.compute_triple_barrier_labels.
  5. Build flat-row features via src.brain.gbm.gbm_features.build_features.
  6. Drop warmup rows (NaN in lag/rolling cols) and align features+labels.
  7. Split into train/val on the train_end timestamp.
  8. With --tune: run Optuna study (per-symbol SQLite store, trial 0 =
     literature defaults on a fresh study), save _default + _tuned
     artifacts. Without --tune: train one model with literature defaults.

Usage:
    python scripts/train_gbm.py --symbols XAUUSD --tune --tune-trials 20
    python scripts/train_gbm.py --symbols XAUUSD EURUSD USDJPY USDCAD
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.brain.gbm.gbm_features import build_features, build_features_rich
from src.brain.gbm.gbm_model import GBMPredictor
from src.brain.hmm_regime import HMMRegimeClassifier
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.data_pipeline.mt5_feed import MT5DataFeed
from src.utils.model_head import validate_kind_head

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_gbm")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models"
TUNING_SPACES_PATH = PROJECT_ROOT / "config" / "tuning_spaces.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.yaml"

# Phase A scope per spec — XAUUSD/EURUSD/USDJPY/USDCAD only
LIVE_SYMBOLS = ("XAUUSD", "EURUSD", "USDJPY", "USDCAD")


def parse_args():
    p = argparse.ArgumentParser(description="Train GBM (LightGBM) primary model")
    p.add_argument("--symbols", nargs="+", default=list(LIVE_SYMBOLS))
    p.add_argument("--bars", type=int, default=0,
                   help="H4 bars of history (0 = all available within TVT clip)")

    # --- Phase A: explicit train/val/test windows (invariant #14) ---
    p.add_argument("--train-start", default="2021-01-01")
    p.add_argument("--train-end", default="2024-06-30")
    p.add_argument("--val-start", default="2024-07-01")
    p.add_argument("--val-end", default="2025-04-30")
    p.add_argument("--test-start", default="2025-05-01")
    p.add_argument("--test-end", default="2026-04-30")

    # --- Phase A: Optuna hyperparameter tuning ---
    p.add_argument(
        "--tune", action="store_true",
        help="Run Optuna study per symbol. Trial 0 = literature defaults "
             "(only on fresh study); trials 1..N-1 sample from "
             "config/tuning_spaces.yaml gbm.search. Saves both "
             "gbm_{symbol}_default.pkl and gbm_{symbol}_tuned.pkl."
    )
    p.add_argument(
        "--tune-trials", type=int, default=20,
        help="Number of Optuna trials when --tune is set (default 20)."
    )
    p.add_argument(
        "--rich-features", action="store_true",
        help="DEPRECATED alias for --features rich. Phase 2B Q1: rich "
             "single-timeframe technical surface."
    )
    p.add_argument(
        "--features", choices=("thin", "rich", "parity"), default=None,
        help="Feature surface for GBM training (Phase 2B Q1.5):\n"
             "  thin    — 36 features (lag + rolling stats + regime + "
             "crosses on close stats). Default; what the model bake-off used.\n"
             "  rich    — 96 features (FeatureEngineer.transform single-TF "
             "+ regime + lag-on-log-return + crosses on top indicators).\n"
             "  parity  — same feature surface as the production LSTM: "
             "transform_multi_timeframe(H4+D1+W1) + calendar features + "
             "zero-fill cross-asset placeholders + regime injection. NO "
             "GBM-specific augmentations — the GBM and LSTM see exactly "
             "the same columns. Q1.5 fairness experiment."
    )
    args = p.parse_args()
    # Backward-compat: --rich-features → --features rich.
    if args.rich_features and args.features is None:
        args.features = "rich"
    if args.features is None:
        args.features = "thin"
    return args


# ---------------------------------------------------------------------------
# Data loading + feature build (DB-direct, MT5-free)
# ---------------------------------------------------------------------------

async def _load_features_and_labels(
    symbol: str,
    args: argparse.Namespace,
    *,
    feed: MT5DataFeed,
    engineer: FeatureEngineer,
    train_start_ts: pd.Timestamp,
    val_end_ts_exclusive: pd.Timestamp,
    test_start_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Build the GBM feature matrix + TB labels on the train+val window.

    Returns:
        (features, labels_012, n_bars_after_clip) where features is a
        DataFrame indexed by H4 bar timestamp (naive UTC), labels_012 is
        a numpy array of int8 in {0, 1, 2} mapped from TB {-1, 0, +1}.
    """
    bars_arg = args.bars if args.bars > 0 else 99999

    # --- H4 OHLCV ---
    h4 = await feed.get_historical_db_only(symbol, "H4", bars=bars_arg)
    if h4 is None or h4.empty:
        raise RuntimeError(
            f"[{symbol}] no H4 bars in DB. Has the live bot or backfill ever "
            f"populated this symbol? See backfill_ohlcv.py."
        )
    # H4 column normalization — mt5_feed's DB path returns 'tick_volume',
    # but build_features expects 'volume'. Normalize once.
    if "tick_volume" in h4.columns and "volume" not in h4.columns:
        h4 = h4.rename(columns={"tick_volume": "volume"})

    # Phase A invariant #14 clip — strict-< exclusive upper bound.
    h4 = h4[(h4.index >= train_start_ts) & (h4.index < val_end_ts_exclusive)]
    if len(h4) == 0:
        raise RuntimeError(
            f"[{symbol}] H4 dataset is empty after clip "
            f"[{train_start_ts}, {val_end_ts_exclusive}). Check TVT flags."
        )
    # Defense-in-depth: explicit raise (not assert, won't strip under -O).
    if h4.index.max() >= test_start_ts:
        raise ValueError(
            f"[invariant #14] {symbol} H4 extends into test window: "
            f"max bar={h4.index.max()} >= test_start={test_start_ts}."
        )

    # --- D1 OHLCV (for HMM regime inference + multi-TF features) ---
    d1 = await feed.get_historical_db_only(symbol, "D1", bars=bars_arg // 6 or 99999)
    if "tick_volume" in d1.columns and "volume" not in d1.columns:
        d1 = d1.rename(columns={"tick_volume": "volume"})
    d1 = d1[(d1.index >= train_start_ts) & (d1.index < val_end_ts_exclusive)]

    # --- W1 OHLCV (only needed for --features parity, but cheap to fetch
    # always so the multi-TF transform matches the LSTM input frame). ---
    w1 = None
    if args.features == "parity":
        w1 = await feed.get_historical_db_only(
            symbol, "W1", bars=bars_arg // 42 or 99999,
        )
        if w1 is not None and not w1.empty:
            if "tick_volume" in w1.columns and "volume" not in w1.columns:
                w1 = w1.rename(columns={"tick_volume": "volume"})
            w1 = w1[(w1.index >= train_start_ts) & (w1.index < val_end_ts_exclusive)]
            if w1.empty:
                w1 = None

    logger.info(
        "[%s] H4=%d bars D1=%d bars W1=%s bars (clipped to [%s, %s])",
        symbol, len(h4), len(d1),
        len(w1) if w1 is not None else "skipped",
        args.train_start, args.val_end,
    )

    # --- Regime injection (6 columns appended) ---
    hmm = HMMRegimeClassifier()
    if not hmm.load(symbol):
        raise RuntimeError(
            f"[{symbol}] HMM not found at data/models/hmm_{symbol}.pkl. "
            f"Run scripts/train_hmm.py first."
        )
    # --- TB labels (computed on H4 OHLCV — done BEFORE feature transforms
    # since transform() drops the raw OHLCV columns) ---
    tb_labels = FeatureEngineer.compute_triple_barrier_labels(h4)
    # tb_labels: numpy array shape (len(h4),) with values {-1, 0, +1}
    # The last `time_limit_bars` entries are NaN (insufficient forward
    # window); we drop them when masking warmup below.

    # --- Build flat-row features ---
    if args.features == "rich":
        # Phase 2B Q1 — rich single-timeframe technical surface from
        # FeatureEngineer.transform() (ADX, ER, MACD, RSI, MFI, OBV, BB,
        # ATR, Hurst, skew/kurt, …), plus regime cols, plus GBM-specific
        # lag-on-log_return + crosses on top indicators. H4 ONLY.
        h4_rich = engineer.transform(h4.copy())
        h4_with_regime = engineer.inject_regime_features(
            h4_rich, hmm, symbol, d1,
        )
        features = build_features_rich(h4_with_regime)
    elif args.features == "parity":
        # Phase 2B Q1.5 — INFORMATION-EQUAL to the production LSTM. Mirrors
        # the LSTM serving pipeline in scripts/backtest_full.py exactly:
        # multi-TF (H4+D1+W1) transform, calendar features, zero-fill
        # cross-asset placeholders, regime injection. NO GBM-specific
        # augmentations — the GBM gets exactly the same columns the LSTM
        # consumes. Lets us answer "is the $30K LSTM-vs-GBM gap structural
        # (architecture) or accidental (we starved the GBM of D1+W1+
        # calendar+cross-asset)?".
        ohlcv_by_tf = {"H4": h4.copy()}
        if d1 is not None and len(d1) > 50:
            ohlcv_by_tf["D1"] = d1.copy()
        if w1 is not None and len(w1) > 10:
            ohlcv_by_tf["W1"] = w1.copy()
        feat = engineer.transform_multi_timeframe(
            ohlcv_by_tf, primary_tf="H4",
        )
        # Calendar features
        try:
            from src.data_pipeline.market.calendar_features import (
                CalendarFeatureBuilder,
            )
            cal = CalendarFeatureBuilder()
            cal_df = cal.get_historical_calendar_features(feat.index)
            feat = feat.join(cal_df, how="left")
        except Exception as exc:
            logger.warning(
                "[%s] calendar feature build failed (non-fatal): %s",
                symbol, exc,
            )
        # Zero-fill cross-asset placeholders that survived Phase A revert.
        for col in engineer.get_zero_fill_feature_names(symbol):
            if col not in feat.columns:
                feat[col] = 0.0
        # Regime
        feat = engineer.inject_regime_features(feat, hmm, symbol, d1)
        feat = feat.fillna(0.0)
        features = feat
    else:
        # Legacy thin path — lag + rolling stats on raw OHLCV + regime crosses.
        h4_with_regime = engineer.inject_regime_features(h4.copy(), hmm, symbol, d1)
        features = build_features(h4_with_regime)

    # --- Align labels onto features index ---
    labels = pd.Series(tb_labels, index=h4.index, name="tb_label")

    # --- Drop NaN warmup rows + NaN tail labels ---
    feature_mask = ~features.isna().any(axis=1)
    label_mask = labels.notna()
    keep = feature_mask & label_mask
    features = features[keep]
    labels = labels[keep]

    if len(features) == 0:
        raise RuntimeError(
            f"[{symbol}] no usable rows after warmup/label-alignment drop."
        )

    # Map {-1, 0, +1} → {0, 1, 2} for LightGBM multiclass
    y = labels.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(np.int8).to_numpy()

    return features, y, len(h4)


def _split_train_val(
    features: pd.DataFrame, y: np.ndarray, train_end_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Timestamp-based split: train = [<= train_end], val = [> train_end].

    ``DatetimeIndex <= Timestamp`` returns a numpy bool ndarray (not a
    pandas Series), so we index ``y`` with the mask directly — no
    ``.values`` accessor.
    """
    train_mask = np.asarray(features.index <= train_end_ts)
    X_train = features.loc[train_mask]
    X_val = features.loc[~train_mask]
    y_train = y[train_mask]
    y_val = y[~train_mask]
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Single GBM training call
# ---------------------------------------------------------------------------

def _train_one_gbm(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    params: dict,
    model_cfg: dict,
) -> tuple["lgb.Booster", float]:
    """Train one GBM. Returns (booster, val_loss=multi_logloss).

    Class imbalance handled via inverse-frequency sample weights —
    symmetric to the LSTM track's --class-weight flag (anchor 8).
    """
    import lightgbm as lgb

    # Compose params: model_config statics + tuning overrides.
    base = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbose": -1,
        "max_depth": model_cfg.get("max_depth", -1),
        "bagging_freq": model_cfg.get("bagging_freq", 5),
        "reg_alpha": model_cfg.get("reg_alpha", 0.0),
    }
    base.update(params)
    early_stopping_rounds = int(
        params.get("early_stopping_rounds",
                   model_cfg.get("early_stopping_rounds", 50))
    )
    n_estimators = int(base.pop("n_estimators", 500))
    base.pop("early_stopping_rounds", None)

    # Inverse-frequency sample weights so a rare class doesn't get
    # washed out. Counts clipped at 1 to avoid div-by-zero in the rare
    # case a class is entirely absent from a fold.
    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    cw = (counts.sum() / (3.0 * np.clip(counts, 1.0, None)))
    sample_weight = cw[y_train]

    train_set = lgb.Dataset(X_train.values, label=y_train, weight=sample_weight)
    val_set = lgb.Dataset(X_val.values, label=y_val, reference=train_set)
    booster = lgb.train(
        params=base,
        train_set=train_set,
        num_boost_round=n_estimators,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(period=0),  # silent
        ],
    )

    # Best iteration is recorded by early_stopping callback.
    val_pred = booster.predict(X_val.values)
    eps = 1e-15
    # multi_logloss = -mean(log(P(y_true)))
    val_loss = float(
        -np.mean(np.log(np.clip(val_pred[np.arange(len(y_val)), y_val], eps, 1.0)))
    )
    return booster, val_loss


# ---------------------------------------------------------------------------
# Artifact persistence (mirrors the LSTM training_dist.json convention)
# ---------------------------------------------------------------------------

def _save_artifact(
    booster: "lgb.Booster",
    feature_names: list[str],
    symbol: str,
    suffix: str,
    X_train: pd.DataFrame,
    val_loss: float,
    n_train_bars: int,
) -> Path:
    """Save .pkl model + .training_dist.json (drift-monitor compat)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = MODEL_DIR / f"gbm_{symbol}{suffix}.pkl"
    GBMPredictor.save(booster, feature_names, pkl_path)

    # training_dist mirrors the LSTM convention so the A-8 drift monitor
    # can score GBM features against their training distribution. Means
    # / stds saved per-feature; downstream PSI/KS computation reads these.
    dist_path = MODEL_DIR / f"gbm_{symbol}{suffix}.training_dist.json"
    dist = {
        "model_kind": "gbm",
        "feature_names": list(feature_names),
        "feature_means": {c: float(X_train[c].mean()) for c in feature_names},
        "feature_stds": {c: float(X_train[c].std(ddof=0)) for c in feature_names},
        "n_train_rows": int(len(X_train)),
        "n_h4_bars_pre_warmup": int(n_train_bars),
        "earliest_train_ts": str(X_train.index.min()),
        "latest_train_ts": str(X_train.index.max()),
        "val_loss": float(val_loss),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    dist_path.write_text(json.dumps(dist, indent=2), encoding="utf-8")
    return pkl_path


# ---------------------------------------------------------------------------
# Optuna study (per-symbol, mirrors LSTM Sprint 2 pattern)
# ---------------------------------------------------------------------------

def _run_optuna_study_for_symbol(
    symbol: str,
    args: argparse.Namespace,
    *,
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    n_train_bars: int,
    spaces: dict,
    model_cfg: dict,
) -> None:
    """Run Optuna study for one symbol. Saves both _default and _tuned.

    Mirrors scripts/train_deep_learning.py::_run_optuna_study_for_symbol:
      - Per-symbol SQLite study at data/models/gbm_{symbol}_optuna_study.db
      - Trial 0 = literature defaults verbatim on a fresh study only
        (computed from len(study.trials), NOT trial.number — resumed
        studies start above 0 and would silently skip the defaults branch).
      - try/except/else split on study.tell so success-tell and fail-tell
        are mutually exclusive (avoids double-tell RuntimeError).
      - Cleanup trial intermediates in finally so disk stays bounded.
    """
    import optuna
    from optuna.trial import TrialState

    # Phase 2B Q1/Q1.5 — feature-mode keeps studies separate so a wider
    # feature surface doesn't poison the narrower-mode sqlite (param spaces
    # are identical but val_loss landscape is different).
    feat_mode = getattr(args, "features", "thin") or "thin"
    if feat_mode == "rich":
        rich_suffix = "_rich"
    elif feat_mode == "parity":
        rich_suffix = "_parity"
    else:
        rich_suffix = ""

    storage_path = MODEL_DIR / f"gbm_{symbol}{rich_suffix}_optuna_study.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_uri = f"sqlite:///{storage_path.as_posix()}"

    study = optuna.create_study(
        direction="minimize",
        study_name=f"gbm_{symbol}_phase_a",
        storage=storage_uri,
        load_if_exists=True,
    )

    existing_finished = [t for t in study.trials if t.state.is_finished()]
    needs_defaults_pin = len(existing_finished) == 0
    if needs_defaults_pin:
        # Canonical Optuna pattern for "pin a known-good config as trial 0":
        # enqueue defaults BEFORE any study.ask() so they pop off the queue
        # on the first ask, but suggest_* still records the standard
        # search-space distribution. The naive alternative
        # (suggest_categorical(k, [v]) on trial 0, suggest_categorical(k,
        # search_list) on trial 1+) hits Optuna's
        # `CategoricalDistribution does not support dynamic value space`
        # error because the per-param distribution is locked after the
        # first call. enqueue_trial sidesteps this entirely.
        study.enqueue_trial(dict(spaces["defaults"]))
    else:
        logger.info(
            "[%s] Resuming Optuna study with %d existing trials — defaults "
            "already pinned in trial 0; new trials sample from search space.",
            symbol, len(existing_finished),
        )

    n_trials = int(args.tune_trials)
    logger.info(
        "[%s] starting GBM Optuna study (n_trials=%d, storage=%s)",
        symbol, n_trials, storage_path,
    )

    feature_names = list(X_train.columns)
    trial_numbers_this_run: list[int] = []
    try:
        for _ in range(n_trials):
            trial = study.ask()
            trial_numbers_this_run.append(trial.number)
            try:
                # Same suggest_* path for every trial — when the
                # enqueue_trial queue is non-empty (trial 0 of a fresh
                # study), suggest_* returns the enqueued value. After
                # that, the sampler takes over.
                params = {}
                for k, spec in spaces["search"].items():
                    if isinstance(spec, list):
                        params[k] = trial.suggest_categorical(k, spec)
                    elif isinstance(spec, dict):
                        low = spec["low"]
                        high = spec["high"]
                        if spec.get("log"):
                            params[k] = trial.suggest_float(
                                k, low, high, log=True,
                            )
                        else:
                            params[k] = trial.suggest_float(k, low, high)
                    else:
                        # ValueError (not SystemExit) so the per-trial
                        # except below catches it and keeps the loop
                        # alive. SystemExit inherits BaseException
                        # and would orphan a RUNNING trial.
                        raise ValueError(
                            f"[{symbol}] unsupported search spec for "
                            f"{k!r}: {spec!r}"
                        )

                booster, val_loss = _train_one_gbm(
                    X_train, y_train, X_val, y_val, params, model_cfg,
                )
                # Save trial intermediate so a crash mid-study doesn't
                # lose the work; cleaned up in the finally below.
                _save_artifact(
                    booster, feature_names, symbol,
                    suffix=f"{rich_suffix}_trial_{trial.number}",
                    X_train=X_train, val_loss=val_loss,
                    n_train_bars=n_train_bars,
                )
            except Exception as exc:
                logger.exception(
                    "[%s] trial %d failed: %s", symbol, trial.number, exc,
                )
                study.tell(trial, state=TrialState.FAIL)
            else:
                study.tell(trial, val_loss)
                logger.info(
                    "[%s] trial %d val_loss=%.6f params=%s",
                    symbol, trial.number, val_loss, params,
                )

        # Two finals at top level so they're queryable / restorable.
        default_params = dict(spaces["defaults"])
        try:
            tuned_params = dict(study.best_params)
        except ValueError:
            logger.warning(
                "[%s] no completed trials — falling back to defaults for tuned",
                symbol,
            )
            tuned_params = default_params

        for label, params in (("default", default_params),
                              ("tuned", tuned_params)):
            logger.info(
                "[%s] training %s final with params=%s", symbol, label, params,
            )
            booster, val_loss = _train_one_gbm(
                X_train, y_train, X_val, y_val, params, model_cfg,
            )
            pkl = _save_artifact(
                booster, feature_names, symbol,
                suffix=f"{rich_suffix}_{label}",
                X_train=X_train, val_loss=val_loss, n_train_bars=n_train_bars,
            )
            logger.info(
                "[%s] saved %s artifact: val_loss=%.6f model=%s",
                symbol, label, val_loss, pkl,
            )
    finally:
        # Cleanup trial intermediates — iterate the trials we created
        # this run (resumed studies start above 0, so range(n_trials)
        # would miss the actual file numbers).
        removed = 0
        for trial_num in trial_numbers_this_run:
            base = f"gbm_{symbol}_trial_{trial_num}"
            for p in (MODEL_DIR / f"{base}.pkl",
                      MODEL_DIR / f"{base}.training_dist.json"):
                existed = p.exists()
                try:
                    p.unlink(missing_ok=True)
                    if existed:
                        removed += 1
                except OSError as exc:
                    logger.warning(
                        "[%s] failed to remove %s: %s", symbol, p, exc,
                    )
        if removed:
            logger.info(
                "[%s] cleaned up %d trial-intermediate artifacts",
                symbol, removed,
            )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def main_async() -> None:
    args = parse_args()

    # Validate (kind, head) once per symbol up front — single head type
    # for GBM per spec anchor 7. Catches typos in --symbols early.
    for symbol in args.symbols:
        validate_kind_head(symbol, kind="gbm", head="classifier")

    # Load tuning spaces + model config once.
    import yaml
    spaces_full = yaml.safe_load(TUNING_SPACES_PATH.read_text(encoding="utf-8"))
    spaces = spaces_full.get("gbm", {})
    if "defaults" not in spaces or "search" not in spaces:
        raise SystemExit(
            f"tuning_spaces.yaml missing gbm.defaults or gbm.search "
            f"(path={TUNING_SPACES_PATH})"
        )
    model_cfg_full = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    model_cfg = model_cfg_full.get("gbm", {})

    # DB connect once; reused across all symbols.
    from src.data_pipeline.data_store import DataStore
    data_store = DataStore()
    await data_store.connect()
    logger.info("DataStore connected — DB-direct OHLCV reads (MT5-free)")

    feed = MT5DataFeed(connector=None, data_store=data_store)
    engineer = FeatureEngineer(data_store=data_store)

    # TVT timestamps — same val_end_ts_exclusive trick as Sprint 2 LSTM.
    train_start_ts = pd.Timestamp(args.train_start)
    train_end_ts = pd.Timestamp(args.train_end)
    val_end_ts_exclusive = pd.Timestamp(args.val_end) + pd.Timedelta(days=1)
    test_start_ts = pd.Timestamp(args.test_start)

    # Phase 2B Q1/Q1.5 feature-mode suffix marker. Prepended onto whatever
    # _default / _tuned / _trial_N suffix the caller is using.
    #   thin    → ""           → gbm_X_default.pkl
    #   rich    → "_rich"      → gbm_X_rich_default.pkl
    #   parity  → "_parity"    → gbm_X_parity_default.pkl
    if args.features == "rich":
        rich_suffix = "_rich"
    elif args.features == "parity":
        rich_suffix = "_parity"
    else:
        rich_suffix = ""
    if rich_suffix:
        logger.info(
            "[Phase 2B Q1/Q1.5] feature_mode=%r — artifacts will be saved "
            "with %r suffix (e.g. gbm_XAUUSD%s_default.pkl).",
            args.features, rich_suffix, rich_suffix,
        )

    try:
        for symbol in args.symbols:
            logger.info("=" * 60)
            logger.info("[%s] starting GBM training", symbol)

            features, y, n_bars = await _load_features_and_labels(
                symbol, args,
                feed=feed, engineer=engineer,
                train_start_ts=train_start_ts,
                val_end_ts_exclusive=val_end_ts_exclusive,
                test_start_ts=test_start_ts,
            )
            X_train, y_train, X_val, y_val = _split_train_val(
                features, y, train_end_ts,
            )
            logger.info(
                "[%s] train=%d rows val=%d rows (%d features)",
                symbol, len(X_train), len(X_val), X_train.shape[1],
            )

            if args.tune:
                _run_optuna_study_for_symbol(
                    symbol, args,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    n_train_bars=n_bars,
                    spaces=spaces, model_cfg=model_cfg,
                )
            else:
                booster, val_loss = _train_one_gbm(
                    X_train, y_train, X_val, y_val,
                    dict(spaces["defaults"]), model_cfg,
                )
                pkl = _save_artifact(
                    booster, list(X_train.columns), symbol,
                    suffix=f"{rich_suffix}_default",
                    X_train=X_train, val_loss=val_loss, n_train_bars=n_bars,
                )
                logger.info(
                    "[%s] saved default artifact: val_loss=%.6f model=%s",
                    symbol, val_loss, pkl,
                )
    finally:
        await data_store.close()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
