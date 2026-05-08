"""
analyze_xau_regime_persistence.py — E-7 v2 a-priori falsifier (spec §4.6)

Tests the structural-trend-prone argument for XAU empirically. Loads each
symbol's HMM model, runs predict_proba over the full 2021-2026 D1 history,
finds runs of consecutive Bull/Euphoria regimes, and computes median run
length per symbol. The test passes if XAU's median Bull regime duration
is at least 1.5× the median across the USD-axis FX pairs.

If this test fails, the structural-trend-prone argument is empirically
refuted on the 2021-2026 sample. Per spec §4.6: shelve v2 entirely
rather than weaken the threshold (would be reverse-curve-fitting).

Usage:
    python scripts/analyze_xau_regime_persistence.py
    python scripts/analyze_xau_regime_persistence.py --start 2021-01-01

Reads from DB only (DB-direct OHLCV via feed.get_historical_db_only).
No MT5 connection — safe to run while prod bot is active.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import List

import numpy as np
import pandas as pd

from src.brain.hmm_regime import HMMRegimeClassifier
from src.data_pipeline.data_store import DataStore
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.data_pipeline.mt5_feed import MT5DataFeed

# Per spec §4.6: XAU vs USD-axis FX (5 pairs). USDCAD included since HMM
# uses pure technical features — unaffected by today's CAD-block addition
# to cross_asset.py (that change only affects LSTM input pipeline).
XAU_SYMBOL = "XAUUSD"
FX_SYMBOLS = ["EURUSD", "USDJPY", "USDCAD", "GBPUSD", "AUDUSD"]

# regime_index ∈ {3, 4} = Bull / Euphoria (per src/strategy/trend_mode.py)
BULL_REGIMES = {3, 4}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def predict_regime_sequence(
    classifier: HMMRegimeClassifier,
    symbol: str,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """Run HMM predict_proba on the full matrix and return canonical regime
    indices per bar. Mirrors the per-bar logic from HMMRegimeClassifier.predict()
    but keeps every bar's prediction instead of just the last.

    Returns shape (n_bars,) with int regime indices in [0, 4].
    """
    if symbol not in classifier._models:
        raise RuntimeError(f"No HMM model loaded for {symbol}")

    model = classifier._models[symbol]
    label_map = classifier._state_label_maps.get(symbol, {})

    # Z-score normalize using stored stats (mirrors predict() lines 279-289).
    if symbol in classifier._norm_means and symbol in classifier._norm_stds:
        norm_matrix = (
            feature_matrix - classifier._norm_means[symbol]
        ) / classifier._norm_stds[symbol]
    else:
        logger.warning(
            "[%s] No saved normalization stats — falling back to batch z-score",
            symbol,
        )
        means = feature_matrix.mean(axis=0)
        stds = feature_matrix.std(axis=0)
        stds[stds == 0] = 1.0
        norm_matrix = (feature_matrix - means) / stds

    norm_matrix = np.nan_to_num(norm_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Forward-backward smoothed posterior over ALL bars
    state_probs_all = model.predict_proba(norm_matrix)  # (n_bars, n_states)
    raw_states = state_probs_all.argmax(axis=1)         # (n_bars,)

    # Map raw → canonical regime index per bar
    canonical = np.array(
        [label_map.get(int(rs), int(rs)) for rs in raw_states],
        dtype=np.int32,
    )
    return canonical


def find_bull_regime_runs(regime_seq: np.ndarray) -> List[int]:
    """Return durations (in bars) of every maximal run of consecutive
    Bull/Euphoria regimes in the sequence.

    Example:
        [2, 3, 3, 4, 2, 3, 3, 3, 1] → [3, 3]   (run of {3,3,4} = 3, then {3,3,3} = 3)
    """
    in_bull = np.isin(regime_seq, list(BULL_REGIMES))
    if not in_bull.any():
        return []

    durations: list[int] = []
    current_run = 0
    for is_bull in in_bull:
        if is_bull:
            current_run += 1
        else:
            if current_run > 0:
                durations.append(current_run)
            current_run = 0
    if current_run > 0:  # tail run
        durations.append(current_run)

    return durations


async def compute_durations_for_symbol(
    symbol: str,
    classifier: HMMRegimeClassifier,
    engineer: FeatureEngineer,
    feed: MT5DataFeed,
    start: str | None = None,
) -> List[int]:
    """Load D1 OHLCV + HMM model, compute regime sequence, return run durations."""
    if not classifier.load(symbol):
        raise RuntimeError(f"HMM model file not found for {symbol}")

    ohlcv = await feed.get_historical_db_only(symbol, "D1", bars=99999)
    if start:
        cutoff = pd.Timestamp(start)
        ohlcv = ohlcv[ohlcv.index >= cutoff]

    features_df = engineer.transform(ohlcv)
    if features_df.empty:
        raise RuntimeError(f"No features computed for {symbol}")

    # Use the HMM's saved feature_manifest to pick exactly the columns the
    # model was trained on. FeatureEngineer may have evolved (new columns
    # added) since training; normalization stats are sized to the manifest.
    manifest = classifier._feature_manifests.get(symbol, [])
    if manifest:
        # Some manifests may include columns no longer produced; missing → fill 0.
        cols_present = [c for c in manifest if c in features_df.columns]
        cols_missing = [c for c in manifest if c not in features_df.columns]
        if cols_missing:
            logger.warning(
                "[%s] %d manifest columns missing from current features: %s",
                symbol, len(cols_missing), cols_missing[:5],
            )
        sorted_df = features_df[cols_present]
        # If any manifest columns missing, pad with zeros to match training shape
        for c in cols_missing:
            sorted_df = sorted_df.assign(**{c: 0.0})
        sorted_df = sorted_df[manifest]  # final order matches training
    else:
        sorted_df = features_df[sorted(features_df.columns)]

    raw_matrix = np.nan_to_num(
        sorted_df.values.astype(np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )

    regime_seq = predict_regime_sequence(classifier, symbol, raw_matrix)
    durations = find_bull_regime_runs(regime_seq)

    bull_bars = int(np.isin(regime_seq, list(BULL_REGIMES)).sum())
    total_bars = len(regime_seq)
    logger.info(
        "[%s] %d D1 bars (%s → %s); %d in Bull/Euphoria (%.1f%%); "
        "%d Bull runs, lengths: median=%d, max=%d, mean=%.1f",
        symbol, total_bars,
        sorted_df.index[0].date(), sorted_df.index[-1].date(),
        bull_bars, 100 * bull_bars / max(total_bars, 1),
        len(durations),
        int(np.median(durations)) if durations else 0,
        int(np.max(durations)) if durations else 0,
        float(np.mean(durations)) if durations else 0.0,
    )
    return durations


async def amain(args: argparse.Namespace) -> int:
    data_store = DataStore()
    await data_store.connect()
    feed = MT5DataFeed(connector=None, data_store=data_store)
    engineer = FeatureEngineer()

    classifier = HMMRegimeClassifier(data_store=data_store)

    print()
    print("=" * 80)
    print(" E-7 v2 a-priori falsifier — XAU regime persistence vs FX (spec §4.6)")
    print("=" * 80)
    print()

    # XAU
    try:
        xau_durations = await compute_durations_for_symbol(
            XAU_SYMBOL, classifier, engineer, feed, start=args.start,
        )
    except Exception as exc:
        logger.error("XAU analysis failed: %s", exc)
        return 1

    # FX (pooled)
    fx_durations: list[int] = []
    fx_per_symbol: dict[str, list[int]] = {}
    for sym in FX_SYMBOLS:
        try:
            d = await compute_durations_for_symbol(
                sym, classifier, engineer, feed, start=args.start,
            )
        except Exception as exc:
            logger.warning("[%s] skipped: %s", sym, exc)
            continue
        fx_per_symbol[sym] = d
        fx_durations.extend(d)

    await data_store.close()

    # Verdict
    print()
    print("=" * 80)
    print(" VERDICT")
    print("=" * 80)

    if not xau_durations:
        print(" XAU has no Bull regime runs in the window — test cannot evaluate.")
        return 1
    if not fx_durations:
        print(" FX pool has no Bull regime runs in the window — test cannot evaluate.")
        return 1

    xau_median = float(np.median(xau_durations))
    fx_median = float(np.median(fx_durations))
    ratio = xau_median / fx_median if fx_median > 0 else float("inf")
    threshold = 1.5

    print(f"  XAU Bull regime runs:          {len(xau_durations):4d}")
    print(f"  XAU median duration (D1 bars): {xau_median:6.1f}")
    print(f"  XAU max duration:              {int(np.max(xau_durations)):6d}")
    print()
    print(f"  FX (pooled across {len(fx_per_symbol)} pairs) Bull runs: {len(fx_durations):4d}")
    print(f"  FX median duration (D1 bars):  {fx_median:6.1f}")
    print(f"  FX max duration:               {int(np.max(fx_durations)):6d}")
    print()
    print(f"  Ratio XAU/FX:                  {ratio:6.2f}x")
    print(f"  Threshold (spec sec 4.6):      >= {threshold:6.2f}x")
    print()

    print(" Per-symbol FX breakdown:")
    for sym, d in fx_per_symbol.items():
        if d:
            med = int(np.median(d))
            mx = int(np.max(d))
            print(f"   {sym}: {len(d):3d} runs, median={med:3d}, max={mx:4d}")
        else:
            print(f"   {sym}: 0 runs")
    print()

    if ratio >= threshold:
        print(f" [PASS] XAU regimes are {ratio:.2f}x longer than FX, >= {threshold:.2f} threshold.")
        print("    Structural-trend-prone argument is empirically supported.")
        print("    v2 may proceed to live wiring (v2.5).")
        return 0
    else:
        print(f" [FAIL] XAU regimes are only {ratio:.2f}x longer, BELOW the {threshold:.2f} threshold.")
        print("    Structural-trend-prone argument is empirically refuted on 2021-2026 sample.")
        print("    Per spec sec 4.6: SHELVE v2 entirely. Do NOT weaken the threshold.")
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", type=str, default="2021-01-01",
        help="Start date (YYYY-MM-DD). Default: 2021-01-01 (canonical baseline window)",
    )
    args = parser.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
