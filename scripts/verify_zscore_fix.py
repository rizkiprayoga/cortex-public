"""
verify_zscore_fix.py — A/B test the z-score normalization fix.

Runs backtest twice on USDCAD:
  A: baseline (current z-score normalizer)
  B: patched (raw clip to [-1, 1], no rolling z-score)

Prints a side-by-side comparison. No DB writes. No source changes.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import numpy as np

load_dotenv()
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from src.brain.signal_combiner import SignalCombiner
from src.broker.mt5_connector import MT5Connector
from src.data_pipeline.mt5_feed import MT5DataFeed
from scripts.backtest import run_backtest, compute_summary

SYMBOL = "USDCAD"
START = "2022-01-01"
END = "2026-04-01"
INITIAL_EQUITY = 10_000.0


def run_one(ohlcv, d1, w1, h1, label: str) -> dict:
    equity_curve, trades = run_backtest(
        SYMBOL, ohlcv, INITIAL_EQUITY,
        mode="full",
        d1_ohlcv=d1, w1_ohlcv=w1, h1_ohlcv=h1,
    )
    summary = compute_summary(equity_curve, trades)
    summary["_label"] = label
    summary["_n_trades"] = len(trades)
    return summary


def main():
    connector = MT5Connector()
    connector.connect()
    feed = MT5DataFeed(connector)

    start_dt = datetime.strptime(START, "%Y-%m-%d")
    end_dt = datetime.strptime(END, "%Y-%m-%d")

    print(f"Fetching {SYMBOL} data {START} -> {END}...")
    ohlcv = feed.get_historical(SYMBOL, "H4", start_date=start_dt)
    ohlcv = ohlcv[ohlcv.index <= end_dt]
    d1 = feed.get_historical(SYMBOL, "D1", bars=5000)
    d1 = d1[d1.index <= end_dt] if d1 is not None else None
    w1 = feed.get_historical(SYMBOL, "W1", bars=1000)
    w1 = w1[w1.index <= end_dt] if w1 is not None else None
    h1 = feed.get_historical(SYMBOL, "H1", start_date=start_dt)
    h1 = h1[h1.index <= end_dt] if h1 is not None else None
    print(f"  H4 bars: {len(ohlcv)}  H1 bars: {len(h1) if h1 is not None else 0}")

    # ---- Save baseline method reference -----------------------------------
    baseline_method = SignalCombiner._normalize_lstm_prediction

    # ---- Run A: BASELINE (rolling z-score) --------------------------------
    print(f"\n[A] Running baseline (rolling z-score)...")
    a = run_one(ohlcv, d1, w1, h1, "baseline_zscore")

    # ---- Run B: PATCHED (raw clip to [-1, 1]) -----------------------------
    def _raw_clip(self, symbol: str, raw_prediction: float) -> float:
        return float(np.clip(float(raw_prediction), -1.0, 1.0))

    SignalCombiner._normalize_lstm_prediction = _raw_clip
    print(f"[B] Running patched (raw clip, no z-score)...")
    b = run_one(ohlcv, d1, w1, h1, "patched_raw_clip")

    # Restore
    SignalCombiner._normalize_lstm_prediction = baseline_method

    # ---- Print comparison -------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  A/B — {SYMBOL}   {START} -> {END}")
    print("=" * 72)
    keys = sorted(set(a.keys()) | set(b.keys()))
    print(f"  {'metric':30} {'baseline':>18} {'patched':>18}")
    print(f"  {'-'*30} {'-'*18} {'-'*18}")
    for k in keys:
        if k.startswith("_") and k not in ("_n_trades",):
            continue
        va, vb = a.get(k), b.get(k)
        fa = f"{va:.4f}" if isinstance(va, float) else str(va)
        fb = f"{vb:.4f}" if isinstance(vb, float) else str(vb)
        print(f"  {k:30} {fa:>18} {fb:>18}")
    print("=" * 72)

    connector.disconnect()


if __name__ == "__main__":
    main()
