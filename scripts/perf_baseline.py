"""
perf_baseline.py — capture per-bar pipeline timing before Phase 2 expansion.

Phase 1G (P-2 lite). Doubles symbol count + expanded features in Phase 2
can 2-3× the per-bar compute cost. This script captures "before"
measurements so Phase 2 model training can detect regressions early.

Pipeline phases timed (matches what main.py runs every H4 close):

    1. FeatureEngineer.transform — 6 sub-phases (price/trend/momentum/
       volatility/volume/statistical features).
    2. LSTM input prep — StandardScaler.transform + PCA.transform on
       the feature matrix.
    3. DataStore.save_feature_vector — JSONB write per bar.

Usage:

    python -m scripts.perf_baseline                        # all 5 live symbols
    python -m scripts.perf_baseline --symbols XAUUSD       # one symbol
    python -m scripts.perf_baseline --iterations 100       # more samples
    python -m scripts.perf_baseline --output docs/perf_baseline_pre_expansion.md

Output: a Markdown table written to the given path (default
``docs/perf_baseline_pre_expansion.md``) plus a stdout summary.

Caveats:
    - Runs against the same Postgres + same in-memory cache the live bot
      uses; nothing is mocked. The DB write phase will idempotently
      upsert rows that already exist (PK conflict → DO UPDATE), so
      repeated runs are safe.
    - First-iteration outliers are dropped (warmup=5 by default) so
      cold-cache effects don't skew the median.
    - Measures wall-clock via ``time.perf_counter()``. Don't run other
      heavy workloads on the same machine during the baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.data_store import DataStore  # noqa: E402
from src.data_pipeline.feature_engineering import FeatureEngineer  # noqa: E402

logger = logging.getLogger(__name__)


_LIVE_SYMBOLS = ["XAUUSD", "EURUSD", "USDJPY", "USDCAD", "ETHUSD"]
_TIMEFRAME = "H4"
_BARS_TO_LOAD = 500           # rolling window the live bot keeps in memory
_DEFAULT_ITERATIONS = 50
_DEFAULT_WARMUP = 5


# ---------------------------------------------------------------------------
# Phase timers — instrumented copy of FeatureEngineer.transform's flow
# ---------------------------------------------------------------------------


def _time_transform_phases(
    fe: FeatureEngineer, ohlcv: pd.DataFrame
) -> dict[str, float]:
    """
    Run the same 6 phases as FeatureEngineer.transform and return per-phase
    elapsed seconds. Mirrors the logic in feature_engineering.py:130-141 —
    if that gets refactored, update here.
    """
    df = ohlcv.copy()
    if "volume" in df.columns and "tick_volume" not in df.columns:
        df = df.rename(columns={"volume": "tick_volume"})

    phases = {}

    t0 = time.perf_counter()
    df = fe._compute_price_features(df)
    phases["price"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = fe._compute_trend_features(df)
    phases["trend"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = fe._compute_momentum_features(df)
    phases["momentum"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = fe._compute_volatility_features(df)
    phases["volatility"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = fe._compute_volume_features(df)
    phases["volume"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = fe._compute_statistical_features(df)
    phases["statistical"] = time.perf_counter() - t0

    return phases


def _time_lstm_prep(
    feature_matrix: np.ndarray, scaler, pca,
) -> dict[str, float]:
    """Time scaler.transform + pca.transform (LSTM input pipeline)."""
    phases = {}

    t0 = time.perf_counter()
    scaled = scaler.transform(feature_matrix)
    phases["scaler"] = time.perf_counter() - t0

    if pca is not None:
        t0 = time.perf_counter()
        pca.transform(scaled)
        phases["pca"] = time.perf_counter() - t0
    else:
        phases["pca"] = 0.0

    return phases


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _summarize(samples: list[float]) -> dict[str, float]:
    """Return {p50, p95, max, mean} in milliseconds."""
    if not samples:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0}
    sorted_s = sorted(samples)
    p50 = sorted_s[len(sorted_s) // 2]
    p95 = sorted_s[int(len(sorted_s) * 0.95)] if len(sorted_s) >= 20 else sorted_s[-1]
    return {
        "p50_ms":  p50 * 1000,
        "p95_ms":  p95 * 1000,
        "max_ms":  max(sorted_s) * 1000,
        "mean_ms": statistics.mean(sorted_s) * 1000,
    }


# ---------------------------------------------------------------------------
# Per-symbol benchmark
# ---------------------------------------------------------------------------


async def _benchmark_symbol(
    store: DataStore,
    symbol: str,
    iterations: int,
    warmup: int,
) -> dict:
    """Run the full pipeline `iterations + warmup` times, return stats."""
    fe = FeatureEngineer(data_store=store)

    # Load OHLCV — most recent 500 H4 bars, same window the live bot uses.
    end = datetime.utcnow()
    start = end - timedelta(days=500 * 4 // 24 * 4)   # ~333 days for 500 H4
    ohlcv = await store.get_ohlcv_range(symbol, _TIMEFRAME, start, end)
    if ohlcv is None or len(ohlcv) < 100:
        logger.warning("skipping %s — only %d bars available", symbol, len(ohlcv) if ohlcv is not None else 0)
        return {"symbol": symbol, "skipped": True, "reason": "insufficient OHLCV"}

    ohlcv = ohlcv.tail(_BARS_TO_LOAD).copy()

    # Try loading scaler + PCA. Skipped symbols (e.g. ETH if model missing)
    # are reported as such; transform timing still runs.
    try:
        import joblib
        scaler_path = _PROJECT_ROOT / "data" / "models" / f"lstm_scaler_{symbol}.pkl"
        pca_path    = _PROJECT_ROOT / "data" / "models" / f"lstm_{symbol}.pca.pkl"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        pca    = joblib.load(pca_path)    if pca_path.exists() else None
    except Exception as exc:
        logger.warning("scaler/pca load failed for %s: %s", symbol, exc)
        scaler = pca = None

    # Build a sample feature matrix the LSTM-prep timer can reuse (computed once,
    # same shape every iteration to isolate scaler+pca cost).
    sample_features = fe.transform(ohlcv)
    if sample_features.empty:
        logger.warning("skipping %s — transform returned empty", symbol)
        return {"symbol": symbol, "skipped": True, "reason": "empty transform output"}
    feature_matrix = sample_features.tail(60).to_numpy(dtype=np.float64, na_value=0.0)

    # Pad feature matrix to match the scaler's trained dim. The live bot
    # injects ~36-49 fundamental features (macro/COT/yields/etc.) before
    # LSTM inference; the scaler was trained on the full padded vector.
    # Zeros for the fundamentals are representative — the scaler+pca cost
    # is shape-determined, not value-determined, so timing is honest.
    if scaler is not None:
        expected = getattr(scaler, "n_features_in_", feature_matrix.shape[1])
        if feature_matrix.shape[1] < expected:
            pad = np.zeros((feature_matrix.shape[0], expected - feature_matrix.shape[1]), dtype=np.float64)
            feature_matrix = np.concatenate([feature_matrix, pad], axis=1)
        elif feature_matrix.shape[1] > expected:
            feature_matrix = feature_matrix[:, :expected]

    samples_transform: dict[str, list[float]] = {
        "price": [], "trend": [], "momentum": [],
        "volatility": [], "volume": [], "statistical": [],
        "total_transform": [],
    }
    samples_lstm: dict[str, list[float]] = {"scaler": [], "pca": []}
    samples_persist: list[float] = []

    for i in range(iterations + warmup):
        # Phase A — transform
        t0 = time.perf_counter()
        phases = _time_transform_phases(fe, ohlcv)
        total = time.perf_counter() - t0
        if i >= warmup:
            for k, v in phases.items():
                samples_transform[k].append(v)
            samples_transform["total_transform"].append(total)

        # Phase B — LSTM prep
        if scaler is not None:
            lstm_phases = _time_lstm_prep(feature_matrix, scaler, pca)
            if i >= warmup:
                for k, v in lstm_phases.items():
                    samples_lstm[k].append(v)

        # Phase C — DB write (just one bar — the latest — to mirror the
        # live H4-tick path. PK conflict on re-runs becomes an UPDATE.)
        latest_ts = sample_features.index[-1].strftime("%Y-%m-%dT%H:%M:%S")
        latest_features = {
            name: (None if pd.isna(val) else float(val))
            for name, val in sample_features.iloc[-1].items()
        }
        t0 = time.perf_counter()
        await store.save_feature_vector(
            symbol=symbol, timeframe=_TIMEFRAME,
            bar_timestamp=latest_ts, feature_dict=latest_features,
        )
        if i >= warmup:
            samples_persist.append(time.perf_counter() - t0)

    return {
        "symbol":        symbol,
        "skipped":       False,
        "n_bars_loaded": len(ohlcv),
        "n_features":    feature_matrix.shape[1],
        "transform":     {k: _summarize(v) for k, v in samples_transform.items()},
        "lstm_prep":     {k: _summarize(v) for k, v in samples_lstm.items()},
        "persist":       _summarize(samples_persist),
        "iterations":    iterations,
        "warmup":        warmup,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format_md(results: list[dict], iterations: int) -> str:
    """Render the per-symbol stats as a Markdown report."""
    lines: list[str] = []
    lines.append("# Per-bar pipeline performance baseline")
    lines.append("")
    lines.append(f"**Generated:** {datetime.utcnow().isoformat(timespec='seconds')}Z  ")
    lines.append(f"**Iterations:** {iterations} per symbol (after 5-iter warmup)  ")
    lines.append("**Workload:** 500 H4 bars per symbol, real models (scaler + PCA), real DB writes  ")
    lines.append("**Symbols:** 5 live pairs (XAU, EUR, JPY, CAD, ETH) — pre-Phase-2 baseline")
    lines.append("")
    lines.append("Use this file as the point-of-comparison for Phase 2 perf runs:")
    lines.append("`git show 722ec5e:docs/perf_baseline_pre_expansion.md` (or whichever commit)")
    lines.append("...will give you the numbers below for direct diffing.")
    lines.append("")

    lines.append("## Transform — 6 sub-phases (median per call)")
    lines.append("")
    lines.append("| Symbol | bars | feats | price | trend | momentum | volatility | volume | statistical | TOTAL transform |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        if r.get("skipped"):
            lines.append(f"| {r['symbol']} | — | — | — | — | — | — | — | — | _{r.get('reason','skipped')}_ |")
            continue
        t = r["transform"]
        lines.append(
            f"| {r['symbol']} | {r['n_bars_loaded']} | {r['n_features']} | "
            f"{t['price']['p50_ms']:.2f} | {t['trend']['p50_ms']:.2f} | "
            f"{t['momentum']['p50_ms']:.2f} | {t['volatility']['p50_ms']:.2f} | "
            f"{t['volume']['p50_ms']:.2f} | {t['statistical']['p50_ms']:.2f} | "
            f"**{t['total_transform']['p50_ms']:.2f} ms** |"
        )
    lines.append("")
    lines.append("All values in milliseconds (p50). p95 and max in the JSON dump (stdout).")
    lines.append("")

    lines.append("## LSTM input prep (per inference)")
    lines.append("")
    lines.append("| Symbol | scaler.transform p50 | pca.transform p50 |")
    lines.append("|---|---:|---:|")
    for r in results:
        if r.get("skipped"):
            continue
        lp = r["lstm_prep"]
        if lp["scaler"]["p50_ms"] == 0 and lp["pca"]["p50_ms"] == 0:
            lines.append(f"| {r['symbol']} | _model artifacts not loaded_ | — |")
        else:
            lines.append(
                f"| {r['symbol']} | {lp['scaler']['p50_ms']:.3f} ms | "
                f"{lp['pca']['p50_ms']:.3f} ms |"
            )
    lines.append("")

    lines.append("## DB write — `save_feature_vector` (1 bar, JSONB upsert)")
    lines.append("")
    lines.append("| Symbol | p50 | p95 | max |")
    lines.append("|---|---:|---:|---:|")
    for r in results:
        if r.get("skipped"):
            continue
        p = r["persist"]
        lines.append(
            f"| {r['symbol']} | {p['p50_ms']:.2f} ms | {p['p95_ms']:.2f} ms | "
            f"{p['max_ms']:.2f} ms |"
        )
    lines.append("")

    # Aggregate budget — what does ONE H4 close cost across all 5 symbols?
    total_transform_p50 = sum(
        r["transform"]["total_transform"]["p50_ms"]
        for r in results if not r.get("skipped")
    )
    total_lstm_p50 = sum(
        r["lstm_prep"]["scaler"]["p50_ms"] + r["lstm_prep"]["pca"]["p50_ms"]
        for r in results if not r.get("skipped")
    )
    total_persist_p50 = sum(
        r["persist"]["p50_ms"] for r in results if not r.get("skipped")
    )

    lines.append("## Aggregate per-tick budget (5 live symbols, p50)")
    lines.append("")
    lines.append(f"- Transform total:   **{total_transform_p50:.1f} ms** (sum across 5 symbols)")
    lines.append(f"- LSTM input prep:   **{total_lstm_p50:.1f} ms**")
    lines.append(f"- DB write (1 bar/sym): **{total_persist_p50:.1f} ms**")
    lines.append(f"- **Sum:** **{total_transform_p50 + total_lstm_p50 + total_persist_p50:.1f} ms** per H4 tick")
    lines.append("")
    lines.append("Phase 2 doubles symbol count to 11 (5 live + 6 expansion). Naive scaling")
    lines.append("would put the same workload at ~2.2× this number. Watch for regressions")
    lines.append("vs that linear-scaling expectation — anything materially higher signals")
    lines.append("a non-linear hotspot worth optimizing.")
    lines.append("")

    lines.append("## Known hotspots (from forex_expansion_plan §1G)")
    lines.append("")
    lines.append("- Feature engineering ~200ms × 5 symbols/bar (this baseline confirms or refutes)")
    lines.append("- PCA not vectorized across symbols — runs once per symbol")
    lines.append("- Equity-curve writer synchronous on every tick (not measured here)")
    lines.append("")
    lines.append("Phase 2 should re-run this script on the same machine + DB after model")
    lines.append("retraining and feature changes. Compare per-phase p50/p95 deltas against")
    lines.append("the baseline above.")
    lines.append("")

    return "\n".join(lines)


def _format_table(results: list[dict]) -> str:
    """Compact text table for stdout."""
    out = []
    for r in results:
        if r.get("skipped"):
            out.append(f"{r['symbol']:8s}  SKIPPED ({r.get('reason')})")
            continue
        t = r["transform"]["total_transform"]
        lp = r["lstm_prep"]
        p = r["persist"]
        out.append(
            f"{r['symbol']:8s}  transform p50={t['p50_ms']:7.2f}ms  p95={t['p95_ms']:7.2f}ms  "
            f"|  scaler={lp['scaler']['p50_ms']:5.3f}ms  pca={lp['pca']['p50_ms']:5.3f}ms  "
            f"|  persist p50={p['p50_ms']:5.2f}ms"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    if not os.environ.get("POSTGRES_DSN"):
        print("ERROR: POSTGRES_DSN not set", file=sys.stderr)
        return 1

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    iterations = args.iterations
    warmup = args.warmup

    print(f"Running perf baseline: {len(symbols)} symbols × {iterations} iter × {warmup} warmup")
    print()

    store = DataStore()
    await store.connect()

    results = []
    try:
        for sym in symbols:
            print(f"  benchmarking {sym}...", end="", flush=True)
            t0 = time.perf_counter()
            r = await _benchmark_symbol(store, sym, iterations, warmup)
            elapsed = time.perf_counter() - t0
            print(f"  ({elapsed:.1f}s)")
            results.append(r)
    finally:
        await store.close()

    print()
    print(_format_table(results))
    print()

    md = _format_md(results, iterations)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Markdown report -> {out_path}")
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Capture per-bar pipeline timing for Phase 1 → Phase 2 regression checks.",
    )
    parser.add_argument(
        "--symbols", default=",".join(_LIVE_SYMBOLS),
        help="Comma-separated symbols (default: 5 live pairs)",
    )
    parser.add_argument(
        "--iterations", type=int, default=_DEFAULT_ITERATIONS,
        help=f"Iterations per symbol after warmup (default: {_DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--warmup", type=int, default=_DEFAULT_WARMUP,
        help=f"Warmup iterations (discarded; default: {_DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--output", default="docs/perf_baseline_pre_expansion.md",
        help="Output Markdown path (default: docs/perf_baseline_pre_expansion.md)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
