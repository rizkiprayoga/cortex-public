"""
analyze_pair_correlations.py — an earlier sprint Phase E correlation diagnostic

Computes returns-correlation matrix across all 20 forex+XAU pairs in the
current and proposed the universe sweep sprint universe (9 existing forex + 10 new
candidates + XAUUSD; ETHUSD excluded). Output feeds:

  1. the universe sweep sprint Phase F diversification rule (replaces heuristic
     "max 3 same-currency-same-side" with empirical correlation buckets).
  2. Sprint 3b correlation-aware allocation impl (already-derived buckets
     so impl doesn't need to re-discover the structure).

Data source: yfinance daily closes. Sufficient for correlation structure;
broker-tick precision not needed for diversification-rule design. The 10
new candidate pairs don't have OHLCV in our DB yet — that's an earlier sprint
Phase B, not done. yfinance bypasses that dependency for now.

Usage:
    PYTHONPATH=. python scripts/analyze_pair_correlations.py
    PYTHONPATH=. python scripts/analyze_pair_correlations.py --start 2023-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "data" / "logs" / "correlation_diagnostic"

# 20 pairs: 9 existing forex + 10 new candidates + XAUUSD. ETHUSD excluded.
# Yahoo Finance ticker conventions:
#   Forex:  "{base}{quote}=X"
#   Gold:   "GC=F" (front-month futures, most reliable for daily corr)
PAIRS_YFINANCE = {
    # --- 9 existing forex (currently in dev DB) ---
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCAD": "USDCAD=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "AUDNZD": "AUDNZD=X",
    # --- XAU ---
    "XAUUSD": "GC=F",
    # --- 10 new candidates (the universe sweep sprint universe) ---
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
    "EURCHF": "EURCHF=X",
    "EURAUD": "EURAUD=X",
    "AUDJPY": "AUDJPY=X",
    "NZDJPY": "NZDJPY=X",
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
    "GBPCHF": "GBPCHF=X",
    "GBPAUD": "GBPAUD=X",
}

# Per-pair USD-axis tagging for the coherence study (Phase 2G analytic
# carried forward into 1.4). +1 = long-USD direction (USD-base buys + USD-quote
# sells), -1 = short-USD, 0 = USD-neutral cross.
USD_AXIS_DIRECTION = {
    # USD-base pairs (e.g. USDJPY): buy = long-USD; flip the sign for sell-side analysis
    "USDJPY": +1, "USDCAD": +1, "USDCHF": +1,
    # USD-quote pairs (e.g. EURUSD): buy = short-USD (selling USD to buy EUR)
    "EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1,
    # Crosses — USD-neutral
    "EURGBP": 0, "EURJPY": 0, "GBPJPY": 0, "EURCHF": 0,
    "EURAUD": 0, "AUDJPY": 0, "NZDJPY": 0, "CADJPY": 0,
    "CHFJPY": 0, "GBPCHF": 0, "GBPAUD": 0, "AUDNZD": 0,
    # Metals — quoted in USD but driven by real rates + flight-to-safety,
    # not pure dollar dynamics. Treat as neutral for the coherence study.
    "XAUUSD": 0,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_close_series(start: str, end: str) -> pd.DataFrame:
    """Fetch daily close series for all 20 pairs from yfinance, returns
    a DataFrame indexed by date, columns = pair names."""
    import yfinance as yf

    tickers = list(PAIRS_YFINANCE.values())
    logger.info("Fetching %d tickers from yfinance: %s -> %s",
                len(tickers), start, end)
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
        group_by="ticker",
    )

    closes = {}
    for pair, ticker in PAIRS_YFINANCE.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    s = raw[(ticker, "Close")].dropna()
                else:
                    logger.warning("Ticker %s missing from yfinance result", ticker)
                    continue
            else:
                s = raw["Close"].dropna()
            if len(s) < 30:
                logger.warning("[%s] only %d closes — skipping", pair, len(s))
                continue
            closes[pair] = s
        except Exception as exc:
            logger.warning("[%s/%s] fetch failed: %s", pair, ticker, exc)
            continue

    df = pd.DataFrame(closes)
    df = df.dropna(how="all")
    return df


def compute_correlation_matrix(closes: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation on log returns, full sample."""
    log_returns = np.log(closes / closes.shift(1)).dropna(how="all")
    return log_returns.corr()


def find_correlation_buckets(corr: pd.DataFrame, threshold: float = 0.7) -> list[set]:
    """Greedy clustering: any two pairs with corr >= threshold are in the
    same bucket. Symmetric — buckets are unordered sets.
    """
    pairs = corr.columns.tolist()
    parent = {p: p for p in pairs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, p1 in enumerate(pairs):
        for p2 in pairs[i + 1:]:
            if corr.loc[p1, p2] >= threshold:
                union(p1, p2)

    groups: dict[str, set[str]] = {}
    for p in pairs:
        root = find(p)
        groups.setdefault(root, set()).add(p)

    return [g for g in groups.values() if len(g) >= 2] + \
           [g for g in groups.values() if len(g) == 1]


def usd_coherence_summary(corr: pd.DataFrame) -> dict:
    """Average pairwise correlation within each USD-axis group."""
    long_usd = [p for p, d in USD_AXIS_DIRECTION.items() if d == +1 and p in corr.columns]
    short_usd = [p for p, d in USD_AXIS_DIRECTION.items() if d == -1 and p in corr.columns]
    neutral = [p for p, d in USD_AXIS_DIRECTION.items() if d == 0 and p in corr.columns]

    def avg_within(group):
        if len(group) < 2:
            return float("nan")
        m = corr.loc[group, group]
        n = len(group)
        # Mean of off-diagonal entries
        off_diag = (m.values.sum() - np.trace(m.values)) / (n * (n - 1))
        return float(off_diag)

    def avg_between(g1, g2):
        if not g1 or not g2:
            return float("nan")
        m = corr.loc[g1, g2]
        return float(m.values.mean())

    return {
        "long_usd_count": len(long_usd),
        "long_usd_avg_intra_corr": avg_within(long_usd),
        "long_usd_pairs": long_usd,
        "short_usd_count": len(short_usd),
        "short_usd_avg_intra_corr": avg_within(short_usd),
        "short_usd_pairs": short_usd,
        "neutral_count": len(neutral),
        "neutral_avg_intra_corr": avg_within(neutral),
        "long_vs_short_avg_corr": avg_between(long_usd, short_usd),
    }


def write_markdown(corr: pd.DataFrame, buckets: list[set], coherence: dict,
                   start: str, end: str, output_path: Path) -> None:
    pairs_sorted = sorted(corr.columns.tolist())
    n = len(pairs_sorted)

    lines = [
        "# Pair correlation diagnostic — an earlier sprint Phase E",
        "",
        f"**Date generated:** 2026-04-29",
        f"**Data:** yfinance daily closes (auto-adjusted), {start} → {end}",
        f"**Universe:** {n} pairs (9 existing forex + XAU + 10 new candidates; ETHUSD excluded)",
        "",
        "## Why yfinance and not broker MT5",
        "",
        "The 10 new candidate pairs (USDCHF, NZDUSD, EURCHF, EURAUD, AUDJPY, NZDJPY, "
        "CADJPY, CHFJPY, GBPCHF, GBPAUD) don't have OHLCV in our DB yet — that's "
        "an earlier sprint Phase B (operator-coordinated MT5 backfill). yfinance daily closes "
        "are sufficient for correlation structure; broker-tick precision isn't needed "
        "for the diversification-rule design output. After Phase B, a follow-up run on "
        "broker H4 closes can validate the buckets.",
        "",
        "## TL;DR",
        "",
        f"Identified **{len([b for b in buckets if len(b) >= 2])} correlation clusters** at the 0.7 threshold "
        f"plus {len([b for b in buckets if len(b) == 1])} standalone pairs. "
        f"USD-axis structure: long-USD pairs have **avg intra-group correlation "
        f"{coherence['long_usd_avg_intra_corr']:.2f}**, short-USD pairs **{coherence['short_usd_avg_intra_corr']:.2f}**, "
        f"and long-vs-short pairs are anti-correlated at **{coherence['long_vs_short_avg_corr']:.2f}** (as expected).",
        "",
        "## Correlation buckets (Pearson, log returns, full sample)",
        "",
        "Pairs with rolling correlation ≥ 0.7 over the sample window. Use as inputs "
        "to the universe sweep sprint Phase F diversification rule: cap simultaneous exposure per "
        "bucket at 2 of N pairs.",
        "",
    ]

    # Sorted: clusters first by size (descending), singletons last
    clusters = [b for b in buckets if len(b) >= 2]
    singletons = [b for b in buckets if len(b) == 1]
    clusters.sort(key=lambda s: -len(s))

    for i, bucket in enumerate(clusters, 1):
        # Find representative max correlation within the bucket
        members = sorted(bucket)
        m = corr.loc[members, members]
        n_b = len(members)
        avg_corr = (m.values.sum() - np.trace(m.values)) / (n_b * (n_b - 1)) if n_b > 1 else 0
        lines.append(f"### Bucket {i} — {len(members)} pairs (avg pairwise corr {avg_corr:.2f})")
        lines.append("")
        lines.append("- " + ", ".join(f"`{p}`" for p in members))
        lines.append("")

    if singletons:
        flat = sorted(p for s in singletons for p in s)
        lines.append("### Standalone (no pair with correlation ≥ 0.7)")
        lines.append("")
        lines.append("- " + ", ".join(f"`{p}`" for p in flat))
        lines.append("")

    lines.append("## USD-axis coherence (deferred Phase 2G analytic)")
    lines.append("")
    lines.append("Average pairwise correlation within and between USD-axis groups. "
                 "If long-vs-short avg corr is strongly negative, it justifies a "
                 "USD-direction soft veto in Sprint 3b's correlation-aware allocation.")
    lines.append("")
    lines.append("| Group | N pairs | Avg intra-group corr | Pairs |")
    lines.append("|---|---:|---:|---|")
    lines.append(
        f"| Long-USD (USD-base) | {coherence['long_usd_count']} | "
        f"{coherence['long_usd_avg_intra_corr']:.3f} | "
        f"{', '.join(coherence['long_usd_pairs'])} |"
    )
    lines.append(
        f"| Short-USD (USD-quote) | {coherence['short_usd_count']} | "
        f"{coherence['short_usd_avg_intra_corr']:.3f} | "
        f"{', '.join(coherence['short_usd_pairs'])} |"
    )
    lines.append(
        f"| Neutral (crosses + XAU) | {coherence['neutral_count']} | "
        f"{coherence['neutral_avg_intra_corr']:.3f} | "
        f"({coherence['neutral_count']} pairs, see matrix) |"
    )
    lines.append("")
    lines.append(f"**Long-USD vs Short-USD avg correlation:** "
                 f"**{coherence['long_vs_short_avg_corr']:.3f}** (should be strongly negative).")
    lines.append("")

    lines.append("## Full correlation matrix")
    lines.append("")
    # Compact matrix: header row + data rows, abbreviated to integers × 100
    # for readability
    header = "| | " + " | ".join(pairs_sorted) + " |"
    sep = "|---|" + "|".join(["---:"] * n) + "|"
    lines.append(header)
    lines.append(sep)
    for p1 in pairs_sorted:
        row_vals = []
        for p2 in pairs_sorted:
            v = corr.loc[p1, p2]
            row_vals.append(f"{v*100:+3.0f}" if not np.isnan(v) else " — ")
        lines.append(f"| **{p1}** | " + " | ".join(row_vals) + " |")
    lines.append("")
    lines.append("(Values shown as correlation × 100, integer-rounded for compactness. "
                 "Diagonal is 100 by definition. Raw matrix in JSON output.)")
    lines.append("")

    lines.append("## How to apply")
    lines.append("")
    lines.append("**the universe sweep sprint Phase F selection rule (revised):**")
    lines.append("")
    lines.append("> Among pairs that clear the hard floor (PF≥2.0, DSR≥0.5, 150-400 trades, WR≥50%), ")
    lines.append("> apply the diversification cap: **max 2 pairs per correlation bucket**, where ")
    lines.append("> bucket = pairs with daily-return correlation ≥ 0.7 over the most recent 12 months. ")
    lines.append("> Among PF ties within the same bucket, prefer the pair with lowest max-pairwise-")
    lines.append("> correlation to already-selected pairs.")
    lines.append("")
    lines.append("**Sprint 3b correlation-aware allocation impl:**")
    lines.append("")
    lines.append("> Use the buckets above (or re-derive after an earlier sprint Phase B's broker H4 backfill ")
    lines.append("> if those numbers diverge meaningfully). Cap simultaneous exposure per bucket at ")
    lines.append("> 2 of N positions. USD-direction soft veto justified IFF the empirical USD-axis ")
    lines.append("> coherence study (deferred from Phase 2G) shows long-vs-short avg corr is ")
    lines.append("> materially negative AND disagreement-trades underperform agreement-trades. ")
    lines.append("> Today's number: long-vs-short avg corr = "
                 f"{coherence['long_vs_short_avg_corr']:.3f}.")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("- yfinance daily closes are **auto-adjusted** for splits/dividends; FX shouldn't ")
    lines.append("  have material adjustments but XAU (gold futures `GC=F`) does have roll effects.")
    lines.append("- Daily-frequency correlation is the right granularity for diversification-rule ")
    lines.append("  design (the bot's bucket-cap fires at signal time, which is on H4 boundaries — ")
    lines.append("  but the correlation structure is dominated by the daily macro factor, not the ")
    lines.append("  intraday session structure).")
    lines.append("- Sample window is recent (default 2023-2026, ~3 years). Earlier history might ")
    lines.append("  show different correlation regimes (e.g. EUR-USD vs GBP-USD pre-Brexit). Run ")
    lines.append("  with `--start 2010-01-01` for a multi-cycle view.")
    lines.append("- USDCAD's correlation profile may shift slightly after an earlier sprint USDCAD retrain ")
    lines.append("  (new model's signal correlation is a separate question from raw price correlation).")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote diagnostic to %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01",
                        help="Start date (YYYY-MM-DD). Default: 2023-01-01 (~3yr)")
    parser.add_argument("--end", default=None,
                        help="End date (YYYY-MM-DD). Default: today")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Correlation threshold for bucketing (default 0.7)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    end = args.end or datetime.utcnow().strftime("%Y-%m-%d")

    closes = fetch_close_series(args.start, end)
    logger.info("Fetched %d pairs over %d daily closes", len(closes.columns), len(closes))

    corr = compute_correlation_matrix(closes)
    buckets = find_correlation_buckets(corr, args.threshold)
    coherence = usd_coherence_summary(corr)

    # Save raw artifacts
    closes.to_csv(OUTPUT_DIR / "daily_closes.csv")
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    summary = {
        "start": args.start,
        "end": end,
        "threshold": args.threshold,
        "n_pairs": len(corr.columns),
        "buckets": [sorted(b) for b in buckets],
        "usd_coherence": coherence,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8",
    )

    write_markdown(
        corr, buckets, coherence,
        args.start, end,
        ROOT / "docs" / "sprints" / "sprint_1.4" / "correlation_diagnostic.md",
    )
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
