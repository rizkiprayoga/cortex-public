"""
train_meta_labeler.py — Train per-symbol LightGBM meta-labelers (M-1 + Phase A Sprint 4).

Pulls labeled trades from ``backtest_trades``, enriches each trade with 17
fundamental features via ``read_feature_store_safe(as_of=entry_ts)``
(lookahead-safe per spec invariant #11), trains a binary classifier that
predicts *whether the trade will be profitable*, and saves the bundle to
``data/models/meta_labeler_{symbol}_{primary}.pkl``. The artifact carries
``feature_schema_hash`` so signal_combiner refuses to load on schema
drift (spec invariant #12).

Why backtest_trades rather than live ``trades`` — live has ≈ 19 rows;
backtest_trades has ~9.7k labeled samples across the 5 live symbols.
This matches López de Prado's AFML pattern: train on historical
outcomes where ground truth is known, then use it live as a filter.

The ``--primary {lstm,gbm}`` flag filters trades by ``primary_kind``
(spec §1 anchor 4: 8 meta-labelers trained — 4 symbols × 2 primaries).
Legacy rows default to ``primary_kind='lstm'`` via the ORM column
default + a backfilling DB migration.

Usage:
    python scripts/train_meta_labeler.py
    python scripts/train_meta_labeler.py --symbols XAUUSD --primary lstm
    python scripts/train_meta_labeler.py --primary gbm  # after Sprint 6 backtests
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_meta_labeler")

LIVE_SYMBOLS = ("XAUUSD", "EURUSD", "USDJPY", "USDCAD", "ETHUSD")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(LIVE_SYMBOLS))
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold for the 'take trade' decision "
                        "at inference time (default: 0.5).")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Fraction of the most-recent trades held out for "
                        "validation (default: 0.20).")
    p.add_argument("--min-trades", type=int, default=200,
                   help="Skip symbols with fewer than this many trades "
                        "(default: 200).")
    p.add_argument("--primary", choices=("lstm", "gbm"), default="lstm",
                   help="Which primary's backtest_trades to train against "
                        "(spec §1 anchor 4 — primary-agnostic interface, "
                        "but training data is primary-specific).")
    p.add_argument("--no-fundamentals", action="store_true",
                   help="Skip the feature_store fundamentals enrichment. "
                        "Trains a 5-feature (BASE_FEATURE_NAMES only) bundle "
                        "with a different schema_hash. Use this until "
                        "fundamentals plumbing reaches the live inference "
                        "hot path — gives train/serve parity at the cost "
                        "of 17 features. Bundle is signed with "
                        "BASE_ONLY_SCHEMA_HASH; signal_combiner accepts it "
                        "via ACCEPTED_SCHEMA_HASHES.")
    p.add_argument("--no-exec-features", action="store_true",
                   help="Skip the 4 execution-conditional features (rolling "
                        "RV + score_avg). Used for ablation/legacy-22-feature "
                        "bundles. Default is to compute them — they're cheap "
                        "(one H4 OHLCV slice per symbol per training run).")
    p.add_argument("--max-entry-date", default=None,
                   help="Train ONLY on trades with entry_time < this date "
                        "(YYYY-MM-DD). Used for clean OOS evaluation: set "
                        "to the start of the OOS window so the training set "
                        "is strictly pre-OOS. Without this, training data "
                        "overlaps with the OOS test window (López de Prado "
                        "Ch. 18 cautions against this contamination).")
    return p.parse_args()


async def load_backtest_trades(
    symbol: str, primary: str, max_entry_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch labeled trades for one (symbol, primary) from ``backtest_trades``.

    If ``max_entry_date`` is set, only returns trades with
    ``entry_time < max_entry_date`` (clean OOS — training set is strictly
    before the OOS window).
    """
    import asyncpg
    dsn = os.environ["POSTGRES_DSN"].replace(
        "postgresql+asyncpg://", "postgresql://",
    )
    conn = await asyncpg.connect(dsn)
    try:
        if max_entry_date:
            # entry_time is stored as varchar (ISO 8601), so string compare
            # works correctly. Pad the cutoff date to a full ISO timestamp
            # to avoid prefix-matching ambiguity.
            cutoff_str = max_entry_date if "T" in max_entry_date or " " in max_entry_date else f"{max_entry_date} 00:00:00"
            rows = await conn.fetch(
                """
                SELECT symbol, direction, entry_time, combined_score,
                       regime_label, pnl, primary_kind
                FROM backtest_trades
                WHERE symbol = $1
                  AND primary_kind = $2
                  AND pnl IS NOT NULL
                  AND combined_score IS NOT NULL
                  AND entry_time < $3
                ORDER BY entry_time ASC
                """,
                symbol, primary, cutoff_str,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT symbol, direction, entry_time, combined_score,
                       regime_label, pnl, primary_kind
                FROM backtest_trades
                WHERE symbol = $1
                  AND primary_kind = $2
                  AND pnl IS NOT NULL
                  AND combined_score IS NOT NULL
                ORDER BY entry_time ASC
                """,
                symbol, primary,
            )
    finally:
        await conn.close()
    return pd.DataFrame(rows, columns=[
        "symbol", "direction", "entry_time", "combined_score",
        "regime_label", "pnl", "primary_kind",
    ])


def _log_to_mlflow(symbol: str, primary: str, result, path: Path,
                   threshold: float, n_total: int) -> None:
    """Log the training run to MLflow registry under experiment 'meta_labeler'."""
    try:
        import mlflow
        from src.ml.registry import start_run
    except ImportError:
        logger.warning("mlflow not available; skipping registry logging")
        return

    run_name = f"{symbol}-{primary}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    with start_run(
        experiment="meta_labeler",
        run_name=run_name,
        tags={
            "symbol": symbol,
            "primary_kind": primary,
            "model_head": "lightgbm_binary",
            "training_script": "train_meta_labeler.py",
            "phase": "phase_a",
        },
    ):
        mlflow.log_params({
            "symbol": symbol,
            "primary": primary,
            "threshold": threshold,
            "n_train": result.n_train,
            "n_val": result.n_val,
            "n_total": n_total,
        })
        mlflow.log_metrics({
            "val_accuracy": result.val_accuracy,
            "val_precision": result.val_precision,
            "val_recall": result.val_recall,
            "coverage": result.coverage_at_default_threshold,
            "pf_without_gate": result.pf_without_gate,
            "pf_with_gate": (0.0 if result.pf_with_gate == float("inf")
                              else result.pf_with_gate),
        })
        if path.exists():
            mlflow.log_artifact(str(path))


async def _train_one_symbol(symbol: str, args: argparse.Namespace) -> dict | None:
    """Train one (symbol, primary) bundle. Returns summary dict or None on skip."""
    from src.data_pipeline.data_store import DataStore
    from src.ml.meta_labeler import (
        _enrich_with_exec_features, _enrich_with_fundamentals,
        save_meta_labeler, train_meta_labeler,
    )

    logger.info("=== %s (primary=%s) ===", symbol, args.primary)
    df = await load_backtest_trades(symbol, args.primary, args.max_entry_date)
    if df.empty:
        logger.warning("[%s/%s] no trades; skipping", symbol, args.primary)
        return None
    if len(df) < args.min_trades:
        logger.warning(
            "[%s/%s] only %d trades (need %d); skipping",
            symbol, args.primary, len(df), args.min_trades,
        )
        return None

    # Spec invariant #11 — lookahead-safe fundamental enrichment.
    # When --no-fundamentals is passed, skip the enrichment entirely and
    # train on the 5 base features only. This gives train/serve parity for
    # the inference path (which currently passes fundamentals=None) until
    # the live fundamentals plumbing lands.
    if args.no_fundamentals:
        logger.info("[%s] --no-fundamentals: skipping enrichment, "
                     "training on 5 base features only", symbol)
        enriched = df
    else:
        store = DataStore()
        await store.connect()
        try:
            logger.info("[%s] enriching %d trades with fundamentals "
                         "(may take a minute on cold partitions)", symbol, len(df))
            enriched = await _enrich_with_fundamentals(df, symbol, store)
            # Phase 2B Option 2: exec features (rolling RV + score_avg).
            # Same DataStore connection — one H4 OHLCV slice per symbol.
            if not args.no_exec_features:
                logger.info("[%s] enriching with 4 exec-conditional features "
                             "(rolling RV + score_avg)", symbol)
                enriched = await _enrich_with_exec_features(enriched, symbol, store)
        finally:
            await store.close()

    try:
        clf, result = train_meta_labeler(
            symbol, enriched, val_fraction=args.val_fraction,
            threshold=args.threshold,
        )
    except ValueError as exc:
        logger.warning("[%s/%s] %s", symbol, args.primary, exc)
        return None

    path = save_meta_labeler(
        clf, symbol, threshold=args.threshold, primary=args.primary,
    )
    _log_to_mlflow(symbol, args.primary, result, path, args.threshold,
                   n_total=len(df))

    return {
        "symbol": symbol,
        "primary": args.primary,
        "n_trades": len(df),
        "val_acc": f"{result.val_accuracy * 100:.1f}%",
        "precision": f"{result.val_precision * 100:.1f}%",
        "recall": f"{result.val_recall * 100:.1f}%",
        "coverage": f"{result.coverage_at_default_threshold * 100:.1f}%",
        "pf_no_gate": f"{result.pf_without_gate:.2f}",
        "pf_gated": f"{result.pf_with_gate:.2f}",
    }


async def main_async() -> int:
    args = parse_args()
    summary_rows: list[dict] = []

    for symbol in args.symbols:
        row = await _train_one_symbol(symbol, args)
        if row is not None:
            summary_rows.append(row)

    if summary_rows:
        print("\n" + "=" * 100)
        print(f"META-LABELER TRAINING SUMMARY (primary={args.primary})")
        print("=" * 100)
        print(pd.DataFrame(summary_rows).to_string(index=False))
        print("=" * 100)
        print(
            f"\nThreshold: {args.threshold}  "
            f"Val fraction: {args.val_fraction}  "
            f"Saved to data/models/meta_labeler_{{symbol}}_{args.primary}.pkl"
        )
    else:
        logger.error("No symbols trained. Check --symbols and backtest_trades data.")
        return 1
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
