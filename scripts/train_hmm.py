"""
train_hmm.py — Standalone HMM Training Script

Trains and saves a GaussianHMM regime classifier for each symbol
using the full available historical data from MT5.

Usage:
    python scripts/train_hmm.py
    python scripts/train_hmm.py --symbols XAUUSD --bars 1000

Steps:
    1. Connect to MT5
    2. Fetch D1 historical OHLCV for each symbol
    3. Compute HMM features (log_return, volatility, RSI, ATR, volume_ratio)
    4. Train GaussianHMM with n_init random starts
    5. Print regime statistics (state means, transition matrix)
    6. Save model to data/models/hmm_{symbol}.pkl
    7. Disconnect from MT5
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from dotenv import load_dotenv
load_dotenv()

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.data_store import DataStore
from src.data_pipeline.mt5_feed import MT5DataFeed
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.brain.hmm_regime import HMMRegimeClassifier

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train HMM regime classifier")
    parser.add_argument("--symbols", nargs="+",
                        default=["XAUUSD", "EURUSD", "USDJPY", "USDCAD"])
    parser.add_argument("--bars", type=int, default=5000,
                        help="D1 bars of history (0 = all available)")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--no-snapshot", action="store_true",
                        help="Skip auto-snapshot before training")
    parser.add_argument("--snapshot-label", default=None,
                        help="Override auto-snapshot label")
    parser.add_argument("--end-date", default=None,
                        help="Truncate training data at this date (YYYY-MM-DD). "
                             "Used for walk-forward OOS validation.")
    return parser.parse_args()


async def main_async():
    args = parse_args()

    # Auto-snapshot existing models before retraining (rollback safety)
    if not args.no_snapshot:
        from datetime import datetime, timezone
        try:
            from scripts.model_snapshot import cmd_save
            label = (args.snapshot_label or
                     f"auto-pretrain-hmm-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            cmd_save(label, note="Auto-snapshot before HMM retraining")
            logger.info("Pre-training snapshot saved: %s", label)
        except Exception as exc:
            logger.warning("Auto-snapshot failed (continuing): %s", exc)

    # DB-only OHLCV reads — no MT5 contact, eliminates the shared-terminal
    # hijack risk that polluted prod equity_history on 2026-04-25.
    # See feedback_dev_mt5_steals_prod_terminal.md.
    data_store = DataStore()
    await data_store.connect()
    logger.info("DataStore connected — training will read OHLCV from DB only.")

    feed = MT5DataFeed(connector=None, data_store=data_store)
    engineer = FeatureEngineer()
    clf = HMMRegimeClassifier(n_components=args.n_components, n_init=args.n_init)

    bars = args.bars if args.bars > 0 else 99999

    for symbol in args.symbols:
        logger.info("Training HMM for %s (%s D1 bars)...",
                     symbol, "all" if args.bars == 0 else args.bars)

        ohlcv = await feed.get_historical_db_only(symbol, "D1", bars=bars)
        if args.end_date:
            import pandas as pd
            cutoff = pd.Timestamp(args.end_date)
            ohlcv = ohlcv[ohlcv.index <= cutoff]
            logger.info("  Truncated D1 to %d bars (<= %s)",
                         len(ohlcv), args.end_date)
        features_df = engineer.transform(ohlcv)
        feature_manifest = engineer.get_feature_columns(features_df)
        # Pass raw (un-normalized) matrix — train() handles z-scoring
        # internally and stores the normalization stats for inference.
        sorted_df = features_df[sorted(features_df.columns)]
        raw_matrix = np.nan_to_num(
            sorted_df.values.astype(np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        logger.info("  Feature matrix: %s (%d features)",
                     raw_matrix.shape, len(feature_manifest))

        # MLflow run per (symbol, training invocation) — T-8
        from datetime import datetime, timezone
        import mlflow
        from src.ml.registry import start_run, dataset_fingerprint

        run_name = f"{symbol}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        fp = dataset_fingerprint(
            symbol=symbol,
            timeframe="D1",
            first_bar_ts=str(ohlcv.index[0]),
            last_bar_ts=str(ohlcv.index[-1]),
            closes=ohlcv["close"].values,
        )
        with start_run(
            experiment="hmm_regime",
            run_name=run_name,
            tags={
                "symbol": symbol,
                "model_head": "hmm",
                "training_script": "train_hmm.py",
            },
        ):
            mlflow.log_params({
                "symbol": symbol,
                "n_components": args.n_components,
                "n_init": args.n_init,
                "bars_requested": args.bars,
                "bars_actual": len(ohlcv),
                "end_date": args.end_date or "",
                "dataset_fingerprint": fp,
            })

            clf.train(symbol, raw_matrix, feature_manifest=feature_manifest)
            clf.save(symbol)

            # Metrics: stationary regime weights (len = n_components)
            try:
                # HMMRegimeClassifier exposes transmat_ on the fitted hmm
                # via clf._models[symbol] — stay defensive in case the
                # attribute shape drifts.
                model = clf._models.get(symbol) if hasattr(clf, "_models") else None
                if model is not None and hasattr(model, "transmat_"):
                    import numpy as _np
                    # Solve for stationary distribution: left eigenvector at 1
                    eigvals, eigvecs = _np.linalg.eig(model.transmat_.T)
                    stat = _np.real(eigvecs[:, _np.argmin(_np.abs(eigvals - 1.0))])
                    stat = stat / stat.sum()
                    for i, w in enumerate(stat):
                        mlflow.log_metric(f"regime_{i}_weight", float(w))
            except Exception as _exc:
                logger.debug("Skipping regime-weight metrics: %s", _exc)
            mlflow.log_metric("n_training_bars", float(len(ohlcv)))

            # Artifact: the saved .pkl + a fingerprint sidecar
            pkl_path = Path("data/models") / f"hmm_{symbol}.pkl"
            if pkl_path.exists():
                mlflow.log_artifact(str(pkl_path))

            import json, tempfile
            with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8",
            ) as tmp:
                json.dump({
                    "symbol": symbol,
                    "timeframe": "D1",
                    "first_bar": str(ohlcv.index[0]),
                    "last_bar": str(ohlcv.index[-1]),
                    "bars": len(ohlcv),
                    "fingerprint_sha256": fp,
                }, tmp, indent=2)
                tmp_path = tmp.name
            try:
                mlflow.log_artifact(tmp_path, artifact_path="dataset")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            # Summary line for operator log
            result = clf.predict(symbol, raw_matrix[-60:])
            logger.info(f"  → Current regime: {result.regime_label} "
                        f"(p={result.state_probability:.2f}, "
                        f"multiplier={result.position_multiplier})")

    logger.info("HMM training complete.")


def main():
    """Sync wrapper — entry point for the script."""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
