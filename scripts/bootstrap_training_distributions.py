"""
bootstrap_training_distributions.py — A-8 one-shot seed helper.

Creates ``data/models/lstm_{symbol}.training_dist.json`` for each live
symbol using the same feature pipeline that ``train_deep_learning.py``
uses, so the daily drift-monitor has a baseline to compare against
starting the very next cron firing.

Without this, you'd have to wait until the next monthly retrain
(2026-05-01 by default) before the first PSI score could be computed.

Approximation: uses all H4 history available *up to now*, rather than
the exact window the models were trained on. This is within a few days
of the actual training cutoff (models retrained 2026-04-17) and the
gap is dwarfed by the 5+ years of history the distribution is built
from. Proper training-distribution snapshots are saved atomically
alongside the model from the next retrain onward.

Defaults to a DB-backed feed (reads ``ohlcv_bars`` directly) so the
script is safe to run while the live bot is attached — it does NOT
touch the MT5 terminal. Pass ``--mt5`` to fall back to the legacy
MT5-backed path if DB coverage is missing.

Usage:
    python scripts/bootstrap_training_distributions.py
    python scripts/bootstrap_training_distributions.py --symbols XAUUSD
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import numpy as np

from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bootstrap_training_distributions")


LIVE_SYMBOLS = ("XAUUSD", "EURUSD", "USDJPY", "USDCAD", "ETHUSD")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(LIVE_SYMBOLS))
    p.add_argument(
        "--mt5", action="store_true",
        help="Use MT5Connector instead of the DB-backed feed. "
             "Unsafe while the live bot is attached — prefer the default.",
    )
    return p.parse_args()


class DBBackedFeed:
    """Minimal feed shim that reads ``ohlcv_bars`` directly from Postgres.

    Mirrors ``MT5DataFeed.get_historical()``'s return shape so the rest
    of the feature pipeline doesn't care where bars came from. Safe to
    run concurrently with the live bot — never touches MT5.
    """

    def __init__(self, dsn: str) -> None:
        from src.data_pipeline.data_store import DataStore
        self._store = DataStore(dsn=dsn)
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._store.connect())

    def close(self) -> None:
        self._loop.run_until_complete(self._store.close())
        self._loop.close()

    def get_historical(self, symbol, timeframe, bars=500, start_date=None):
        async def go():
            df = await self._store.get_ohlcv_range(
                symbol, timeframe, start=None, end=None, limit=bars,
            )
            if "volume" in df.columns and "tick_volume" not in df.columns:
                df = df.rename(columns={"volume": "tick_volume"})
            if "tick_volume" not in df.columns:
                df["tick_volume"] = 0
            return df
        return self._loop.run_until_complete(go())


def _build_feed(use_mt5: bool):
    """Return (feed, cleanup_callable)."""
    if use_mt5:
        from src.broker.mt5_connector import MT5Connector
        from src.data_pipeline.mt5_feed import MT5DataFeed
        connector = MT5Connector()
        connector.connect()
        return MT5DataFeed(connector), connector.disconnect
    feed = DBBackedFeed(dsn=os.environ["POSTGRES_DSN"])
    return feed, feed.close


def main() -> int:
    args = parse_args()

    from src.data_pipeline.feature_engineering import FeatureEngineer
    from src.brain.hmm_regime import HMMRegimeClassifier
    from src.ml.drift import save_training_distribution

    feed, cleanup = _build_feed(use_mt5=args.mt5)
    engineer = FeatureEngineer()

    try:
        for symbol in args.symbols:
            logger.info("=== %s ===", symbol)
            h4 = feed.get_historical(symbol, "H4", bars=99999)
            if h4 is None or len(h4) < 500:
                logger.warning("[%s] too little H4 data (%s); skipping",
                               symbol, len(h4) if h4 is not None else 0)
                continue
            d1 = feed.get_historical(symbol, "D1", bars=h4.shape[0] // 6)
            w1 = feed.get_historical(symbol, "W1", bars=h4.shape[0] // 30)

            ohlcv_by_tf = {"H4": h4}
            if d1 is not None and len(d1) > 50:
                ohlcv_by_tf["D1"] = d1
            if w1 is not None and len(w1) > 10:
                ohlcv_by_tf["W1"] = w1

            features_df = engineer.transform_multi_timeframe(
                ohlcv_by_tf, primary_tf="H4",
            )
            # Calendar + zero-fill match train_deep_learning.py
            try:
                from src.data_pipeline.market.calendar_features import (
                    CalendarFeatureBuilder,
                )
                cal = CalendarFeatureBuilder()
                cal_df = cal.get_historical_calendar_features(features_df.index)
                features_df = features_df.join(cal_df, how="left")
            except Exception as exc:
                logger.warning("Calendar features unavailable: %s", exc)
            zero_fill = engineer.get_zero_fill_feature_names(symbol)
            for col in zero_fill:
                if col not in features_df.columns:
                    features_df[col] = 0.0
            features_df = features_df.fillna(0.0)

            # Inject HMM regime features if the HMM exists — matches
            # the LSTM feature-space without actually training the LSTM.
            hmm = HMMRegimeClassifier()
            if hmm.load(symbol) and d1 is not None:
                features_df = engineer.inject_regime_features(
                    features_df, hmm, symbol, d1,
                )

            # Save RAW feature values (pre-``to_matrix`` z-score). The
            # drift monitor compares raw distributions because
            # ``to_matrix`` normalizes each batch against its own scale,
            # which makes PSI/KS between training and drift-window
            # batches meaningless.
            feature_manifest = engineer.get_feature_columns(features_df)
            raw_matrix = features_df[feature_manifest].to_numpy(
                dtype=float, copy=True,
            )
            raw_matrix = np.nan_to_num(
                raw_matrix, nan=0.0, posinf=0.0, neginf=0.0,
            )

            dist_path = Path("data/models") / f"lstm_{symbol}.training_dist.json"
            save_training_distribution(
                dist_path,
                symbol=symbol, timeframe="H4",
                feature_matrix=raw_matrix,
                feature_names=tuple(feature_manifest),
            )
            logger.info("[%s] bootstrapped training dist: %d rows, %d features",
                        symbol, raw_matrix.shape[0], raw_matrix.shape[1])
    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
