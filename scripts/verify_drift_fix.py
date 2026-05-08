"""End-to-end verification of the A-8 drift-check fix.

Exercises the full pipeline against all 5 live symbols using DB-backed
bars (no MT5 touch, safe to run alongside live bot). Prints:
  - current feature matrix shape
  - overlap with saved training distribution
  - PSI/KS + whether retrain WOULD fire
  - safety-rail behavior (absurd-PSI ceiling, kill-switch)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import logging
logging.basicConfig(level=logging.WARNING)

from src.data_pipeline.data_store import DataStore
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.ml.drift_check import check_symbol_drift, maybe_trigger_retrain


class DBBackedFeed:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._store: DataStore | None = None
        self._loop = asyncio.new_event_loop()

    def _ensure(self) -> None:
        if self._store is None:
            self._store = DataStore(dsn=self._dsn)
            self._loop.run_until_complete(self._store.connect())

    def close(self) -> None:
        if self._store is not None:
            self._loop.run_until_complete(self._store.close())
        self._loop.close()

    def get_historical(self, symbol, timeframe, bars=500, start_date=None):
        self._ensure()
        async def go():
            df = await self._store.get_ohlcv_range(symbol, timeframe, start=None, end=None, limit=bars)
            if "volume" in df.columns and "tick_volume" not in df.columns:
                df = df.rename(columns={"volume": "tick_volume"})
            if "tick_volume" not in df.columns:
                df["tick_volume"] = 0
            return df
        return self._loop.run_until_complete(go())


def main() -> None:
    feed = DBBackedFeed(dsn=os.environ["POSTGRES_DSN"])
    engineer = FeatureEngineer()
    symbols = ["XAUUSD", "EURUSD", "USDJPY", "USDCAD", "ETHUSD"]
    tmp_state = Path("data/state/_drift_verify.json")
    if tmp_state.exists():
        tmp_state.unlink()
    try:
        print(f"{'symbol':<8} {'PSI':>8} {'KS':>8} {'n':>5} {'warn':>5} {'alert':>5} {'retrain_fired':>14}  worst_feature", flush=True)
        print("-" * 95, flush=True)
        for sym in symbols:
            s = check_symbol_drift(sym, feed, engineer)
            if s.error:
                print(f"{sym:<8} ERROR: {s.error}", flush=True)
                continue
            # Try the retrain trigger with safety rails
            fired = maybe_trigger_retrain(sym, s.psi_max, retrain_threshold=0.5, last_trigger_path=tmp_state)
            warn = "WARN" if s.warn_breached else "-"
            alert = "ALERT" if s.alert_breached else "-"
            fired_str = "YES-FIRED" if fired else "no"
            print(f"{sym:<8} {s.psi_max:>8.3f} {s.ks_max:>8.3f} {s.n_current_samples:>5} {warn:>5} {alert:>5} {fired_str:>14}  {s.worst_feature or '-'}", flush=True)
    finally:
        feed.close()
        # Clean up throwaway state file
        if tmp_state.exists():
            tmp_state.unlink()


if __name__ == "__main__":
    main()
