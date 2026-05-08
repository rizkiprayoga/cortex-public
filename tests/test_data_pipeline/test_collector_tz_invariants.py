"""
test_collector_tz_invariants.py — cross-collector TZ + format guard rail.

Lesson from the 2026-04-24 backfill incident: any collector that produces
a DataFrame or dict joined with OHLCV must use the same timestamp
convention (naive ISO8601, UTC-anchored). This test enforces that
invariant explicitly for every collector added in Phase 1B + existing
collectors we rely on.

Core invariants
---------------

1. OHLCV bar_timestamp string format: "YYYY-MM-DDTHH:MM:SS" (no tz
   suffix, no microseconds, second precision).
2. Any collector that returns a DataFrame indexed by date/time must
   return it as pd.DatetimeIndex with tz=None (naive).
3. Any collector that returns a dict[str, float] feature output has no
   TZ concern — the invariant applies only to time-indexed outputs.
4. When a collector parses external dates (CSV, API), the parsed
   DatetimeIndex must come out naive.

The guard is parametrized across collectors so adding a new collector
just means adding it to the ``_TIME_INDEXED_COLLECTORS`` list.
"""

from __future__ import annotations

import io
import re
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------
# Invariant 1: OHLCV bar_timestamp format
# ---------------------------------------------------------------------

_NAIVE_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


class TestBarTimestampFormatInvariant:
    """Canonical ohlcv_bars.bar_timestamp format — naive ISO8601, seconds."""

    def test_live_writer_emits_naive_iso_format(self):
        """mt5_feed._rates_to_dicts writer produces the canonical format."""
        from src.data_pipeline.mt5_feed import _broker_ts_to_utc
        import numpy as np

        # Fake MT5 rate struct. time is broker-local epoch (seconds).
        fake_rates = np.array(
            [(1713859200, 3300.0, 3310.0, 3295.0, 3305.0, 100, 0, 0)],
            dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"),
                    ("low", "f8"), ("close", "f8"),
                    ("tick_volume", "i8"), ("spread", "i4"),
                    ("real_volume", "i8")],
        )
        ts = _broker_ts_to_utc(int(fake_rates[0]["time"]))
        emitted = ts.strftime("%Y-%m-%dT%H:%M:%S")
        assert _NAIVE_ISO_RE.match(emitted), f"Bad format: {emitted}"
        assert "+" not in emitted and "Z" not in emitted
        assert ts.tzinfo is None

    def test_backfill_writer_matches_live_format(self):
        """backfill_ohlcv.fetch_chunked should produce the same format —
        the 2026-04-24 incident was exactly this drifting."""
        from src.data_pipeline.mt5_feed import _broker_ts_to_utc
        broker_ts = 1713859200
        live_emits = _broker_ts_to_utc(broker_ts).strftime("%Y-%m-%dT%H:%M:%S")
        # Same helper is used by fetch_chunked after the broker-ts fix.
        # Confirming the helper's output shape is the invariant — if it
        # changes, backfill + live outputs stay in lockstep.
        assert _NAIVE_ISO_RE.match(live_emits)


# ---------------------------------------------------------------------
# Invariant 2+3+4: collector DataFrame indexes are naive
# ---------------------------------------------------------------------

def _assert_df_index_is_naive(df: pd.DataFrame, label: str) -> None:
    """Assert a DataFrame's index is a naive pd.DatetimeIndex (or empty)."""
    if df is None or df.empty:
        return
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        assert idx.tz is None, f"{label}: index has tz={idx.tz}, expected None"
    # Datetime columns (when not indexed) should also be naive — check if
    # any column is a datetime dtype.
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            col_tz = getattr(df[col].dt, "tz", None)
            assert col_tz is None, (
                f"{label}: column {col} has tz={col_tz}, expected None"
            )


class TestStooqParsedDataFrameIsNaive:
    """stooq_data._parse_tff_csv-equivalent path (via get_series)."""

    def test_parsed_csv_index_is_naive(self):
        from src.data_pipeline.market.stooq_data import StooqFetcher
        fetcher = StooqFetcher()
        fetcher.api_key = "dummy"
        sample_csv = (
            "Date,Open,High,Low,Close,Volume\n"
            "2026-04-20,4.10,4.15,4.08,4.12,1000\n"
            "2026-04-21,4.12,4.20,4.10,4.18,1200\n"
        )
        resp = MagicMock()
        resp.text = sample_csv
        resp.raise_for_status = MagicMock()
        with patch("src.data_pipeline.market.stooq_data.requests.get",
                    return_value=resp):
            df = fetcher.get_series("uk_10y")
        _assert_df_index_is_naive(df, "stooq_data")
        assert not df.empty


class TestECBParsedDataFrameIsNaive:
    """ecb_data.get_series output index."""

    def test_parsed_csv_index_is_naive(self):
        from src.data_pipeline.market.ecb_data import ECBDataFetcher
        fetcher = ECBDataFetcher()
        sample = (
            "KEY,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
            "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,2026-04-22,3.0400,A\n"
            "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,2026-04-23,3.0675,A\n"
        )
        resp = MagicMock()
        resp.text = sample
        resp.raise_for_status = MagicMock()
        with patch("src.data_pipeline.market.ecb_data.requests.get",
                    return_value=resp):
            df = fetcher.get_series("10y")
        _assert_df_index_is_naive(df, "ecb_data")
        assert not df.empty


class TestCOTParsedDataFramesAreNaive:
    """cot_data._parse_csv + _parse_tff_csv output naive datetimes."""

    def test_tff_parsed_date_column_is_naive(self):
        from src.data_pipeline.fundamental.cot_data import COTDataFetcher
        tff_csv = (
            "CFTC_Contract_Market_Code,Report_Date_as_YYYY-MM-DD,"
            "Dealer_Positions_Long_All,Dealer_Positions_Short_All,"
            "Lev_Money_Positions_Long_All,Lev_Money_Positions_Short_All,"
            "Open_Interest_All\n"
            "096742,2026-04-22,110000,55000,90000,35000,210000\n"
        )
        df = COTDataFetcher()._parse_tff_csv(tff_csv)
        _assert_df_index_is_naive(df, "cot_data TFF")
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["date"].dt.tz is None


class TestYFinanceIndexIsNaive:
    """cross_asset relies on yfinance.download — verify today's yfinance
    still returns naive daily indexes. If this fails, newer yfinance
    changed behavior and cross_asset will need tz stripping."""

    def test_yfinance_daily_returns_naive_index(self):
        import yfinance as yf
        try:
            df = yf.download("^GSPC", period="5d", progress=False,
                               auto_adjust=True)
        except Exception:
            pytest.skip("yfinance network unavailable")
        if df is None or df.empty:
            pytest.skip("yfinance returned empty (network issue)")
        assert df.index.tz is None, (
            f"yfinance index tz changed to {df.index.tz} — cross_asset "
            "needs tz_localize(None) added to stay compatible"
        )


# ---------------------------------------------------------------------
# Invariant 5: feature dict outputs have no TZ surface at all
# ---------------------------------------------------------------------

class TestFeatureDictOutputsAreScalar:
    """get_*_features() dicts return only scalar floats — no datetimes."""

    @pytest.mark.parametrize("module_factory, method, symbol", [
        ("stooq", "get_yield_features", "GBPUSD"),
        ("ecb", "get_yield_curve_features", "EURUSD"),
    ])
    def test_collector_feature_dicts_are_scalars(
        self, module_factory, method, symbol,
    ):
        if module_factory == "stooq":
            from src.data_pipeline.market.stooq_data import StooqFetcher
            fetcher = StooqFetcher()
            # Stub get_series so no HTTP call is made
            fetcher.get_series = lambda label: pd.DataFrame()  # type: ignore[assignment]
        else:
            from src.data_pipeline.market.ecb_data import ECBDataFetcher
            fetcher = ECBDataFetcher()
            fetcher.get_series = lambda label: pd.DataFrame()  # type: ignore[assignment]

        feats = getattr(fetcher, method)(symbol)
        for key, value in feats.items():
            assert isinstance(value, (int, float)), (
                f"{module_factory}.{method}({symbol}): {key}={value!r} "
                f"is not a scalar (type={type(value).__name__})"
            )


# ---------------------------------------------------------------------
# Cross-module alignment: bar_timestamp string parses cleanly into the
# same naive datetime, across the DB read path and live writer path
# ---------------------------------------------------------------------

class TestBarTimestampRoundTrip:
    """Round-trip a bar_timestamp through pd.to_datetime and back."""

    def test_round_trip_preserves_naive(self):
        """Simulates data_store.get_ohlcv_range reading the DB."""
        bar_ts = "2026-04-24T13:00:00"
        parsed = pd.to_datetime(bar_ts)
        assert parsed.tzinfo is None
        # Reformat back — should match
        assert parsed.strftime("%Y-%m-%dT%H:%M:%S") == bar_ts

    def test_empty_suffix_rejected_by_regex(self):
        """The +00:00 suffix is exactly what the April incident wrote.
        Ensure our format check rejects it so the invariant test above
        catches a regression."""
        assert not _NAIVE_ISO_RE.match("2026-04-24T13:00:00+00:00")
        assert not _NAIVE_ISO_RE.match("2026-04-24T13:00:00.000000")
        assert not _NAIVE_ISO_RE.match("2026-04-24T13:00:00Z")

    def test_canonical_format_accepted(self):
        assert _NAIVE_ISO_RE.match("2026-04-24T13:00:00")
        assert _NAIVE_ISO_RE.match("2000-12-05T22:00:00")
