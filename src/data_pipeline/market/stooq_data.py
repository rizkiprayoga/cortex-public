"""
stooq_data.py — Daily sovereign bond yields via Stooq CSV endpoints.

Stooq (stooq.com) offers free CSV downloads of global market data. We use
it for **daily** 2Y + 10Y government bond yields across the 6 countries
that matter for the 10-symbol universe:

    Country  2Y symbol   10Y symbol   Currency exposure
    -------  ---------   ----------   -------------------------
    UK       2YUKY.B     10YUKY.B     GBP  (GBPUSD, EURGBP, GBPJPY)
    AU       2YAUY.B     10YAUY.B     AUD  (AUDUSD, AUDNZD)
    NZ       2YNZY.B     10YNZY.B     NZD  (AUDNZD)
    JP       2YJPY.B     10YJPY.B     JPY  (USDJPY, EURJPY, GBPJPY)
    DE       2YDEY.B     10YDEY.B     EUR  (EURUSD, EURGBP, EURJPY)
    US       2YUSY.B     10YUSY.B     USD  (cross-check vs FRED DGS10)

This gives us daily-resolution yields (vs FRED's monthly OECD series),
plus the 2Y-10Y slope per country — useful as a recession/expectations
indicator analogous to the US T10Y2Y the macro block already emits.

Collect-breadth mandate: we fetch every tenor Stooq exposes and every
country in the list, regardless of whether the current symbol uses it.
Emission is per-currency-exposure only.

Stooq API notes
---------------

As of 2025, Stooq's CSV endpoints require a free API key (captcha-gated).

    1. Visit https://stooq.com/q/d/?s=10yuky.b&get_apikey
    2. Solve captcha, copy the key.
    3. Export STOOQ_API_KEY=... in your .env

When the key is missing OR a request fails, the fetcher logs a warning
and returns empty Series / zero-filled defaults — no blocking errors
at bot startup. This matches the pattern used by news_sentiment.py +
macro_data.py for optional external data sources.
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.data_pipeline.data_store import DataStore

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# Stooq symbol catalog. Key = internal label used downstream; value =
# (stooq_symbol, tenor_years, country_code).
STOOQ_SERIES: dict[str, tuple[str, int, str]] = {
    # UK
    "uk_2y":  ("2YUKY.B",  2,  "UK"),
    "uk_10y": ("10YUKY.B", 10, "UK"),
    # Australia
    "au_2y":  ("2YAUY.B",  2,  "AU"),
    "au_10y": ("10YAUY.B", 10, "AU"),
    # New Zealand
    "nz_2y":  ("2YNZY.B",  2,  "NZ"),
    "nz_10y": ("10YNZY.B", 10, "NZ"),
    # Japan
    "jp_2y":  ("2YJPY.B",  2,  "JP"),
    "jp_10y": ("10YJPY.B", 10, "JP"),
    # Germany (EUR benchmark — Bund)
    "de_2y":  ("2YDEY.B",  2,  "DE"),
    "de_10y": ("10YDEY.B", 10, "DE"),
    # US (cross-check against FRED DGS10)
    "us_2y":  ("2YUSY.B",  2,  "US"),
    "us_10y": ("10YUSY.B", 10, "US"),
}

# Currency exposure — re-exported from the shared source of truth at
# fundamental/_currency_exposure.py.
from src.data_pipeline.fundamental._currency_exposure import (  # noqa: E402
    AUD_EXPOSURE as _AUD_EXPOSURE,
    EUR_EXPOSURE as _EUR_EXPOSURE,
    GBP_EXPOSURE as _GBP_EXPOSURE,
    JPY_EXPOSURE as _JPY_EXPOSURE,
    NZD_EXPOSURE as _NZD_EXPOSURE,
)

_STOOQ_CSV_URL = "https://stooq.com/q/d/l/"
_HTTP_TIMEOUT_SEC = 15.0
_APIKEY_PROMPT_MARKER = "get_apikey"   # body text when key is missing


class StooqFetcher:
    """
    Daily sovereign-yield fetcher using Stooq CSV endpoints.

    Usage:
        fetcher = StooqFetcher()
        df = fetcher.get_series("uk_10y")              # raw series
        feats = fetcher.get_yield_features("GBPUSD")   # per-symbol dict

    Cache TTL defaults to 1 day — yields update once per trading session.
    """

    def __init__(self, cache_ttl_hours: float = 24.0):
        self.api_key = os.getenv("STOOQ_API_KEY", "").strip()
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_ts: dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        if not self.api_key:
            logger.info(
                "STOOQ_API_KEY not set — Stooq yields will return empty. "
                "Get a free key at https://stooq.com/q/d/?s=10yuky.b&get_apikey"
            )

    # ------------------------------------------------------------------
    # Raw CSV fetch
    # ------------------------------------------------------------------

    def get_series(self, label: str) -> pd.DataFrame:
        """
        Fetch one Stooq series by internal label. Returns a DataFrame
        indexed by Date with OHLCV columns (or empty DataFrame on any
        error — never raises).
        """
        if label not in STOOQ_SERIES:
            logger.warning("Unknown Stooq label: %s", label)
            return pd.DataFrame()

        # Cache hit?
        ts = self._cache_ts.get(label)
        if ts and (datetime.utcnow() - ts) < self._cache_ttl:
            return self._cache[label]

        stooq_symbol, _tenor, _country = STOOQ_SERIES[label]
        params = {"s": stooq_symbol.lower(), "i": "d"}
        if self.api_key:
            params["apikey"] = self.api_key

        try:
            resp = requests.get(_STOOQ_CSV_URL, params=params,
                                 timeout=_HTTP_TIMEOUT_SEC)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Stooq fetch failed for %s: %s", label, exc)
            return pd.DataFrame()

        body = resp.text
        if _APIKEY_PROMPT_MARKER in body.lower():
            # Stooq returned the "get an API key" page instead of CSV
            logger.warning("Stooq requires API key for %s — skipping", label)
            return pd.DataFrame()

        try:
            df = pd.read_csv(io.StringIO(body))
        except Exception as exc:
            logger.warning("Stooq CSV parse failed for %s: %s", label, exc)
            return pd.DataFrame()

        if df.empty or "Date" not in df.columns or "Close" not in df.columns:
            logger.warning("Stooq returned unusable body for %s", label)
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        self._cache[label] = df
        self._cache_ts[label] = datetime.utcnow()
        return df

    # ------------------------------------------------------------------
    # Per-symbol feature emission
    # ------------------------------------------------------------------

    def get_yield_features(self, symbol: str) -> dict[str, float]:
        """
        Return daily-yield features for the trading symbol, routed by
        currency exposure. Every symbol gets US yields as the USD-axis
        reference; country-specific blocks layer on top.

        Emitted per country block: `{country}_2y_daily`, `{country}_10y_daily`,
        `{country}_slope_daily` (= 10Y - 2Y).
        """
        features: dict[str, float] = {}
        sym_upper = symbol.upper()

        # USD-axis baseline — always emitted so downstream can always
        # compute cross-country spreads.
        self._emit_country_block(features, "us", "us_2y", "us_10y")

        if sym_upper in _GBP_EXPOSURE:
            self._emit_country_block(features, "uk", "uk_2y", "uk_10y")
        if sym_upper in _EUR_EXPOSURE:
            self._emit_country_block(features, "de", "de_2y", "de_10y")
        if sym_upper in _JPY_EXPOSURE:
            self._emit_country_block(features, "jp", "jp_2y", "jp_10y")
        if sym_upper in _AUD_EXPOSURE:
            self._emit_country_block(features, "au", "au_2y", "au_10y")
        if sym_upper in _NZD_EXPOSURE:
            self._emit_country_block(features, "nz", "nz_2y", "nz_10y")

        return features

    def _emit_country_block(
        self,
        features: dict[str, float],
        country: str,
        label_2y: str,
        label_10y: str,
    ) -> None:
        """Populate `{country}_2y_daily`, `_10y_daily`, `_slope_daily`."""
        y2 = self._latest_close(label_2y)
        y10 = self._latest_close(label_10y)
        features[f"{country}_2y_daily"] = y2
        features[f"{country}_10y_daily"] = y10
        # Slope defaults to 0.0 when either side is missing — neutral.
        features[f"{country}_slope_daily"] = (y10 - y2) if (y2 and y10) else 0.0

    def _latest_close(self, label: str) -> float:
        """Return latest Close value for a Stooq series, or 0.0 on miss."""
        df = self.get_series(label)
        if df.empty or "Close" not in df.columns:
            return 0.0
        try:
            return float(df["Close"].iloc[-1])
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # feature_store backfill (Phase 1F)
    # ------------------------------------------------------------------

    FEATURE_GROUP = "stooq_yields"
    SCHEMA_VERSION = 1

    async def persist_raw_history_to_feature_store(
        self, store: "DataStore", symbol: str,
        *,
        force: bool = False,
        lookback_days: Optional[int] = None,
    ) -> int:
        """
        Write the daily yield curve for this symbol's routed countries
        to ``feature_store``. Mirrors the per-symbol routing of
        ``get_yield_features`` — every symbol gets the US-axis pair and
        layers on its currency-exposed countries.

        Args:
            force: When True, switches conflict policy from DO NOTHING to
                DO UPDATE — used by the weekly TTL safety-net job.
            lookback_days: When set, only persist rows from the last N days.
                Default None = full history.

        Snapshot model: one row per (symbol, date). Returns rows touched.
        """
        sym_upper = symbol.upper()

        # Same routing as _emit_country_block. US-axis is always emitted.
        countries: list[tuple[str, str, str]] = [("us", "us_2y", "us_10y")]
        if sym_upper in _GBP_EXPOSURE:
            countries.append(("uk", "uk_2y", "uk_10y"))
        if sym_upper in _EUR_EXPOSURE:
            countries.append(("de", "de_2y", "de_10y"))
        if sym_upper in _JPY_EXPOSURE:
            countries.append(("jp", "jp_2y", "jp_10y"))
        if sym_upper in _AUD_EXPOSURE:
            countries.append(("au", "au_2y", "au_10y"))
        if sym_upper in _NZD_EXPOSURE:
            countries.append(("nz", "nz_2y", "nz_10y"))

        # Build wide DataFrame: one column per (country)_(tenor)_daily.
        per_label: dict[str, pd.Series] = {}
        for _country, label_2y, label_10y in countries:
            for label in (label_2y, label_10y):
                df = self.get_series(label)
                if df.empty or "Close" not in df.columns:
                    continue
                per_label[label] = df["Close"]
        if not per_label:
            logger.warning("Stooq: no series available for %s, skipping", symbol)
            return 0

        wide = pd.concat(per_label, axis=1).sort_index().ffill().dropna(how="all")
        if lookback_days is not None and not wide.empty:
            cutoff = wide.index.max() - pd.Timedelta(days=lookback_days)
            wide = wide[wide.index >= cutoff]
        if wide.empty:
            return 0

        rows = []
        for ts, row in wide.iterrows():
            values: dict[str, float] = {}
            for country, label_2y, label_10y in countries:
                y2 = row.get(label_2y)
                y10 = row.get(label_10y)
                if y2 is not None and not pd.isna(y2):
                    values[f"{country}_2y_daily"] = float(y2)
                if y10 is not None and not pd.isna(y10):
                    values[f"{country}_10y_daily"] = float(y10)
                # Slope only well-defined when both legs are present.
                if (y2 is not None and not pd.isna(y2)
                        and y10 is not None and not pd.isna(y10)):
                    values[f"{country}_slope_daily"] = float(y10) - float(y2)
            if not values:
                continue
            rows.append({
                "symbol":         symbol,
                "timestamp":      ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                "feature_group":  self.FEATURE_GROUP,
                "values":         values,
                "schema_version": self.SCHEMA_VERSION,
            })

        if not rows:
            return 0
        return await store.upsert_feature_store_bulk(
            rows, mode=("overwrite" if force else "skip"),
        )

    # ------------------------------------------------------------------
    # Historical engineered features (Phase 2A — read from feature_store)
    # ------------------------------------------------------------------

    async def get_historical_yield_features(
        self,
        store: "DataStore",
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        feeds_config: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical sovereign-yield features over a date range.

        Returns a DataFrame indexed by observation timestamp with the SAME
        feature columns ``get_yield_features(symbol)`` emits at live time
        (``{country}_2y_daily``, ``{country}_10y_daily``, ``{country}_slope_daily``
        per routed country).

        Lookahead-safety: each raw observation's timestamp is shifted forward
        by ``release_lag_hours`` (24h for stooq_yields, from data_feeds.yaml).
        Slopes recomputed from legs when missing in stored row, matching
        ``_emit_country_block``'s neutral-on-missing behavior.
        """
        from src.data_pipeline.feature_engineering import _load_data_feeds_yaml

        cfg = feeds_config if feeds_config is not None else _load_data_feeds_yaml()
        src_cfg = cfg.get("sources", {}).get(self.FEATURE_GROUP)
        if src_cfg is None:
            raise ValueError(
                f"feature_group {self.FEATURE_GROUP!r} not in data_feeds.yaml — "
                "refusing to query without a release-lag bound (lookahead risk)."
            )
        lag_hours = float(src_cfg.get("release_lag_hours") or 0.0)

        # Generous warmup so the daily grid forward-fills cleanly across
        # weekend / holiday gaps in any one country's series.
        raw_start = start - timedelta(days=60)
        raw_df = await store.read_feature_store(
            symbol=symbol,
            feature_group=self.FEATURE_GROUP,
            start=raw_start,
            end=end,
        )
        if raw_df.empty:
            logger.warning(
                "feature_store[%s] returned no rows for %s in [%s, %s] — "
                "training will see only zero defaults for this block",
                self.FEATURE_GROUP, symbol, raw_start, end,
            )
            return pd.DataFrame()

        raw_df = raw_df.copy()
        raw_df.index = raw_df.index + pd.Timedelta(hours=lag_hours)

        engineered = self._engineer_yield_features_from_raw(raw_df, symbol)
        if engineered.empty:
            return engineered

        engineered = engineered[
            (engineered.index >= pd.Timestamp(start))
            & (engineered.index <= pd.Timestamp(end))
        ]
        return engineered

    def _engineer_yield_features_from_raw(
        self, raw: pd.DataFrame, symbol: str,
    ) -> pd.DataFrame:
        """
        Compute the engineered feature columns from a wide raw-observation
        DataFrame (one column per ``{country}_{tenor}_daily`` label).

        Stooq's persist already pre-computes slopes when both legs are
        present, so most rows already carry them. For robustness, slope
        is recomputed from legs when it's absent, matching live behavior.
        """
        if raw.empty:
            return pd.DataFrame()

        sym_upper = symbol.upper()

        # Same routing as live get_yield_features. US-axis is always emitted.
        countries: list[str] = ["us"]
        if sym_upper in _GBP_EXPOSURE:
            countries.append("uk")
        if sym_upper in _EUR_EXPOSURE:
            countries.append("de")
        if sym_upper in _JPY_EXPOSURE:
            countries.append("jp")
        if sym_upper in _AUD_EXPOSURE:
            countries.append("au")
        if sym_upper in _NZD_EXPOSURE:
            countries.append("nz")

        idx = raw.index
        out = pd.DataFrame(index=idx)

        for country in countries:
            col_2y = f"{country}_2y_daily"
            col_10y = f"{country}_10y_daily"
            col_slope = f"{country}_slope_daily"

            y2 = raw.get(col_2y)
            y10 = raw.get(col_10y)
            slope = raw.get(col_slope)

            # Forward-fill legs across weekend/holiday gaps. Default 0.0
            # mirrors live ``_latest_close()`` returning 0.0 on a missing
            # series — which the slope guard then treats as neutral.
            y2_ffilled = (
                y2.reindex(idx).ffill().fillna(0.0).astype(float)
                if y2 is not None else pd.Series(0.0, index=idx, dtype=float)
            )
            y10_ffilled = (
                y10.reindex(idx).ffill().fillna(0.0).astype(float)
                if y10 is not None else pd.Series(0.0, index=idx, dtype=float)
            )
            out[col_2y] = y2_ffilled
            out[col_10y] = y10_ffilled

            # Use the persisted slope when available; otherwise reconstruct
            # from legs with the same neutral-on-missing rule the live
            # ``_emit_country_block`` applies (slope = 0 when either leg = 0).
            if slope is not None:
                slope_series = slope.reindex(idx).ffill()
                # Where the persisted slope is NaN, fall back to legs.
                fallback = (y10_ffilled - y2_ffilled).where(
                    (y2_ffilled != 0.0) & (y10_ffilled != 0.0), 0.0,
                )
                slope_series = slope_series.fillna(fallback).astype(float)
            else:
                slope_series = (y10_ffilled - y2_ffilled).where(
                    (y2_ffilled != 0.0) & (y10_ffilled != 0.0), 0.0,
                ).astype(float)
            out[col_slope] = slope_series

        return out.sort_index()

    # ------------------------------------------------------------------
    # Defaults (used by downstream when fetcher is unavailable)
    # ------------------------------------------------------------------

    @staticmethod
    def default_yield_features(symbol: str) -> dict[str, float]:
        """
        Neutral defaults when Stooq is unavailable (no key, network
        failure, etc.). Mirrors the routing of get_yield_features().
        """
        feats: dict[str, float] = {
            "us_2y_daily": 0.0, "us_10y_daily": 0.0, "us_slope_daily": 0.0,
        }
        sym_upper = symbol.upper()
        if sym_upper in _GBP_EXPOSURE:
            feats.update({"uk_2y_daily": 0.0, "uk_10y_daily": 0.0, "uk_slope_daily": 0.0})
        if sym_upper in _EUR_EXPOSURE:
            feats.update({"de_2y_daily": 0.0, "de_10y_daily": 0.0, "de_slope_daily": 0.0})
        if sym_upper in _JPY_EXPOSURE:
            feats.update({"jp_2y_daily": 0.0, "jp_10y_daily": 0.0, "jp_slope_daily": 0.0})
        if sym_upper in _AUD_EXPOSURE:
            feats.update({"au_2y_daily": 0.0, "au_10y_daily": 0.0, "au_slope_daily": 0.0})
        if sym_upper in _NZD_EXPOSURE:
            feats.update({"nz_2y_daily": 0.0, "nz_10y_daily": 0.0, "nz_slope_daily": 0.0})
        return feats
