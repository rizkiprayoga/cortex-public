"""
ecb_data.py — ECB Data Portal AAA yield curve via SDMX 2.1 REST.

The ECB publishes daily-estimated euro-area government bond yield curves
at noon Frankfurt time. We fetch the AAA spot curve (essentially German
Bunds — the euro-area risk-free benchmark) across 10 tenors from 3M to
30Y, giving us a full daily yield curve going back to 2004-09-06.

This complements existing EUR data:
  - macro_data (FRED): monthly ECB policy rate + Eurozone CPI
  - cross_asset (yfinance): DAX daily
  - stooq_data: DE 2Y / 10Y daily (narrower, two points)

This module adds: full euro-area AAA yield curve, 10 daily tenors from
which downstream features can compute curve shape (slope, butterfly,
level) — standard toolkit for modeling EUR rate-driven flows into
EUR-bearing pairs (EURUSD, EURGBP, EURJPY).

API notes
---------

- Endpoint: https://data-api.ecb.europa.eu/service/data/YC/{series_key}
- Format: ``?format=csvdata`` returns flat CSV with TIME_PERIOD + OBS_VALUE
  columns (plus ~38 metadata columns we discard on parse)
- **No API key required** — fully public, rate-limited by IP
- Plan (docs/forex_expansion_plan.md line 75) mentions the `ecbdata`
  Python lib, but the REST API is simple enough that we use requests
  directly. No new dependency.

Series key format for AAA spot curve
    YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{tenor}
where tenor ∈ {3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y}. Key parts:
    B       — Business frequency (daily business-week)
    U2      — Reference area: euro area (changing composition)
    EUR     — Currency
    4F      — Provider: ECB
    G_N_A   — Instrument: Government bond, nominal, triple-A issuers only
    SV_C_YM — Svensson model, continuous compounding, yield error min
    SR_{T}  — Spot rate at tenor T
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.data_pipeline.data_store import DataStore

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# Tenor → internal label + SDMX series suffix.
# Value = (suffix, tenor_years_for_sorting)
_TENORS: dict[str, tuple[str, float]] = {
    "3m":  ("SR_3M",  0.25),
    "6m":  ("SR_6M",  0.5),
    "1y":  ("SR_1Y",  1.0),
    "2y":  ("SR_2Y",  2.0),
    "3y":  ("SR_3Y",  3.0),
    "5y":  ("SR_5Y",  5.0),
    "7y":  ("SR_7Y",  7.0),
    "10y": ("SR_10Y", 10.0),
    "20y": ("SR_20Y", 20.0),
    "30y": ("SR_30Y", 30.0),
}

# Build the catalog mapping internal label → SDMX dimension-value key.
# NOTE: the URL path is ``/service/data/{FLOW}/{KEY}`` so the flow (YC)
# goes in the path segment, not the key. The key is just the dimension
# values joined by dots. The ECB Data Portal displays the full name with
# the YC prefix ("YC.B.U2...") but the REST API rejects that as a 400.
_SERIES_KEY_PREFIX = "B.U2.EUR.4F.G_N_A.SV_C_YM."
ECB_SERIES: dict[str, str] = {
    label: _SERIES_KEY_PREFIX + suffix
    for label, (suffix, _tenor) in _TENORS.items()
}

# Currency exposure — re-exported from the shared source of truth at
# fundamental/_currency_exposure.py.
from src.data_pipeline.fundamental._currency_exposure import (  # noqa: E402
    EUR_EXPOSURE as _EUR_EXPOSURE,
)

_ECB_BASE_URL = "https://data-api.ecb.europa.eu/service/data/YC/"
_HTTP_TIMEOUT_SEC = 20.0


class ECBDataFetcher:
    """
    ECB AAA yield curve fetcher. 10 tenors × daily from 2004-09-06.

    Usage:
        fetcher = ECBDataFetcher()
        df = fetcher.get_series("10y")                       # raw 10Y series
        feats = fetcher.get_yield_curve_features("EURUSD")   # full curve dict

    No API key required. Cache TTL defaults to 24h — ECB publishes once
    per trading day around noon Frankfurt time.
    """

    def __init__(self, cache_ttl_hours: float = 24.0):
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_ts: dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=cache_ttl_hours)

    # ------------------------------------------------------------------
    # Raw CSV fetch
    # ------------------------------------------------------------------

    def get_series(self, label: str) -> pd.DataFrame:
        """
        Fetch one tenor by internal label (e.g. "10y"). Returns a DataFrame
        indexed by date with a single "Close" column (renamed from
        ECB's OBS_VALUE for consistency with stooq_data).

        Returns empty DataFrame on any error — never raises.
        """
        if label not in ECB_SERIES:
            logger.warning("Unknown ECB tenor label: %s", label)
            return pd.DataFrame()

        ts = self._cache_ts.get(label)
        if ts and (datetime.utcnow() - ts) < self._cache_ttl:
            return self._cache[label]

        series_key = ECB_SERIES[label]
        url = _ECB_BASE_URL + series_key

        try:
            resp = requests.get(
                url,
                params={"format": "csvdata"},
                headers={"Accept": "text/csv"},
                timeout=_HTTP_TIMEOUT_SEC,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("ECB fetch failed for %s: %s", label, exc)
            return pd.DataFrame()

        try:
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:
            logger.warning("ECB CSV parse failed for %s: %s", label, exc)
            return pd.DataFrame()

        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            logger.warning("ECB returned unexpected schema for %s", label)
            return pd.DataFrame()

        # Keep only the two columns we need; rename for consistency.
        df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
        df = df.dropna(subset=["TIME_PERIOD"]).rename(
            columns={"TIME_PERIOD": "Date", "OBS_VALUE": "Close"}
        )
        df = df.set_index("Date").sort_index()
        # Coerce Close to numeric — ECB sometimes emits missing values as
        # blanks that pd.read_csv reads as NaN. Drop those.
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])

        self._cache[label] = df
        self._cache_ts[label] = datetime.utcnow()
        return df

    # ------------------------------------------------------------------
    # Per-symbol feature emission
    # ------------------------------------------------------------------

    def get_yield_curve_features(self, symbol: str) -> dict[str, float]:
        """
        Return daily AAA curve features for the trading symbol, routed by
        EUR exposure. Non-EUR pairs get an empty dict (ECB is a euro-area
        signal; USD/JPY/GBP-only pairs don't need it).

        Emitted keys (EUR pairs only):
            eu_aaa_3m_daily, eu_aaa_6m_daily, eu_aaa_1y_daily,
            eu_aaa_2y_daily, eu_aaa_3y_daily, eu_aaa_5y_daily,
            eu_aaa_7y_daily, eu_aaa_10y_daily, eu_aaa_20y_daily,
            eu_aaa_30y_daily
            eu_aaa_slope_2y10y       (10Y − 2Y, recession proxy)
            eu_aaa_slope_3m10y       (10Y − 3M, fed-analog slope)

        Downstream training can derive additional curve-shape features
        (butterfly, level, PC1) from the raw tenor readings.
        """
        if symbol.upper() not in _EUR_EXPOSURE:
            return {}

        features: dict[str, float] = {}
        for label in ECB_SERIES:
            features[f"eu_aaa_{label}_daily"] = self._latest_close(label)

        # Pre-derived slopes — convenient and always well-defined.
        y_2 = features.get("eu_aaa_2y_daily", 0.0)
        y_3m = features.get("eu_aaa_3m_daily", 0.0)
        y_10 = features.get("eu_aaa_10y_daily", 0.0)
        features["eu_aaa_slope_2y10y"] = (y_10 - y_2) if (y_2 and y_10) else 0.0
        features["eu_aaa_slope_3m10y"] = (y_10 - y_3m) if (y_3m and y_10) else 0.0
        return features

    def _latest_close(self, label: str) -> float:
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

    FEATURE_GROUP = "ecb_yield_curve"
    SCHEMA_VERSION = 1

    async def persist_raw_history_to_feature_store(
        self,
        store: "DataStore",
        *,
        force: bool = False,
        lookback_days: Optional[int] = None,
    ) -> int:
        """
        Write the full daily AAA yield curve to ``feature_store``.

        ECB AAA is symbol-independent — one curve, applies to every EUR pair.
        Writes under ``symbol="_GLOBAL"`` so no per-symbol duplication.

        Pulls each tenor's cached series, joins on date, forward-fills
        missing tenors, writes one feature_store row per unique date.

        Args:
            force: When True, switches the bulk upsert from ``DO NOTHING`` to
                ``DO UPDATE`` so upstream revisions land. Used by the
                weekly TTL safety-net job (scripts/ttl_check_feature_store.py).
            lookback_days: When set, only persist rows from the last N days.
                Used by the TTL job to avoid re-writing 21 yrs of history
                weekly. Default ``None`` = full history (initial backfill).

        Returns the number of rows touched.
        """
        return await self._persist_curve(
            store, symbol="_GLOBAL", force=force, lookback_days=lookback_days,
        )

    async def _persist_curve(
        self, store: "DataStore", *, symbol: str,
        force: bool = False, lookback_days: Optional[int] = None,
    ) -> int:
        # Build a wide DataFrame: index=date, columns=tenor labels.
        per_tenor: dict[str, pd.Series] = {}
        for label in ECB_SERIES:
            df = self.get_series(label)   # uses 24h cache; re-fetches if cold
            if df.empty:
                continue
            per_tenor[label] = df["Close"]

        if not per_tenor:
            logger.warning("ECB: no tenors available, skipping persist")
            return 0

        wide = pd.concat(per_tenor, axis=1).sort_index()
        # Forward-fill so every row has the latest known value for each tenor.
        # New tenors (e.g. 30Y added later) will leave leading NaN — drop those
        # rows where ALL tenors are still NaN.
        wide = wide.ffill().dropna(how="all")
        if lookback_days is not None and not wide.empty:
            cutoff = wide.index.max() - pd.Timedelta(days=lookback_days)
            wide = wide[wide.index >= cutoff]
        if wide.empty:
            return 0

        rows = []
        for ts, row in wide.iterrows():
            values = {}
            for label in ECB_SERIES:
                v = row.get(label)
                if v is None or pd.isna(v):
                    continue
                values[f"eu_aaa_{label}_daily"] = float(v)
            # Pre-derived slopes — same as get_yield_curve_features.
            y_2 = values.get("eu_aaa_2y_daily", 0.0)
            y_3m = values.get("eu_aaa_3m_daily", 0.0)
            y_10 = values.get("eu_aaa_10y_daily", 0.0)
            values["eu_aaa_slope_2y10y"] = (y_10 - y_2) if (y_2 and y_10) else 0.0
            values["eu_aaa_slope_3m10y"] = (y_10 - y_3m) if (y_3m and y_10) else 0.0

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

    async def get_historical_curve_features(
        self,
        store: "DataStore",
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        feeds_config: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical AAA-curve features over a date range.

        Returns a DataFrame indexed by observation timestamp with the SAME
        columns the live ``get_yield_curve_features(symbol)`` emits — 10
        tenor levels + 2 pre-derived slopes — but only for EUR-exposure
        symbols. Non-EUR symbols (USDJPY, USDCAD, GBPUSD, …) get an empty
        DataFrame, mirroring the live early-return for non-EUR pairs.

        ECB AAA is symbol-independent: data is stored under
        ``symbol="_GLOBAL"`` regardless of which EUR pair is asking. The
        same global rows feed every EUR pair's training run.

        Lookahead-safety: each raw observation's timestamp is shifted
        forward by ``release_lag_hours`` (24h for ecb_yield_curve, from
        data_feeds.yaml). Slopes recomputed from legs when missing in
        stored row, matching live's neutral-on-missing rule.
        """
        # Non-EUR symbols get nothing — matches the live early return.
        if symbol.upper() not in _EUR_EXPOSURE:
            return pd.DataFrame()

        from src.data_pipeline.feature_engineering import _load_data_feeds_yaml

        cfg = feeds_config if feeds_config is not None else _load_data_feeds_yaml()
        src_cfg = cfg.get("sources", {}).get(self.FEATURE_GROUP)
        if src_cfg is None:
            raise ValueError(
                f"feature_group {self.FEATURE_GROUP!r} not in data_feeds.yaml — "
                "refusing to query without a release-lag bound (lookahead risk)."
            )
        lag_hours = float(src_cfg.get("release_lag_hours") or 0.0)

        raw_start = start - timedelta(days=60)
        raw_df = await store.read_feature_store(
            symbol="_GLOBAL",   # ECB curve is symbol-independent
            feature_group=self.FEATURE_GROUP,
            start=raw_start,
            end=end,
        )
        if raw_df.empty:
            logger.warning(
                "feature_store[%s] returned no rows for _GLOBAL in [%s, %s] — "
                "training will see only zero defaults for this block",
                self.FEATURE_GROUP, raw_start, end,
            )
            return pd.DataFrame()

        raw_df = raw_df.copy()
        raw_df.index = raw_df.index + pd.Timedelta(hours=lag_hours)

        engineered = self._engineer_curve_features_from_raw(raw_df)
        if engineered.empty:
            return engineered

        engineered = engineered[
            (engineered.index >= pd.Timestamp(start))
            & (engineered.index <= pd.Timestamp(end))
        ]
        return engineered

    def _engineer_curve_features_from_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the engineered feature columns from a wide raw-observation
        DataFrame (one column per ``eu_aaa_{tenor}_daily`` label, plus
        the two pre-computed slopes).

        Mirrors live ``get_yield_curve_features`` exactly for the symbol-
        independent EUR-pair case: 10 tenor levels forward-filled, slopes
        recomputed from 2y/3m/10y legs when missing.
        """
        if raw.empty:
            return pd.DataFrame()

        idx = raw.index
        out = pd.DataFrame(index=idx)

        # 10 tenor levels — forward-fill across business-day gaps,
        # default 0.0 on missing tenor (matches live ``_latest_close``).
        for label in ECB_SERIES:
            col = f"eu_aaa_{label}_daily"
            raw_col = raw.get(col)
            out[col] = (
                raw_col.reindex(idx).ffill().fillna(0.0).astype(float)
                if raw_col is not None else pd.Series(0.0, index=idx, dtype=float)
            )

        # Slopes: prefer persisted; fall back to leg arithmetic with
        # the same neutral-on-missing rule as live.
        y_2 = out["eu_aaa_2y_daily"]
        y_3m = out["eu_aaa_3m_daily"]
        y_10 = out["eu_aaa_10y_daily"]

        for slope_col, leg_short, y_short in (
            ("eu_aaa_slope_2y10y", y_2, y_2),
            ("eu_aaa_slope_3m10y", y_3m, y_3m),
        ):
            persisted = raw.get(slope_col)
            fallback = (y_10 - y_short).where(
                (y_short != 0.0) & (y_10 != 0.0), 0.0,
            )
            if persisted is not None:
                slope = persisted.reindex(idx).ffill().fillna(fallback)
            else:
                slope = fallback
            out[slope_col] = slope.astype(float)

        return out.sort_index()

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @staticmethod
    def default_yield_curve_features(symbol: str) -> dict[str, float]:
        """Neutral defaults when ECB is unavailable."""
        if symbol.upper() not in _EUR_EXPOSURE:
            return {}
        feats: dict[str, float] = {
            f"eu_aaa_{label}_daily": 0.0 for label in ECB_SERIES
        }
        feats["eu_aaa_slope_2y10y"] = 0.0
        feats["eu_aaa_slope_3m10y"] = 0.0
        return feats
