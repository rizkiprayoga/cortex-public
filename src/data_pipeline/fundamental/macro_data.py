"""
macro_data.py — Macroeconomic Data (FRED API)

Fetches macroeconomic indicators relevant to Gold and Bitcoin pricing:

    For Gold (XAUUSD):
        - DXY (US Dollar Index) — inverse correlation with gold
        - Fed Funds Rate (FEDFUNDS) — interest rate expectations
        - CPI YoY (CPIAUCSL) — inflation proxy
        - 10Y Treasury Yield (DGS10) — real yield = yield - CPI
        - Real Yield (DFII10) — direct inflation-adjusted yield

    For Bitcoin (BTCUSD):
        - M2 Money Supply (M2SL) — liquidity indicator
        - Fed Funds Rate — risk-off sentiment
        - VIX (via FRED VIXCLS) — risk appetite

    Additional (both symbols):
        - Yield curve slope (T10Y2Y) — recession indicator
        - Breakeven inflation (T10YIE) — inflation expectations
        - High yield credit spread (BAMLH0A0HYM2) — credit stress
        - Initial claims (ICSA) — labor market health
        - Fed balance sheet (WALCL) — monetary policy proxy

API: FRED (Federal Reserve Economic Data)
    Free key at: https://fred.stlouisfed.org/docs/api/api_key.html
    Python client: fredapi

Output:
    - get_macro_score(): backward-compatible single float in [-1, 1]
    - get_macro_features(): dict of ~12 individual features for ML pipeline
"""

import logging
import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.data_pipeline.data_store import DataStore

import numpy as np
import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)

FRED_SERIES = {
    # --- Common (all symbols) ---
    "dxy":                "DTWEXBGS",       # USD index (broad)
    "fed_funds":          "FEDFUNDS",       # Federal Funds rate
    "cpi_yoy":            "CPIAUCSL",       # CPI (all urban, US)
    "t10y":               "DGS10",          # 10Y Treasury yield
    "real_yield":         "DFII10",         # 10Y TIPS yield
    "m2":                 "M2SL",           # M2 money supply
    "vix":                "VIXCLS",         # VIX volatility
    "yield_curve":        "T10Y2Y",         # 10Y-2Y spread
    "breakeven_inflation": "T10YIE",        # 10Y breakeven inflation
    "hy_spread":          "BAMLH0A0HYM2",  # HY credit spread
    "initial_claims":     "ICSA",           # Weekly initial claims
    "fed_balance_sheet":  "WALCL",          # Fed total assets
    "t2y":                "DGS2",           # 2Y Treasury yield
    "t3m":                "DTB3",           # 3-month T-bill
    # --- EUR (EURUSD, EURGBP, EURJPY) ---
    "ecb_rate":           "ECBDFR",         # ECB deposit facility rate
    "eu_cpi_yoy":         "CP0000EZ19M086NEST",  # Eurozone HICP YoY
    "eu_unemployment":    "LRHUTTTTEZM156S",     # Eurozone unemployment rate
    # --- JPY (USDJPY, EURJPY, GBPJPY) ---
    "boj_rate":           "IRSTCB01JPM156N",     # BoJ policy rate (call rate)
    "japan_cpi_yoy":      "JPNCPIALLMINMEI",     # Japan CPI all items
    # --- CAD (USDCAD) ---
    "boc_rate":           "IRSTCB01CAM156N",     # BoC policy rate (call rate)
    "canada_cpi_yoy":     "CANCPIALLMINMEI",     # Canada CPI all items
    "wti_oil":            "DCOILWTICO",          # WTI Crude Oil (for USDCAD)
    # --- GBP (GBPUSD, EURGBP, GBPJPY) ---
    # boe_rate: was IRSTCB01GBM156N (OECD short rate) — discontinued at FRED.
    # IUDSOIA = SONIA, daily, the BoE-anchored overnight rate (Bank Rate proxy).
    # SONIA tracks Bank Rate to within a few bps; close enough for regime
    # modeling and gives daily resolution instead of OECD's monthly cadence.
    "boe_rate":           "IUDSOIA",             # SONIA (BoE proxy, daily)
    "uk_10y":             "IRLTLT01GBM156N",     # UK 10Y Gilt yield (OECD)
    "uk_cpi_yoy":         "GBRCPIALLMINMEI",     # UK CPI all items
    # --- AUD (AUDUSD, AUDNZD) ---
    # rba_rate: was IRSTCB01AUM156N — discontinued. Switched to OECD's
    # "immediate rates" series for AU which is the closest available
    # cash-rate proxy at FRED (monthly).
    "rba_rate":           "IRSTCI01AUM156N",     # RBA cash rate proxy (OECD, monthly)
    "au_10y":             "IRLTLT01AUM156N",     # Australia 10Y bond (OECD)
    "au_cpi_yoy":         "AUSCPIALLQINMEI",     # Australia CPI (quarterly)
    # AU is commodity-heavy with China as #1 trading partner. These three
    # feed into the AUD block via the China-demand channel: iron ore is
    # AU's largest export, CNY strength translates into AUD flow, and
    # Chinese inflation is the leading indicator for commodity demand.
    "china_cpi_yoy":      "CHNCPIALLMINMEI",     # China CPI (all items)
    "cny_usd":            "EXCHUS",              # China / US FX (CNY per USD)
    "iron_ore":           "PIORECRUSDM",         # Global iron ore spot (USD/mt, monthly)
    # --- NZD (AUDNZD) ---
    # rbnz_rate: was IR3TCR01NZM156N — discontinued. Switched to OECD's
    # 3-month interbank rate for NZ — the closest still-published proxy
    # for the RBNZ Official Cash Rate at FRED (monthly).
    "rbnz_rate":          "IR3TIB01NZM156N",     # RBNZ rate proxy (OECD 3M interbank, monthly)
    "nz_10y":             "IRLTLT01NZM156N",     # NZ 10Y bond (OECD)
    "nz_cpi_yoy":         "NZLCPIALLQINMEI",     # NZ CPI (quarterly)
}

# Currency-exposure sets — re-exported under the legacy private names from
# the shared source of truth at fundamental/_currency_exposure.py.
from src.data_pipeline.fundamental._currency_exposure import (  # noqa: E402
    AUD_EXPOSURE as _AUD_EXPOSURE,
    CAD_EXPOSURE as _CAD_EXPOSURE,
    EUR_EXPOSURE as _EUR_EXPOSURE,
    GBP_EXPOSURE as _GBP_EXPOSURE,
    JPY_EXPOSURE as _JPY_EXPOSURE,
    NZD_EXPOSURE as _NZD_EXPOSURE,
)


class MacroDataFetcher:
    """
    Fetches and normalizes macroeconomic data from FRED.

    Usage:
        fetcher = MacroDataFetcher()
        score = fetcher.get_macro_score("XAUUSD")
        features = fetcher.get_macro_features("XAUUSD")
    """

    def __init__(self):
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise EnvironmentError("FRED_API_KEY not set in environment")
        self.fred = Fred(api_key=api_key)
        self._cache: dict[str, pd.Series] = {}
        self._cache_ts: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=4)

    def get_series(self, series_id: str, lookback_days: int = 365) -> pd.Series:
        """
        Fetch a FRED time series for the past lookback_days days.

        Args:
            series_id:    FRED series identifier (e.g. "FEDFUNDS")
            lookback_days: Days of history to retrieve

        Returns:
            pd.Series with DatetimeIndex.
        """
        start = datetime.utcnow() - timedelta(days=lookback_days)
        try:
            data = self.fred.get_series(series_id, observation_start=start)
            if data is None or data.empty:
                logger.warning("FRED returned no data for %s", series_id)
                return pd.Series(dtype=float)
            return data.dropna()
        except Exception as exc:
            logger.error("FRED fetch failed for %s: %s", series_id, exc)
            return pd.Series(dtype=float)

    def _get_cached(self, key: str) -> pd.Series:
        """Get a FRED series with caching."""
        now = datetime.utcnow()
        if (
            self._cache_ts
            and (now - self._cache_ts) < self._cache_ttl
            and key in self._cache
        ):
            return self._cache[key]

        series_id = FRED_SERIES.get(key)
        if not series_id:
            return pd.Series(dtype=float)

        data = self.get_series(series_id, lookback_days=365)
        self._cache[key] = data
        self._cache_ts = now
        return data

    def get_macro_score(self, symbol: str) -> float:
        """
        Compute a composite macro score (backward compatible).

        XAUUSD: weighted combination of DXY, real yield, CPI signals.
        BTCUSD: weighted combination of M2, Fed Funds, VIX signals.

        Returns:
            Float in [-1, 1] — positive = bullish conditions.
        """
        try:
            if symbol.upper().startswith("XAU"):
                dxy = self._normalize_series(self._get_cached("dxy"))
                real_y = self._normalize_series(self._get_cached("real_yield"))
                cpi = self._normalize_series(self._get_cached("cpi_yoy"))
                score = -0.35 * dxy + -0.35 * real_y + 0.30 * cpi

            elif symbol.upper().startswith("BTC"):
                m2 = self._normalize_series(self._get_cached("m2"))
                fed = self._normalize_series(self._get_cached("fed_funds"))
                vix = self._normalize_series(self._get_cached("vix"))
                score = 0.40 * m2 + -0.30 * fed + -0.30 * vix

            else:
                return 0.0

            return float(max(-1.0, min(1.0, score)))

        except Exception as exc:
            logger.error("Macro score failed for %s: %s", symbol, exc)
            return 0.0

    def get_macro_features(self, symbol: str) -> dict[str, float]:
        """
        Return individual macro features for ML pipeline.

        Returns ~12 features as a dict for any symbol.
        """
        try:
            features: dict[str, float] = {}

            # Fed Funds rate level and recent change
            fed = self._get_cached("fed_funds")
            features["fed_funds_level"] = self._latest_val(fed, default=5.0)
            features["fed_funds_change"] = self._recent_change(fed)

            # CPI YoY z-score
            cpi = self._get_cached("cpi_yoy")
            features["cpi_yoy_zscore"] = self._normalize_series(cpi)

            # Real yield level
            ry = self._get_cached("real_yield")
            features["real_yield_level"] = self._latest_val(ry, default=2.0)

            # M2 money supply year-over-year change
            m2 = self._get_cached("m2")
            features["m2_yoy_change"] = self._yoy_change(m2)

            # Yield curve slope (10Y - 2Y)
            yc = self._get_cached("yield_curve")
            features["yield_curve_slope"] = self._latest_val(yc, default=0.0)

            # Breakeven inflation
            bei = self._get_cached("breakeven_inflation")
            features["breakeven_inflation"] = self._latest_val(bei, default=2.0)

            # HY credit spread
            hy = self._get_cached("hy_spread")
            features["hy_credit_spread"] = self._latest_val(hy, default=4.0)

            # Initial claims z-score
            claims = self._get_cached("initial_claims")
            features["initial_claims_zscore"] = self._normalize_series(claims)

            # Fed balance sheet change (3-month)
            fbs = self._get_cached("fed_balance_sheet")
            features["fed_balance_sheet_change"] = self._recent_pct_change(fbs, periods=12)

            # DXY z-score (useful for all symbols)
            dxy = self._get_cached("dxy")
            features["dxy_zscore"] = self._normalize_series(dxy)

            # VIX level
            vix = self._get_cached("vix")
            features["vix_fred_level"] = self._latest_val(vix, default=20.0)

            sym_upper = symbol.upper()

            # --- EUR macro block ---
            # Fires for EURUSD + crosses (EURGBP, EURJPY). ECB rate and EU
            # inflation/unemployment drive EUR-side pricing regardless of
            # what's on the other side of the pair.
            if sym_upper in _EUR_EXPOSURE:
                ecb = self._get_cached("ecb_rate")
                features["ecb_rate_level"] = self._latest_val(ecb, default=3.0)
                eu_cpi = self._get_cached("eu_cpi_yoy")
                features["eu_cpi_yoy_zscore"] = self._normalize_series(eu_cpi)
                eu_unemp = self._get_cached("eu_unemployment")
                features["eu_unemployment_level"] = self._latest_val(eu_unemp, default=6.5)
                # Fed-ECB diff is only meaningful for EURUSD. For crosses,
                # the diff involving the OTHER currency is emitted below.
                if sym_upper in ("EURUSD", "EUR/USD"):
                    features["eur_usd_rate_diff"] = (
                        features["fed_funds_level"] - features["ecb_rate_level"]
                    )

            # --- JPY macro block ---
            # Fires for USDJPY + crosses (EURJPY, GBPJPY). BoJ rate and JP
            # CPI are foundational for all JPY-bearing pairs.
            if sym_upper in _JPY_EXPOSURE:
                boj = self._get_cached("boj_rate")
                features["boj_rate_level"] = self._latest_val(boj, default=0.5)
                jp_cpi = self._get_cached("japan_cpi_yoy")
                features["japan_cpi_yoy_zscore"] = self._normalize_series(jp_cpi)
                if sym_upper in ("USDJPY", "USD/JPY"):
                    features["usd_jpy_rate_diff"] = (
                        features["fed_funds_level"] - features["boj_rate_level"]
                    )
                    # Classic carry trade indicator — only for USDJPY; cross
                    # pairs have their own carry below.
                    features["carry_trade_indicator"] = features["usd_jpy_rate_diff"]

            # --- CAD macro block ---
            # Canada is major oil exporter — USDCAD inversely correlated
            # with oil. Rate spread (Fed-BoC) drives multi-month trends.
            if sym_upper in _CAD_EXPOSURE:
                boc = self._get_cached("boc_rate")
                features["boc_rate_level"] = self._latest_val(boc, default=3.0)
                features["usd_cad_rate_diff"] = (
                    features["fed_funds_level"] - features["boc_rate_level"]
                )
                ca_cpi = self._get_cached("canada_cpi_yoy")
                features["canada_cpi_yoy_zscore"] = self._normalize_series(ca_cpi)
                oil = self._get_cached("wti_oil")
                features["wti_oil_zscore"] = self._normalize_series(oil)

            # --- GBP macro block ---
            # Fires for GBPUSD + crosses (EURGBP, GBPJPY). UK has its own
            # yield curve worth tracking (Gilt 10Y = BoE expectations proxy).
            if sym_upper in _GBP_EXPOSURE:
                boe = self._get_cached("boe_rate")
                features["boe_rate_level"] = self._latest_val(boe, default=4.0)
                uk_10y = self._get_cached("uk_10y")
                features["uk_10y_level"] = self._latest_val(uk_10y, default=3.5)
                uk_cpi = self._get_cached("uk_cpi_yoy")
                features["uk_cpi_yoy_zscore"] = self._normalize_series(uk_cpi)
                if sym_upper in ("GBPUSD", "GBP/USD"):
                    features["gbp_usd_rate_diff"] = (
                        features["fed_funds_level"] - features["boe_rate_level"]
                    )

            # --- AUD macro block ---
            # Fires for AUDUSD + AUDNZD. RBA cash rate + AU 10Y yield.
            # Commodity-linked currency — includes the China-demand channel
            # via iron ore + Chinese CPI + CNY/USD.
            if sym_upper in _AUD_EXPOSURE:
                rba = self._get_cached("rba_rate")
                features["rba_rate_level"] = self._latest_val(rba, default=4.0)
                au_10y = self._get_cached("au_10y")
                features["au_10y_level"] = self._latest_val(au_10y, default=4.0)
                au_cpi = self._get_cached("au_cpi_yoy")
                features["au_cpi_yoy_zscore"] = self._normalize_series(au_cpi)
                # China / commodity channel — AU's #1 trading partner drives
                # AUD even when US-AU rate differentials don't move.
                iron = self._get_cached("iron_ore")
                features["iron_ore_zscore"] = self._normalize_series(iron)
                china_cpi = self._get_cached("china_cpi_yoy")
                features["china_cpi_yoy_zscore"] = self._normalize_series(china_cpi)
                cny = self._get_cached("cny_usd")
                features["cny_usd_zscore"] = self._normalize_series(cny)
                if sym_upper in ("AUDUSD", "AUD/USD"):
                    features["aud_usd_rate_diff"] = (
                        features["fed_funds_level"] - features["rba_rate_level"]
                    )

            # --- NZD macro block ---
            # Only AUDNZD exposes NZD in our universe. RBNZ OCR + NZ 10Y.
            if sym_upper in _NZD_EXPOSURE:
                rbnz = self._get_cached("rbnz_rate")
                features["rbnz_rate_level"] = self._latest_val(rbnz, default=4.0)
                nz_10y = self._get_cached("nz_10y")
                features["nz_10y_level"] = self._latest_val(nz_10y, default=4.0)
                nz_cpi = self._get_cached("nz_cpi_yoy")
                features["nz_cpi_yoy_zscore"] = self._normalize_series(nz_cpi)

            # --- Cross-pair rate differentials ---
            # For crosses, the relevant rate diff is between the two QUOTE
            # currencies, not either vs USD. Emit one diff per cross.
            if sym_upper in ("EURGBP", "EUR/GBP"):
                features["eur_gbp_rate_diff"] = (
                    features["ecb_rate_level"] - features["boe_rate_level"]
                )
            if sym_upper in ("EURJPY", "EUR/JPY"):
                features["eur_jpy_rate_diff"] = (
                    features["ecb_rate_level"] - features["boj_rate_level"]
                )
                features["carry_trade_indicator"] = features["eur_jpy_rate_diff"]
            if sym_upper in ("GBPJPY", "GBP/JPY"):
                features["gbp_jpy_rate_diff"] = (
                    features["boe_rate_level"] - features["boj_rate_level"]
                )
                features["carry_trade_indicator"] = features["gbp_jpy_rate_diff"]
            if sym_upper in ("AUDNZD", "AUD/NZD"):
                features["aud_nzd_rate_diff"] = (
                    features["rba_rate_level"] - features["rbnz_rate_level"]
                )

            return features

        except Exception as exc:
            logger.error("Macro features failed for %s: %s", symbol, exc)
            return self._default_features()

    def _normalize_series(self, series: pd.Series, lookback: int = 60) -> float:
        """
        Z-score the latest value relative to recent history,
        then clip to [-1, 1].
        """
        if series is None or len(series) < 2:
            return 0.0

        recent = series.tail(lookback)
        mean = recent.mean()
        std = recent.std()
        if std == 0 or pd.isna(std):
            return 0.0

        z = (recent.iloc[-1] - mean) / std
        return float(max(-1.0, min(1.0, z)))

    @staticmethod
    def _latest_val(series: pd.Series, default: float = 0.0) -> float:
        """Return latest non-NaN value from series."""
        if series is None or series.empty:
            return default
        val = series.iloc[-1]
        return float(val) if not pd.isna(val) else default

    @staticmethod
    def _recent_change(series: pd.Series, periods: int = 1) -> float:
        """Return the absolute change over the last `periods` observations."""
        if series is None or len(series) < periods + 1:
            return 0.0
        return float(series.iloc[-1] - series.iloc[-1 - periods])

    @staticmethod
    def _recent_pct_change(series: pd.Series, periods: int = 1) -> float:
        """Return percentage change over last `periods` observations."""
        if series is None or len(series) < periods + 1:
            return 0.0
        prev = series.iloc[-1 - periods]
        if prev == 0 or pd.isna(prev):
            return 0.0
        return float((series.iloc[-1] - prev) / abs(prev))

    @staticmethod
    def _yoy_change(series: pd.Series) -> float:
        """Compute year-over-year percentage change."""
        if series is None or len(series) < 12:
            return 0.0
        # Monthly data: look back ~12 observations for YoY
        current = series.iloc[-1]
        # Find the observation closest to 12 months ago
        target_date = series.index[-1] - timedelta(days=365)
        idx = series.index.get_indexer([target_date], method="nearest")[0]
        if idx < 0 or idx >= len(series):
            return 0.0
        prev = series.iloc[idx]
        if prev == 0 or pd.isna(prev):
            return 0.0
        return float((current - prev) / abs(prev))

    # ------------------------------------------------------------------
    # feature_store backfill (Phase 1F)
    # ------------------------------------------------------------------

    FEATURE_GROUP = "fred_macro"
    SCHEMA_VERSION = 1

    # Series labels routed by currency exposure. Mirrors the if-blocks in
    # get_macro_features (). Stored centrally so the persist path uses
    # exactly the same set without re-implementing the routing logic.
    _COMMON_SERIES = (
        "fed_funds", "cpi_yoy", "real_yield", "m2", "yield_curve",
        "breakeven_inflation", "hy_spread", "initial_claims",
        "fed_balance_sheet", "dxy", "vix", "t10y", "t2y", "t3m",
    )
    _EUR_SERIES = ("ecb_rate", "eu_cpi_yoy", "eu_unemployment")
    _JPY_SERIES = ("boj_rate", "japan_cpi_yoy")
    _CAD_SERIES = ("boc_rate", "canada_cpi_yoy", "wti_oil")
    _GBP_SERIES = ("boe_rate", "uk_10y", "uk_cpi_yoy")
    _AUD_SERIES = ("rba_rate", "au_10y", "au_cpi_yoy",
                   "iron_ore", "china_cpi_yoy", "cny_usd")
    _NZD_SERIES = ("rbnz_rate", "nz_10y", "nz_cpi_yoy")

    def _routed_series_labels(self, symbol: str) -> list[str]:
        """Return FRED series labels relevant to this symbol's exposure."""
        sym_upper = symbol.upper()
        labels = list(self._COMMON_SERIES)
        if sym_upper in _EUR_EXPOSURE:
            labels.extend(self._EUR_SERIES)
        if sym_upper in _JPY_EXPOSURE:
            labels.extend(self._JPY_SERIES)
        if sym_upper in _CAD_EXPOSURE:
            labels.extend(self._CAD_SERIES)
        if sym_upper in _GBP_EXPOSURE:
            labels.extend(self._GBP_SERIES)
        if sym_upper in _AUD_EXPOSURE:
            labels.extend(self._AUD_SERIES)
        if sym_upper in _NZD_EXPOSURE:
            labels.extend(self._NZD_SERIES)
        return labels

    # Backfill lookback: 25 yrs covers Phase 2's planned 2010-2026 backtest
    # window with a comfortable warmup. The live fetcher's _get_cached
    # uses a 365-day window (sufficient for z-score + YoY); persist needs
    # a much longer history.
    BACKFILL_LOOKBACK_DAYS = 365 * 25

    async def persist_raw_history_to_feature_store(
        self, store: "DataStore", symbol: str,
        *,
        force: bool = False,
        lookback_days: Optional[int] = None,
    ) -> int:
        """
        Write the routed FRED series for this symbol to ``feature_store``
        as a per-date snapshot.

        Stores RAW observation values, not z-scores or normalizations —
        downstream consumers re-derive (z-scores, YoY changes) from the
        raw history. This keeps the cache stable across feature-engineering
        changes and matches the operator's collect-breadth mandate.

        Snapshot model: one row per (symbol, date) with the latest known
        raw value of each routed series forward-filled to that date.

        Args:
            force: When True, switches conflict policy from DO NOTHING to
                DO UPDATE — used by the weekly TTL safety-net job.
            lookback_days: When set, only persist rows from the last N days.
                Default None = full 25-yr history (initial backfill mode).

        Bypasses ``_get_cached`` (365-day lookback) and calls ``get_series``
        directly with a 25-yr lookback so backfill covers the historical
        window Phase 2 backtests need.
        """
        labels = self._routed_series_labels(symbol)

        per_label: dict[str, pd.Series] = {}
        for label in labels:
            series_id = FRED_SERIES.get(label)
            if not series_id:
                continue
            series = self.get_series(series_id, lookback_days=self.BACKFILL_LOOKBACK_DAYS)
            if series is None or series.empty:
                continue
            per_label[label] = series
        if not per_label:
            logger.warning("FRED: no routed series for %s, skipping", symbol)
            return 0

        wide = pd.concat(per_label, axis=1).sort_index().ffill().dropna(how="all")
        if lookback_days is not None and not wide.empty:
            cutoff = wide.index.max() - pd.Timedelta(days=lookback_days)
            wide = wide[wide.index >= cutoff]
        if wide.empty:
            return 0

        rows = []
        for ts, row in wide.iterrows():
            values = {
                label: float(v)
                for label, v in row.items()
                if v is not None and not pd.isna(v)
            }
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

    # Rolling-window length for z-score normalization. Matches the live
    # path's _normalize_series(lookback=60) — 60 most recent observations,
    # natural cadence (so monthly CPI gets 5-yr window, daily DXY gets 3-mo).
    _ZSCORE_LOOKBACK = 60

    # Period for YoY-style changes on monthly cadence series. Matches live
    # _yoy_change which targets ~365 days back via index nearest-match.
    _YOY_PERIODS_MONTHLY = 12

    async def get_historical_macro_features(
        self,
        store: "DataStore",
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        feeds_config: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical macro features over a date range from feature_store.

        Returns a DataFrame indexed by observation timestamp with the SAME
        feature columns ``get_macro_features(symbol)`` emits at live time.
        Used by the LSTM training pipeline to inject historical macro context
        in place of zero-fill placeholders, eliminating train/serve skew.

        Lookahead-safety: each raw observation's timestamp is shifted forward
        by ``release_lag_hours`` (from ``config/data_feeds.yaml``) so a feature
        row at time ``t`` only reflects FRED data that would have been
        publicly knowable by ``t``. Rolling z-scores / YoY changes are
        point-in-time-correct by construction (pandas rolling windows).

        Args:
            store:        Async ``DataStore`` instance (already connected).
            symbol:       Trading symbol — drives currency-conditional blocks.
            start, end:   Naive UTC date range (inclusive).
            feeds_config: Optional pre-loaded ``data_feeds.yaml`` dict (tests).

        Returns:
            DataFrame indexed by observation timestamp, sorted ascending.
            Columns match the live ``get_macro_features(symbol)`` keys.
            Empty DataFrame if feature_store has no rows for this symbol.
        """
        # Lazy import to avoid circular import (feature_engineering imports
        # from data_pipeline modules; data_pipeline doesn't import from it).
        from src.data_pipeline.feature_engineering import _load_data_feeds_yaml

        cfg = feeds_config if feeds_config is not None else _load_data_feeds_yaml()
        src_cfg = cfg.get("sources", {}).get(self.FEATURE_GROUP)
        if src_cfg is None:
            raise ValueError(
                f"feature_group {self.FEATURE_GROUP!r} not in data_feeds.yaml — "
                "refusing to query without a release-lag bound (lookahead risk)."
            )
        lag_hours = float(src_cfg.get("release_lag_hours") or 0.0)

        # Read raw history with a generous warmup so the first emitted bar
        # has clean rolling stats. 400 days >> 60-obs daily window and the
        # ~12-period YoY span on monthly series.
        raw_start = start - timedelta(days=400)
        raw_df = await store.read_feature_store(
            symbol=symbol,
            feature_group=self.FEATURE_GROUP,
            start=raw_start,
            end=end,
        )
        if raw_df.empty:
            logger.warning(
                "feature_store[%s] returned no rows for %s in [%s, %s] — "
                "training will see only defaults for this block",
                self.FEATURE_GROUP, symbol, raw_start, end,
            )
            return pd.DataFrame()

        # Apply release-lag shift: each raw observation becomes "knowable"
        # only after timestamp + release_lag_hours. After this shift, a
        # rolling window over the resulting series is automatically
        # lookahead-safe — feature[t] uses only original observations whose
        # publication date <= t.
        raw_df = raw_df.copy()
        raw_df.index = raw_df.index + pd.Timedelta(hours=lag_hours)

        engineered = self._engineer_macro_features_from_raw(raw_df, symbol)
        if engineered.empty:
            return engineered

        # Trim to requested range; rolling-warmup rows before `start` were
        # only there to seed the stats. Keep the final emitted index inside
        # [start, end].
        engineered = engineered[
            (engineered.index >= pd.Timestamp(start))
            & (engineered.index <= pd.Timestamp(end))
        ]
        return engineered

    def _engineer_macro_features_from_raw(
        self, raw: pd.DataFrame, symbol: str,
    ) -> pd.DataFrame:
        """
        Compute the engineered feature columns from a wide raw-observation
        DataFrame (one column per FRED label).

        Mirrors the live ``get_macro_features()`` schema exactly: same column
        names, same default-on-missing values from ``_default_features()``,
        same z-score lookback, same currency-conditional blocks routed via
        the consolidated _currency_exposure module.
        """
        if raw.empty:
            return pd.DataFrame()

        sym_upper = symbol.upper()
        idx = raw.index
        out = pd.DataFrame(index=idx)
        defaults = self._default_features()

        # ----- Common (all symbols) ----------------------------------------
        out["fed_funds_level"] = self._level_or_default(
            raw.get("fed_funds"), defaults["fed_funds_level"], idx,
        )
        out["fed_funds_change"] = self._diff_or_zero(raw.get("fed_funds"), idx)
        out["cpi_yoy_zscore"] = self._rolling_zscore_or_zero(
            raw.get("cpi_yoy"), self._ZSCORE_LOOKBACK, idx,
        )
        out["real_yield_level"] = self._level_or_default(
            raw.get("real_yield"), defaults["real_yield_level"], idx,
        )
        out["m2_yoy_change"] = self._yoy_change_or_zero(
            raw.get("m2"), self._YOY_PERIODS_MONTHLY, idx,
        )
        out["yield_curve_slope"] = self._level_or_default(
            raw.get("yield_curve"), defaults["yield_curve_slope"], idx,
        )
        out["breakeven_inflation"] = self._level_or_default(
            raw.get("breakeven_inflation"), defaults["breakeven_inflation"], idx,
        )
        out["hy_credit_spread"] = self._level_or_default(
            raw.get("hy_spread"), defaults["hy_credit_spread"], idx,
        )
        out["initial_claims_zscore"] = self._rolling_zscore_or_zero(
            raw.get("initial_claims"), self._ZSCORE_LOOKBACK, idx,
        )
        out["fed_balance_sheet_change"] = self._yoy_change_or_zero(
            raw.get("fed_balance_sheet"), self._YOY_PERIODS_MONTHLY, idx,
        )
        out["dxy_zscore"] = self._rolling_zscore_or_zero(
            raw.get("dxy"), self._ZSCORE_LOOKBACK, idx,
        )
        out["vix_fred_level"] = self._level_or_default(
            raw.get("vix"), defaults["vix_fred_level"], idx,
        )

        # ----- EUR block ---------------------------------------------------
        if sym_upper in _EUR_EXPOSURE:
            out["ecb_rate_level"] = self._level_or_default(
                raw.get("ecb_rate"), 3.0, idx,
            )
            out["eu_cpi_yoy_zscore"] = self._rolling_zscore_or_zero(
                raw.get("eu_cpi_yoy"), self._ZSCORE_LOOKBACK, idx,
            )
            out["eu_unemployment_level"] = self._level_or_default(
                raw.get("eu_unemployment"), 6.5, idx,
            )
            if sym_upper in ("EURUSD", "EUR/USD"):
                out["eur_usd_rate_diff"] = (
                    out["fed_funds_level"] - out["ecb_rate_level"]
                )

        # ----- JPY block ---------------------------------------------------
        if sym_upper in _JPY_EXPOSURE:
            out["boj_rate_level"] = self._level_or_default(
                raw.get("boj_rate"), 0.5, idx,
            )
            out["japan_cpi_yoy_zscore"] = self._rolling_zscore_or_zero(
                raw.get("japan_cpi_yoy"), self._ZSCORE_LOOKBACK, idx,
            )
            if sym_upper in ("USDJPY", "USD/JPY"):
                out["usd_jpy_rate_diff"] = (
                    out["fed_funds_level"] - out["boj_rate_level"]
                )
                out["carry_trade_indicator"] = out["usd_jpy_rate_diff"]

        # ----- CAD block ---------------------------------------------------
        if sym_upper in _CAD_EXPOSURE:
            out["boc_rate_level"] = self._level_or_default(
                raw.get("boc_rate"), 3.0, idx,
            )
            out["usd_cad_rate_diff"] = (
                out["fed_funds_level"] - out["boc_rate_level"]
            )
            out["canada_cpi_yoy_zscore"] = self._rolling_zscore_or_zero(
                raw.get("canada_cpi_yoy"), self._ZSCORE_LOOKBACK, idx,
            )
            out["wti_oil_zscore"] = self._rolling_zscore_or_zero(
                raw.get("wti_oil"), self._ZSCORE_LOOKBACK, idx,
            )

        # ----- GBP block ---------------------------------------------------
        if sym_upper in _GBP_EXPOSURE:
            out["boe_rate_level"] = self._level_or_default(
                raw.get("boe_rate"), 4.0, idx,
            )
            out["uk_10y_level"] = self._level_or_default(
                raw.get("uk_10y"), 3.5, idx,
            )
            out["uk_cpi_yoy_zscore"] = self._rolling_zscore_or_zero(
                raw.get("uk_cpi_yoy"), self._ZSCORE_LOOKBACK, idx,
            )
            if sym_upper in ("GBPUSD", "GBP/USD"):
                out["gbp_usd_rate_diff"] = (
                    out["fed_funds_level"] - out["boe_rate_level"]
                )

        # ----- AUD block (with China-demand channel) -----------------------
        if sym_upper in _AUD_EXPOSURE:
            out["rba_rate_level"] = self._level_or_default(
                raw.get("rba_rate"), 4.0, idx,
            )
            out["au_10y_level"] = self._level_or_default(
                raw.get("au_10y"), 4.0, idx,
            )
            out["au_cpi_yoy_zscore"] = self._rolling_zscore_or_zero(
                raw.get("au_cpi_yoy"), self._ZSCORE_LOOKBACK, idx,
            )
            out["iron_ore_zscore"] = self._rolling_zscore_or_zero(
                raw.get("iron_ore"), self._ZSCORE_LOOKBACK, idx,
            )
            out["china_cpi_yoy_zscore"] = self._rolling_zscore_or_zero(
                raw.get("china_cpi_yoy"), self._ZSCORE_LOOKBACK, idx,
            )
            out["cny_usd_zscore"] = self._rolling_zscore_or_zero(
                raw.get("cny_usd"), self._ZSCORE_LOOKBACK, idx,
            )
            if sym_upper in ("AUDUSD", "AUD/USD"):
                out["aud_usd_rate_diff"] = (
                    out["fed_funds_level"] - out["rba_rate_level"]
                )

        # ----- NZD block ---------------------------------------------------
        if sym_upper in _NZD_EXPOSURE:
            out["rbnz_rate_level"] = self._level_or_default(
                raw.get("rbnz_rate"), 4.0, idx,
            )
            out["nz_10y_level"] = self._level_or_default(
                raw.get("nz_10y"), 4.0, idx,
            )
            out["nz_cpi_yoy_zscore"] = self._rolling_zscore_or_zero(
                raw.get("nz_cpi_yoy"), self._ZSCORE_LOOKBACK, idx,
            )

        # ----- Cross-pair rate differentials -------------------------------
        if sym_upper in ("EURGBP", "EUR/GBP"):
            out["eur_gbp_rate_diff"] = (
                out["ecb_rate_level"] - out["boe_rate_level"]
            )
        if sym_upper in ("EURJPY", "EUR/JPY"):
            out["eur_jpy_rate_diff"] = (
                out["ecb_rate_level"] - out["boj_rate_level"]
            )
            out["carry_trade_indicator"] = out["eur_jpy_rate_diff"]
        if sym_upper in ("GBPJPY", "GBP/JPY"):
            out["gbp_jpy_rate_diff"] = (
                out["boe_rate_level"] - out["boj_rate_level"]
            )
            out["carry_trade_indicator"] = out["gbp_jpy_rate_diff"]
        if sym_upper in ("AUDNZD", "AUD/NZD"):
            out["aud_nzd_rate_diff"] = (
                out["rba_rate_level"] - out["rbnz_rate_level"]
            )

        return out.sort_index()

    # ----- Engineering helpers (rolling, point-in-time-correct) -----------
    # Each helper accepts the raw column (or None if missing) and returns
    # a Series aligned to ``idx``. Missing columns or NaN gaps fall back
    # to the matching live default — keeps train/serve aligned end-to-end.

    @staticmethod
    def _level_or_default(
        col: Optional[pd.Series], default: float, idx: pd.Index,
    ) -> pd.Series:
        """Forward-fill the raw observation; fill remaining NaN with default."""
        if col is None:
            return pd.Series(default, index=idx, dtype=float)
        return col.reindex(idx).ffill().fillna(default).astype(float)

    @staticmethod
    def _diff_or_zero(col: Optional[pd.Series], idx: pd.Index) -> pd.Series:
        """First-difference of forward-filled raw observation; zero-fill NaN."""
        if col is None:
            return pd.Series(0.0, index=idx, dtype=float)
        ffilled = col.reindex(idx).ffill()
        return ffilled.diff().fillna(0.0).astype(float)

    @staticmethod
    def _rolling_zscore_or_zero(
        col: Optional[pd.Series], lookback: int, idx: pd.Index, clip: float = 1.0,
    ) -> pd.Series:
        """Rolling z-score clipped to ``[-clip, +clip]``. Mirrors live
        ``_normalize_series`` (which takes the most recent ``lookback``
        observations including current) over a vectorized rolling window."""
        if col is None:
            return pd.Series(0.0, index=idx, dtype=float)
        ffilled = col.reindex(idx).ffill()
        # min_periods=2 matches live's `if len(series) < 2: return 0.0` guard.
        roll = ffilled.rolling(lookback, min_periods=2)
        mean = roll.mean()
        std = roll.std()
        z = (ffilled - mean) / std
        z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return z.clip(-clip, clip).astype(float)

    @staticmethod
    def _yoy_change_or_zero(
        col: Optional[pd.Series], periods: int, idx: pd.Index,
    ) -> pd.Series:
        """Percentage change over ``periods`` observations; zero on missing/inf.

        Matches live ``_yoy_change`` / ``_recent_pct_change(periods=12)`` —
        for monthly-cadence series, periods=12 ≈ year-over-year.
        """
        if col is None:
            return pd.Series(0.0, index=idx, dtype=float)
        ffilled = col.reindex(idx).ffill()
        prev = ffilled.shift(periods)
        # Match live's `if prev == 0: return 0.0` — divide by abs(prev).
        out = (ffilled - prev) / prev.abs().replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)

    @staticmethod
    def _default_features() -> dict[str, float]:
        """Neutral defaults when FRED is unavailable."""
        return {
            "fed_funds_level": 5.0,
            "fed_funds_change": 0.0,
            "cpi_yoy_zscore": 0.0,
            "real_yield_level": 2.0,
            "m2_yoy_change": 0.0,
            "yield_curve_slope": 0.0,
            "breakeven_inflation": 2.0,
            "hy_credit_spread": 4.0,
            "initial_claims_zscore": 0.0,
            "fed_balance_sheet_change": 0.0,
            "dxy_zscore": 0.0,
            "vix_fred_level": 20.0,
        }
