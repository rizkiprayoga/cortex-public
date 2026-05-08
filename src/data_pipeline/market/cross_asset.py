"""
cross_asset.py — Cross-Asset Feature Fetcher (yfinance, no API key)

Fetches daily OHLCV data for correlated assets and computes features
that capture inter-market relationships. Feature routing is by currency
exposure — every pair touching a currency gets that country's equity
index and relevant commodity equities.

USD-axis (all symbols):
    DX-Y.NYB (DXY)   — USD Dollar Index
    ^GSPC (SPX)      — S&P 500 risk proxy
    ^VIX             — CBOE volatility / fear gauge
    CL=F             — WTI crude oil
    ^TNX / ^IRX      — US 10Y / 3M yields

XAU-specific:
    GC=F / SI=F      — gold/silver ratio

GBP (GBPUSD/EURGBP/GBPJPY):
    ^FTSE            — FTSE 100
    RIO.L            — Rio Tinto (LSE, commodity miner)

EUR (EURUSD/EURGBP/EURJPY):
    ^GDAXI           — Germany DAX

JPY (USDJPY/EURJPY/GBPJPY):
    ^N225            — Nikkei 225

AUD (AUDUSD/AUDNZD) — China-heavy because AU exports ~30% to China:
    ^AXJO            — ASX 200
    ^SSEC / ^HSI     — Shanghai / Hang Seng (China demand)
    BHP.AX / VALE    — iron ore majors
    HG=F             — copper (global growth proxy)
    CNH=X            — offshore yuan

NZD (AUDNZD):
    ^NZ50            — NZX 50

CAD (USDCAD/CADJPY) — petro-currency, oil already in USD-axis:
    ^GSPTSE          — TSX Composite (Canadian equity benchmark)

CHF (USDCHF/EURCHF/CHFJPY/GBPCHF) — safe-haven currency:
    ^SSMI            — Swiss Market Index (SMI)

Data source: yfinance (Yahoo Finance) — free, no API key required.
Rate limits: ~2000 requests/hour, batched download minimizes calls.

Output: dict of per-symbol features — common (USD-axis) plus any
country-specific features matching the symbol's currency exposure.
"""

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.data_pipeline.data_store import DataStore

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tickers to fetch and their roles. Ordering is cosmetic — yfinance
# downloads them in one batch.
CROSS_ASSET_TICKERS = {
    # --- USD-axis (used by all symbols) ---
    "DX-Y.NYB":  "dxy",         # US Dollar Index
    "^GSPC":     "spx",          # S&P 500
    "^VIX":      "vix",          # CBOE Volatility Index
    "CL=F":      "oil",          # WTI Crude Oil
    "^TNX":      "us10y",        # 10-Year Treasury Yield
    "^IRX":      "us3m",         # 3-Month Treasury Yield (for yield curve)
    # --- XAU-specific ---
    "GC=F":      "gold_fut",     # Gold futures (for gold/silver ratio)
    "SI=F":      "silver",       # Silver futures
    # --- GBP exposure ---
    "^FTSE":     "ftse",         # FTSE 100 (UK)
    "RIO.L":     "rio_l",        # Rio Tinto LSE (UK/AU commodity miner)
    # --- EUR exposure ---
    "^GDAXI":    "dax",          # Germany DAX (EU benchmark)
    # --- JPY exposure ---
    "^N225":     "nikkei",       # Nikkei 225 (Japan)
    # --- AUD exposure (commodity + China) ---
    "^AXJO":     "axjo",         # ASX 200 (Australia)
    "^SSEC":     "ssec",         # Shanghai Composite (China demand proxy)
    "^HSI":      "hsi",          # Hang Seng (China/HK)
    "BHP.AX":    "bhp",          # BHP ASX (iron ore producer)
    "VALE":      "vale",         # Vale NYSE (iron ore producer)
    "HG=F":      "copper",       # Copper futures (Dr Copper — global growth)
    "CNH=X":     "cnh",          # CNH offshore yuan (China FX proxy)
    # --- NZD exposure ---
    "^NZ50":     "nz50",         # NZX 50 (New Zealand)
    # --- CAD exposure ---
    "^GSPTSE":   "tsx",          # TSX Composite (Canada)
    # --- CHF exposure ---
    "^SSMI":     "smi",          # Swiss Market Index
}

# Currency exposure — re-exported from the shared source of truth at
# fundamental/_currency_exposure.py. Any pair containing that currency
# gets the block.
from src.data_pipeline.fundamental._currency_exposure import (  # noqa: E402
    AUD_EXPOSURE as _AUD_EXPOSURE,
    CAD_EXPOSURE as _CAD_EXPOSURE,
    CHF_EXPOSURE as _CHF_EXPOSURE,
    EUR_EXPOSURE as _EUR_EXPOSURE,
    GBP_EXPOSURE as _GBP_EXPOSURE,
    JPY_EXPOSURE as _JPY_EXPOSURE,
    NZD_EXPOSURE as _NZD_EXPOSURE,
)


class CrossAssetFetcher:
    """
    Fetches cross-asset market data via yfinance and computes
    inter-market features.

    Usage:
        fetcher = CrossAssetFetcher()
        features = fetcher.get_cross_asset_features("XAUUSD")
    """

    def __init__(self, lookback_days: int = 60):
        self._lookback_days = lookback_days
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_ts: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)

    def get_historical_cross_asset_features(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch historical cross-asset features over a full date range.

        Used by training scripts to add inter-market context to historical
        OHLCV data.  Returns a DataFrame indexed by trading day with the
        same feature columns that ``get_cross_asset_features()`` produces.

        Args:
            symbol:     Trading symbol (e.g. "XAUUSD") — affects gold/silver ratio
            start_date: Inclusive start
            end_date:   Inclusive end

        Returns:
            DataFrame with ~11-12 feature columns, one row per trading day.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed — pip install yfinance")
            return pd.DataFrame()

        tickers = list(CROSS_ASSET_TICKERS.keys())
        start = (start_date - timedelta(days=30)).strftime("%Y-%m-%d")
        end = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            raw = yf.download(
                tickers, start=start, end=end,
                group_by="ticker", auto_adjust=True,
                threads=True, progress=False,
            )
        except Exception as exc:
            logger.error("yfinance historical download failed: %s", exc)
            return pd.DataFrame()

        # Parse per-ticker close series
        close_data: dict[str, pd.Series] = {}
        for ticker, label in CROSS_ASSET_TICKERS.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker in raw.columns.get_level_values(0):
                        s = raw[(ticker, "Close")].dropna()
                    else:
                        continue
                else:
                    s = raw["Close"].dropna()
                if len(s) > 0:
                    close_data[label] = s
            except Exception:
                continue

        if not close_data:
            return pd.DataFrame()

        # Build vectorized features across the whole date range
        rows: list[dict[str, float]] = []
        # Use the DXY index as the reference date axis (most liquid). Fall
        # back to any other available series if DXY's download failed.
        # Note: `close_data.get("dxy") or ...` would raise on truthiness of
        # a pandas Series — must check explicitly.
        ref = close_data.get("dxy")
        if ref is None or ref.empty:
            ref = next(iter(close_data.values()), None)
        if ref is None or ref.empty:
            return pd.DataFrame()
        dates = ref.index

        for i in range(len(dates)):
            dt = dates[i]
            f: dict[str, float] = {}

            # DXY
            dxy_s = close_data.get("dxy")
            if dxy_s is not None and i >= 14:
                window = dxy_s.loc[:dt]
                if len(window) >= 2:
                    f["dxy_log_return"] = float(np.log(window.iloc[-1] / window.iloc[-2]))
                    f["dxy_rsi_14"] = float(self._rsi(window, 14))
                    f["dxy_zscore"] = float(self._zscore(window, 20))

            # SPX
            spx_s = close_data.get("spx")
            if spx_s is not None and i >= 14:
                window = spx_s.loc[:dt]
                if len(window) >= 2:
                    f["spx_log_return"] = float(np.log(window.iloc[-1] / window.iloc[-2]))
                    f["spx_zscore"] = float(self._zscore(window, 20))

            # VIX
            vix_s = close_data.get("vix")
            if vix_s is not None:
                window = vix_s.loc[:dt]
                if len(window) >= 2 and window.iloc[-2] != 0:
                    f["vix_level"] = float(window.iloc[-1])
                    f["vix_change"] = float(
                        (window.iloc[-1] - window.iloc[-2]) / window.iloc[-2]
                    )

            # Oil
            oil_s = close_data.get("oil")
            if oil_s is not None:
                window = oil_s.loc[:dt]
                if len(window) >= 2:
                    f["oil_log_return"] = float(np.log(window.iloc[-1] / window.iloc[-2]))

            # Gold/Silver ratio
            if symbol.upper().startswith("XAU"):
                gold_s = close_data.get("gold_fut")
                silver_s = close_data.get("silver")
                if gold_s is not None and silver_s is not None:
                    gw = gold_s.loc[:dt]
                    sw = silver_s.loc[:dt]
                    if len(gw) >= 1 and len(sw) >= 1 and sw.iloc[-1] > 0:
                        f["gold_silver_ratio"] = float(gw.iloc[-1] / sw.iloc[-1])

            # US 10Y yield
            y10_s = close_data.get("us10y")
            if y10_s is not None:
                window = y10_s.loc[:dt]
                if len(window) >= 1:
                    f["us10y_level"] = float(window.iloc[-1])

            # Yield curve
            y3m_s = close_data.get("us3m")
            if y10_s is not None and y3m_s is not None:
                w10 = y10_s.loc[:dt]
                w3m = y3m_s.loc[:dt]
                if len(w10) >= 1 and len(w3m) >= 1:
                    f["yield_curve_10y_2y"] = float(w10.iloc[-1] - w3m.iloc[-1])

            # DXY-SPX correlation
            if "dxy" in close_data and "spx" in close_data:
                dr = close_data["dxy"].loc[:dt].pct_change().tail(20)
                sr = close_data["spx"].loc[:dt].pct_change().tail(20)
                aligned = pd.concat([dr, sr], axis=1).dropna()
                if len(aligned) >= 10:
                    f["corr_dxy_spx_20"] = float(aligned.corr().iloc[0, 1])

            # Currency-specific country + commodity blocks — same routing
            # as the live path. Emits log_return + zscore per ticker.
            def _hist_emit(label: str, key_prefix: str, z_window: int = 20) -> None:
                s = close_data.get(label)
                if s is None:
                    return
                window = s.loc[:dt]
                if len(window) < 2:
                    return
                prev = window.iloc[-2]
                if prev > 0 and not pd.isna(prev):
                    f[f"{key_prefix}_log_return"] = float(np.log(window.iloc[-1] / prev))
                if len(window) >= z_window:
                    f[f"{key_prefix}_zscore"] = float(self._zscore(window, z_window))

            sym_upper = symbol.upper()

            if sym_upper in _GBP_EXPOSURE:
                _hist_emit("ftse",  "ftse")
                _hist_emit("rio_l", "rio_l")

            if sym_upper in _EUR_EXPOSURE:
                _hist_emit("dax", "dax")

            if sym_upper in _JPY_EXPOSURE:
                _hist_emit("nikkei", "nikkei")

            if sym_upper in _AUD_EXPOSURE:
                _hist_emit("axjo",   "axjo")
                _hist_emit("ssec",   "ssec")
                _hist_emit("hsi",    "hsi")
                _hist_emit("bhp",    "bhp")
                _hist_emit("vale",   "vale")
                _hist_emit("copper", "copper")
                _hist_emit("cnh",    "cnh")

            if sym_upper in _NZD_EXPOSURE:
                _hist_emit("nz50", "nz50")

            if sym_upper in _CAD_EXPOSURE:
                _hist_emit("tsx", "tsx")

            if sym_upper in _CHF_EXPOSURE:
                _hist_emit("smi", "smi")

            rows.append(f)

        result = pd.DataFrame(rows, index=dates)
        # Fill defaults for any missing columns
        defaults = self._default_features(symbol)
        for col, default_val in defaults.items():
            if col not in result.columns:
                result[col] = default_val
        result = result.fillna(method="ffill").fillna(0.0)

        # Trim to requested range
        mask = (result.index >= pd.Timestamp(start_date)) & (
            result.index <= pd.Timestamp(end_date)
        )
        return result.loc[mask]

    def get_cross_asset_features(self, symbol: str) -> dict[str, float]:
        """
        Compute cross-asset features for a trading symbol.

        Returns:
            Dict of ~15 feature_name → float values.
        """
        try:
            data = self._fetch_data()
            if not data:
                return self._default_features(symbol)
            return self._compute_features(data, symbol)
        except Exception as exc:
            logger.error("Cross-asset features failed: %s", exc)
            return self._default_features(symbol)

    def _fetch_data(self) -> dict[str, pd.DataFrame]:
        """Fetch or return cached cross-asset OHLCV data."""
        now = datetime.utcnow()
        if (
            self._cache
            and self._cache_ts
            and (now - self._cache_ts) < self._cache_ttl
        ):
            return self._cache

        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed — pip install yfinance")
            return {}

        tickers = list(CROSS_ASSET_TICKERS.keys())
        start = (now - timedelta(days=self._lookback_days)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        try:
            raw = yf.download(
                tickers, start=start, end=end,
                group_by="ticker", auto_adjust=True,
                threads=True, progress=False,
            )
        except Exception as exc:
            logger.error("yfinance download failed: %s", exc)
            return {}

        result = {}
        for ticker, label in CROSS_ASSET_TICKERS.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker in raw.columns.get_level_values(0):
                        df = raw[ticker].dropna(subset=["Close"])
                    else:
                        continue
                else:
                    df = raw.dropna(subset=["Close"])
                if not df.empty:
                    result[label] = df
            except Exception:
                continue

        self._cache = result
        self._cache_ts = now
        logger.debug("Fetched cross-asset data for %d tickers", len(result))
        return result

    def _compute_features(
        self, data: dict[str, pd.DataFrame], symbol: str
    ) -> dict[str, float]:
        """Compute all cross-asset features from fetched data."""
        features: dict[str, float] = {}

        # DXY features
        if "dxy" in data and len(data["dxy"]) >= 5:
            dxy = data["dxy"]["Close"]
            features["dxy_log_return"] = float(np.log(dxy.iloc[-1] / dxy.iloc[-2]))
            features["dxy_rsi_14"] = float(self._rsi(dxy, 14))
            features["dxy_zscore"] = float(self._zscore(dxy, 20))
        else:
            features.update({"dxy_log_return": 0.0, "dxy_rsi_14": 50.0, "dxy_zscore": 0.0})

        # SPX features
        if "spx" in data and len(data["spx"]) >= 5:
            spx = data["spx"]["Close"]
            features["spx_log_return"] = float(np.log(spx.iloc[-1] / spx.iloc[-2]))
            features["spx_zscore"] = float(self._zscore(spx, 20))
        else:
            features.update({"spx_log_return": 0.0, "spx_zscore": 0.0})

        # VIX features
        if "vix" in data and len(data["vix"]) >= 2:
            vix = data["vix"]["Close"]
            features["vix_level"] = float(vix.iloc[-1])
            features["vix_change"] = float(
                (vix.iloc[-1] - vix.iloc[-2]) / vix.iloc[-2]
            ) if vix.iloc[-2] != 0 else 0.0
        else:
            features.update({"vix_level": 20.0, "vix_change": 0.0})

        # Oil features
        if "oil" in data and len(data["oil"]) >= 2:
            oil = data["oil"]["Close"]
            features["oil_log_return"] = float(np.log(oil.iloc[-1] / oil.iloc[-2]))
        else:
            features["oil_log_return"] = 0.0

        # Gold/Silver ratio (XAU only)
        if symbol.upper().startswith("XAU"):
            if "gold_fut" in data and "silver" in data:
                gold = data["gold_fut"]["Close"]
                silver = data["silver"]["Close"]
                if len(gold) >= 2 and len(silver) >= 2 and silver.iloc[-1] > 0:
                    features["gold_silver_ratio"] = float(gold.iloc[-1] / silver.iloc[-1])
                else:
                    features["gold_silver_ratio"] = 80.0  # typical ratio
            else:
                features["gold_silver_ratio"] = 80.0

        # US 10Y yield
        if "us10y" in data and len(data["us10y"]) >= 2:
            y10 = data["us10y"]["Close"]
            features["us10y_level"] = float(y10.iloc[-1])
        else:
            features["us10y_level"] = 4.0

        # Yield curve slope: 10Y - 3M
        if "us10y" in data and "us3m" in data:
            y10 = data["us10y"]["Close"]
            y3m = data["us3m"]["Close"]
            if len(y10) >= 1 and len(y3m) >= 1:
                features["yield_curve_10y_2y"] = float(y10.iloc[-1] - y3m.iloc[-1])
            else:
                features["yield_curve_10y_2y"] = 0.0
        else:
            features["yield_curve_10y_2y"] = 0.0

        # Cross-correlations (20-day rolling)
        # Requires the trading symbol's close prices from the caller context.
        # For now, compute DXY-SPX correlation as a proxy.
        if "dxy" in data and "spx" in data:
            dxy_ret = data["dxy"]["Close"].pct_change().tail(20)
            spx_ret = data["spx"]["Close"].pct_change().tail(20)
            aligned = pd.concat([dxy_ret, spx_ret], axis=1).dropna()
            if len(aligned) >= 10:
                features["corr_dxy_spx_20"] = float(aligned.corr().iloc[0, 1])
            else:
                features["corr_dxy_spx_20"] = 0.0
        else:
            features["corr_dxy_spx_20"] = 0.0

        # --------------------------------------------------------------
        # Currency-specific country + commodity blocks
        # --------------------------------------------------------------
        # Each block fires only for symbols that expose that currency. Emit
        # log_return + zscore for each ticker — per collect-breadth mandate,
        # keep raw features wide; downstream training picks what matters.

        sym_upper = symbol.upper()

        def _emit_price_features(
            label: str, key_prefix: str, z_window: int = 20
        ) -> None:
            """Emit log_return + zscore for a ticker if data present."""
            if label not in data or len(data[label]) < 2:
                return
            s = data[label]["Close"]
            prev = s.iloc[-2]
            if prev > 0 and not pd.isna(prev):
                features[f"{key_prefix}_log_return"] = float(np.log(s.iloc[-1] / prev))
            else:
                features[f"{key_prefix}_log_return"] = 0.0
            features[f"{key_prefix}_zscore"] = float(self._zscore(s, z_window))

        if sym_upper in _GBP_EXPOSURE:
            _emit_price_features("ftse",  "ftse")
            _emit_price_features("rio_l", "rio_l")

        if sym_upper in _EUR_EXPOSURE:
            _emit_price_features("dax", "dax")

        if sym_upper in _JPY_EXPOSURE:
            _emit_price_features("nikkei", "nikkei")

        if sym_upper in _AUD_EXPOSURE:
            _emit_price_features("axjo",   "axjo")
            _emit_price_features("ssec",   "ssec")
            _emit_price_features("hsi",    "hsi")
            _emit_price_features("bhp",    "bhp")
            _emit_price_features("vale",   "vale")
            _emit_price_features("copper", "copper")
            _emit_price_features("cnh",    "cnh")

        if sym_upper in _NZD_EXPOSURE:
            _emit_price_features("nz50", "nz50")

        if sym_upper in _CAD_EXPOSURE:
            _emit_price_features("tsx", "tsx")

        if sym_upper in _CHF_EXPOSURE:
            _emit_price_features("smi", "smi")

        return features

    def _default_features(self, symbol: str) -> dict[str, float]:
        """Return neutral defaults when data is unavailable."""
        defaults = {
            "dxy_log_return": 0.0, "dxy_rsi_14": 50.0, "dxy_zscore": 0.0,
            "spx_log_return": 0.0, "spx_zscore": 0.0,
            "vix_level": 20.0, "vix_change": 0.0,
            "oil_log_return": 0.0,
            "us10y_level": 4.0, "yield_curve_10y_2y": 0.0,
            "corr_dxy_spx_20": 0.0,
        }
        sym_upper = symbol.upper()
        if sym_upper.startswith("XAU"):
            defaults["gold_silver_ratio"] = 80.0

        # Currency-specific defaults — zeros for log_return + zscore keep
        # downstream pipelines deterministic when yfinance errors.
        def _zero_pair(prefix: str) -> dict[str, float]:
            return {f"{prefix}_log_return": 0.0, f"{prefix}_zscore": 0.0}

        if sym_upper in _GBP_EXPOSURE:
            defaults.update(_zero_pair("ftse"))
            defaults.update(_zero_pair("rio_l"))
        if sym_upper in _EUR_EXPOSURE:
            defaults.update(_zero_pair("dax"))
        if sym_upper in _JPY_EXPOSURE:
            defaults.update(_zero_pair("nikkei"))
        if sym_upper in _AUD_EXPOSURE:
            for pfx in ("axjo", "ssec", "hsi", "bhp", "vale", "copper", "cnh"):
                defaults.update(_zero_pair(pfx))
        if sym_upper in _NZD_EXPOSURE:
            defaults.update(_zero_pair("nz50"))
        if sym_upper in _CAD_EXPOSURE:
            defaults.update(_zero_pair("tsx"))
        if sym_upper in _CHF_EXPOSURE:
            defaults.update(_zero_pair("smi"))

        return defaults

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> float:
        """Compute current RSI value for a price series."""
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
        loss = (-delta).clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0

    # ------------------------------------------------------------------
    # feature_store backfill (Phase 1F)
    # ------------------------------------------------------------------

    FEATURE_GROUP = "yfinance_cross_asset"
    SCHEMA_VERSION = 1

    async def persist_raw_history_to_feature_store(
        self,
        store: "DataStore",
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        *,
        force: bool = False,
        lookback_days: Optional[int] = None,
    ) -> int:
        """
        Write daily cross-asset features to ``feature_store`` for one symbol.

        Reuses ``get_historical_cross_asset_features`` which already
        downloads the routed yfinance tickers in one batch and emits the
        full per-day feature set.

        Args:
            start_date / end_date: Explicit window. ``end_date`` defaults
                to now; ``start_date`` defaults to ``end_date - 25 yr``.
            force: When True, switches conflict policy from DO NOTHING to
                DO UPDATE — used by the weekly TTL safety-net job.
            lookback_days: Convenience that overrides ``start_date`` to
                ``end_date - lookback_days``. Set by the TTL job.

        Returns rows touched.
        """
        if end_date is None:
            end_date = datetime.utcnow()
        if lookback_days is not None:
            start_date = end_date - timedelta(days=lookback_days)
        elif start_date is None:
            # Backfill default: pull the longest history yfinance can return.
            # Live bot's `_lookback_days` (60d) is for the per-tick cache —
            # not appropriate for populating feature_store. 25 yr matches
            # the OHLCV window in `ohlcv_bars`. yfinance returns whatever
            # subset it has for younger tickers (e.g. CNH=X started 2010).
            start_date = end_date - timedelta(days=365 * 25)

        df = self.get_historical_cross_asset_features(symbol, start_date, end_date)
        if df.empty:
            logger.warning("yfinance: no cross-asset history for %s, skipping", symbol)
            return 0

        rows = []
        for ts, row in df.iterrows():
            values = {
                col: float(val)
                for col, val in row.items()
                if val is not None and not pd.isna(val)
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

    @staticmethod
    def _zscore(series: pd.Series, window: int = 20) -> float:
        """Z-score of latest value vs rolling window."""
        if len(series) < window:
            return 0.0
        recent = series.tail(window)
        mean = recent.mean()
        std = recent.std()
        if std == 0 or pd.isna(std):
            return 0.0
        z = (series.iloc[-1] - mean) / std
        return float(max(-3.0, min(3.0, z)))
