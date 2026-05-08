"""
cot_data.py — CFTC Commitment of Traders (COT) Reports.

Fetches weekly positioning data from the CFTC — two different reports
depending on what's being modelled:

Gold (XAUUSD)
    Disaggregated Futures-Only Report. Categories: Producer/Merchant/
    Processor/User (commercials), Swap Dealers, Managed Money
    (speculators), Other Reportables.

FX pairs (EUR, JPY, CAD, GBP, AUD, NZD IMM futures)
    Traders in Financial Futures (TFF) Report. Categories: Dealer
    Intermediaries (hedgers), Asset Manager/Institutional, Leveraged
    Funds (speculators), Other Reportables, Non-Reportable.

Both reports update weekly (Tuesday snapshot, released Friday 15:30 ET).
No API key required.

Data source URLs:
    Disaggregated (commodities — Gold):
      https://www.cftc.gov/dea/newcot/f_disagg.txt                    (current)
      https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip (history)
    TFF (financial — FX):
      https://www.cftc.gov/dea/newcot/f_tff.txt                        (current)
      https://www.cftc.gov/files/dea/history/fin_fut_txt_{year}.zip    (history)

Contract codes (CFTC):
    XAU:  088691  COMEX Gold
    EUR:  099741  EuroFX IMM (CME)
    JPY:  097741  Japanese Yen IMM
    CAD:  090741  Canadian Dollar IMM
    GBP:  096742  British Pound IMM
    AUD:  232741  Australian Dollar IMM
    NZD:  112741  New Zealand Dollar IMM

Per-symbol emission:
    XAUUSD             → cot_net_position / cot_net_zscore_52w / cot_wow_change
                         / cot_commercial_ratio / cot_extreme_flag
    EUR pair (EURUSD)  → cot_eur_{...} (5 features)
    JPY pair (USDJPY)  → cot_jpy_{...}
    CAD pair (USDCAD)  → cot_cad_{...}
    GBP pair (GBPUSD)  → cot_gbp_{...}
    AUD pair (AUDUSD)  → cot_aud_{...}
    AUDNZD             → cot_aud_{...} + cot_nzd_{...}  (both currencies)
    EURGBP             → cot_eur_{...} + cot_gbp_{...}
    EURJPY             → cot_eur_{...} + cot_jpy_{...}
    GBPJPY             → cot_gbp_{...} + cot_jpy_{...}
    ETH + others       → empty dict
"""

import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.data_pipeline.data_store import DataStore

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# CFTC data URLs
CFTC_CURRENT_YEAR_URL = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
CFTC_HISTORY_URL_TEMPLATE = (
    "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
)

# TFF (Traders in Financial Futures) report URLs — FX contracts live here.
# CFTC renamed both endpoints around April 2026:
#   OLD: f_tff.txt              -> NEW: FinFutWk.txt
#   OLD: fin_fut_txt_{year}.zip -> NEW: fut_fin_txt_{year}.zip
# The yearly zip's internal CSV is still named FinFutYY.txt and still
# carries column headers, so _parse_tff_csv works unchanged.
CFTC_TFF_CURRENT_URL = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"
CFTC_TFF_HISTORY_URL_TEMPLATE = (
    "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
)

# COMEX Gold contract code in CFTC data
GOLD_CONTRACT_CODE = "088691"

# CME IMM FX contract codes (TFF report). Key = currency code.
FX_CONTRACT_CODES: dict[str, str] = {
    "EUR": "099741",   # EuroFX
    "JPY": "097741",   # Japanese Yen
    "CAD": "090741",   # Canadian Dollar
    "GBP": "096742",   # British Pound
    "AUD": "232741",   # Australian Dollar
    "NZD": "112741",   # New Zealand Dollar
}

# Per trading-symbol currency exposure (which FX futures the pair should
# pull COT data for). Re-exported from the shared source of truth at
# fundamental/_currency_exposure.py.
from src.data_pipeline.fundamental._currency_exposure import (  # noqa: E402
    SYMBOL_CURRENCIES as _SYMBOL_CURRENCIES,
)

# Column names we need from the disaggregated report
# These are positional in the fixed-width CSV
COL_MAP = {
    "Market_and_Exchange_Names": "market",
    "CFTC_Contract_Market_Code": "code",
    "Report_Date_as_YYYY-MM-DD": "date",
    "Pct_of_OI_Prod_Merc_Long_All": "comm_long_pct",
    "Pct_of_OI_Prod_Merc_Short_All": "comm_short_pct",
    "Pct_of_OI_M_Money_Long_All": "mm_long_pct",
    "Pct_of_OI_M_Money_Short_All": "mm_short_pct",
    "Prod_Merc_Positions_Long_All": "comm_long",
    "Prod_Merc_Positions_Short_All": "comm_short",
    "M_Money_Positions_Long_All": "mm_long",
    "M_Money_Positions_Short_All": "mm_short",
    "Open_Interest_All": "open_interest",
}


class COTDataFetcher:
    """
    Fetches CFTC COT report data for Gold futures directly from
    the CFTC website. No API key required.

    Only relevant for XAUUSD — returns empty dict for other symbols.

    Usage:
        fetcher = COTDataFetcher()
        features = fetcher.get_cot_features("XAUUSD")
        score = fetcher.get_cot_score("XAUUSD")  # backward compat
    """

    def __init__(self):
        # Separate caches for the two report types — they live at
        # different URLs and have different schemas.
        self._gold_cache: Optional[pd.DataFrame] = None
        self._gold_cache_ts: Optional[datetime] = None
        # FX TFF cache: full parsed report, filtered per-currency on read.
        self._tff_cache: Optional[pd.DataFrame] = None
        self._tff_cache_ts: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=12)  # COT updates weekly

    def get_cot_score(self, symbol: str) -> float:
        """
        Backward-compatible: return single contrarian score in [-1, 1].

        +1 = speculators very short → contrarian bullish signal
        -1 = speculators very long → contrarian bearish signal

        XAU uses the XAU feature set; FX pairs use the first currency's
        zscore (e.g. EURGBP uses EUR, not GBP — this is a heuristic for
        the backward-compat single-score API).
        """
        features = self.get_cot_features(symbol)
        # Try XAU key first, then the first FX currency key present.
        for key in ("cot_net_zscore_52w",
                     "cot_eur_net_zscore_52w", "cot_jpy_net_zscore_52w",
                     "cot_cad_net_zscore_52w", "cot_gbp_net_zscore_52w",
                     "cot_aud_net_zscore_52w", "cot_nzd_net_zscore_52w"):
            if key in features:
                return features[key] * -1  # invert for contrarian
        return 0.0

    def get_cot_features(self, symbol: str) -> dict[str, float]:
        """
        Return COT features for a symbol. Dispatches by symbol type:
          - XAUUSD → 5 features from Disaggregated Gold report
          - FX pairs → 5 features per exposed currency from TFF report
          - Other (ETH, etc.) → empty dict

        On any fetch/parse error, returns zero defaults matching the
        expected key set (never raises).
        """
        sym_upper = symbol.upper()

        if sym_upper.startswith("XAU"):
            try:
                df = self._get_gold_data()
                if df.empty:
                    return self._default_xau_features()
                return self._compute_features_xau(df)
            except Exception as exc:
                logger.error("COT XAU features failed: %s", exc)
                return self._default_xau_features()

        currencies = _SYMBOL_CURRENCIES.get(sym_upper)
        if not currencies:
            return {}

        try:
            df = self._get_tff_data()
            if df.empty:
                return self._default_fx_features(currencies)
            features: dict[str, float] = {}
            for ccy in currencies:
                features.update(self._compute_features_fx(df, ccy))
            return features
        except Exception as exc:
            logger.error("COT FX features failed for %s: %s", symbol, exc)
            return self._default_fx_features(currencies)

    def _get_gold_data(self) -> pd.DataFrame:
        """Fetch or return cached XAU COT data (Disaggregated report)."""
        now = datetime.utcnow()
        if (
            self._gold_cache is not None
            and not self._gold_cache.empty
            and self._gold_cache_ts
            and (now - self._gold_cache_ts) < self._cache_ttl
        ):
            return self._gold_cache

        df = self._fetch_current_year()
        if df.empty:
            return df

        self._gold_cache = df
        self._gold_cache_ts = now
        return df

    # Kept for backward compat with any external callers.
    _get_data = _get_gold_data

    def _get_tff_data(self) -> pd.DataFrame:
        """Fetch or return cached TFF FX data (all 6 currencies)."""
        now = datetime.utcnow()
        if (
            self._tff_cache is not None
            and not self._tff_cache.empty
            and self._tff_cache_ts
            and (now - self._tff_cache_ts) < self._cache_ttl
        ):
            return self._tff_cache

        df = self._fetch_tff_current_year()
        if df.empty:
            return df

        self._tff_cache = df
        self._tff_cache_ts = now
        return df

    def _fetch_tff_current_year(self) -> pd.DataFrame:
        """
        Download current year's TFF report. Mirrors the XAU path — tries
        the yearly ZIP first (always header-ed) and falls back to the
        headerless current-year feed.
        """
        year = datetime.now(tz=timezone.utc).year
        zip_df = self._fetch_tff_historical_year(year)
        if not zip_df.empty:
            return zip_df
        try:
            resp = requests.get(CFTC_TFF_CURRENT_URL, timeout=30)
            resp.raise_for_status()
            df = self._parse_tff_csv(resp.text)
            if df.empty:
                df = self._fetch_tff_historical_year(year - 1)
            return df
        except Exception as exc:
            logger.warning("CFTC TFF current year fetch failed: %s", exc)
            return self._fetch_tff_historical_year(year - 1)

    def _fetch_tff_historical_year(self, year: int) -> pd.DataFrame:
        """Download a historical year's TFF zip from CFTC."""
        url = CFTC_TFF_HISTORY_URL_TEMPLATE.format(year=year)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    text = f.read().decode("utf-8", errors="replace")
                    return self._parse_tff_csv(text)
        except Exception as exc:
            logger.error("CFTC TFF historical fetch failed for %d: %s", year, exc)
            return pd.DataFrame()

    def _parse_tff_csv(self, text: str) -> pd.DataFrame:
        """
        Parse TFF CSV text and return a DataFrame filtered to just the 6
        FX currency contracts. One row per (date, currency) with
        dealer/leveraged/asset-manager positions.
        """
        try:
            raw = pd.read_csv(io.StringIO(text), low_memory=False)
        except Exception as exc:
            logger.error("Failed to parse CFTC TFF CSV: %s", exc)
            return pd.DataFrame()

        raw.columns = [c.strip() for c in raw.columns]

        code_col = None
        for candidate in ["CFTC_Contract_Market_Code",
                           "CFTC Contract Market Code"]:
            if candidate in raw.columns:
                code_col = candidate
                break
        if code_col is None:
            for c in raw.columns:
                if "contract" in c.lower() and "code" in c.lower():
                    code_col = c
                    break
        if code_col is None:
            logger.error("Cannot find contract code column in TFF data")
            return pd.DataFrame()

        fx_codes = set(FX_CONTRACT_CODES.values())
        # CFTC codes are 6-digit strings. Pandas may infer them as int64
        # (stripping the leading zero on "096742" → "96742") — zero-pad
        # back to 6 digits so the filter matches.
        raw[code_col] = raw[code_col].astype(str).str.strip().str.zfill(6)
        fx = raw[raw[code_col].isin(fx_codes)].copy()
        if fx.empty:
            logger.warning("No FX futures rows found in TFF data")
            return pd.DataFrame()

        # Reverse-lookup: contract code → currency
        code_to_ccy = {code: ccy for ccy, code in FX_CONTRACT_CODES.items()}

        result = pd.DataFrame()
        result["code"] = fx[code_col]
        result["currency"] = result["code"].map(code_to_ccy)
        result["date"] = pd.to_datetime(
            self._find_col(fx, ["Report_Date_as_YYYY-MM-DD",
                                  "As of Date in Form YYYY-MM-DD"]),
            errors="coerce",
        )
        # TFF category columns — dealer = commercial hedger analogue,
        # leveraged funds = the speculator class we care about.
        result["dealer_long"] = pd.to_numeric(
            self._find_col(fx, ["Dealer_Positions_Long_All",
                                  "Dealer Longs"]),
            errors="coerce",
        )
        result["dealer_short"] = pd.to_numeric(
            self._find_col(fx, ["Dealer_Positions_Short_All",
                                  "Dealer Shorts"]),
            errors="coerce",
        )
        result["lev_long"] = pd.to_numeric(
            self._find_col(fx, ["Lev_Money_Positions_Long_All",
                                  "Leveraged Funds Longs"]),
            errors="coerce",
        )
        result["lev_short"] = pd.to_numeric(
            self._find_col(fx, ["Lev_Money_Positions_Short_All",
                                  "Leveraged Funds Shorts"]),
            errors="coerce",
        )
        result["open_interest"] = pd.to_numeric(
            self._find_col(fx, ["Open_Interest_All", "Open Interest (All)"]),
            errors="coerce",
        )
        result["net_spec"] = result["lev_long"] - result["lev_short"]
        result["net_dealer"] = result["dealer_long"] - result["dealer_short"]
        result = result.dropna(subset=["date", "currency", "net_spec"])
        result = result.sort_values(["currency", "date"]).reset_index(drop=True)
        return result

    def _fetch_current_year(self) -> pd.DataFrame:
        """
        Download the current year's disaggregated COT report from CFTC.

        The current-year endpoint (f_disagg.txt) changed sometime in 2025
        to return a **header-less** CSV, which broke column lookup and
        spammed "Cannot find contract code column" errors. Prefer the
        yearly ZIP (always header-ed); fall back to the headerless
        current-year feed only if the ZIP is unavailable.
        """
        year = datetime.now(tz=timezone.utc).year
        zip_df = self._fetch_historical_year(year)
        if not zip_df.empty:
            return zip_df
        # Fallback: raw headerless current-year feed.
        try:
            resp = requests.get(CFTC_CURRENT_YEAR_URL, timeout=30)
            resp.raise_for_status()
            df = self._parse_csv(resp.text)
            if df.empty:
                df = self._fetch_historical_year(year - 1)
            return df
        except Exception as exc:
            logger.warning("CFTC current year fetch failed: %s", exc)
            return self._fetch_historical_year(year - 1)

    def _fetch_historical_year(self, year: int) -> pd.DataFrame:
        """Download a historical year's COT zip from CFTC."""
        url = CFTC_HISTORY_URL_TEMPLATE.format(year=year)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                # The zip contains a single CSV file
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    text = f.read().decode("utf-8", errors="replace")
                    return self._parse_csv(text)
        except Exception as exc:
            logger.error("CFTC historical fetch failed for %d: %s", year, exc)
            return pd.DataFrame()

    def _parse_csv(self, text: str) -> pd.DataFrame:
        """Parse CFTC CSV text and return Gold-filtered DataFrame."""
        try:
            raw = pd.read_csv(io.StringIO(text), low_memory=False)
        except Exception as exc:
            logger.error("Failed to parse CFTC CSV: %s", exc)
            return pd.DataFrame()

        # Clean column names (CFTC CSVs have inconsistent whitespace)
        raw.columns = [c.strip() for c in raw.columns]

        # Find the contract code column
        code_col = None
        for candidate in ["CFTC_Contract_Market_Code", "CFTC Contract Market Code"]:
            if candidate in raw.columns:
                code_col = candidate
                break

        if code_col is None:
            # Try matching by partial name
            for c in raw.columns:
                if "contract" in c.lower() and "code" in c.lower():
                    code_col = c
                    break

        if code_col is None:
            logger.error("Cannot find contract code column in CFTC data")
            return pd.DataFrame()

        # Filter to Gold futures. Zero-pad codes to 6 digits to guard
        # against pandas reading numeric codes as int64 and losing a
        # leading zero (088691 → 88691). Same guard used in TFF parse.
        raw[code_col] = raw[code_col].astype(str).str.strip().str.zfill(6)
        gold = raw[raw[code_col] == GOLD_CONTRACT_CODE].copy()

        if gold.empty:
            logger.warning("No Gold futures rows found in CFTC data")
            return pd.DataFrame()

        # Extract the columns we need
        result = pd.DataFrame()
        result["date"] = pd.to_datetime(
            self._find_col(gold, ["Report_Date_as_YYYY-MM-DD", "As of Date in Form YYYY-MM-DD"]),
            errors="coerce",
        )
        result["mm_long"] = pd.to_numeric(
            self._find_col(gold, ["M_Money_Positions_Long_All", "Money Manager Longs"]),
            errors="coerce",
        )
        result["mm_short"] = pd.to_numeric(
            self._find_col(gold, ["M_Money_Positions_Short_All", "Money Manager Shorts"]),
            errors="coerce",
        )
        result["comm_long"] = pd.to_numeric(
            self._find_col(gold, ["Prod_Merc_Positions_Long_All", "Producer/Merchant/Processor/User Longs"]),
            errors="coerce",
        )
        result["comm_short"] = pd.to_numeric(
            self._find_col(gold, ["Prod_Merc_Positions_Short_All", "Producer/Merchant/Processor/User Shorts"]),
            errors="coerce",
        )
        result["open_interest"] = pd.to_numeric(
            self._find_col(gold, ["Open_Interest_All", "Open Interest (All)"]),
            errors="coerce",
        )

        result["net_spec"] = result["mm_long"] - result["mm_short"]
        result["net_comm"] = result["comm_long"] - result["comm_short"]
        result = result.dropna(subset=["date", "net_spec"]).sort_values("date")
        result.reset_index(drop=True, inplace=True)

        logger.debug("Parsed %d weeks of Gold COT data", len(result))
        return result

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
        """Find a column by trying multiple name variants."""
        for name in candidates:
            if name in df.columns:
                return df[name]
            # Try stripped/lowered match
            for col in df.columns:
                if col.strip().lower() == name.strip().lower():
                    return df[col]
        # Return zeros if not found
        return pd.Series(0, index=df.index)

    def _compute_features_xau(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute 5 COT features from parsed Gold Disaggregated data."""
        if len(df) < 2:
            return self._default_xau_features()

        latest = df.iloc[-1]
        net = latest["net_spec"]
        oi = latest["open_interest"] if latest["open_interest"] > 0 else 1

        # 1. Normalized net position
        max_net = df["net_spec"].abs().max()
        cot_net_position = float(net / max_net) if max_net > 0 else 0.0

        # 2. Z-score vs 52-week history
        recent = df["net_spec"].tail(52)
        mean = recent.mean()
        std = recent.std()
        if std > 0 and not pd.isna(std):
            zscore = float((net - mean) / std)
            cot_net_zscore = max(-2.0, min(2.0, zscore))
        else:
            cot_net_zscore = 0.0

        # 3. Week-over-week change
        prev_net = df["net_spec"].iloc[-2]
        if oi > 0:
            cot_wow_change = float((net - prev_net) / oi)
        else:
            cot_wow_change = 0.0

        # 4. Commercial ratio (hedger net as fraction of OI)
        net_comm = latest["net_comm"]
        cot_commercial_ratio = float(net_comm / oi)

        # 5. Extreme flag (contrarian)
        if cot_net_zscore > 1.5:
            cot_extreme_flag = -1.0  # specs very long → bearish contrarian
        elif cot_net_zscore < -1.5:
            cot_extreme_flag = 1.0   # specs very short → bullish contrarian
        else:
            cot_extreme_flag = 0.0

        return {
            "cot_net_position": cot_net_position,
            "cot_net_zscore_52w": cot_net_zscore,
            "cot_wow_change": cot_wow_change,
            "cot_commercial_ratio": cot_commercial_ratio,
            "cot_extreme_flag": cot_extreme_flag,
        }

    # Kept for backward compat with any external callers.
    _compute_features = _compute_features_xau

    def _compute_features_fx(
        self, df: pd.DataFrame, currency: str,
    ) -> dict[str, float]:
        """
        Compute 5 COT features for one FX currency from parsed TFF data.

        Returns keys prefixed with the currency (lowercase), e.g.
        cot_eur_net_position / cot_eur_net_zscore_52w / ... / cot_eur_extreme_flag.
        """
        sub = df[df["currency"] == currency]
        if len(sub) < 2:
            return self._default_fx_features((currency,))

        sub = sub.sort_values("date")
        latest = sub.iloc[-1]
        net = latest["net_spec"]
        oi = latest["open_interest"] if latest["open_interest"] > 0 else 1

        max_net = sub["net_spec"].abs().max()
        cot_net_position = float(net / max_net) if max_net > 0 else 0.0

        recent = sub["net_spec"].tail(52)
        mean = recent.mean()
        std = recent.std()
        if std > 0 and not pd.isna(std):
            zscore = float((net - mean) / std)
            cot_net_zscore = max(-2.0, min(2.0, zscore))
        else:
            cot_net_zscore = 0.0

        prev_net = sub["net_spec"].iloc[-2]
        cot_wow_change = float((net - prev_net) / oi) if oi > 0 else 0.0

        # Dealer ratio (TFF analogue of commercial_ratio)
        net_dealer = latest["net_dealer"]
        cot_dealer_ratio = float(net_dealer / oi)

        if cot_net_zscore > 1.5:
            cot_extreme_flag = -1.0
        elif cot_net_zscore < -1.5:
            cot_extreme_flag = 1.0
        else:
            cot_extreme_flag = 0.0

        ccy = currency.lower()
        return {
            f"cot_{ccy}_net_position":     cot_net_position,
            f"cot_{ccy}_net_zscore_52w":   cot_net_zscore,
            f"cot_{ccy}_wow_change":       cot_wow_change,
            f"cot_{ccy}_dealer_ratio":     cot_dealer_ratio,
            f"cot_{ccy}_extreme_flag":     cot_extreme_flag,
        }

    # ------------------------------------------------------------------
    # feature_store backfill (Phase 1F)
    # ------------------------------------------------------------------

    FEATURE_GROUP_XAU = "cot_disagg"
    FEATURE_GROUP_FX  = "cot_tff"
    SCHEMA_VERSION    = 1
    # Backfill: fetch 25 yearly zip files (CFTC publishes per-year archives
    # going back to 2006 for TFF, ~1995 for Disaggregated). Years before the
    # archive cutoff return empty and are skipped.
    BACKFILL_YEARS    = 25

    def _fetch_multi_year_disagg(self, years: int) -> pd.DataFrame:
        """Concatenate the last N years of disaggregated (XAU) reports."""
        cur = datetime.now(tz=timezone.utc).year
        frames = []
        for yr in range(cur - years + 1, cur + 1):
            df = self._fetch_historical_year(yr)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _fetch_multi_year_tff(self, years: int) -> pd.DataFrame:
        """Concatenate the last N years of TFF (FX) reports."""
        cur = datetime.now(tz=timezone.utc).year
        frames = []
        for yr in range(cur - years + 1, cur + 1):
            df = self._fetch_tff_historical_year(yr)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    async def persist_raw_history_to_feature_store(
        self, store: "DataStore", symbol: str,
        *,
        force: bool = False,
        lookback_days: Optional[int] = None,
    ) -> int:
        """
        Write the weekly raw COT report rows for this symbol to feature_store.

        XAU paths: Disaggregated report → ``cot_disagg`` group, one row per
            weekly report date with raw position counts.
        FX paths: TFF report → ``cot_tff`` group, one row per weekly report
            date with raw dealer + leveraged-fund positions for each
            currency this symbol is exposed to.

        Args:
            force: When True, switches conflict policy from DO NOTHING to
                DO UPDATE — used by the weekly TTL safety-net job.
            lookback_days: When set, only persist rows from the last N days.
                Default None = full 25-yr history.

        Returns rows touched. Stores RAW positions; z-scores derived
        downstream so retraining doesn't require refetching CFTC.
        """
        sym_upper = symbol.upper()
        if sym_upper.startswith("XAU"):
            return await self._persist_xau(
                store, symbol, force=force, lookback_days=lookback_days,
            )

        currencies = _SYMBOL_CURRENCIES.get(sym_upper)
        if not currencies:
            return 0   # symbols outside FX universe (ETH etc.) get nothing
        return await self._persist_fx(
            store, symbol, currencies, force=force, lookback_days=lookback_days,
        )

    async def _persist_xau(
        self, store: "DataStore", symbol: str,
        *, force: bool = False, lookback_days: Optional[int] = None,
    ) -> int:
        df = self._fetch_multi_year_disagg(self.BACKFILL_YEARS)
        if df.empty:
            return 0
        if lookback_days is not None:
            cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
            df = df[df["date"] >= cutoff]
            if df.empty:
                return 0

        rows = []
        for _, raw in df.iterrows():
            ts = raw.get("date")
            if pd.isna(ts):
                continue
            values = {
                k: float(raw[k])
                for k in (
                    "mm_long", "mm_short", "comm_long", "comm_short",
                    "open_interest", "net_spec", "net_comm",
                )
                if k in raw and not pd.isna(raw[k])
            }
            if not values:
                continue
            rows.append({
                "symbol":         symbol,
                "timestamp":      ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                "feature_group":  self.FEATURE_GROUP_XAU,
                "values":         values,
                "schema_version": self.SCHEMA_VERSION,
            })
        if not rows:
            return 0
        return await store.upsert_feature_store_bulk(
            rows, mode=("overwrite" if force else "skip"),
        )

    async def _persist_fx(
        self, store: "DataStore", symbol: str, currencies: tuple[str, ...],
        *, force: bool = False, lookback_days: Optional[int] = None,
    ) -> int:
        df = self._fetch_multi_year_tff(self.BACKFILL_YEARS)
        if df.empty:
            return 0

        # Filter to the symbol's exposed currencies, then reshape to one
        # row per date with all currencies' positions in a single values dict.
        sub = df[df["currency"].isin(currencies)].copy()
        if lookback_days is not None and not sub.empty:
            cutoff = sub["date"].max() - pd.Timedelta(days=lookback_days)
            sub = sub[sub["date"] >= cutoff]
        if sub.empty:
            return 0

        per_date: dict[pd.Timestamp, dict[str, float]] = {}
        for _, raw in sub.iterrows():
            ts = raw.get("date")
            if pd.isna(ts):
                continue
            ccy = str(raw["currency"]).lower()
            bucket = per_date.setdefault(pd.Timestamp(ts), {})
            for k in ("dealer_long", "dealer_short", "lev_long", "lev_short",
                      "open_interest", "net_spec", "net_dealer"):
                if k in raw and not pd.isna(raw[k]):
                    bucket[f"{ccy}_{k}"] = float(raw[k])

        if not per_date:
            return 0

        rows = []
        for ts, values in per_date.items():
            if not values:
                continue
            rows.append({
                "symbol":         symbol,
                "timestamp":      ts.to_pydatetime(),
                "feature_group":  self.FEATURE_GROUP_FX,
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

    # Live engineering uses .tail(52) for z-score window and clips to ±2.
    _COT_ZSCORE_LOOKBACK = 52
    _COT_ZSCORE_CLIP     = 2.0
    _COT_EXTREME_THRESH  = 1.5

    async def get_historical_cot_features(
        self,
        store: "DataStore",
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        feeds_config: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical COT features over a date range from feature_store.

        Returns a DataFrame indexed by report timestamp with the SAME
        feature columns ``get_cot_features(symbol)`` emits at live time.
        Dispatches by symbol type (mirrors live):

            * XAUUSD → 5 features from cot_disagg feature_group
            * FX pairs → 5 features per exposed currency from cot_tff
            * Other (XXX/USD where XXX not FX, ETH, …) → empty DataFrame

        Lookahead-safety: each raw observation's timestamp is shifted
        forward by ``release_lag_hours`` (504h for COT — CFTC publishes
        Friday afternoon for Tuesday data, plus weekend buffer). Rolling
        z-score / max / shift operations are point-in-time-correct.
        """
        sym_upper = symbol.upper()

        if sym_upper.startswith("XAU"):
            return await self._historical_xau(
                store, symbol, start, end, feeds_config=feeds_config,
            )

        currencies = _SYMBOL_CURRENCIES.get(sym_upper)
        if not currencies:
            return pd.DataFrame()

        return await self._historical_fx(
            store, symbol, start, end, currencies, feeds_config=feeds_config,
        )

    async def _historical_xau(
        self,
        store: "DataStore",
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        feeds_config: Optional[dict],
    ) -> pd.DataFrame:
        from src.data_pipeline.feature_engineering import _load_data_feeds_yaml

        cfg = feeds_config if feeds_config is not None else _load_data_feeds_yaml()
        src_cfg = cfg.get("sources", {}).get(self.FEATURE_GROUP_XAU)
        if src_cfg is None:
            raise ValueError(
                f"feature_group {self.FEATURE_GROUP_XAU!r} not in data_feeds.yaml — "
                "refusing to query without a release-lag bound (lookahead risk)."
            )
        lag_hours = float(src_cfg.get("release_lag_hours") or 0.0)

        # COT is weekly. 52 obs warmup ≈ 1 year — extra-pad to 400 days
        # to be safe across holiday-shifted publication weeks.
        raw_start = start - timedelta(days=400)
        raw_df = await store.read_feature_store(
            symbol=symbol,
            feature_group=self.FEATURE_GROUP_XAU,
            start=raw_start,
            end=end,
        )
        if raw_df.empty:
            logger.warning(
                "feature_store[%s] returned no rows for %s in [%s, %s]",
                self.FEATURE_GROUP_XAU, symbol, raw_start, end,
            )
            return pd.DataFrame()

        raw_df = raw_df.copy()
        raw_df.index = raw_df.index + pd.Timedelta(hours=lag_hours)

        engineered = self._engineer_xau_features_from_raw(raw_df)
        if engineered.empty:
            return engineered
        engineered = engineered[
            (engineered.index >= pd.Timestamp(start))
            & (engineered.index <= pd.Timestamp(end))
        ]
        return engineered

    async def _historical_fx(
        self,
        store: "DataStore",
        symbol: str,
        start: datetime,
        end: datetime,
        currencies: tuple[str, ...],
        *,
        feeds_config: Optional[dict],
    ) -> pd.DataFrame:
        from src.data_pipeline.feature_engineering import _load_data_feeds_yaml

        cfg = feeds_config if feeds_config is not None else _load_data_feeds_yaml()
        src_cfg = cfg.get("sources", {}).get(self.FEATURE_GROUP_FX)
        if src_cfg is None:
            raise ValueError(
                f"feature_group {self.FEATURE_GROUP_FX!r} not in data_feeds.yaml — "
                "refusing to query without a release-lag bound (lookahead risk)."
            )
        lag_hours = float(src_cfg.get("release_lag_hours") or 0.0)

        raw_start = start - timedelta(days=400)
        raw_df = await store.read_feature_store(
            symbol=symbol,
            feature_group=self.FEATURE_GROUP_FX,
            start=raw_start,
            end=end,
        )
        if raw_df.empty:
            logger.warning(
                "feature_store[%s] returned no rows for %s in [%s, %s]",
                self.FEATURE_GROUP_FX, symbol, raw_start, end,
            )
            return pd.DataFrame()

        raw_df = raw_df.copy()
        raw_df.index = raw_df.index + pd.Timedelta(hours=lag_hours)

        engineered = self._engineer_fx_features_from_raw(raw_df, currencies)
        if engineered.empty:
            return engineered
        engineered = engineered[
            (engineered.index >= pd.Timestamp(start))
            & (engineered.index <= pd.Timestamp(end))
        ]
        return engineered

    def _engineer_xau_features_from_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the 5 XAU COT features from a wide raw-observation DataFrame
        (columns: net_spec, open_interest, net_comm, plus the underlying
        long/short positions). Mirrors live ``_compute_features_xau``.
        """
        if raw.empty or "net_spec" not in raw.columns:
            return pd.DataFrame()

        out = pd.DataFrame(index=raw.index)
        net = raw["net_spec"].astype(float)
        oi = raw.get("open_interest")
        if oi is None:
            oi = pd.Series(1.0, index=raw.index)
        # Match live's `if open_interest > 0 else 1` guard
        oi = oi.astype(float).where(oi.astype(float) > 0, 1.0)

        # 1. cot_net_position = net / expanding-max(|net|) — point-in-time
        max_abs_net = net.abs().expanding(min_periods=1).max()
        out["cot_net_position"] = (net / max_abs_net.replace(0.0, pd.NA)).fillna(0.0).astype(float)

        # 2. cot_net_zscore_52w — rolling z, clip ±2
        out["cot_net_zscore_52w"] = self._rolling_zscore_clipped(
            net, self._COT_ZSCORE_LOOKBACK, self._COT_ZSCORE_CLIP,
        )

        # 3. cot_wow_change = (net - prev_net) / open_interest
        out["cot_wow_change"] = ((net - net.shift(1)) / oi).fillna(0.0).astype(float)

        # 4. cot_commercial_ratio = net_comm / open_interest
        net_comm = raw.get("net_comm")
        if net_comm is not None:
            out["cot_commercial_ratio"] = (net_comm.astype(float) / oi).fillna(0.0).astype(float)
        else:
            out["cot_commercial_ratio"] = 0.0

        # 5. cot_extreme_flag from zscore (contrarian)
        out["cot_extreme_flag"] = self._extreme_flag_from_zscore(out["cot_net_zscore_52w"])

        return out.sort_index()

    def _engineer_fx_features_from_raw(
        self, raw: pd.DataFrame, currencies: tuple[str, ...],
    ) -> pd.DataFrame:
        """
        Compute 5 features per exposed currency from a wide TFF raw DataFrame
        (columns prefixed by lowercase currency code, e.g. eur_net_spec,
        eur_open_interest, eur_net_dealer). Mirrors live ``_compute_features_fx``
        but vectorized over time per currency.
        """
        if raw.empty:
            return pd.DataFrame()

        out = pd.DataFrame(index=raw.index)
        emitted = False

        for ccy in currencies:
            ccy_lc = ccy.lower()
            net_col = f"{ccy_lc}_net_spec"
            oi_col = f"{ccy_lc}_open_interest"
            dealer_col = f"{ccy_lc}_net_dealer"

            if net_col not in raw.columns:
                # No data for this currency — emit zeros so the schema still matches.
                for k in ("net_position", "net_zscore_52w", "wow_change",
                          "dealer_ratio", "extreme_flag"):
                    out[f"cot_{ccy_lc}_{k}"] = 0.0
                continue

            net = raw[net_col].astype(float)
            oi = raw.get(oi_col)
            if oi is None:
                oi = pd.Series(1.0, index=raw.index)
            oi = oi.astype(float).where(oi.astype(float) > 0, 1.0)

            max_abs_net = net.abs().expanding(min_periods=1).max()
            out[f"cot_{ccy_lc}_net_position"] = (
                net / max_abs_net.replace(0.0, pd.NA)
            ).fillna(0.0).astype(float)

            out[f"cot_{ccy_lc}_net_zscore_52w"] = self._rolling_zscore_clipped(
                net, self._COT_ZSCORE_LOOKBACK, self._COT_ZSCORE_CLIP,
            )

            out[f"cot_{ccy_lc}_wow_change"] = (
                (net - net.shift(1)) / oi
            ).fillna(0.0).astype(float)

            net_dealer = raw.get(dealer_col)
            if net_dealer is not None:
                out[f"cot_{ccy_lc}_dealer_ratio"] = (
                    net_dealer.astype(float) / oi
                ).fillna(0.0).astype(float)
            else:
                out[f"cot_{ccy_lc}_dealer_ratio"] = 0.0

            out[f"cot_{ccy_lc}_extreme_flag"] = self._extreme_flag_from_zscore(
                out[f"cot_{ccy_lc}_net_zscore_52w"],
            )
            emitted = True

        if not emitted and not currencies:
            return pd.DataFrame()
        return out.sort_index()

    @staticmethod
    def _rolling_zscore_clipped(
        series: pd.Series, lookback: int, clip: float,
    ) -> pd.Series:
        """Rolling z-score with [-clip, +clip] clamping. Matches live's
        ``zscore = (net - mean) / std; clip(±2.0)`` over ``df.tail(52)``."""
        # min_periods=2 matches live's `if std > 0 and not pd.isna(std)` guard
        # (std needs at least 2 observations to be defined).
        roll = series.rolling(lookback, min_periods=2)
        mean = roll.mean()
        std = roll.std()
        z = (series - mean) / std
        import numpy as np
        z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return z.clip(-clip, clip).astype(float)

    def _extreme_flag_from_zscore(self, zscore: pd.Series) -> pd.Series:
        """Match live's contrarian flag:
            zscore > +1.5 → -1.0 (specs very long → bearish contrarian)
            zscore < -1.5 → +1.0 (specs very short → bullish contrarian)
            else → 0.0
        """
        flag = pd.Series(0.0, index=zscore.index)
        flag = flag.where(zscore <= self._COT_EXTREME_THRESH, -1.0)
        flag = flag.where(zscore >= -self._COT_EXTREME_THRESH, 1.0)
        return flag.astype(float)

    @staticmethod
    def _default_xau_features() -> dict[str, float]:
        """Neutral defaults when Gold COT data is unavailable."""
        return {
            "cot_net_position": 0.0,
            "cot_net_zscore_52w": 0.0,
            "cot_wow_change": 0.0,
            "cot_commercial_ratio": 0.0,
            "cot_extreme_flag": 0.0,
        }

    # Kept for backward compat with any external callers.
    _default_features = _default_xau_features

    @staticmethod
    def _default_fx_features(currencies: tuple[str, ...]) -> dict[str, float]:
        """Neutral defaults when FX TFF data is unavailable."""
        feats: dict[str, float] = {}
        for ccy in currencies:
            ccy_lc = ccy.lower()
            feats[f"cot_{ccy_lc}_net_position"] = 0.0
            feats[f"cot_{ccy_lc}_net_zscore_52w"] = 0.0
            feats[f"cot_{ccy_lc}_wow_change"] = 0.0
            feats[f"cot_{ccy_lc}_dealer_ratio"] = 0.0
            feats[f"cot_{ccy_lc}_extreme_flag"] = 0.0
        return feats
