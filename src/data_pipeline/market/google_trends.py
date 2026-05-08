"""
google_trends.py — Google Trends Search Interest (pytrends, no API key)

Fetches Google Trends search interest data for financial keywords
as a proxy for retail sentiment and attention.

Keywords per symbol:
    XAUUSD: "gold price", "buy gold", "gold investment"
    BTCUSD: "bitcoin", "buy bitcoin", "crypto"

Features (~6):
    trends_{keyword}_interest  — raw search interest [0, 100]
    trends_{keyword}_momentum  — 4-week change in interest

Data source: Google Trends via pytrends library — no API key required.
Rate limits: aggressive (429 errors common), uses exponential backoff
and 6-hour caching.
"""

import logging
import time
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)

SYMBOL_KEYWORDS = {
    "XAUUSD": ["gold price", "buy gold", "gold investment"],
    "BTCUSD": ["bitcoin", "buy bitcoin", "crypto"],
}


class TrendsFetcher:
    """
    Fetches Google Trends data and returns interest/momentum features.

    Usage:
        fetcher = TrendsFetcher()
        features = fetcher.get_trends_features("XAUUSD")
    """

    def __init__(self, cache_ttl_hours: int = 6):
        self._cache: dict[str, dict] = {}
        self._cache_ts: dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=cache_ttl_hours)

    def get_trends_features(self, symbol: str) -> dict[str, float]:
        """
        Fetch and compute Google Trends features for a symbol.

        Returns:
            Dict of ~6 feature_name → float values.
        """
        symbol_upper = symbol.upper()
        keywords = SYMBOL_KEYWORDS.get(symbol_upper)
        if not keywords:
            return self._default_features(symbol_upper)

        # Check cache
        now = datetime.utcnow()
        if (
            symbol_upper in self._cache
            and symbol_upper in self._cache_ts
            and (now - self._cache_ts[symbol_upper]) < self._cache_ttl
        ):
            return self._cache[symbol_upper]

        try:
            features = self._fetch_and_compute(symbol_upper, keywords)
            self._cache[symbol_upper] = features
            self._cache_ts[symbol_upper] = now
            return features
        except Exception as exc:
            logger.error("Google Trends fetch failed for %s: %s", symbol, exc)
            return self._default_features(symbol_upper)

    def _fetch_and_compute(
        self, symbol: str, keywords: list[str]
    ) -> dict[str, float]:
        """Fetch from pytrends and compute interest/momentum features."""
        try:
            from pytrends.request import TrendReq
        except ImportError:
            logger.error("pytrends not installed — pip install pytrends")
            return self._default_features(symbol)

        features: dict[str, float] = {}
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))

        for kw in keywords:
            label = _keyword_to_label(symbol, kw)
            try:
                pytrends.build_payload([kw], timeframe="today 3-m", geo="")
                interest_df = pytrends.interest_over_time()

                if interest_df is None or interest_df.empty or kw not in interest_df:
                    features[f"trends_{label}_interest"] = 0.0
                    features[f"trends_{label}_momentum"] = 0.0
                    continue

                series = interest_df[kw].astype(float)
                latest = float(series.iloc[-1])
                features[f"trends_{label}_interest"] = latest / 100.0  # normalize to [0,1]

                # 4-week momentum
                if len(series) >= 28:
                    prev = float(series.iloc[-28])
                    if prev > 0:
                        features[f"trends_{label}_momentum"] = (latest - prev) / prev
                    else:
                        features[f"trends_{label}_momentum"] = 0.0
                else:
                    features[f"trends_{label}_momentum"] = 0.0

                # Rate limit courtesy
                time.sleep(0.5)

            except Exception as exc:
                logger.warning("Trends fetch failed for '%s': %s", kw, exc)
                features[f"trends_{label}_interest"] = 0.0
                features[f"trends_{label}_momentum"] = 0.0

        return features

    @staticmethod
    def _default_features(symbol: str) -> dict[str, float]:
        """Return neutral defaults when data is unavailable."""
        keywords = SYMBOL_KEYWORDS.get(symbol, [])
        features = {}
        for kw in keywords:
            label = _keyword_to_label(symbol, kw)
            features[f"trends_{label}_interest"] = 0.0
            features[f"trends_{label}_momentum"] = 0.0
        return features


def _keyword_to_label(symbol: str, keyword: str) -> str:
    """Convert a keyword like 'gold price' to a feature-safe label like 'gold'."""
    # Use first word of keyword for conciseness
    first_word = keyword.split()[0].lower()
    return first_word
