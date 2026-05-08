"""
fear_greed.py — Crypto Fear & Greed Index (alternative.me, no API key)

Fetches the Crypto Fear & Greed Index — a composite sentiment
metric combining:
    - Volatility (25%)
    - Market momentum/volume (25%)
    - Social media (15%)
    - Surveys (15%)
    - BTC dominance (10%)
    - Google Trends (10%)

Data source: alternative.me API — free, no API key required.
Rate limits: generous (no documented limit, but we cache for 1 hour).

Output: 2 features (BTC only, returns 0 for other symbols):
    fear_greed_index   — [0, 100], 0 = extreme fear, 100 = extreme greed
    fear_greed_change  — day-over-day change in index
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/"


class FearGreedFetcher:
    """
    Fetches Crypto Fear & Greed Index from alternative.me.

    Only relevant for BTCUSD — returns empty dict for other symbols.

    Usage:
        fetcher = FearGreedFetcher()
        features = fetcher.get_fear_greed_features("BTCUSD")
    """

    def __init__(self, cache_ttl_hours: int = 1):
        self._cache: Optional[dict[str, float]] = None
        self._cache_ts: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=cache_ttl_hours)

    def get_fear_greed_features(self, symbol: str) -> dict[str, float]:
        """
        Fetch Fear & Greed Index features for BTCUSD.

        Returns empty dict for non-BTC symbols.

        Returns:
            Dict with keys: fear_greed_index, fear_greed_change
        """
        if not symbol.upper().startswith("BTC"):
            return {}

        now = datetime.utcnow()
        if (
            self._cache
            and self._cache_ts
            and (now - self._cache_ts) < self._cache_ttl
        ):
            return self._cache

        try:
            features = self._fetch()
            self._cache = features
            self._cache_ts = now
            return features
        except Exception as exc:
            logger.error("Fear & Greed fetch failed: %s", exc)
            return {"fear_greed_index": 50.0, "fear_greed_change": 0.0}

    def _fetch(self) -> dict[str, float]:
        """Fetch last 2 days of F&G data from alternative.me."""
        resp = requests.get(
            FEAR_GREED_URL,
            params={"limit": 2, "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        if not data:
            return {"fear_greed_index": 50.0, "fear_greed_change": 0.0}

        today_val = float(data[0].get("value", 50))
        yesterday_val = float(data[1].get("value", 50)) if len(data) > 1 else today_val

        # Normalize to [0, 1] for model consumption
        features = {
            "fear_greed_index": today_val / 100.0,
            "fear_greed_change": (today_val - yesterday_val) / 100.0,
        }

        logger.debug(
            "Fear & Greed: %.0f (change: %+.0f)",
            today_val, today_val - yesterday_val,
        )
        return features
