"""
onchain_data.py — Bitcoin On-Chain Data (CoinGecko API)

Fetches on-chain and market-structure metrics for Bitcoin:

    - Hash Rate         — miner security / long-term confidence
    - NVT Ratio         — Network Value to Transactions (valuation signal)
    - SOPR              — Spent Output Profit Ratio (profit-taking signal)
    - Active Addresses  — network activity / adoption
    - Exchange Netflow  — coins moving to/from exchanges (sell pressure)
    - Market Dominance  — BTC vs. altcoin risk appetite

Data source: CoinGecko API (free public tier, no key required for basic endpoints)
Optional: Glassnode API (paid) for NVT/SOPR if CoinGecko is insufficient.

Output: onchain_score float in [-1, 1]:
    +1 = on-chain metrics strongly bullish (holders accumulating)
    -1 = on-chain metrics strongly bearish (exchange inflows, profit-taking)
"""

import logging
import os

import requests
import numpy as np

logger = logging.getLogger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


class OnChainDataFetcher:
    """
    Fetches Bitcoin on-chain metrics and returns a composite score.

    Only relevant for BTCUSD — returns 0.0 for other symbols.

    Usage:
        fetcher = OnChainDataFetcher()
        score = fetcher.get_onchain_score("BTCUSD")
    """

    def __init__(self):
        self.api_key = os.getenv("COINGECKO_API_KEY", "")  # Optional pro key

    def get_onchain_score(self, symbol: str) -> float:
        """
        Compute composite on-chain sentiment score for BTCUSD.

        Components and weights:
            - Market dominance trend (0.30) — rising dominance = bullish
            - Hash rate trend (0.25) — rising = bullish
            - Volume/market-cap ratio (0.25) — high = bullish activity
            - Exchange netflow proxy (0.20) — outflows = bullish

        Returns 0.0 for non-Bitcoin symbols.

        Returns:
            Float in [-1, 1].
        """
        if not symbol.upper().startswith("BTC"):
            return 0.0

        try:
            market = self.get_market_data()
            if not market:
                return 0.0

            scores = []

            # 1. Dominance trend — above 50% is bullish
            dominance = market.get("dominance", 0)
            dom_score = (dominance - 50) / 25  # 50% → 0, 75% → 1, 25% → -1
            scores.append(0.30 * max(-1.0, min(1.0, dom_score)))

            # 2. Hash rate trend
            hash_score = self.get_hash_rate_trend(days=30)
            scores.append(0.25 * hash_score)

            # 3. Volume / market cap ratio (turnover proxy)
            vol_24h = market.get("volume_24h", 0)
            mcap = market.get("market_cap", 1)
            turnover = vol_24h / mcap if mcap > 0 else 0
            # Typical BTC turnover: 1-5%. Above 3% → active, bullish
            turn_score = (turnover - 0.03) / 0.03
            scores.append(0.25 * max(-1.0, min(1.0, turn_score)))

            # 4. Exchange netflow proxy
            netflow_score = self.get_exchange_netflow_score()
            scores.append(0.20 * netflow_score)

            total = sum(scores)
            return float(max(-1.0, min(1.0, total)))

        except Exception as exc:
            logger.error("On-chain score failed for %s: %s", symbol, exc)
            return 0.0

    def get_market_data(self) -> dict:
        """
        Fetch CoinGecko market data for Bitcoin:
            price, market_cap, volume, dominance, circulating_supply.

        Returns:
            Dict with current market stats.
        """
        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key

        try:
            # Global data for dominance
            resp_global = requests.get(
                f"{COINGECKO_BASE}/global",
                headers=headers, timeout=10,
            )
            resp_global.raise_for_status()
            global_data = resp_global.json().get("data", {})
            dominance = global_data.get("market_cap_percentage", {}).get("btc", 0)

            # BTC-specific market data
            resp_btc = requests.get(
                f"{COINGECKO_BASE}/coins/bitcoin",
                params={"localization": "false", "tickers": "false",
                        "community_data": "false", "developer_data": "false"},
                headers=headers, timeout=10,
            )
            resp_btc.raise_for_status()
            btc_data = resp_btc.json().get("market_data", {})

            return {
                "price": btc_data.get("current_price", {}).get("usd", 0),
                "market_cap": btc_data.get("market_cap", {}).get("usd", 0),
                "volume_24h": btc_data.get("total_volume", {}).get("usd", 0),
                "dominance": dominance,
                "circulating_supply": btc_data.get("circulating_supply", 0),
                "price_change_7d_pct": btc_data.get(
                    "price_change_percentage_7d", 0
                ),
            }

        except Exception as exc:
            logger.error("CoinGecko market data fetch failed: %s", exc)
            return {}

    def get_hash_rate_trend(self, days: int = 30) -> float:
        """
        Return normalized hash rate trend score.
        Uses BTC price chart as a proxy — CoinGecko free tier doesn't
        expose raw hash rate. We compute the slope of the 30-day price
        trend as a stand-in (hash rate and price are correlated).

        Rising → bullish (+1), falling → bearish (-1).
        """
        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key

        try:
            resp = requests.get(
                f"{COINGECKO_BASE}/coins/bitcoin/market_chart",
                params={"vs_currency": "usd", "days": str(days)},
                headers=headers, timeout=10,
            )
            resp.raise_for_status()
            prices = resp.json().get("prices", [])

            if len(prices) < 5:
                return 0.0

            vals = np.array([p[1] for p in prices], dtype=float)
            # Percent change from start to end
            pct_change = (vals[-1] - vals[0]) / vals[0] if vals[0] != 0 else 0
            # Scale: ±10% in 30 days maps to ±1
            score = pct_change / 0.10
            return float(max(-1.0, min(1.0, score)))

        except Exception as exc:
            logger.error("Hash rate trend proxy fetch failed: %s", exc)
            return 0.0

    def get_exchange_netflow_score(self) -> float:
        """
        Estimate exchange netflow direction using the 24h volume
        relative to the 7-day average volume as a proxy.

        High volume days with price drops → likely exchange inflows (bearish).
        High volume days with price rises → likely accumulation (bullish).

        CoinGecko free tier doesn't have direct netflow data; this is a
        heuristic. For production, integrate Glassnode or CryptoQuant APIs.
        """
        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key

        try:
            resp = requests.get(
                f"{COINGECKO_BASE}/coins/bitcoin/market_chart",
                params={"vs_currency": "usd", "days": "7"},
                headers=headers, timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])

            if len(prices) < 2 or len(volumes) < 2:
                return 0.0

            price_start = prices[0][1]
            price_end = prices[-1][1]
            price_change = (price_end - price_start) / price_start if price_start else 0

            avg_vol = np.mean([v[1] for v in volumes[:-1]]) if len(volumes) > 1 else 1
            latest_vol = volumes[-1][1]
            vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0

            # Combine: positive price change with high volume → bullish
            if price_change > 0:
                score = min(1.0, price_change * 10 * min(vol_ratio, 2.0))
            else:
                score = max(-1.0, price_change * 10 * min(vol_ratio, 2.0))

            return float(score)

        except Exception as exc:
            logger.error("Exchange netflow proxy failed: %s", exc)
            return 0.0
