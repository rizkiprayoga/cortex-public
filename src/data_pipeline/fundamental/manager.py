"""
manager.py — FundamentalDataManager (coordinator for all data fetchers)

Coordinates all fundamental, cross-asset, calendar, and alternative
data fetchers. Runs on a schedule (1-6h refresh), caches results in
memory, and exposes a single method to get all external features for
a given symbol and bar timestamp.

Design principle: No external API call ever blocks the trading tick.
All data is pre-fetched and cached. The trading loop reads only from
cache — if a fetcher fails, its features gracefully degrade to neutral
defaults (0.0 or category-specific neutral values).

Feature breakdown by source (~55 external features):
    Cross-asset (yfinance):  ~12 features
    Macro (FRED):            ~12 features
    Calendar (computed):     ~10 features
    COT (CFTC CSV):          ~5 features (XAU only)
    Sentiment (NewsAPI):     ~1 feature
    On-chain (CoinGecko):    ~1 feature
    Trends (pytrends):       ~6 features
    Fear & Greed:            ~2 features (BTC only)
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class FundamentalDataManager:
    """
    Coordinates all external data fetchers and returns a unified
    feature dict for any symbol at any bar timestamp.

    Usage:
        manager = FundamentalDataManager()
        features = manager.get_all_features("XAUUSD", bar_timestamp)
    """

    def __init__(self):
        # Lazy-init fetchers to avoid import errors if optional deps missing
        self._cross_asset = None
        self._calendar = None
        self._cot = None
        self._macro = None
        self._stooq = None       # Phase 2A: sovereign yields (UK/EU/JP/AU/NZ)
        self._ecb = None         # Phase 2A: ECB AAA curve (EUR pairs only)
        self._sentiment = None
        self._onchain = None
        self._trends = None
        self._fear_greed = None
        self._init_errors: list[str] = []

    def init_fetchers(self) -> None:
        """
        Initialize all fetchers. Call once at startup.
        Logs errors for unavailable fetchers but doesn't crash.
        """
        # Calendar — always available (no deps)
        try:
            from src.data_pipeline.market.calendar_features import CalendarFeatureBuilder
            self._calendar = CalendarFeatureBuilder()
        except Exception as exc:
            self._init_errors.append(f"Calendar: {exc}")

        # Cross-asset (yfinance)
        try:
            from src.data_pipeline.market.cross_asset import CrossAssetFetcher
            self._cross_asset = CrossAssetFetcher()
        except Exception as exc:
            self._init_errors.append(f"CrossAsset: {exc}")

        # COT (CFTC direct — no API key)
        try:
            from src.data_pipeline.fundamental.cot_data import COTDataFetcher
            self._cot = COTDataFetcher()
        except Exception as exc:
            self._init_errors.append(f"COT: {exc}")

        # Macro (FRED — needs FRED_API_KEY)
        try:
            from src.data_pipeline.fundamental.macro_data import MacroDataFetcher
            self._macro = MacroDataFetcher()
        except EnvironmentError:
            self._init_errors.append("Macro: FRED_API_KEY not set")
        except Exception as exc:
            self._init_errors.append(f"Macro: {exc}")

        # Stooq (sovereign yields — STOOQ_API_KEY optional, falls back to
        # zero-defaults on failure). Phase 2A: previously only persisted to
        # feature_store; now also fed into the live feature dict so live and
        # historical training paths see the same column set.
        try:
            from src.data_pipeline.market.stooq_data import StooqFetcher
            self._stooq = StooqFetcher()
        except Exception as exc:
            self._init_errors.append(f"Stooq: {exc}")

        # ECB AAA curve (no key — public ECB Data Portal). Phase 2A: same
        # rationale as Stooq. Only EUR pairs receive features at live time.
        try:
            from src.data_pipeline.market.ecb_data import ECBDataFetcher
            self._ecb = ECBDataFetcher()
        except Exception as exc:
            self._init_errors.append(f"ECB: {exc}")

        # News sentiment (NewsAPI + FinBERT)
        try:
            from src.data_pipeline.fundamental.news_sentiment import NewsSentimentAnalyzer
            self._sentiment = NewsSentimentAnalyzer()
        except EnvironmentError:
            self._init_errors.append("Sentiment: NEWS_API_KEY not set")
        except Exception as exc:
            self._init_errors.append(f"Sentiment: {exc}")

        # On-chain (CoinGecko — no key required)
        try:
            from src.data_pipeline.fundamental.onchain_data import OnChainDataFetcher
            self._onchain = OnChainDataFetcher()
        except Exception as exc:
            self._init_errors.append(f"OnChain: {exc}")

        # Google Trends (pytrends — no key)
        try:
            from src.data_pipeline.market.google_trends import TrendsFetcher
            self._trends = TrendsFetcher()
        except Exception as exc:
            self._init_errors.append(f"Trends: {exc}")

        # Fear & Greed (alternative.me — no key)
        try:
            from src.data_pipeline.market.fear_greed import FearGreedFetcher
            self._fear_greed = FearGreedFetcher()
        except Exception as exc:
            self._init_errors.append(f"FearGreed: {exc}")

        if self._init_errors:
            logger.warning(
                "FundamentalDataManager: %d fetchers unavailable: %s",
                len(self._init_errors),
                "; ".join(self._init_errors),
            )
        else:
            logger.info("FundamentalDataManager: all fetchers initialized")

    def get_all_features(
        self,
        symbol: str,
        bar_timestamp: Optional[datetime] = None,
    ) -> dict[str, float]:
        """
        Return all external features for a symbol at a given bar timestamp.

        Each fetcher runs independently — if one fails, the rest still
        contribute their features. Failed features default to neutral values.

        Args:
            symbol:        Trading symbol ("XAUUSD" or "BTCUSD")
            bar_timestamp: UTC datetime of the current bar (for calendar features)

        Returns:
            Dict of feature_name → float (typically ~40-55 features
            depending on symbol).
        """
        if bar_timestamp is None:
            bar_timestamp = datetime.utcnow()

        features: dict[str, float] = {}

        # 1. Calendar features (pure computation, always works)
        if self._calendar:
            try:
                features.update(self._calendar.get_calendar_features(bar_timestamp))
            except Exception as exc:
                logger.error("Calendar features failed: %s", exc)

        # 2. Cross-asset features (yfinance)
        if self._cross_asset:
            try:
                features.update(self._cross_asset.get_cross_asset_features(symbol))
            except Exception as exc:
                logger.error("Cross-asset features failed: %s", exc)

        # 3. Macro features (FRED)
        if self._macro:
            try:
                features.update(self._macro.get_macro_features(symbol))
                # Also include backward-compat composite score
                features["macro_score"] = self._macro.get_macro_score(symbol)
            except Exception as exc:
                logger.error("Macro features failed: %s", exc)

        # 4. COT features (XAU only)
        if self._cot:
            try:
                cot_feats = self._cot.get_cot_features(symbol)
                features.update(cot_feats)
                # Backward compat
                if cot_feats:
                    features["cot_score"] = self._cot.get_cot_score(symbol)
            except Exception as exc:
                logger.error("COT features failed: %s", exc)

        # 4a. Stooq sovereign yields (Phase 2A — was persisted but never
        # fed live until now). US-axis always emitted; country blocks
        # layered by symbol's currency exposure.
        if self._stooq:
            try:
                features.update(self._stooq.get_yield_features(symbol))
            except Exception as exc:
                logger.error("Stooq features failed: %s", exc)

        # 4b. ECB AAA curve (Phase 2A — same rationale; EUR pairs only).
        if self._ecb:
            try:
                features.update(self._ecb.get_yield_curve_features(symbol))
            except Exception as exc:
                logger.error("ECB features failed: %s", exc)

        # 5. Sentiment score
        if self._sentiment:
            try:
                features["sentiment_score"] = self._sentiment.get_sentiment_score(symbol)
            except Exception as exc:
                logger.error("Sentiment failed: %s", exc)

        # 6. On-chain score (BTC only)
        if self._onchain:
            try:
                features["onchain_score"] = self._onchain.get_onchain_score(symbol)
            except Exception as exc:
                logger.error("On-chain failed: %s", exc)

        # 7. Trends features (pytrends)
        if self._trends:
            try:
                features.update(self._trends.get_trends_features(symbol))
            except Exception as exc:
                logger.error("Trends features failed: %s", exc)

        # 8. Fear & Greed (BTC only)
        if self._fear_greed:
            try:
                features.update(self._fear_greed.get_fear_greed_features(symbol))
            except Exception as exc:
                logger.error("Fear & Greed failed: %s", exc)

        logger.debug(
            "FundamentalDataManager: %d features for %s", len(features), symbol
        )
        return features

    def get_backward_compat_scores(self, symbol: str) -> dict[str, float]:
        """
        Return the 4 backward-compatible composite scores used by the
        existing transform() API:
            macro_score, sentiment_score, onchain_score, cot_score

        This allows the existing trading loop to work unchanged while
        the expanded feature pipeline is being integrated.
        """
        scores: dict[str, float] = {}

        if self._macro:
            try:
                scores["macro_score"] = self._macro.get_macro_score(symbol)
            except Exception:
                scores["macro_score"] = 0.0
        else:
            scores["macro_score"] = 0.0

        if self._sentiment:
            try:
                scores["sentiment_score"] = self._sentiment.get_sentiment_score(symbol)
            except Exception:
                scores["sentiment_score"] = 0.0
        else:
            scores["sentiment_score"] = 0.0

        if self._onchain:
            try:
                scores["onchain_score"] = self._onchain.get_onchain_score(symbol)
            except Exception:
                scores["onchain_score"] = 0.0
        else:
            scores["onchain_score"] = 0.0

        if self._cot:
            try:
                scores["cot_score"] = self._cot.get_cot_score(symbol)
            except Exception:
                scores["cot_score"] = 0.0
        else:
            scores["cot_score"] = 0.0

        return scores
