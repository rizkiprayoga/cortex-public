"""
news_sentiment.py — News Sentiment via FinBERT NLP

Fetches recent financial news headlines for a symbol and scores
them using FinBERT (a BERT model fine-tuned on financial text).

Pipeline:
    1. NewsAPI → fetch last 24h headlines for the asset keyword
    2. FinBERT (HuggingFace) → classify each headline:
       ["positive", "negative", "neutral"] with confidence score
    3. Aggregate → weighted mean sentiment score in [-1, 1]

NewsAPI free tier: 100 requests/day, headlines only, 1-month history.
FinBERT model: "ProsusAI/finbert" from HuggingFace Hub.

Output: sentiment_score float in [-1, 1]:
    +1 = strongly positive news sentiment
    -1 = strongly negative news sentiment
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
from transformers import pipeline

logger = logging.getLogger(__name__)

# Keywords used for NewsAPI queries
SYMBOL_KEYWORDS = {
    "XAUUSD": ["gold price", "XAU", "gold commodity", "precious metals"],
    "BTCUSD": ["bitcoin", "BTC", "cryptocurrency", "crypto market"],
}


class NewsSentimentAnalyzer:
    """
    Fetches news and returns a sentiment score using FinBERT.

    Usage:
        analyzer = NewsSentimentAnalyzer()
        score = analyzer.get_sentiment_score("XAUUSD")
    """

    # Class-level cache shared across instances. NewsAPI free tier is
    # 100 req / rolling-24h; with the bot's ~15min refresh cadence we
    # would burn ~96 req/day per symbol and stay permanently 429.
    # 4h TTL caps each symbol at ~6 req/day.
    _cache: dict[str, tuple[float, float]] = {}   # symbol -> (score, fetched_at_epoch)
    _CACHE_TTL_SEC = 4 * 3600

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise EnvironmentError("NEWS_API_KEY not set in environment")
        self._model_name = model_name
        self._sentiment_pipeline = None   # Lazy-load model on first use

    def get_sentiment_score(self, symbol: str, hours_back: int = 24) -> float:
        """
        Fetch news and return composite sentiment score.

        Cached per-symbol for `_CACHE_TTL_SEC` to avoid burning the NewsAPI
        rolling-24h quota. On fetch failure, returns the last cached value
        if present (keeps the bot on stale-but-real sentiment instead of
        flipping to neutral 0.0 mid-session).

        Args:
            symbol:     Trading symbol ("XAUUSD" or "BTCUSD")
            hours_back: Look back this many hours for news

        Returns:
            Float in [-1, 1].
        """
        now = time.time()
        cached = self._cache.get(symbol)
        if cached is not None and now - cached[1] < self._CACHE_TTL_SEC:
            return cached[0]
        try:
            headlines = self.fetch_headlines(symbol, hours_back)
            if not headlines:
                logger.info("No headlines found for %s, returning 0.0", symbol)
                score = 0.0
            else:
                score = self.score_headlines(headlines)
        except Exception as exc:
            logger.error("Sentiment score failed for %s: %s", symbol, exc)
            if cached is not None:
                return cached[0]
            return 0.0
        self._cache[symbol] = (score, now)
        return score

    def fetch_headlines(self, symbol: str, hours_back: int = 24) -> list[str]:
        """
        Call NewsAPI and return a list of headline strings.

        Args:
            symbol:     Used to select query keywords
            hours_back: Time window for article search

        Returns:
            List of headline strings (may be empty if no news found).
        """
        keywords = SYMBOL_KEYWORDS.get(symbol.upper())
        if not keywords:
            logger.info("No keywords configured for %s", symbol)
            return []

        query = " OR ".join(f'"{kw}"' for kw in keywords)
        from_dt = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()

        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_dt,
                    "sortBy": "relevancy",
                    "language": "en",
                    "pageSize": 50,
                    "apiKey": self.api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])

            headlines = [
                a["title"]
                for a in articles
                if a.get("title") and a["title"] != "[Removed]"
            ]
            logger.debug("Fetched %d headlines for %s", len(headlines), symbol)
            return headlines

        except Exception as exc:
            logger.error("NewsAPI fetch failed for %s: %s", symbol, exc)
            return []

    def score_headlines(self, headlines: list[str]) -> float:
        """
        Run FinBERT on a list of headlines and return the mean score.

        Positive label → +confidence, Negative label → -confidence.
        Neutral label → 0.

        Returns:
            Mean sentiment score in [-1, 1].
        """
        if not headlines:
            return 0.0

        self._load_model()

        scores: list[float] = []
        # Process in batches to handle large headline lists
        batch_size = 16
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i + batch_size]
            # Truncate long headlines to avoid tokenizer issues
            batch = [h[:512] for h in batch]
            try:
                results = self._sentiment_pipeline(batch)
                for res in results:
                    label = res["label"].lower()
                    conf = res["score"]
                    if label == "positive":
                        scores.append(conf)
                    elif label == "negative":
                        scores.append(-conf)
                    else:
                        scores.append(0.0)
            except Exception as exc:
                logger.warning("FinBERT batch failed: %s", exc)

        if not scores:
            return 0.0

        mean_score = float(np.mean(scores))
        return max(-1.0, min(1.0, mean_score))

    def _load_model(self) -> None:
        """Lazy-load FinBERT pipeline on first call (avoids slow startup)."""
        if self._sentiment_pipeline is not None:
            return

        logger.info("Loading FinBERT model '%s'...", self._model_name)
        self._sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self._model_name,
            tokenizer=self._model_name,
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT model loaded.")
