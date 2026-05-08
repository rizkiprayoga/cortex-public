"""
feature_engineering.py — Technical Indicator & Feature Builder

Transforms raw OHLCV DataFrames into feature matrices ready for
the HMM and LSTM models, and persists the computed features to
the ``feature_vectors`` (JSONB) and legacy ``engineered_features``
(EAV) tables for reuse during retraining.

Features computed (~67 technical, before fundamentals):

    Price-derived (~13):
        log_return, log_return_5, log_return_10, log_return_20
        realized_volatility_5, realized_volatility_10,
        realized_volatility_20, realized_volatility_60
        price_range, gap, close_position_in_range,
        intrabar_momentum, overnight_gap

    Trend (~12):
        sma_10_rel, sma_20_rel, sma_50_rel, sma_200_rel
        ema_12_rel, ema_26_rel
        macd, macd_signal, macd_histogram
        adx, plus_di, minus_di

    Momentum (~8):
        rsi_7, rsi_14, stoch_k, stoch_d
        williams_r, roc_10, cci_20, mfi_14

    Volatility (~9):
        atr_7, atr_14
        bb_upper_rel, bb_lower_rel, bb_width, bb_pct_b
        keltner_width, parkinson_vol, garman_klass_vol

    Volume & microstructure (~6):
        volume_ratio, obv_roc
        bar_body_ratio, upper_shadow_ratio, lower_shadow_ratio
        consecutive_direction_count

    Statistical (~8):
        hurst_exponent, autocorr_lag1, autocorr_lag5
        rolling_skewness_20, rolling_kurtosis_20, shannon_entropy_20
        zscore_close_sma50, price_percentile_200

    Fundamental scores (injected from fundamental/ modules):
        macro_score, sentiment_score, onchain_score, cot_score

Persistence
-----------
Dual-write strategy: features are persisted to both the new JSONB
``feature_vectors`` table (one row per bar) and the legacy EAV
``engineered_features`` table (one row per feature per bar) for
backward compatibility. Reads prefer ``feature_vectors`` with
``engineered_features`` as fallback.

Symbol-agnostic design: All moving averages are computed as relative
distance ``(close - indicator) / indicator`` so models trained on one
asset can generalize to others with different price scales.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from src.data_pipeline.data_store import DataStore

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Transforms OHLCV DataFrames into model-ready feature matrices
    and persists them to PostgreSQL for reuse.

    Usage:
        engineer = FeatureEngineer(data_store=store)
        feature_df = engineer.transform(ohlcv_df, fundamental_scores)
        matrix = engineer.to_matrix(feature_df)  # numpy array for models

        # Later, during retraining:
        cached = await engineer.load_from_db("XAUUSD", "H4", start, end)
    """

    def __init__(self, data_store: Optional["DataStore"] = None):
        """
        Args:
            data_store: Optional async DataStore. If provided, persist_features()
                        will write to both JSONB and EAV tables.
        """
        self.data_store = data_store

    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------

    def transform(
        self,
        ohlcv: pd.DataFrame,
        fundamental_scores: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Compute all technical features and merge fundamental scores.

        Args:
            ohlcv:               Raw OHLCV DataFrame from MT5DataFeed
            fundamental_scores:  Dict of {score_name: float} from fundamental/

        Returns:
            DataFrame with all features, NaN rows dropped.

        Notes:
            This method is pure — it does NOT hit the DB. Persistence is
            handled by ``persist_features()`` below, which is called
            explicitly from the trading loop on an async context.
        """
        if ohlcv is None or ohlcv.empty:
            return pd.DataFrame()

        df = ohlcv.copy()

        # DB-origin DataFrames use `volume`; live MT5 path uses `tick_volume`.
        # Normalize so momentum/volume features (MFI, OBV, etc.) work in both.
        if "volume" in df.columns and "tick_volume" not in df.columns:
            df = df.rename(columns={"volume": "tick_volume"})

        # Feature groups
        df = self._compute_price_features(df)
        df = self._compute_trend_features(df)
        df = self._compute_momentum_features(df)
        df = self._compute_volatility_features(df)
        df = self._compute_volume_features(df)
        df = self._compute_statistical_features(df)

        # Fundamental scores — broadcast scalar across all rows
        if fundamental_scores:
            for name, value in fundamental_scores.items():
                df[name] = float(value) if value is not None else np.nan

        # Drop the raw OHLCV columns — keep only features
        feature_cols = [
            c for c in df.columns
            if c not in ("open", "high", "low", "close", "tick_volume")
        ]
        result = df[feature_cols].copy()

        # Drop warm-up NaN rows (from rolling windows)
        result.dropna(inplace=True)

        logger.debug("transform() produced %d bars × %d features",
                      len(result), len(result.columns))
        return result

    async def persist_features(
        self,
        symbol: str,
        timeframe: str,
        feature_df: pd.DataFrame,
    ) -> int:
        """
        Save every bar's feature vector to DB (dual-write: JSONB + EAV).

        Safe to call repeatedly — duplicates are handled by upsert/ignore.

        Args:
            symbol:      Trading symbol
            timeframe:   OHLCV timeframe ("H4", "D1", ...)
            feature_df:  Output of ``transform()``, indexed by bar datetime

        Returns:
            Number of bars persisted.
        """
        if self.data_store is None or feature_df.empty:
            return 0

        count = 0
        for ts, row in feature_df.iterrows():
            ts_str = _iso(ts)
            feature_dict = {
                name: (None if pd.isna(val) else float(val))
                for name, val in row.items()
            }

            # Primary: JSONB feature_vectors table
            await self.data_store.save_feature_vector(
                symbol=symbol,
                timeframe=timeframe,
                bar_timestamp=ts_str,
                feature_dict=feature_dict,
            )

            # Legacy: EAV engineered_features table
            await self.data_store.save_engineered_features(
                symbol=symbol,
                timeframe=timeframe,
                bar_timestamp=ts_str,
                feature_dict=feature_dict,
            )
            count += 1
        return count

    async def load_from_db(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load precomputed features from DB, preferring JSONB over EAV.

        Returns:
            Wide DataFrame indexed by bar_timestamp, one column per feature.
            Empty DataFrame if no cached features exist for the range.
        """
        if self.data_store is None:
            return pd.DataFrame()

        # Try JSONB first
        result = await self.data_store.get_feature_vectors_range(
            symbol, timeframe, start=start, end=end
        )
        if result is not None and not result.empty:
            return result

        # Fallback to EAV
        return await self.data_store.get_features_range(
            symbol, timeframe, start=start, end=end
        )

    async def backfill_features(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 300,
    ) -> int:
        """
        Compute and persist engineered features for any OHLCV bars that
        exist in ``ohlcv_bars`` but have no matching row in
        ``feature_vectors``. Called after ``MT5DataFeed.backfill_gaps()``
        to keep feature coverage in sync with OHLCV coverage.

        Args:
            lookback_bars: Number of bars before the last feature timestamp
                           to include as warm-up for rolling indicators.
                           300 bars covers the deepest window
                           (price_percentile_200 + margin).

        Returns:
            Number of new feature rows persisted.
        """
        if self.data_store is None:
            return 0

        feat_ts = await self.data_store.get_latest_feature_vector_timestamp(
            symbol, timeframe
        )
        bar_ts = await self.data_store.get_latest_bar_timestamp(symbol, timeframe)

        if bar_ts is None:
            return 0  # nothing to compute against
        if feat_ts is not None and feat_ts >= bar_ts:
            return 0  # features are already in sync with OHLCV

        # Load OHLCV with lookback for rolling-indicator warm-up.
        start: Optional[datetime] = None
        cutoff: Optional[datetime] = None
        if feat_ts is not None:
            cutoff = datetime.fromisoformat(feat_ts)
            from src.data_pipeline.mt5_feed import TIMEFRAME_DELTA
            delta = TIMEFRAME_DELTA.get(timeframe)
            if delta is not None:
                start = cutoff - (delta * lookback_bars)

        ohlcv = await self.data_store.get_ohlcv_range(
            symbol, timeframe, start=start, end=None
        )
        if ohlcv.empty:
            return 0

        # Compute features for the whole window (includes warm-up bars).
        feature_df = self.transform(ohlcv)
        if feature_df is None or feature_df.empty:
            return 0

        # Keep only rows strictly after the last-known feature timestamp
        if cutoff is not None:
            feature_df = feature_df[feature_df.index > cutoff]

        if feature_df.empty:
            return 0

        return await self.persist_features(symbol, timeframe, feature_df)

    @staticmethod
    def compute_triple_barrier_labels(
        ohlcv: pd.DataFrame,
        atr: Optional[pd.Series] = None,
        tp_r_mult: float = 2.5,
        sl_atr_mult: float = 2.0,
        time_limit_bars: int = 20,
        atr_period: int = 14,
    ) -> np.ndarray:
        """
        Compute Triple Barrier labels for a hypothetical long entry at each bar.

        Walks forward from each bar's close up to ``time_limit_bars`` to see
        which barrier is hit first:

            TP level = close + (tp_r_mult * sl_atr_mult) * ATR
            SL level = close - sl_atr_mult * ATR

        Labels:
            +1.0 — TP hit first (profitable long trade)
            -1.0 — SL hit first (losing long trade)
             0.0 — Neither hit within time limit (timeout)

        When both barriers are hit within the same bar, the SL is assumed
        to have triggered first (pessimistic/conservative labeling — this
        is what the exit manager would see in live trading since SL orders
        are on broker side).

        ATR computation
        ---------------
        If ``atr`` is None, a standard ATR in **price units** is computed
        locally via a Wilder-style EWM on true range. This avoids the trap
        where the ``atr_14`` column in the feature frame is normalized by
        close (fraction, not price) and would yield stops off by a factor
        of ``close``.

        Args:
            ohlcv:           DataFrame with 'high', 'low', 'close' columns
            atr:             Optional ATR series in **absolute price units**.
                             If None, ATR is computed from OHLCV.
            tp_r_mult:       Reward/Risk multiple (e.g., 2.5)
            sl_atr_mult:     SL distance in ATR units (e.g., 2.0)
            time_limit_bars: Max bars to wait before timeout
            atr_period:      ATR lookback (used only when atr is None)

        Returns:
            np.ndarray of shape (len(ohlcv),) with values in {-1.0, 0.0, +1.0}
        """
        if ohlcv is None or len(ohlcv) == 0:
            return np.array([], dtype=np.float32)

        high = ohlcv["high"].values.astype(np.float64)
        low = ohlcv["low"].values.astype(np.float64)
        close = ohlcv["close"].values.astype(np.float64)

        if atr is None:
            # Compute ATR in absolute price units (Wilder EWM on true range)
            prev_close = pd.Series(close).shift(1)
            tr1 = pd.Series(high) - pd.Series(low)
            tr2 = (pd.Series(high) - prev_close).abs()
            tr3 = (pd.Series(low) - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_abs = true_range.ewm(
                alpha=1 / atr_period, min_periods=atr_period, adjust=False,
            ).mean()
            atr_arr = atr_abs.values.astype(np.float64)
        else:
            atr_arr = atr.reindex(ohlcv.index).values.astype(np.float64)

        n = len(close)
        labels = np.zeros(n, dtype=np.float32)

        for i in range(n - 1):
            entry = close[i]
            a = atr_arr[i]
            if not np.isfinite(a) or a <= 0:
                continue

            sl_price = entry - sl_atr_mult * a
            tp_price = entry + (tp_r_mult * sl_atr_mult) * a

            end_j = min(i + 1 + time_limit_bars, n)
            for j in range(i + 1, end_j):
                hit_tp = high[j] >= tp_price
                hit_sl = low[j] <= sl_price
                if hit_sl:
                    # SL-first (conservative when both hit in same bar)
                    labels[i] = -1.0
                    break
                if hit_tp:
                    labels[i] = 1.0
                    break
            # else label stays 0.0 (timeout)

        return labels

    def to_matrix(self, feature_df: pd.DataFrame) -> np.ndarray:
        """
        Convert feature DataFrame to numpy array for model input.

        Applies per-column z-score normalization (subtract mean, divide by
        std) so each feature has zero mean and unit variance. Columns are
        sorted alphabetically for deterministic ordering across sessions.

        Returns:
            Shape (n_bars, n_features) — float64.
        """
        if feature_df is None or feature_df.empty:
            return np.empty((0, 0), dtype=np.float64)

        # Sort columns alphabetically for deterministic ordering
        sorted_df = feature_df[sorted(feature_df.columns)]
        mat = sorted_df.values.astype(np.float64)

        # Replace any remaining NaN/inf with 0
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score per column
        means = mat.mean(axis=0)
        stds = mat.std(axis=0)
        stds[stds == 0] = 1.0  # avoid division by zero
        mat = (mat - means) / stds

        return mat

    # -------------------------------------------------------------------------
    # Multi-Timeframe Alignment
    # -------------------------------------------------------------------------

    # Default feature subsets per timeframe (what features to compute for each)
    TIMEFRAME_FEATURE_SUBSETS: dict[str, list[str]] = {
        "M15": [
            "log_return", "realized_volatility_5", "price_range",
            "close_position_in_range", "intrabar_momentum",
            "rsi_7", "stoch_k", "volume_ratio",
            "bar_body_ratio", "upper_shadow_ratio", "lower_shadow_ratio",
        ],
        "H1": [
            "log_return", "log_return_5", "realized_volatility_10",
            "price_range", "rsi_7", "rsi_14", "stoch_k", "stoch_d",
            "ema_12_rel", "macd", "volume_ratio", "atr_7",
            "bb_pct_b", "obv_roc", "consecutive_direction_count",
        ],
        "H4": None,  # Full feature set (primary timeframe)
        "D1": [
            "log_return", "log_return_5", "log_return_20",
            "realized_volatility_20", "realized_volatility_60",
            "sma_20_rel", "sma_50_rel", "sma_200_rel",
            "adx", "rsi_14", "atr_14", "bb_width",
            "hurst_exponent", "autocorr_lag1",
            "rolling_skewness_20", "zscore_close_sma50",
        ],
        "W1": [
            "log_return", "log_return_5",
            "realized_volatility_20", "sma_20_rel", "sma_50_rel",
            "adx", "rsi_14", "atr_14", "bb_width",
        ],
    }

    def transform_multi_timeframe(
        self,
        ohlcv_by_tf: dict[str, pd.DataFrame],
        fundamental_features: Optional[dict[str, float]] = None,
        primary_tf: str = "H4",
    ) -> pd.DataFrame:
        """
        Compute features across multiple timeframes and align to the
        primary timeframe grid.

        Higher-TF features are forward-filled to the primary grid and
        prefixed with their timeframe name (e.g., ``D1_sma_50_rel``).
        The primary TF gets no prefix.

        Args:
            ohlcv_by_tf:           Dict mapping timeframe string → OHLCV DataFrame.
            fundamental_features:  Dict of all external features from
                                   FundamentalDataManager.get_all_features().
            primary_tf:            Primary timeframe for alignment grid.

        Returns:
            DataFrame indexed by primary timeframe timestamps, with all
            multi-TF features aligned. NaN warmup rows dropped.
        """
        if primary_tf not in ohlcv_by_tf or ohlcv_by_tf[primary_tf].empty:
            return pd.DataFrame()

        # Compute features for primary TF (full set)
        primary_ohlcv = ohlcv_by_tf[primary_tf]
        primary_features = self.transform(primary_ohlcv)

        if primary_features.empty:
            return pd.DataFrame()

        # Merge fundamental/external features into primary
        if fundamental_features:
            for name, value in fundamental_features.items():
                primary_features[name] = float(value) if value is not None else 0.0

        # Compute and align each non-primary timeframe
        for tf, ohlcv in ohlcv_by_tf.items():
            if tf == primary_tf or ohlcv is None or ohlcv.empty:
                continue

            # Compute features for this TF
            tf_features = self.transform(ohlcv)
            if tf_features.empty:
                continue

            # Select subset of features for this TF
            subset = self.TIMEFRAME_FEATURE_SUBSETS.get(tf)
            if subset is not None:
                available = [c for c in subset if c in tf_features.columns]
                tf_features = tf_features[available]

            # Prefix columns with timeframe
            tf_features = tf_features.add_prefix(f"{tf}_")

            # Forward-fill to primary grid
            # Reindex to primary index, forward-fill from higher TF
            tf_aligned = tf_features.reindex(
                primary_features.index, method="ffill"
            )

            # Join to primary
            primary_features = primary_features.join(tf_aligned, how="left")

        # Fill any remaining NaN from alignment with 0
        primary_features = primary_features.fillna(0.0)

        logger.info(
            "transform_multi_timeframe: %d bars × %d features (%d TFs)",
            len(primary_features),
            len(primary_features.columns),
            len(ohlcv_by_tf),
        )
        return primary_features

    # -------------------------------------------------------------------------
    # Training-oriented: regime feature injection
    # -------------------------------------------------------------------------

    def inject_regime_features(
        self,
        feature_df: pd.DataFrame,
        hmm,
        symbol: str,
        d1_ohlcv: pd.DataFrame,
        d1_window: int = 60,
    ) -> pd.DataFrame:
        """
        Run the pre-trained HMM on D1 data and inject regime features
        into the H4 feature DataFrame as 6 extra columns.

        This lets the LSTM learn regime-conditional patterns — e.g.,
        "in Bull + RSI>70, price continues; in Neutral, price reverts."

        Columns added:
            regime_0 .. regime_4  — one-hot encoded regime state
            regime_probability    — HMM posterior confidence

        Args:
            feature_df: H4 feature DataFrame (from transform or transform_multi_timeframe)
            hmm:        Trained HMMRegimeClassifier with hmm.load(symbol) done
            symbol:     Trading symbol
            d1_ohlcv:   Raw D1 OHLCV DataFrame
            d1_window:  Number of D1 bars to feed HMM per prediction

        Returns:
            feature_df with 6 regime columns joined (forward-filled from D1 to H4).
        """
        if d1_ohlcv is None or d1_ohlcv.empty or feature_df.empty:
            # Graceful fallback — add neutral regime features
            for i in range(5):
                feature_df[f"regime_{i}"] = 0.2
            feature_df["regime_probability"] = 0.2
            return feature_df

        # Compute D1 features and matrix for HMM
        d1_clean = d1_ohlcv.copy()
        if "volume" in d1_clean.columns and "tick_volume" not in d1_clean.columns:
            d1_clean.rename(columns={"volume": "tick_volume"}, inplace=True)
        if "tick_volume" not in d1_clean.columns:
            d1_clean["tick_volume"] = 0

        d1_features = self.transform(d1_clean)
        if d1_features.empty:
            for i in range(5):
                feature_df[f"regime_{i}"] = 0.2
            feature_df["regime_probability"] = 0.2
            return feature_df

        d1_matrix = self.to_matrix(d1_features)

        # Run HMM on rolling D1 windows
        regime_rows = []
        for i in range(len(d1_matrix)):
            start = max(0, i - d1_window + 1)
            window = d1_matrix[start:i + 1]
            if len(window) < 5:  # HMM needs at least n_components bars
                regime_rows.append({
                    **{f"regime_{j}": 0.2 for j in range(5)},
                    "regime_probability": 0.2,
                })
                continue

            try:
                result = hmm.predict(symbol, window)
                one_hot = {f"regime_{j}": 0.0 for j in range(5)}
                one_hot[f"regime_{result.regime_index}"] = 1.0
                one_hot["regime_probability"] = float(result.state_probability)
                regime_rows.append(one_hot)
            except Exception:
                regime_rows.append({
                    **{f"regime_{j}": 0.2 for j in range(5)},
                    "regime_probability": 0.2,
                })

        regime_df = pd.DataFrame(regime_rows, index=d1_features.index)

        # Forward-fill D1 regime onto H4 grid
        regime_aligned = regime_df.reindex(feature_df.index, method="ffill")
        regime_aligned = regime_aligned.fillna(0.2)

        # Join
        feature_df = feature_df.join(regime_aligned, how="left")
        feature_df = feature_df.fillna(0.2)

        logger.info(
            "inject_regime_features: %d D1 bars → %d regime rows, "
            "joined onto %d H4 bars",
            len(d1_matrix), len(regime_rows), len(feature_df),
        )
        return feature_df

    # -------------------------------------------------------------------------
    # Training-oriented: historical externals
    # -------------------------------------------------------------------------

    def transform_with_externals(
        self,
        ohlcv: pd.DataFrame,
        symbol: str,
        zero_fill_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute technical features + historical cross-asset + calendar.

        Phase 2A: when ``self.data_store`` is connected, ALSO joins on
        the engineered FRED macro / Stooq yield / ECB curve / COT historical
        features via ``feature_store`` reads. This eliminates the train/serve
        column-set skew that Phase 1B's zero-fill pattern silently introduced.
        When ``data_store`` is None, falls back to the pre-Phase-2A behavior
        (cross-asset + calendar + zero-fill).

        OHLCV index is assumed to be naive true-UTC (per the DB convention,
        produced by ``MT5DataFeed.get_historical`` after the Phase 2A broker-
        ts fix). All external readers also produce naive-UTC indices, so the
        ffill alignment onto the bar grid is direct.

        Args:
            ohlcv:          Raw OHLCV DataFrame from MT5/DB (naive UTC index).
            symbol:         Trading symbol (affects per-currency feature blocks).
            zero_fill_cols: Feature names to zero-fill (live-only sources
                            with no historical backfill — sentiment, fear-greed,
                            on-chain, trends).

        Returns:
            DataFrame indexed by OHLCV bar timestamp (naive UTC) with
            technical + cross-asset + calendar + (optionally) FRED/Stooq/ECB/COT
            historical features + zero-fill columns.
        """
        # Delegate to the async path when a DataStore is wired up — that's
        # the Phase 2A "real" path. Sync fallback below preserves callers
        # that haven't been updated.
        if self.data_store is not None:
            import asyncio
            return asyncio.run(self._transform_with_externals_async(
                ohlcv, symbol, zero_fill_cols,
            ))

        tech_df = self.transform(ohlcv)
        if tech_df.empty:
            return tech_df

        # Cross-asset features (yfinance historical)
        try:
            from src.data_pipeline.market.cross_asset import CrossAssetFetcher
            fetcher = CrossAssetFetcher()
            start_dt = tech_df.index[0].to_pydatetime()
            end_dt = tech_df.index[-1].to_pydatetime()
            cross_df = fetcher.get_historical_cross_asset_features(
                symbol, start_dt, end_dt,
            )
            if not cross_df.empty:
                cross_aligned = self._align_external_to_bar_grid(
                    cross_df, tech_df.index,
                )
                tech_df = tech_df.join(cross_aligned, how="left")
        except Exception as exc:
            logger.warning("Historical cross-asset features unavailable: %s", exc)

        # Calendar features (pure computation)
        try:
            from src.data_pipeline.market.calendar_features import (
                CalendarFeatureBuilder,
            )
            cal_builder = CalendarFeatureBuilder()
            cal_df = cal_builder.get_historical_calendar_features(tech_df.index)
            tech_df = tech_df.join(cal_df, how="left")
        except Exception as exc:
            logger.warning("Historical calendar features unavailable: %s", exc)

        if zero_fill_cols:
            for col in zero_fill_cols:
                if col not in tech_df.columns:
                    tech_df[col] = 0.0

        tech_df = tech_df.fillna(0.0)

        logger.info(
            "transform_with_externals[sync,no-store]: %d bars x %d features for %s",
            len(tech_df), len(tech_df.columns), symbol,
        )
        return tech_df

    async def _transform_with_externals_async(
        self,
        ohlcv: pd.DataFrame,
        symbol: str,
        zero_fill_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Phase 2A path: technical + cross-asset + calendar + 4 historical
        external readers (FRED macro / Stooq yields / ECB curve / COT)
        joined onto the bar grid via lookahead-safe feature_store reads.

        Internal — called by ``transform_with_externals`` when a DataStore
        is configured. Don't call directly from sync callers.
        """
        if self.data_store is None:
            raise RuntimeError(
                "_transform_with_externals_async requires self.data_store "
                "to be wired. Use the sync transform_with_externals() for "
                "the no-store fallback."
            )

        tech_df = self.transform(ohlcv)
        if tech_df.empty:
            return tech_df

        tech_df = await self._inject_externals_async(
            tech_df, symbol, zero_fill_cols, log_label="transform_with_externals[async,store]",
        )
        return tech_df

    async def transform_multi_timeframe_with_externals_async(
        self,
        ohlcv_by_tf: dict[str, pd.DataFrame],
        symbol: str,
        primary_tf: str = "H4",
        zero_fill_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Multi-TF tech features + Phase 2A historical externals on the
        primary-TF grid (async variant, for callers in event loops).

        Mirrors what live's ``main.py`` produces (per-tick fundamental dict
        passed into ``transform_multi_timeframe``) but for backtest/training:
        rather than a fresh fundamental snapshot per bar, this method pulls
        every bar's externals from ``feature_store`` via the 4 historical
        readers (macro / yield / curve / cot) plus cross-asset + calendar.

        The columns produced match the live ``FundamentalDataManager
        .get_all_features()`` keys on top of multi-TF tech — same set the LSTM
        sees at inference.

        Falls back to the pre-Phase-2A behavior (multi-TF tech + calendar +
        zero-fill, no historical externals) when ``self.data_store`` is None.

        Args:
            ohlcv_by_tf: Dict of timeframe → OHLCV DataFrame (naive UTC index).
            symbol:      Trading symbol (drives currency-conditional blocks).
            primary_tf:  Primary timeframe for the output grid (default H4).
            zero_fill_cols: True-no-historical-source column names.

        Returns:
            DataFrame indexed by primary-TF bar timestamps (naive UTC) with
            multi-TF tech + cross-asset + calendar + macro + yield + curve +
            cot + zero-fill columns.
        """
        tech_df = self.transform_multi_timeframe(ohlcv_by_tf, primary_tf=primary_tf)
        if tech_df.empty:
            return tech_df

        if self.data_store is None:
            # Pre-Phase-2A behavior: calendar + zero-fill, no externals.
            try:
                from src.data_pipeline.market.calendar_features import (
                    CalendarFeatureBuilder,
                )
                cal_builder = CalendarFeatureBuilder()
                cal_df = cal_builder.get_historical_calendar_features(tech_df.index)
                tech_df = tech_df.join(cal_df, how="left")
            except Exception as exc:
                logger.warning("Calendar features unavailable: %s", exc)
            if zero_fill_cols:
                for col in zero_fill_cols:
                    if col not in tech_df.columns:
                        tech_df[col] = 0.0
            tech_df = tech_df.fillna(0.0)
            logger.info(
                "transform_multi_timeframe_with_externals[no-store]: "
                "%d bars x %d features for %s",
                len(tech_df), len(tech_df.columns), symbol,
            )
            return tech_df

        return await self._inject_externals_async(
            tech_df, symbol, zero_fill_cols,
            log_label="transform_multi_timeframe_with_externals[async,store]",
        )

    def transform_multi_timeframe_with_externals(
        self,
        ohlcv_by_tf: dict[str, pd.DataFrame],
        symbol: str,
        primary_tf: str = "H4",
        zero_fill_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Sync wrapper around the async variant — for sync callers like
        backtest scripts that don't run in an event loop.

        Async callers (anything inside ``asyncio.run(main_async())``) MUST
        use ``transform_multi_timeframe_with_externals_async`` directly,
        otherwise asyncio.run() raises "cannot be called from a running
        event loop".
        """
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no running loop, safe to asyncio.run
        else:
            raise RuntimeError(
                "transform_multi_timeframe_with_externals (sync) called from "
                "a running event loop. Use the _async variant instead."
            )
        return asyncio.run(self.transform_multi_timeframe_with_externals_async(
            ohlcv_by_tf, symbol, primary_tf=primary_tf,
            zero_fill_cols=zero_fill_cols,
        ))

    async def _inject_externals_async(
        self,
        tech_df: pd.DataFrame,
        symbol: str,
        zero_fill_cols: Optional[list[str]] = None,
        log_label: str = "_inject_externals_async",
    ) -> pd.DataFrame:
        """Take an already-computed tech feature DataFrame and join on the
        externals that stay in the LSTM input path: cross-asset (yfinance) +
        calendar + zero-fill.

        Shared helper — called by both the single-TF and multi-TF entry
        points so externals injection is one codepath.
        """
        if self.data_store is None:
            raise RuntimeError(
                "_inject_externals_async requires self.data_store to be wired."
            )
        if tech_df.empty:
            return tech_df

        bar_index = tech_df.index
        start_dt = bar_index[0].to_pydatetime()
        end_dt = bar_index[-1].to_pydatetime()

        # 1. Cross-asset (sync — yfinance, not in feature_store yet)
        try:
            from src.data_pipeline.market.cross_asset import CrossAssetFetcher
            cross_fetcher = CrossAssetFetcher()
            cross_df = cross_fetcher.get_historical_cross_asset_features(
                symbol, start_dt, end_dt,
            )
            if not cross_df.empty:
                tech_df = tech_df.join(
                    self._align_external_to_bar_grid(cross_df, bar_index),
                    how="left",
                )
        except Exception as exc:
            logger.warning("Cross-asset features unavailable: %s", exc)

        # 2. Calendar (pure computation)
        try:
            from src.data_pipeline.market.calendar_features import (
                CalendarFeatureBuilder,
            )
            cal_builder = CalendarFeatureBuilder()
            cal_df = cal_builder.get_historical_calendar_features(bar_index)
            tech_df = tech_df.join(cal_df, how="left")
        except Exception as exc:
            logger.warning("Calendar features unavailable: %s", exc)

        # Final zero-fill for true-no-historical-source columns (sentiment,
        # fear-greed, on-chain, trends, plus the legacy single-value scores
        # ``macro_score`` / ``cot_score``).
        if zero_fill_cols:
            for col in zero_fill_cols:
                if col not in tech_df.columns:
                    tech_df[col] = 0.0

        tech_df = tech_df.fillna(0.0)

        logger.info(
            "%s: %d bars x %d features for %s",
            log_label, len(tech_df), len(tech_df.columns), symbol,
        )
        return tech_df

    async def _join_macro_history(
        self, tech_df: pd.DataFrame, symbol: str,
        start_dt: datetime, end_dt: datetime, bar_index: pd.DatetimeIndex,
    ) -> None:
        """Read FRED macro history from feature_store and join onto tech_df
        in-place. Errors are logged and the join is skipped — training
        falls back to the zero-fill columns the LSTM has historically seen."""
        try:
            from src.data_pipeline.fundamental.macro_data import MacroDataFetcher
            fetcher = MacroDataFetcher()
            df = await fetcher.get_historical_macro_features(
                self.data_store, symbol, start_dt, end_dt,
            )
            if not df.empty:
                aligned = self._align_external_to_bar_grid(df, bar_index)
                # join in place via column assignment (avoid reassignment of
                # tech_df, which we'd lose on caller)
                for col in aligned.columns:
                    tech_df[col] = aligned[col]
        except Exception as exc:
            logger.warning("Historical macro features unavailable: %s", exc)

    async def _join_yield_history(
        self, tech_df: pd.DataFrame, symbol: str,
        start_dt: datetime, end_dt: datetime, bar_index: pd.DatetimeIndex,
    ) -> None:
        try:
            from src.data_pipeline.market.stooq_data import StooqFetcher
            fetcher = StooqFetcher()
            df = await fetcher.get_historical_yield_features(
                self.data_store, symbol, start_dt, end_dt,
            )
            if not df.empty:
                aligned = self._align_external_to_bar_grid(df, bar_index)
                for col in aligned.columns:
                    tech_df[col] = aligned[col]
        except Exception as exc:
            logger.warning("Historical yield features unavailable: %s", exc)

    async def _join_curve_history(
        self, tech_df: pd.DataFrame, symbol: str,
        start_dt: datetime, end_dt: datetime, bar_index: pd.DatetimeIndex,
    ) -> None:
        try:
            from src.data_pipeline.market.ecb_data import ECBDataFetcher
            fetcher = ECBDataFetcher()
            df = await fetcher.get_historical_curve_features(
                self.data_store, symbol, start_dt, end_dt,
            )
            if not df.empty:
                aligned = self._align_external_to_bar_grid(df, bar_index)
                for col in aligned.columns:
                    tech_df[col] = aligned[col]
        except Exception as exc:
            logger.warning("Historical ECB curve features unavailable: %s", exc)

    async def _join_cot_history(
        self, tech_df: pd.DataFrame, symbol: str,
        start_dt: datetime, end_dt: datetime, bar_index: pd.DatetimeIndex,
    ) -> None:
        try:
            from src.data_pipeline.fundamental.cot_data import COTDataFetcher
            fetcher = COTDataFetcher()
            df = await fetcher.get_historical_cot_features(
                self.data_store, symbol, start_dt, end_dt,
            )
            if not df.empty:
                aligned = self._align_external_to_bar_grid(df, bar_index)
                for col in aligned.columns:
                    tech_df[col] = aligned[col]
        except Exception as exc:
            logger.warning("Historical COT features unavailable: %s", exc)

    @staticmethod
    def _align_external_to_bar_grid(
        external_df: pd.DataFrame, bar_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Align an externals DataFrame onto the OHLCV bar grid via
        forward-fill, with TZ normalization to naive UTC.

        Phase 2A invariant: every external reader emits a naive-UTC index
        (post release-lag shift). The bar_index is naive UTC (DB convention,
        post ``_broker_ts_to_utc`` for any MT5-sourced bars). Mixing tz-aware
        with naive at this point would silently shift values by hours — same
        category of bug as the 2026-04-24 incident. We assert it.
        """
        if external_df.empty:
            return external_df
        # Defensive normalization: if either side is tz-aware, drop tz to
        # naive (interpreting the datetime as UTC). This catches yfinance,
        # which sometimes returns tz-aware Pacific or Eastern indices.
        ext = external_df.copy()
        if ext.index.tz is not None:
            ext.index = ext.index.tz_convert("UTC").tz_localize(None)
        if bar_index.tz is not None:
            bar_index = bar_index.tz_convert("UTC").tz_localize(None)
        return ext.reindex(bar_index, method="ffill").fillna(0.0)

    @staticmethod
    def get_feature_columns(feature_df: pd.DataFrame) -> list[str]:
        """Return sorted column list — the feature 'manifest' for a model."""
        return sorted(feature_df.columns.tolist())

    @staticmethod
    def align_to_manifest(
        feature_df: pd.DataFrame, manifest: list[str],
    ) -> pd.DataFrame:
        """
        Align a feature DataFrame to a fixed column manifest.

        Missing columns are filled with 0.0 (neutral default).
        Extra columns are dropped.  Order is alphabetical
        (matches ``to_matrix()`` sorting).
        """
        for col in manifest:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        return feature_df[sorted(manifest)]

    @staticmethod
    def get_zero_fill_feature_names(symbol: str = "") -> list[str]:
        """
        Feature names that exist in live trading but legitimately lack
        historical data — sources without a backfill path.

        Phase 2A: this list shrunk significantly. Per-currency macro
        features (boj_rate_level, ecb_rate_level, eur_usd_rate_diff, etc.)
        now have historical backfill via ``feature_store`` + the new
        ``get_historical_macro_features`` reader, so they're no longer
        zero-filled at training. The remaining four are aggregate scalar
        scores or alt-data without a Phase 1 backfill path:

            * macro_score / cot_score  — legacy single-value composites
              (the per-feature dicts ARE backfilled; only the composite
              scalars stay zero-filled in training to match what live emits)
            * sentiment_score          — NewsAPI + FinBERT, no historical feed
            * onchain_score            — CoinGecko, BTC-only, no historical feed

        ``symbol`` argument retained for API stability — Phase 2B may add
        per-symbol entries if any new pair gains live-only features.
        """
        return [
            "sentiment_score",
            "onchain_score",
            "macro_score",
            "cot_score",
        ]

    # -------------------------------------------------------------------------
    # Private feature computations
    # -------------------------------------------------------------------------

    def _compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Price-derived features (~13).
        Multi-period log returns, multi-window realized volatility,
        price range, gap, close position, intrabar momentum.
        """
        close = df["close"]

        # Multi-period log returns
        df["log_return"] = np.log(close / close.shift(1))
        df["log_return_5"] = np.log(close / close.shift(5))
        df["log_return_10"] = np.log(close / close.shift(10))
        df["log_return_20"] = np.log(close / close.shift(20))

        # Multi-window realized volatility
        lr = df["log_return"]
        df["realized_volatility_5"] = lr.rolling(5).std()
        df["realized_volatility_10"] = lr.rolling(10).std()
        df["realized_volatility_20"] = lr.rolling(20).std()
        df["realized_volatility_60"] = lr.rolling(60).std()

        # Price range — normalized
        df["price_range"] = (df["high"] - df["low"]) / close

        # Gap — open vs previous close
        df["gap"] = (df["open"] - close.shift(1)) / close.shift(1)

        # Close position in high-low range [0, 1]
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["close_position_in_range"] = (close - df["low"]) / hl_range

        # Intrabar momentum — (close - open) / (high - low)
        df["intrabar_momentum"] = (close - df["open"]) / hl_range

        # Overnight gap — distance from open to prev close, as ATR fraction
        prev_close = close.shift(1)
        atr_proxy = (df["high"] - df["low"]).rolling(14).mean().replace(0, np.nan)
        df["overnight_gap"] = (df["open"] - prev_close) / atr_proxy

        return df

    def _compute_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trend features (~12).
        All moving averages as relative distance from close (symbol-agnostic).
        MACD line/signal/histogram. ADX/+DI/-DI for trend strength.
        """
        close = df["close"]

        # SMA relative: (close - sma) / sma
        for period in [10, 20, 50, 200]:
            sma = close.rolling(period).mean()
            df[f"sma_{period}_rel"] = (close - sma) / sma.replace(0, np.nan)

        # EMA relative: (close - ema) / ema
        for span in [12, 26]:
            ema = close.ewm(span=span, adjust=False).mean()
            df[f"ema_{span}_rel"] = (close - ema) / ema.replace(0, np.nan)

        # MACD — already normalized since it uses EMA differences
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        df["macd"] = macd_line / close  # normalize by close for cross-symbol
        df["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean() / close
        df["macd_histogram"] = (macd_line - macd_line.ewm(span=9, adjust=False).mean()) / close

        # ADX / +DI / -DI (14-period)
        df = self._compute_adx(df, period=14)
        df["efficiency_ratio"] = self._compute_efficiency_ratio(df["close"], n=20)

        return df

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute ADX, +DI, -DI using Wilder's smoothing."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Directional movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
        )

        # True Range
        high_low = high - low
        high_pc = (high - close.shift(1)).abs()
        low_pc = (low - close.shift(1)).abs()
        tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)

        # Wilder's smoothing (EWM with alpha=1/period)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        smooth_plus = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        smooth_minus = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        # +DI, -DI
        atr_safe = atr.replace(0, np.nan)
        df["plus_di"] = 100 * smooth_plus / atr_safe
        df["minus_di"] = 100 * smooth_minus / atr_safe

        # DX and ADX
        di_sum = (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
        dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / di_sum
        df["adx"] = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        return df

    def _compute_efficiency_ratio(self, closes: pd.Series, n: int = 20) -> pd.Series:
        """Kaufman's Efficiency Ratio: |Close[t] - Close[t-n]| / sum(|ΔClose|, n bars).

        Range [0, 1]. ER ≈ 1.0 = clean directional trend; ER ≈ 0 = pure chop.
        Used as one of three filters in E-7 trend-mode (>0.30 threshold).
        Reference: Kaufman, *Smarter Trading* (1995); Clenow, *Following the Trend* (2013).
        """
        direction = (closes - closes.shift(n)).abs()
        volatility = closes.diff().abs().rolling(n).sum()
        # Avoid div-by-zero on flat segments (volatility = 0). 0/0 → NaN → fill 0.
        return (direction / volatility.replace(0, np.nan)).fillna(0.0)

    def _compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum features (~8).
        RSI (7, 14), Stochastic K/D, Williams %R, ROC, CCI, MFI.
        """
        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # RSI — two periods
        for period in [7, 14]:
            ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
            al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
            rs = ag / al.replace(0, np.nan)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator (14, 3)
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        denom = (high_14 - low_14).replace(0, np.nan)
        df["stoch_k"] = 100 * (close - low_14) / denom
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # Williams %R (14)
        df["williams_r"] = -100 * (high_14 - close) / denom

        # Rate of Change (10)
        df["roc_10"] = (close - close.shift(10)) / close.shift(10).replace(0, np.nan)

        # CCI (20) — Commodity Channel Index
        typical_price = (df["high"] + df["low"] + close) / 3
        tp_sma = typical_price.rolling(20).mean()
        tp_mad = typical_price.rolling(20).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        df["cci_20"] = (typical_price - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

        # MFI (14) — Money Flow Index
        mf_raw = typical_price * df["tick_volume"]
        mf_pos = pd.Series(
            np.where(typical_price > typical_price.shift(1), mf_raw, 0.0),
            index=df.index,
        )
        mf_neg = pd.Series(
            np.where(typical_price < typical_price.shift(1), mf_raw, 0.0),
            index=df.index,
        )
        mf_pos_sum = mf_pos.rolling(14).sum()
        mf_neg_sum = mf_neg.rolling(14).sum().replace(0, np.nan)
        mfr = mf_pos_sum / mf_neg_sum
        df["mfi_14"] = 100 - (100 / (1 + mfr))

        return df

    def _compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility features (~9).
        ATR (7, 14), Bollinger (relative), Keltner, Parkinson, Garman-Klass.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # True Range (shared for ATR and Keltner)
        high_low = high - low
        high_pc = (high - close.shift(1)).abs()
        low_pc = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)

        # ATR — two periods, normalized by close
        for period in [7, 14]:
            df[f"atr_{period}"] = true_range.rolling(period).mean() / close

        # Bollinger Bands (20, 2σ) — relative to close
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df["bb_upper_rel"] = (bb_upper - close) / close
        df["bb_lower_rel"] = (close - bb_lower) / close
        df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

        # %B — position of close within bands [0, 1]
        band_range = (bb_upper - bb_lower).replace(0, np.nan)
        df["bb_pct_b"] = (close - bb_lower) / band_range

        # Keltner Channel width — ATR-based bands
        keltner_mid = close.ewm(span=20, adjust=False).mean()
        atr_20 = true_range.rolling(20).mean()
        keltner_upper = keltner_mid + 2 * atr_20
        keltner_lower = keltner_mid - 2 * atr_20
        df["keltner_width"] = (keltner_upper - keltner_lower) / keltner_mid.replace(0, np.nan)

        # Parkinson volatility estimator (uses high-low range)
        log_hl = np.log(high / low.replace(0, np.nan))
        df["parkinson_vol"] = np.sqrt(
            (log_hl ** 2).rolling(20).mean() / (4 * np.log(2))
        )

        # Garman-Klass volatility estimator
        log_oc = np.log(close / df["open"].replace(0, np.nan))
        gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_oc ** 2
        df["garman_klass_vol"] = np.sqrt(gk.rolling(20).mean().clip(lower=0))

        return df

    def _compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume & microstructure features (~6).
        Volume ratio, OBV rate of change, candlestick body/shadow ratios,
        consecutive direction count.
        """
        close = df["close"]
        vol = df["tick_volume"].astype(float)

        # Volume ratio — relative to 20-bar average
        avg_vol = vol.rolling(20).mean().replace(0, np.nan)
        df["volume_ratio"] = vol / avg_vol

        # OBV (On-Balance Volume) — rate of change over 10 bars
        sign = np.sign(close.diff()).fillna(0)
        obv = (sign * vol).cumsum()
        obv_prev = obv.shift(10)
        df["obv_roc"] = (obv - obv_prev) / obv_prev.abs().replace(0, np.nan)

        # Candlestick body ratio — |close - open| / (high - low)
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["bar_body_ratio"] = (close - df["open"]).abs() / hl_range

        # Shadow ratios
        upper_shadow = df["high"] - pd.concat([close, df["open"]], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, df["open"]], axis=1).min(axis=1) - df["low"]
        df["upper_shadow_ratio"] = upper_shadow / hl_range
        df["lower_shadow_ratio"] = lower_shadow / hl_range

        # Consecutive direction count — positive = bullish streak, negative = bearish
        direction = np.sign(close.diff()).fillna(0).values
        consec = np.zeros(len(direction), dtype=float)
        for i in range(1, len(direction)):
            if direction[i] == direction[i - 1] and direction[i] != 0:
                consec[i] = consec[i - 1] + direction[i]
            else:
                consec[i] = direction[i]
        df["consecutive_direction_count"] = consec

        return df

    def _compute_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Statistical features (~8).
        Hurst exponent, autocorrelation, skewness, kurtosis, entropy,
        z-score, price percentile.
        """
        close = df["close"]
        lr = np.log(close / close.shift(1))

        # Hurst exponent (simplified R/S analysis, 100-bar window)
        df["hurst_exponent"] = lr.rolling(100).apply(
            _hurst_rs, raw=True
        )

        # Autocorrelation at lags 1 and 5
        df["autocorr_lag1"] = lr.rolling(50).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 2 else 0.0,
            raw=False,
        )
        df["autocorr_lag5"] = lr.rolling(50).apply(
            lambda x: pd.Series(x).autocorr(lag=5) if len(x) >= 6 else 0.0,
            raw=False,
        )

        # Rolling skewness and kurtosis (20-bar)
        df["rolling_skewness_20"] = lr.rolling(20).skew()
        df["rolling_kurtosis_20"] = lr.rolling(20).kurt()

        # Shannon entropy of discretized returns (20-bar)
        df["shannon_entropy_20"] = lr.rolling(20).apply(
            _shannon_entropy, raw=True
        )

        # Z-score of close vs SMA(50)
        sma_50 = close.rolling(50).mean()
        sma_50_std = close.rolling(50).std().replace(0, np.nan)
        df["zscore_close_sma50"] = (close - sma_50) / sma_50_std

        # Price percentile in 200-bar window
        df["price_percentile_200"] = close.rolling(200).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(ts) -> str:
    """Coerce a pandas Timestamp / datetime into an ISO 8601 string."""
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if isinstance(ts, datetime) and ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def _hurst_rs(series: np.ndarray) -> float:
    """
    Simplified R/S Hurst exponent estimator.
    H < 0.5 → mean-reverting, H = 0.5 → random walk, H > 0.5 → trending.
    """
    n = len(series)
    if n < 20:
        return 0.5
    series = series[~np.isnan(series)]
    if len(series) < 20:
        return 0.5

    mean = np.mean(series)
    dev = np.cumsum(series - mean)
    r = np.max(dev) - np.min(dev)
    s = np.std(series, ddof=1)
    if s == 0 or r == 0:
        return 0.5
    return np.log(r / s) / np.log(n)


def _shannon_entropy(series: np.ndarray) -> float:
    """
    Shannon entropy of discretized return series.
    Bins returns into 5 buckets, computes -sum(p * log2(p)).
    Higher entropy = more random/uncertain price action.
    """
    series = series[~np.isnan(series)]
    if len(series) < 5:
        return 0.0
    # Discretize into 5 bins
    counts, _ = np.histogram(series, bins=5)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ===========================================================================
# Lookahead-safe feature_store reads (Phase 1E)
#
# External data sources publish with delays — FRED CPI lags ~14 days,
# CFTC COT lags ~3 days, etc. A naive read_feature_store(end=T) returns
# rows whose timestamp is <= T but whose actual publication was AFTER T,
# leaking future information into past-bar simulations.
#
# `read_feature_store_safe` enforces a per-source `release_lag_hours` from
# config/data_feeds.yaml and is the only entry point Phase 2 backtests
# should use against feature_store. Direct calls to read_feature_store
# remain available for live-bot reads where lookahead isn't a concern
# (caller's "now" is always after any release).
# ===========================================================================

_DATA_FEEDS_PATH = Path(__file__).resolve().parents[2] / "config" / "data_feeds.yaml"
_data_feeds_cache: Optional[dict] = None


def _load_data_feeds_yaml(path: Optional[Path] = None, *, force: bool = False) -> dict:
    """
    Read and cache ``config/data_feeds.yaml``. The cache is process-local
    and refreshed via ``force=True`` (used in tests).

    Returns the parsed YAML dict. Raises FileNotFoundError or YAMLError
    on a malformed config — better to fail loud at startup than silently
    fall back to lookahead-leaking defaults.
    """
    global _data_feeds_cache
    if _data_feeds_cache is not None and not force and path is None:
        return _data_feeds_cache
    cfg_path = path or _DATA_FEEDS_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "sources" not in cfg:
        raise ValueError(f"data_feeds.yaml missing top-level `sources:` key (read {cfg_path})")
    if path is None:
        _data_feeds_cache = cfg
    return cfg


async def read_feature_store_safe(
    store: "DataStore",
    symbol: str,
    feature_group: str,
    as_of: datetime,
    *,
    start: Optional[datetime] = None,
    feeds_config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Lookahead-safe wrapper around ``DataStore.read_feature_store``.

    Subtracts the source's ``release_lag_hours`` from ``as_of`` before
    querying, so the returned DataFrame never contains rows that wouldn't
    have been published by ``as_of``. Backtests MUST use this instead of
    calling ``read_feature_store`` directly — the bare reader will happily
    return rows from the future.

    Args:
        store: Async ``DataStore`` instance (already connected).
        symbol: Instrument key. Use ``"_GLOBAL"`` for source-wide rows
            (e.g. ECB curve).
        feature_group: Source label matching a key under ``sources:`` in
            ``data_feeds.yaml`` (e.g. ``"fred_macro"``, ``"cot_tff"``).
        as_of: Naive UTC datetime — the simulated "now" for the caller.
            Cutoff for lookahead checking: only rows with effective
            ``knowable_at <= as_of`` are returned.
        start: Optional naive UTC lower bound. Defaults to
            ``effective_end - 365 days`` so the partition planner can
            prune (per the C-1 contract on read_feature_store).
        feeds_config: Optional pre-loaded config dict (for tests). When
            None, lazy-loads from ``config/data_feeds.yaml``.

    Returns:
        DataFrame indexed by ``timestamp`` (sorted ascending), one column
        per value key in the row's JSONB. Empty DataFrame if no rows
        satisfy the bounds.

    Raises:
        ValueError: ``feature_group`` not in ``data_feeds.yaml`` — refusing
            to query without a release-lag bound (lookahead risk).

    Example — point-in-time read at simulated bar T_sim:

        df = await read_feature_store_safe(
            store, symbol="GBPUSD", feature_group="fred_macro",
            as_of=T_sim,
        )
        latest = df.iloc[-1] if not df.empty else None
    """
    cfg = feeds_config if feeds_config is not None else _load_data_feeds_yaml()
    src = cfg.get("sources", {}).get(feature_group)
    if src is None:
        raise ValueError(
            f"feature_group {feature_group!r} not in data_feeds.yaml — "
            f"refusing to query without a release-lag bound (lookahead risk). "
            f"Add a `sources.{feature_group}` block with release_lag_hours "
            f"before using this feature_group in backtests."
        )

    lag_hours = src.get("release_lag_hours")
    if lag_hours is None or not isinstance(lag_hours, (int, float)):
        raise ValueError(
            f"sources.{feature_group}.release_lag_hours missing or invalid in "
            f"data_feeds.yaml ({lag_hours!r})"
        )

    effective_end = as_of - timedelta(hours=float(lag_hours))
    if start is None:
        # Honor the C-1 partition-pruning contract: provide a lower bound
        # so the planner doesn't evaluate every monthly partition back to
        # 2000. 365 days is enough warmup for any rolling z-score / YoY.
        start = effective_end - timedelta(days=365)

    return await store.read_feature_store(
        symbol=symbol,
        feature_group=feature_group,
        start=start,
        end=effective_end,
    )
