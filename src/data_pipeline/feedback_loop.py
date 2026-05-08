"""
feedback_loop.py — Prediction Error Tracking & Adaptive Retraining

Closes the learning loop by:
    1. Computing actual outcomes (ground-truth labels) once the next bar arrives
    2. Comparing predictions against outcomes to measure model accuracy
    3. Deciding when retraining is necessary (threshold + scheduled)
    4. Applying exponential decay sample weights so recent errors matter more

Retraining triggers (checked daily via APScheduler):
    - Directional accuracy drops below 52% in a rolling 500-prediction window
    - MSE increases more than 10% relative to the previous window
    - Scheduled: LSTM daily, HMM weekly (regardless of metrics)

Exponential decay weighting (for LSTM retraining):
    w[i] = exp(-λ × days_since_bar[i]),  λ = 0.02
    1-week-old bar  ≈ 87% weight
    1-month-old bar ≈ 55% weight
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.data_pipeline.data_store import DataStore

logger = logging.getLogger(__name__)

# Dead-zone for the 3-class directional-accuracy metric. The LSTM regression
# head outputs a continuous value approximating E[TB label in {-1, 0, +1}].
# Without a dead-zone, sign-matching collapses on label=0 bars (~41% of the
# training distribution is time-exit) because no continuous prediction is
# ever exactly 0. Predictions whose magnitude is below this threshold are
# classified as "flat" (class 0), matching label=0 correctly.
DIRECTION_EPSILON = 0.1


class FeedbackLoop:
    """
    Orchestrates outcome logging, error computation, and adaptive model retraining.

    Usage (inside APScheduler job or trading loop):
        loop = FeedbackLoop(data_store, hmm_classifier, lstm_predictor)

        # Called each H4 bar:
        await loop.run_h4_cycle("XAUUSD", prev_bar_timestamp, timeframe="H4")

        # Called daily by APScheduler:
        await loop.check_and_retrain("XAUUSD", data_feed, feature_engineer)
    """

    # Retraining thresholds
    DIRECTIONAL_ACCURACY_THRESHOLD: float = 0.52   # below this → retrain
    MSE_INCREASE_THRESHOLD: float = 0.10           # 10% relative increase → retrain

    # Phase C.3: when False, ``check_and_retrain`` logs errors/thresholds
    # but NEVER triggers an in-process lstm.retrain. The in-process path
    # bypasses the TB+PCA+multi-TF pipeline and would corrupt the
    # production model. Full retrains are handled by the monthly
    # subprocess job in main.py that invokes the proper training scripts.
    enable_inprocess_retrain: bool = False
    MIN_PREDICTIONS_TO_EVALUATE: int = 50          # skip check if fewer predictions

    def __init__(self, data_store: "DataStore", hmm=None, lstm=None):
        """
        Args:
            data_store: Async PostgreSQL data layer (DataStore instance)
            hmm:        HMMRegimeClassifier — used to label actual next regime
            lstm:       LSTMPricePredictor — retrained when thresholds breached
        """
        self.data_store = data_store
        self.hmm = hmm
        self.lstm = lstm
        self._prev_metrics: dict[str, dict] = {}   # symbol → last rolling metrics snapshot

    # -------------------------------------------------------------------------
    # Outcome Computation
    # -------------------------------------------------------------------------

    async def compute_actual_outcomes(
        self,
        symbol: str,
        bar_timestamp: str,
        timeframe: str = "H4",
    ) -> Optional[int]:
        """
        Compute and persist the actual log return for bar ``bar_timestamp``.

        The actual_next_return for bar t is:
            log(close[t+1] / close[t])
        where close[t+1] comes from the bar immediately after ``bar_timestamp``.

        This method is idempotent — calling it twice for the same bar is safe
        (the DataStore uses ON CONFLICT DO NOTHING for actual_outcomes).

        Args:
            symbol:        Trading symbol (e.g. "XAUUSD")
            bar_timestamp: ISO 8601 timestamp string of the bar to label
            timeframe:     OHLCV timeframe to look up (default "H4")

        Returns:
            Inserted actual_outcomes row id, or None if next bar not yet available.
        """
        # Get next bar (t+1) — returns None if it hasn't closed yet
        next_bar = await self.data_store.get_next_bar(symbol, timeframe, bar_timestamp)
        if next_bar is None:
            logger.debug(f"[{symbol}] Next bar after {bar_timestamp} not yet available")
            return None

        # Get current bar (t) close price from DB
        try:
            bar_dt = _parse_iso(bar_timestamp)
        except ValueError as e:
            logger.error(f"[{symbol}] Cannot parse bar_timestamp '{bar_timestamp}': {e}")
            return None

        current_bar_df = await self.data_store.get_ohlcv_range(
            symbol, timeframe,
            start=bar_dt - timedelta(seconds=1),
            end=bar_dt + timedelta(seconds=1),
        )
        if current_bar_df.empty:
            logger.warning(f"[{symbol}] Bar {bar_timestamp} not found in ohlcv_bars")
            return None

        current_close = float(current_bar_df.iloc[-1]["close"])
        next_close = float(next_bar["close"])

        if current_close <= 0 or next_close <= 0:
            logger.error(
                f"[{symbol}] Invalid close prices: current={current_close}, next={next_close}"
            )
            return None

        actual_next_return = float(np.log(next_close / current_close))

        # Optionally label the HMM regime at bar t+1
        actual_next_regime: Optional[int] = None
        if self.hmm and symbol in getattr(self.hmm, "_models", {}):
            actual_next_regime = await self._label_regime(symbol, next_bar["bar_timestamp"])

        # Optionally compute the Triple-Barrier label by walking forward.
        # Returns None if not enough future bars exist yet (the label
        # backfills naturally on a later compute_actual_outcomes call —
        # save_actual_outcome upserts the missing field).
        actual_tb_label = await self._compute_tb_label(symbol, bar_timestamp, timeframe)

        outcome_id = await self.data_store.save_actual_outcome(
            symbol=symbol,
            bar_timestamp=bar_timestamp,
            actual_next_return=actual_next_return,
            actual_next_regime=actual_next_regime,
            actual_tb_label=actual_tb_label,
        )
        logger.debug(
            f"[{symbol}] Outcome persisted for {bar_timestamp}: "
            f"return={actual_next_return:.6f}, regime={actual_next_regime}, "
            f"tb={actual_tb_label}"
        )
        return outcome_id

    async def _compute_tb_label(
        self,
        symbol: str,
        bar_timestamp: str,
        timeframe: str = "H4",
        lookahead_bars: int = 20,
        tp_r_mult: float = 2.5,
        sl_atr_mult: float = 2.0,
    ) -> Optional[float]:
        """
        Compute the Triple-Barrier label for ``bar_timestamp`` by walking
        forward in the OHLCV cache. Returns one of {-1.0, 0.0, +1.0}, or
        None when fewer than ``lookahead_bars`` future bars exist.

        Defaults match the LSTM training command line used for monthly
        retrains. Per-symbol overrides are intentionally not applied here
        — the metric is a comparable cross-symbol signal-quality gauge,
        not a strategy backtest.
        """
        try:
            bar_dt = _parse_iso(bar_timestamp)
        except ValueError:
            return None

        # 20 H4 bars of forex trading = ~3.3 market-days, which crosses a
        # weekend half the time — 10 calendar days forward guarantees 20+
        # future bars for all symbols regardless of weekend placement.
        end_dt = bar_dt + timedelta(days=10)
        # ATR inside compute_triple_barrier_labels is a Wilder EWM with
        # min_periods=14, so it returns NaN for the first 14 bars of any
        # window. Without lookback, the target bar sits at index 0 with
        # ATR=NaN, the label computation silently skips it ("continue"),
        # and the entry bar's label defaults to 0.0 — which blind-sourced
        # 91 ghost-zero labels in the 2026-04-24 backfill. Pulling past
        # bars gives the ATR time to warm up before the target row.
        #
        # Window sizing: 14 trading bars of warmup is the hard minimum.
        # Forex is 24×5 (closed Sat), so a 7-day calendar lookback buys
        # ~5 trading days × 6 H4 bars = 30 bars even across a weekend.
        # ETHUSD is 24×7 so even tighter windows suffice, but we use the
        # same 7-day window uniformly.
        start_dt = bar_dt - timedelta(days=7)

        df = await self.data_store.get_ohlcv_range(
            symbol, timeframe,
            start=start_dt,
            end=end_dt,
        )
        if df is None or df.empty:
            return None

        # Locate the target bar within the expanded window.
        try:
            target_idx = int(df.index.get_loc(bar_dt))
        except (KeyError, TypeError):
            # Pre-2026-04-14 rows used broker-local bar_timestamp; they
            # won't match the true-UTC ohlcv_bars index. Nothing we can
            # do about those without rewriting the legacy rows. Return
            # None so they stay null rather than getting fake labels.
            return None

        # Need ≥14 past bars for ATR warmup + ≥lookahead_bars future bars.
        future_bars = len(df) - target_idx - 1
        if target_idx < 14 or future_bars < lookahead_bars:
            return None

        try:
            from src.data_pipeline.feature_engineering import FeatureEngineer
            labels = FeatureEngineer.compute_triple_barrier_labels(
                df,
                tp_r_mult=tp_r_mult,
                sl_atr_mult=sl_atr_mult,
                time_limit_bars=lookahead_bars,
            )
        except Exception as e:
            logger.warning(f"[{symbol}] TB label computation failed at {bar_timestamp}: {e}")
            return None

        if len(labels) <= target_idx:
            return None
        return float(labels[target_idx])

    async def _label_regime(self, symbol: str, bar_timestamp: str) -> Optional[int]:
        """
        Run HMM.predict() on a window of D1 bars ending at ``bar_timestamp``
        to get the actual regime index at that bar.

        Returns:
            Regime index (0–4) or None on failure.
        """
        try:
            bar_dt = _parse_iso(bar_timestamp)
        except ValueError:
            return None

        try:
            regime_df = await self.data_store.get_ohlcv_range(
                symbol, "D1",
                start=bar_dt - timedelta(days=120),
                end=bar_dt,
            )
            if regime_df.empty or len(regime_df) < 30:
                return None

            # Lazy import to avoid circular dependency at module load time
            from src.data_pipeline.feature_engineering import FeatureEngineer
            engineer = FeatureEngineer()
            feat_df = engineer.transform(regime_df)
            feat_matrix = engineer.to_matrix(feat_df)
            regime_result = self.hmm.predict(symbol, feat_matrix)
            return regime_result.regime_index
        except Exception as e:
            logger.debug(f"[{symbol}] Could not label regime at {bar_timestamp}: {e}")
            return None

    # -------------------------------------------------------------------------
    # Error Computation
    # -------------------------------------------------------------------------

    async def backfill_missing_tb_labels(self, symbol: str) -> int:
        """Fill actual_tb_label on past outcome rows where it's still NULL.

        TB labels need ~20 future H4 bars to compute, so outcomes written
        at bar-close always start with tb_label=None. This method walks
        forward through the now-available future bars and populates the
        label so ``compute_prediction_errors`` can score those predictions.

        Idempotent — only touches rows where the label is still NULL.
        Returns the number of rows successfully backfilled.
        """
        from sqlalchemy import text
        # Oldest-first + LIMIT bounds worst-case work if the cron has
        # been down for a while. backfill_tb_label opens its own session
        # (data_store.py:1058) so reading here and writing later is safe.
        async with self.data_store._session_factory() as session:
            rows = await session.execute(text(
                "SELECT id, bar_timestamp FROM actual_outcomes "
                "WHERE symbol = :s AND actual_tb_label IS NULL "
                "ORDER BY bar_timestamp ASC LIMIT 200"
            ), {"s": symbol})
            pending = rows.fetchall()

        filled = 0
        for outcome_id, bar_ts in pending:
            try:
                label = await self._compute_tb_label(symbol, bar_ts, timeframe="H4")
            except Exception as exc:
                logger.warning(f"[{symbol}] TB backfill failed for {bar_ts}: {exc}")
                continue
            if label is None:
                continue  # still too new — will retry tomorrow
            await self.data_store.backfill_tb_label(int(outcome_id), float(label))
            filled += 1
        if filled:
            logger.info(f"[{symbol}] Backfilled {filled} TB labels")
        return filled

    async def compute_prediction_errors(self, symbol: str) -> int:
        """
        Find all price_return predictions that now have matching outcomes but
        no error record yet. Compute and persist ``prediction_errors`` rows.

        This is idempotent — already-processed pairs are skipped because they
        already have a non-NULL error_magnitude in the JOIN result.

        Returns:
            Number of newly created prediction_error records.
        """
        df = await self.data_store.get_predictions_with_outcomes(symbol, limit=10_000)

        if df.empty:
            return 0

        # Rows where outcome exists but no error record yet
        pending = df[
            df["outcome_id"].notna() &
            df["error_magnitude"].isna() &
            (df.get("prediction_type", "price_return") == "price_return")
        ]

        if pending.empty:
            return 0

        count = 0
        skipped_no_tb = 0
        for _, row in pending.iterrows():
            try:
                predicted_val = float(row["predicted_value"])
                if not np.isfinite(predicted_val):
                    # NaN/Inf here silently becomes class -1 under the
                    # dead-zone logic (abs(NaN) < eps is False, NaN > 0
                    # is False → else branch fires). Skip instead.
                    logger.warning(
                        f"[{symbol}] non-finite predicted_value "
                        f"{predicted_val!r} on prediction "
                        f"{row.get('prediction_id')}; skipping"
                    )
                    continue

                # Prefer the Triple-Barrier label (matches the LSTM's
                # training target). Fall back to the raw log-return only
                # when TB hasn't been computed yet (insufficient future
                # bars). Skip rows where neither is available.
                tb_label = row.get("actual_tb_label")
                has_tb = tb_label is not None and not pd.isna(tb_label)
                if has_tb:
                    actual_val = float(tb_label)
                else:
                    # Wait for the TB label rather than recording a
                    # misleading sign-against-log-return result.
                    skipped_no_tb += 1
                    continue

                error_magnitude = abs(predicted_val - actual_val)
                # 3-class dead-zone match — the TB label lives in
                # {-1, 0, +1} where 0 is the most-common class (~41%
                # of training bars are time-exits). Naive sign() gives
                # sign(0)=0 vs sign(±eps)=±1, guaranteeing mismatch on
                # every label=0 bar. Collapse near-zero predictions to
                # the flat class so they can actually match.
                pred_class = (
                    0 if abs(predicted_val) < DIRECTION_EPSILON
                    else (1 if predicted_val > 0 else -1)
                )
                actual_class = int(np.sign(actual_val))
                direction_correct = bool(pred_class == actual_class)

                await self.data_store.save_prediction_error(
                    prediction_id=int(row["prediction_id"]),
                    outcome_id=int(row["outcome_id"]),
                    error_magnitude=error_magnitude,
                    direction_correct=direction_correct,
                )
                count += 1
            except Exception as e:
                logger.warning(
                    f"[{symbol}] Failed to save error for prediction "
                    f"{row.get('prediction_id')}: {e}"
                )

        if count or skipped_no_tb:
            logger.info(
                f"[{symbol}] Computed {count} new prediction errors "
                f"({skipped_no_tb} skipped — TB label not ready yet)"
            )
        return count

    # -------------------------------------------------------------------------
    # Rolling Metrics
    # -------------------------------------------------------------------------

    async def get_rolling_metrics(self, symbol: str, window: int = 500) -> dict:
        """
        Fetch rolling accuracy and error metrics over the last ``window``
        price_return predictions with confirmed outcomes.

        Returns:
            {
                "directional_accuracy": float,   # fraction of correct direction calls
                "mse":  float,                   # mean squared error
                "mae":  float,                   # mean absolute error
                "n_predictions": int,            # actual count used
            }
        """
        return await self.data_store.get_rolling_metrics(symbol, window)

    # -------------------------------------------------------------------------
    # Retraining Decision
    # -------------------------------------------------------------------------

    async def should_retrain(self, symbol: str, window: int = 500) -> tuple[bool, str]:
        """
        Check if adaptive retraining should be triggered for a symbol.

        Threshold 1: directional accuracy < DIRECTIONAL_ACCURACY_THRESHOLD
        Threshold 2: MSE increased > MSE_INCREASE_THRESHOLD relative to previous window

        Returns:
            (True, reason_string) if retraining is warranted.
            (False, reason_string) otherwise.
        """
        try:
            metrics = await self.get_rolling_metrics(symbol, window)
        except Exception as e:
            logger.warning(f"[{symbol}] Could not fetch metrics: {e}")
            return False, f"metrics unavailable: {e}"

        n = metrics.get("n_predictions", 0)
        if n < self.MIN_PREDICTIONS_TO_EVALUATE:
            return False, f"insufficient predictions ({n} < {self.MIN_PREDICTIONS_TO_EVALUATE})"

        directional_acc = metrics.get("directional_accuracy", 1.0)
        current_mse = metrics.get("mse", 0.0)

        # Threshold 1: directional accuracy below minimum
        if directional_acc < self.DIRECTIONAL_ACCURACY_THRESHOLD:
            return (
                True,
                f"directional_accuracy={directional_acc:.3f} < {self.DIRECTIONAL_ACCURACY_THRESHOLD:.2f}"
            )

        # Threshold 2: MSE deteriorated > 10% vs previous window
        prev = self._prev_metrics.get(symbol, {})
        if prev and prev.get("mse", 0.0) > 1e-12:
            mse_increase = (current_mse - prev["mse"]) / prev["mse"]
            if mse_increase > self.MSE_INCREASE_THRESHOLD:
                return (
                    True,
                    f"MSE increased {mse_increase:.1%} "
                    f"(prev={prev['mse']:.6f} → now={current_mse:.6f})"
                )

        # Update snapshot for next comparison
        self._prev_metrics[symbol] = metrics
        return False, f"metrics OK (acc={directional_acc:.3f}, mse={current_mse:.6f})"

    # -------------------------------------------------------------------------
    # Sample Weighting
    # -------------------------------------------------------------------------

    @staticmethod
    def apply_exponential_weights(
        timestamps: list,
        lambda_: float = 0.02,
        reference_time: Optional[datetime] = None,
    ) -> np.ndarray:
        """
        Compute exponential decay sample weights for LSTM training.

        More recent bars receive higher weights so the model focuses on
        learning from recent patterns over stale historical ones.

        Formula:
            w[i] = exp(-λ × days_since_bar[i])
        Weights are normalized to sum to 1.

        Args:
            timestamps:      List of bar datetimes (one per training sample).
                             Can be datetime objects or ISO 8601 strings.
            lambda_:         Decay rate (default 0.02 → 1-week-old ≈ 87% weight,
                             1-month-old ≈ 55% weight)
            reference_time:  The "now" reference point. Defaults to the
                             current UTC time.

        Returns:
            np.ndarray of shape (len(timestamps),) — normalized weights.
        """
        if not timestamps:
            return np.array([], dtype=float)

        # datetime.utcnow() is deprecated in 3.12+. The rest of this
        # method strips tzinfo a few lines down, so we take "now" as
        # tz-aware UTC then drop the tzinfo to match the naive arithmetic.
        ref = reference_time or datetime.now(timezone.utc).replace(tzinfo=None)

        days_old = np.zeros(len(timestamps), dtype=float)
        for i, ts in enumerate(timestamps):
            if isinstance(ts, str):
                ts = _parse_iso(ts)
            # Strip timezone info for arithmetic (treat all as UTC-naive)
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            days_old[i] = max(0.0, (ref - ts).total_seconds() / 86400.0)

        weights = np.exp(-lambda_ * days_old)
        total = weights.sum()
        if total < 1e-12:
            return np.ones(len(timestamps), dtype=float) / len(timestamps)
        return (weights / total).astype(float)

    # -------------------------------------------------------------------------
    # Full Orchestration
    # -------------------------------------------------------------------------

    async def check_and_retrain(
        self,
        symbol: str,
        data_feed,
        feature_engineer,
    ) -> None:
        """
        Full feedback loop orchestration — called by APScheduler daily.

        Steps:
            1. Compute any pending prediction errors
            2. Check if adaptive retraining thresholds are breached
            3. If yes, retrain LSTM with exponential sample weights

        Args:
            symbol:           Trading symbol (e.g. "XAUUSD")
            data_feed:        MT5DataFeed instance
            feature_engineer: FeatureEngineer instance
        """
        logger.info(f"[{symbol}] FeedbackLoop: running daily check...")

        # Step 0: Backfill TB labels on outcome rows that were too new
        # to label at bar-close (needs ~20 future H4 bars). Without this,
        # prediction_errors never populates and the Models screen accuracy
        # chip stays stuck at 0/50 forever.
        await self.backfill_missing_tb_labels(symbol)

        # Step 1: Compute pending errors (idempotent)
        n_new = await self.compute_prediction_errors(symbol)
        if n_new > 0:
            logger.info(f"[{symbol}] Computed {n_new} new prediction errors")

        # Step 2: Evaluate retraining thresholds
        do_retrain, reason = await self.should_retrain(symbol)

        if not do_retrain:
            logger.info(f"[{symbol}] No adaptive retrain needed — {reason}")
            return

        logger.info(f"[{symbol}] Adaptive retrain triggered — {reason}")

        # Phase C.3: in-process retrain is DISABLED by default. It would
        # call ``lstm.retrain`` which uses ``_train_from_feed`` — a plain
        # 56-feature, log-return, no-PCA pipeline that does NOT match the
        # TB+PCA+multi-TF production config. Running it would silently
        # overwrite a good 25-dim TB model with a vanilla 56-dim one.
        # The monthly subprocess job in main.py handles proper retrains.
        if not self.enable_inprocess_retrain:
            logger.info(
                f"[{symbol}] In-process retrain disabled (see "
                f"FeedbackLoop.enable_inprocess_retrain). Next scheduled "
                f"full retrain via monthly_full_retrain cron job."
            )
            return

        # Step 3: Retrain LSTM in thread-pool executor (PyTorch is blocking)
        if self.lstm is not None:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self.lstm.retrain(
                        data_feed, feature_engineer, symbols=[symbol]
                    ),
                )
                logger.info(f"[{symbol}] LSTM adaptive retrain complete")
            except Exception as e:
                logger.error(f"[{symbol}] LSTM retrain failed: {e}", exc_info=True)
        else:
            logger.warning(f"[{symbol}] No LSTM instance configured — skipping retrain")

    async def run_h4_cycle(
        self,
        symbol: str,
        prev_bar_timestamp: str,
        timeframe: str = "H4",
    ) -> None:
        """
        Lightweight per-H4-bar update called at the end of each H4 bar.

        1. Compute the actual outcome for the bar that just closed
        2. Compute any newly matchable prediction errors

        Args:
            symbol:              Trading symbol
            prev_bar_timestamp:  ISO timestamp of the bar that just closed
            timeframe:           Data timeframe (default "H4")
        """
        await self.compute_actual_outcomes(symbol, prev_bar_timestamp, timeframe)
        await self.compute_prediction_errors(symbol)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp string to a timezone-naive datetime."""
    ts = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt
