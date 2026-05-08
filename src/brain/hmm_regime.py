"""
hmm_regime.py — Hidden Markov Model Market Regime Classifier

Classifies the current market into one of 5 states. the trading universe (2026-04-29)
flipped the multipliers from asymmetric (long-only relic) to symmetric so
that bidirectional pairs get matching conviction at both extremes:

    0 = Crash     → position multiplier 1.0   (full short conviction)
    1 = Bear      → position multiplier 0.75
    2 = Neutral   → position multiplier 0.5   (LSTM drives alone, regime sign=0)
    3 = Bull      → position multiplier 0.75
    4 = Euphoria  → position multiplier 1.0   (full long conviction)

The directional sign is applied separately in signal_combiner._regime_score
via REGIME_DIRECTION_SIGN: Crash/Bear → -1, Neutral → 0, Bull/Euphoria → +1.

The HMM is trained on Daily (D1) bars with features:
    - Log returns
    - Realized volatility (rolling std of returns)
    - Volume ratio (current / 20-bar average)
    - RSI(14)
    - ATR(14) normalized by price

Implementation uses hmmlearn.GaussianHMM with multiple random
initializations; the run with highest log-likelihood is kept.

Retraining: every 7 days — loads the full historical D1 window from
PostgreSQL (``ohlcv_bars``) instead of fetching fresh from MT5, so the
model always trains on the complete corpus. After training, a row is
written to ``model_versions`` and the trained pickle is saved to
``data/models/hmm_{symbol}.pkl``.

Prediction logging: each call to ``predict()`` is persisted via
``DataStore.save_prediction(prediction_type='regime', ...)`` so the
feedback loop can later compare it against the realized regime label.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import joblib
from hmmlearn import hmm

if TYPE_CHECKING:
    from src.data_pipeline.data_store import DataStore

logger = logging.getLogger(__name__)

REGIME_LABELS = {0: "Crash", 1: "Bear", 2: "Neutral", 3: "Bull", 4: "Euphoria"}
MODEL_PATH = Path("data/models/hmm_{symbol}.pkl")


@dataclass
class RegimeResult:
    symbol: str
    regime_index: int              # 0–4
    regime_label: str              # "Crash", "Bear", etc.
    state_probability: float       # Probability of the predicted state
    position_multiplier: float     # 1.0, 0.75, 0.50, 0.75, or 1.0 (symmetric)
    all_probabilities: np.ndarray  # Full posterior probability vector (5,)

    # Volatility fields consumed by the strategy layer (StrategyOrchestrator).
    # Both default to safe values so any code constructing a RegimeResult
    # without these (existing tests, older callers) still works.
    expected_volatility: float = 0.0
    #   σ of log_return in the current state's Gaussian emission.
    #   Compute in predict() as sqrt(model.covars_[state_idx][0, 0]).
    all_expected_vols: Optional[np.ndarray] = None
    #   Shape (n_components,) — σ across all states. Lets the orchestrator
    #   compute a vol rank for the current state without re-running the model.
    #   Compute in predict() as np.sqrt(model.covars_[:, 0, 0]).


class HMMRegimeClassifier:
    """
    Gaussian HMM with 5 components for market regime classification.

    Usage:
        hmm_clf = HMMRegimeClassifier(data_store=store)
        await hmm_clf.load_or_train_async(data_feed, feature_engineer, ["XAUUSD"])
        result = hmm_clf.predict("XAUUSD", feature_matrix)
    """

    # Regime → position-size multiplier mapping.
    # Order: Crash / Bear / Neutral / Bull / Euphoria.
    # Production values are tuned per the operator's bidirectional vs long-only
    # universe and are REDACTED in this public template. Replace with your own
    # tuned mapping. Two example shapes:
    #   asymmetric long-only:  {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}
    #   symmetric bidirectional: values mirror around Neutral
    MULTIPLIERS = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}  # PLACEHOLDER
    MODEL_NAME_FMT = "hmm_{symbol}"

    def __init__(
        self,
        n_components: int = 5,
        n_init: int = 10,
        data_store: Optional["DataStore"] = None,
    ):
        """
        Args:
            n_components: Number of hidden states (5 for our regime taxonomy)
            n_init:       Random initializations — best log-likelihood wins
            data_store:   Async DataStore for DB-based training and logging
        """
        self.n_components = n_components
        self.n_init = n_init
        self.data_store = data_store
        self._models: dict[str, hmm.GaussianHMM] = {}
        self._versions: dict[str, int] = {}   # symbol → current model_version int
        # Bug 1 fix: z-score normalization stats from training. At inference
        # the same mean/std are applied so the feature scale matches the
        # HMM's learned emission distributions.
        self._norm_means: dict[str, np.ndarray] = {}
        self._norm_stds: dict[str, np.ndarray] = {}
        # Canonicalized state ordering: raw HMM state idx → regime label idx
        # (0=Crash … 4=Euphoria). Populated by _sort_states_by_mean_return()
        # at the end of fit() / train(). Without this, weekly retrains produce
        # arbitrary state indices ("state 0 means Bull on Monday, Crash on
        # Tuesday"). Consumed by predict() to keep regime_index stable.
        self._state_label_maps: dict[str, dict[int, int]] = {}
        self._feature_manifests: dict[str, list[str]] = {}

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(
        self, symbol: str, feature_matrix: np.ndarray,
        feature_manifest: Optional[list[str]] = None,
    ) -> None:
        """
        Fit a GaussianHMM on the provided feature matrix.

        Runs ``n_init`` random initializations and keeps the model with the
        highest log-likelihood score. After training, state indices are
        remapped via ``_sort_states_by_mean_return()`` so that
        state 0 → Crash and state 4 → Euphoria.

        Required finalization step (MUST be called after the model is fit,
        before ``self._models[symbol]`` is populated — otherwise weekly
        retrains produce arbitrary state→label assignments):

            best_model = ...  # fitted GaussianHMM with highest log-likelihood
            label_map = self._sort_states_by_mean_return(best_model.means_)
            self._state_label_maps[symbol] = label_map
            self._models[symbol] = best_model

        Args:
            symbol:         Trading symbol (e.g. "XAUUSD")
            feature_matrix: Shape (n_bars, n_features) — D1 bars.
                            Accepted either z-scored (from to_matrix()) or raw.
                            If raw, this method z-scores internally and stores
                            the normalization stats for inference.
        """
        if feature_matrix.ndim != 2 or feature_matrix.shape[0] < self.n_components:
            raise ValueError(
                f"Need at least {self.n_components} bars; got {feature_matrix.shape}"
            )

        # Bug 1 fix: compute and store z-score normalization stats from the
        # training data. If the caller already z-scored via to_matrix(), the
        # means will be ~0 and stds ~1, so re-z-scoring is a near-identity.
        # Storing them lets predict() apply the SAME transform at inference.
        means = feature_matrix.mean(axis=0)
        stds = feature_matrix.std(axis=0)
        stds[stds == 0] = 1.0
        normalized = (feature_matrix - means) / stds
        self._norm_means[symbol] = means
        self._norm_stds[symbol] = stds

        best_model = None
        best_score = -np.inf

        for i in range(self.n_init):
            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=200,
                tol=1e-4,
                random_state=i,
            )
            try:
                model.fit(normalized)
                score = model.score(normalized)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                logger.warning(
                    "[%s] HMM init %d/%d failed: %s", symbol, i + 1, self.n_init, e
                )

        if best_model is None:
            raise RuntimeError(f"All {self.n_init} HMM initializations failed for {symbol}")

        # Determine log_return column index from manifest (Bug 2 fix:
        # was hardcoded to 0 which is 'adx' in alphabetically-sorted columns).
        log_return_col = 0
        if feature_manifest is not None:
            sorted_manifest = sorted(feature_manifest)
            if "log_return" in sorted_manifest:
                log_return_col = sorted_manifest.index("log_return")
            else:
                logger.warning(
                    "[%s] 'log_return' not in feature manifest — falling back "
                    "to col 0 for state ordering", symbol,
                )
        label_map = self._sort_states_by_mean_return(
            best_model.means_, log_return_col=log_return_col,
        )
        self._state_label_maps[symbol] = label_map
        self._models[symbol] = best_model
        if feature_manifest is not None:
            self._feature_manifests[symbol] = feature_manifest

        logger.info(
            "[%s] HMM trained: %d bars, %d features, log-likelihood=%.2f",
            symbol, feature_matrix.shape[0], feature_matrix.shape[1], best_score,
        )

    def predict(self, symbol: str, feature_matrix: np.ndarray) -> RegimeResult:
        """
        Predict current market regime for a symbol.

        Args:
            symbol:         Trading symbol
            feature_matrix: Shape (n_bars, n_features) — recent D1 bars

        Returns:
            RegimeResult with regime label, probability, position multiplier,
            and the two volatility fields consumed by the strategy layer.

        When this stub is implemented, it must populate **every** field on
        ``RegimeResult`` — including the two volatility fields. The recipe is:

            model = self._models[symbol]                   # hmmlearn.GaussianHMM
            # CRITICAL: use predict_proba()[-1], NOT model.predict() / Viterbi.
            # predict_proba returns gamma (forward-backward smoothed posterior).
            # At the LAST timestep beta == 1, so gamma[-1] == alpha[-1] / Z, i.e.
            # the filtered posterior at T. This is the only decoding method that
            # is guaranteed causal when the window ends at the target bar, which
            # is what FeedbackLoop._label_regime relies on for honest training
            # labels. Viterbi decoding re-estimates interior states under the
            # max-likelihood path and MUST NOT be used for bar-level labeling.
            state_probs = model.predict_proba(feature_matrix)[-1]  # shape (5,)
            raw_state = int(np.argmax(state_probs))

            # Canonicalize raw HMM state → regime index (0=Crash … 4=Euphoria)
            # using the mapping stored on self._state_label_maps[symbol] after
            # fit(). Without this step, regime indices are arbitrary per-fit
            # and the weekly retrain causes categorical collapse.
            label_map = self._state_label_maps.get(symbol, {})
            state_idx = label_map.get(raw_state, raw_state)

            # Reorder the full posterior into canonical order so that consumers
            # like the strategy orchestrator and the feedback loop see a
            # consistent (n_components,) vector across retrains.
            canonical_probs = np.zeros_like(state_probs)
            for raw_i, canon_i in label_map.items():
                canonical_probs[canon_i] = state_probs[raw_i]

            # Feature 0 is the log_return column — see _sort_states_by_mean_return.
            # hmmlearn stores covars_ as 3D even for 'diag'. Check ndim:
            if model.covars_.ndim == 3:
                raw_vars = model.covars_[:, 0, 0]
            elif model.covars_.ndim == 2:
                raw_vars = model.covars_[:, 0]
            else:
                raw_vars = model.covars_.copy()
            canonical_vars = np.zeros(len(raw_vars), dtype=raw_vars.dtype)
            for raw_i, canon_i in label_map.items():
                canonical_vars[canon_i] = raw_vars[raw_i]
            expected_volatility = float(np.sqrt(max(float(canonical_vars[state_idx]), 0.0)))
            all_expected_vols   = np.sqrt(np.maximum(canonical_vars, 0.0))

        Note:
            This method is synchronous. The async wrapper ``predict_and_log()``
            additionally persists the prediction via ``DataStore.save_prediction()``.
        """
        if symbol not in self._models:
            raise RuntimeError(f"No trained HMM for {symbol}. Call train() first.")

        model = self._models[symbol]

        # Bug 1 fix: z-score normalize using stats from training.
        # If no stats saved (legacy models), fall back to batch normalization
        # which is still better than raw values.
        if symbol in self._norm_means and symbol in self._norm_stds:
            norm_matrix = (feature_matrix - self._norm_means[symbol]) / self._norm_stds[symbol]
        else:
            logger.warning(
                "[%s] No normalization stats — using batch z-score fallback",
                symbol,
            )
            means = feature_matrix.mean(axis=0)
            stds = feature_matrix.std(axis=0)
            stds[stds == 0] = 1.0
            norm_matrix = (feature_matrix - means) / stds
        norm_matrix = np.nan_to_num(norm_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Forward-backward smoothed posterior (NOT Viterbi — causal at last bar)
        state_probs = model.predict_proba(norm_matrix)[-1]  # shape (n_components,)
        raw_state = int(np.argmax(state_probs))

        # Canonicalize: raw HMM state → regime index (0=Crash … 4=Euphoria)
        label_map = self._state_label_maps.get(symbol, {})
        state_idx = label_map.get(raw_state, raw_state)

        # Reorder posterior into canonical order
        canonical_probs = np.zeros_like(state_probs)
        for raw_i, canon_i in label_map.items():
            canonical_probs[canon_i] = state_probs[raw_i]

        # Volatility fields from emission covariance (feature 0 = log_return)
        # hmmlearn stores covars_ as 3D (n_components, n_features, n_features)
        # for all covariance types — even 'diag'. Check ndim to be safe.
        if model.covars_.ndim == 3:
            raw_vars = model.covars_[:, 0, 0]
        elif model.covars_.ndim == 2:
            raw_vars = model.covars_[:, 0]
        else:  # spherical: (n_components,)
            raw_vars = model.covars_.copy()

        # Reorder covars into canonical order
        canonical_vars = np.zeros(len(raw_vars), dtype=raw_vars.dtype)
        for raw_i, canon_i in label_map.items():
            canonical_vars[canon_i] = raw_vars[raw_i]

        expected_volatility = float(np.sqrt(max(float(canonical_vars[state_idx]), 0.0)))
        all_expected_vols = np.sqrt(np.maximum(canonical_vars, 0.0))

        return RegimeResult(
            symbol=symbol,
            regime_index=state_idx,
            regime_label=REGIME_LABELS.get(state_idx, f"State{state_idx}"),
            state_probability=float(canonical_probs[state_idx]),
            position_multiplier=self.MULTIPLIERS.get(state_idx, 0.5),
            all_probabilities=canonical_probs,
            expected_volatility=expected_volatility,
            all_expected_vols=all_expected_vols,
        )

    async def predict_and_log(
        self,
        symbol: str,
        feature_matrix: np.ndarray,
        bar_timestamp: str,
    ) -> RegimeResult:
        """
        Run ``predict()`` and persist the result to ``model_predictions``.

        Args:
            symbol:         Trading symbol
            feature_matrix: Latest D1 feature window
            bar_timestamp:  ISO 8601 timestamp of the bar being predicted

        Returns:
            RegimeResult (also returned for immediate use by SignalCombiner).
        """
        result = self.predict(symbol, feature_matrix)

        if self.data_store is not None:
            try:
                await self.data_store.save_prediction(
                    symbol=symbol,
                    bar_timestamp=bar_timestamp,
                    model_name=self.MODEL_NAME_FMT.format(symbol=symbol),
                    model_version=self._versions.get(symbol, 0),
                    prediction_type="regime",
                    predicted_value=float(result.regime_index),
                    confidence=float(result.state_probability),
                )
            except Exception as e:
                logger.warning(
                    f"[{symbol}] Failed to persist HMM prediction: {e}"
                )

        return result

    # -------------------------------------------------------------------------
    # Lifecycle: load, train, retrain
    # -------------------------------------------------------------------------

    def load_or_train(self, data_feed, feature_engineer, symbols: list[str]) -> None:
        """
        Sync entry point — load saved model from disk if it exists;
        otherwise train from fresh MT5 data. Called at system startup.
        """
        for symbol in symbols:
            if self.load(symbol):
                logger.info("[%s] HMM loaded from disk", symbol)
                continue

            logger.info("[%s] No saved HMM — training from MT5 data", symbol)
            raw_data = data_feed.get_historical(symbol, "D1", bars=500)
            if raw_data.empty:
                logger.warning("[%s] No D1 data available — HMM not trained", symbol)
                continue

            features = feature_engineer.transform(raw_data)
            if features.empty:
                logger.warning("[%s] Feature transform empty — HMM not trained", symbol)
                continue

            self.train(symbol, feature_engineer.to_matrix(features))
            self.save(symbol)

    async def load_or_train_async(
        self, data_feed, feature_engineer, symbols: list[str]
    ) -> None:
        """
        Async entry point using the DB cache: loads the full historical D1
        window from ``ohlcv_bars`` (falling back to MT5 for missing bars)
        before training.
        """
        for symbol in symbols:
            if self.load(symbol):
                logger.info("[%s] HMM loaded from disk", symbol)
                continue

            logger.info("[%s] No saved HMM — training from DB/MT5", symbol)

            # Try DB first, fall back to MT5
            raw_data = await data_feed.get_historical_async(symbol, "D1", bars=500)
            if raw_data.empty:
                raw_data = data_feed.get_historical(symbol, "D1", bars=500)

            if raw_data.empty:
                logger.warning("[%s] No D1 data — HMM not trained", symbol)
                continue

            features = feature_engineer.transform(raw_data)
            if features.empty:
                logger.warning("[%s] Feature transform empty — HMM not trained", symbol)
                continue

            matrix = feature_engineer.to_matrix(features)
            self.train(symbol, matrix)
            self.save(symbol)

            # Persist training metadata
            await self.save_version_record(
                symbol=symbol,
                trained_data_start=str(raw_data.index[0]),
                trained_data_end=str(raw_data.index[-1]),
                val_log_likelihood=float(self._models[symbol].score(matrix)),
                hyperparameters={
                    "n_components": self.n_components,
                    "n_init": self.n_init,
                    "covariance_type": "diag",
                    "n_bars": matrix.shape[0],
                    "n_features": matrix.shape[1],
                },
            )

    def retrain(self, data_feed, feature_engineer, symbols: list[str]) -> None:
        """
        Retrain and save updated models. Called by APScheduler every 7 days.
        """
        for symbol in symbols:
            try:
                raw_data = data_feed.get_historical(symbol, "D1", bars=500)
                if raw_data.empty:
                    logger.warning("[%s] No D1 data — skipping HMM retrain", symbol)
                    continue

                features = feature_engineer.transform(raw_data)
                if features.empty:
                    continue

                # Pass raw (un-normalized) matrix — train() handles z-scoring
                # internally and stores the normalization stats for inference.
                manifest = feature_engineer.get_feature_columns(features)
                sorted_df = features[sorted(features.columns)]
                raw_matrix = np.nan_to_num(
                    sorted_df.values.astype(np.float64),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                self.train(symbol, raw_matrix, feature_manifest=manifest)
                self.save(symbol)
                logger.info("[%s] HMM retrained successfully", symbol)

            except Exception as e:
                logger.error("[%s] HMM retrain failed: %s", symbol, e, exc_info=True)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, symbol: str) -> None:
        """Persist trained model to data/models/hmm_{symbol}.pkl."""
        if symbol not in self._models:
            raise RuntimeError(f"No trained HMM for {symbol} — nothing to save")

        path = Path(str(MODEL_PATH).format(symbol=symbol))
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self._models[symbol],
            "label_map": self._state_label_maps.get(symbol, {}),
            "version": self._versions.get(symbol, 0),
            "feature_manifest": self._feature_manifests.get(symbol, []),
            "norm_means": self._norm_means.get(symbol),
            "norm_stds": self._norm_stds.get(symbol),
        }
        joblib.dump(payload, path)
        logger.info("[%s] HMM saved to %s", symbol, path)

    def load(self, symbol: str) -> bool:
        """Load model from disk. Returns False if file does not exist."""
        path = Path(str(MODEL_PATH).format(symbol=symbol))
        if not path.exists():
            logger.info("[%s] No saved HMM at %s", symbol, path)
            return False

        try:
            payload = joblib.load(path)
            self._models[symbol] = payload["model"]
            self._state_label_maps[symbol] = payload.get("label_map", {})
            self._versions[symbol] = payload.get("version", 0)
            self._feature_manifests[symbol] = payload.get("feature_manifest", [])
            if payload.get("norm_means") is not None:
                self._norm_means[symbol] = payload["norm_means"]
            if payload.get("norm_stds") is not None:
                self._norm_stds[symbol] = payload["norm_stds"]
            logger.info("[%s] HMM loaded from %s", symbol, path)
            return True
        except Exception as e:
            logger.error("[%s] HMM load failed: %s", symbol, e)
            return False

    async def save_version_record(
        self,
        symbol: str,
        trained_data_start,
        trained_data_end,
        val_log_likelihood: float,
        hyperparameters: dict,
    ) -> Optional[int]:
        """
        Write a ``model_versions`` row for a newly trained HMM.
        Returns the new version integer, or None if no data_store configured.
        """
        if self.data_store is None:
            return None
        version = await self.data_store.save_model_version({
            "model_name": self.MODEL_NAME_FMT.format(symbol=symbol),
            "trained_data_start": trained_data_start,
            "trained_data_end": trained_data_end,
            "val_loss": -float(val_log_likelihood),  # neg. log-likelihood as "loss"
            "directional_accuracy": None,            # N/A for regime classifier
            "hyperparameters": hyperparameters,
        })
        self._versions[symbol] = version
        return version

    # -------------------------------------------------------------------------
    # State ordering
    # -------------------------------------------------------------------------

    def _sort_states_by_mean_return(
        self, means: np.ndarray, log_return_col: int = 0
    ) -> dict[int, int]:
        """
        Map raw HMM state indices to regime labels (0=Crash … 4=Euphoria)
        by sorting states on their mean log-return component.

        HMM states have no inherent ordering — without this step, a fresh
        ``GaussianHMM.fit()`` will assign arbitrary integers to each state,
        so two successive retrains on the same data may produce completely
        different mappings. Consumers that index on ``regime_index`` (the
        strategy orchestrator's vol-rank table, the ``MULTIPLIERS`` dict,
        the feedback loop's stored labels) would silently break.

        This function is a pure helper: it takes the fitted ``means_``
        matrix and returns the canonicalization map. ``train()`` / ``fit()``
        must call it and store the result on ``self._state_label_maps[symbol]``
        so that ``predict()`` can apply it per-inference.

        Args:
            means: Shape ``(n_components, n_features)`` — the ``means_``
                   attribute of a fitted ``hmmlearn.GaussianHMM``.
            log_return_col: Column index of ``log_return`` in the feature
                   matrix (default 0, matching :class:`FeatureEngineer`).

        Returns:
            Dict ``{raw_state_idx → canonical_idx}`` where canonical 0 is
            the lowest mean log-return (Crash) and canonical ``n-1`` is the
            highest (Euphoria). Apply via ``canonical = label_map[raw]``.

        Example:
            >>> means = np.array([[0.02, ...],    # raw state 0: highest return
            ...                   [-0.03, ...],   # raw state 1: lowest return
            ...                   [0.001, ...]])  # raw state 2: middling
            >>> self._sort_states_by_mean_return(means)
            {1: 0, 2: 1, 0: 2}
        """
        means_arr = np.asarray(means)
        if means_arr.ndim != 2:
            raise ValueError(
                f"means must be 2D (n_components, n_features); got shape {means_arr.shape}"
            )
        n_components = means_arr.shape[0]
        if log_return_col < 0 or log_return_col >= means_arr.shape[1]:
            raise ValueError(
                f"log_return_col={log_return_col} out of range for {means_arr.shape[1]} features"
            )
        mean_returns = means_arr[:, log_return_col]
        # argsort ascending: position 0 in the sorted order is the lowest
        # mean return → canonical index 0 (Crash); position n-1 is highest
        # → canonical index n-1 (Euphoria).
        sorted_raw_indices = np.argsort(mean_returns, kind="stable")
        return {int(raw): int(canonical) for canonical, raw in enumerate(sorted_raw_indices)}
