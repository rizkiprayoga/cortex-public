"""
lstm_model.py — LSTM Price Predictor (PyTorch)

Predicts the next-bar log return for a given symbol using a
stacked LSTM network trained on H4 bar features.

Architecture:
    Input  → LSTM (2 layers, hidden=128, dropout=0.3)
           → Linear(128 → 64) → ReLU
           → Linear(64 → 1)
    Output → predicted log return (positive = expect price up)

Input features (sequence_length=60 H4 bars):
    OHLCV, RSI, MACD, Bollinger, ATR, volume_ratio,
    log_return, realized_vol, + fundamental scores

Trained with MSE loss (optionally weighted), Adam optimizer,
early stopping on val loss. Retrained daily and whenever the
FeedbackLoop detects degraded performance.

Adaptive retraining
-------------------
Every retrain loads the full H4 history from PostgreSQL
(``ohlcv_bars`` + ``engineered_features``) and passes per-sample
weights computed by ``FeedbackLoop.apply_exponential_weights()`` to
the trainer. Recent bars get higher weights so the model adapts to
the current regime while retaining long-term patterns.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.data_pipeline.data_store import DataStore

logger = logging.getLogger(__name__)

MODEL_PATH = Path("data/models/lstm_{symbol}.pt")
SCALER_PATH = Path("data/models/lstm_scaler_{symbol}.pkl")


class LSTMNetwork(nn.Module):
    """PyTorch LSTM network definition."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, sequence_length, input_size)
        Returns:
            (batch, output_size) — predicted log return
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]    # (batch, hidden_size)
        out = self.fc1(last_hidden)         # (batch, 64)
        out = self.relu(out)
        out = self.fc2(out)                 # (batch, output_size)
        return out


class LSTMPricePredictor:
    """
    Wrapper around LSTMNetwork with training, inference, and persistence.

    Usage:
        predictor = LSTMPricePredictor(data_store=store)
        await predictor.load_or_train_async(data_feed, feature_engineer, ["XAUUSD"])
        result = await predictor.predict_and_log("XAUUSD", feature_sequence, bar_ts)
    """

    MODEL_NAME_FMT = "lstm_{symbol}"

    def __init__(
        self,
        device: Optional[str] = None,
        data_store: Optional["DataStore"] = None,
        feedback_loop=None,
    ):
        """
        Args:
            device:        "cuda" or "cpu" — auto-detected when None
            data_store:    Async DataStore for DB-based training and logging
            feedback_loop: Optional FeedbackLoop — provides sample weights
                           and error history for adaptive retraining
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data_store = data_store
        self.feedback_loop = feedback_loop
        self._models: dict[str, LSTMNetwork] = {}
        self._scalers: dict[str, object] = {}        # sklearn StandardScaler per symbol
        self._pcas: dict[str, object] = {}           # sklearn PCA per symbol (optional)
        self._versions: dict[str, int] = {}          # symbol → current model_version
        self._feature_manifests: dict[str, list[str]] = {}  # symbol → sorted column list
        # Audit H7: per-symbol lock for atomic model + scaler + PCA + manifest swap
        # during retrain. Without this, main loop can read a new model with
        # old manifest → tensor shape mismatch crash.
        import threading as _threading
        self._update_locks: dict[str, _threading.Lock] = {}
        self._locks_lock: _threading.Lock = _threading.Lock()

    def _get_update_lock(self, symbol: str):
        """Get or create the per-symbol update lock (thread-safe)."""
        with self._locks_lock:
            if symbol not in self._update_locks:
                import threading as _threading
                self._update_locks[symbol] = _threading.Lock()
            return self._update_locks[symbol]

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def predict(self, symbol: str, feature_sequence: np.ndarray) -> float:
        """
        Run inference on the latest feature window.

        Args:
            symbol:           Trading symbol
            feature_sequence: Shape (sequence_length, n_features) — H4 bars

        Returns:
            A scalar in [-1, +1]. For regression models (output_size=1) this
            is the raw predicted value. For 3-class softmax models
            (output_size=3 over {-1, 0, +1} Triple-Barrier classes) this is
            ``P(+1) − P(−1)`` so the scale stays compatible with existing
            combined-score arithmetic downstream.
        """
        # Audit H7: read model + scaler + PCA under per-symbol lock
        # so we never mix a new model with an old scaler during retrain.
        update_lock = self._get_update_lock(symbol)
        with update_lock:
            if symbol not in self._models:
                raise RuntimeError(
                    f"No trained LSTM for {symbol}. Call load_or_train() first."
                )
            model = self._models[symbol]
            scaler = self._scalers.get(symbol)
            pca = self._pcas.get(symbol)
        model.eval()

        # Scale features if a scaler is available
        seq = feature_sequence.copy().astype(np.float64)
        if scaler is not None:
            n_bars, n_feat = seq.shape
            seq = scaler.transform(seq.reshape(-1, n_feat)).reshape(n_bars, n_feat)

        # Apply PCA if trained with dimensionality reduction
        if pca is not None:
            n_bars = seq.shape[0]
            seq = pca.transform(seq.reshape(-1, seq.shape[1])).reshape(n_bars, -1)

        # Convert to tensor: (1, seq_len, n_features)
        x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            prediction = model(x)  # (1, output_size)

        # Detect architecture at inference time by output dimension.
        # - output_size=1 → regression (legacy path, scalar return)
        # - output_size=3 → softmax over {-1, 0, +1} TB classes; map to
        #   P(+1) − P(-1) so the downstream signal_combiner arithmetic
        #   (which expects a [-1, +1] directional score) works unchanged.
        output_size = int(prediction.shape[-1]) if prediction.dim() > 1 else 1
        if output_size == 3:
            probs = torch.softmax(prediction, dim=-1).squeeze(0)  # (3,)
            # Class order follows the training mapping: index 0 = -1,
            # index 1 = 0 (timeout), index 2 = +1. Directional score:
            return float((probs[2] - probs[0]).cpu().item())

        return float(prediction.squeeze().cpu().item())

    async def predict_and_log(
        self,
        symbol: str,
        feature_sequence: np.ndarray,
        bar_timestamp: str,
        confidence: float = 1.0,
    ) -> float:
        """
        Run ``predict()`` and persist the result to ``model_predictions``.

        Args:
            symbol:           Trading symbol
            feature_sequence: Latest feature window (seq_len, n_features)
            bar_timestamp:    ISO 8601 timestamp of the bar being predicted
            confidence:       Optional confidence score [0, 1]

        Returns:
            Predicted log return (same as predict()).
        """
        predicted = self.predict(symbol, feature_sequence)

        if self.data_store is not None:
            try:
                await self.data_store.save_prediction(
                    symbol=symbol,
                    bar_timestamp=bar_timestamp,
                    model_name=self.MODEL_NAME_FMT.format(symbol=symbol),
                    model_version=self._versions.get(symbol, 0),
                    prediction_type="price_return",
                    predicted_value=float(predicted),
                    confidence=float(confidence),
                )
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to persist LSTM prediction: {e}")

        return predicted

    # -------------------------------------------------------------------------
    # Training / retraining
    # -------------------------------------------------------------------------

    def load_or_train(self, data_feed, feature_engineer, symbols: list[str]) -> None:
        """Load saved model weights, or train from scratch if not found."""
        for symbol in symbols:
            if self.load(symbol):
                continue
            logger.info("[%s] No saved LSTM — training from MT5 data", symbol)
            self._train_from_feed(data_feed, feature_engineer, symbol)

    async def load_or_train_async(
        self, data_feed, feature_engineer, symbols: list[str]
    ) -> None:
        """
        Async entry point: loads features from the DB cache (falling back
        to MT5 on first run) before training.
        """
        for symbol in symbols:
            if self.load(symbol):
                continue
            # Try DB cache first
            if self.data_store is not None:
                try:
                    fv_rows = await self.data_store.get_feature_vectors_range(
                        symbol=symbol, timeframe="H4", limit=10000,
                    )
                    if fv_rows and len(fv_rows) >= 200:
                        import pandas as pd
                        df = pd.DataFrame([r.features for r in fv_rows])
                        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                        matrix = df.values.astype(np.float64)
                        self._train_on_matrix(symbol, matrix)
                        continue
                except Exception as e:
                    logger.warning("[%s] DB feature load failed, falling back to MT5: %s", symbol, e)

            logger.info("[%s] Training LSTM from MT5 data", symbol)
            self._train_from_feed(data_feed, feature_engineer, symbol)

    def retrain(self, data_feed, feature_engineer, symbols: list[str]) -> None:
        """
        Retrain with the latest data. Called by APScheduler daily AND by
        ``FeedbackLoop.check_and_retrain()`` when thresholds are breached.
        """
        for symbol in symbols:
            logger.info("[%s] LSTM retrain started", symbol)
            sample_weights = None

            # Get sample weights from feedback loop if available
            if self.feedback_loop is not None:
                try:
                    sample_weights = self.feedback_loop.apply_exponential_weights(symbol)
                except Exception as e:
                    logger.warning("[%s] Could not get sample weights: %s", symbol, e)

            self._train_from_feed(
                data_feed, feature_engineer, symbol,
                sample_weights=sample_weights,
            )

    # -------------------------------------------------------------------------
    # Internal training helpers
    # -------------------------------------------------------------------------

    def _train_from_feed(
        self, data_feed, feature_engineer, symbol: str,
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        """Fetch OHLCV from MT5, compute features, and train."""

        ohlcv = data_feed.get_historical(symbol, "H4", bars=5000)
        if ohlcv is None or len(ohlcv) < 300:
            logger.warning("[%s] Not enough H4 bars for LSTM training (%s)",
                           symbol, len(ohlcv) if ohlcv is not None else 0)
            return

        fe_result = feature_engineer.transform(ohlcv)
        matrix = feature_engineer.to_matrix(fe_result)
        self._train_on_matrix(symbol, matrix, sample_weights)

    def _train_on_matrix(
        self, symbol: str, matrix: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        feature_manifest: Optional[list[str]] = None,
        pca_components: Optional[int] = None,
        targets_override: Optional[np.ndarray] = None,
        softmax: bool = False,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        explicit_split: Optional[tuple[int, int]] = None,
        artifact_suffix: str = "",
        hidden_size_override: Optional[int] = None,
        num_layers_override: Optional[int] = None,
        dropout_override: Optional[float] = None,
        learning_rate_override: Optional[float] = None,
        batch_size_override: Optional[int] = None,
    ) -> "TrainingResult":
        """Scale features, optionally apply PCA, build model, train, save.

        If ``targets_override`` is provided, it is used as the training target
        instead of the default column-0 log_return. This enables Phase B.3
        Triple Barrier label training where targets are {-1, 0, +1}.

        When ``softmax=True`` (Phase 18), the LSTM is built with
        ``output_size=3`` and trained with cross-entropy loss over mapped
        class labels {-1, 0, +1} → {0, 1, 2}. Inference side handles the
        output-head detection automatically via ``predict()``.

        Returns the ``TrainingResult`` from ``ModelTrainer.fit()`` so callers
        (e.g. the T-8 MLflow instrumentation in ``train_deep_learning.py``)
        can log metrics. Result carries ``best_val_loss``, ``epochs_trained``,
        ``train_losses``, ``val_losses``, and ``directional_accuracy``.

        Task 2.2b-2b additions (the model bake-off):
        - ``artifact_suffix`` — when non-empty, the saved files become
          ``lstm_{symbol}{suffix}.pt`` / ``lstm_scaler_{symbol}{suffix}.pkl``
          / ``lstm_{symbol}{suffix}.pca.pkl`` and the live ``self._models``
          registry is NOT updated (tune-mode artifacts must not enter the
          production predictor). With the default empty suffix the legacy
          behavior is unchanged.
        - ``hidden_size_override`` / ``num_layers_override`` /
          ``dropout_override`` — forwarded to ``LSTMNetwork.__init__``
          when set, otherwise the network falls back to its constructor
          defaults (128 / 2 / 0.3).
        - ``learning_rate_override`` / ``batch_size_override`` — applied
          to the ``TrainingConfig`` after construction.
        """
        from sklearn.preprocessing import StandardScaler
        from src.brain.deep_learning.trainer import (
            ModelTrainer, TrainingConfig,
        )

        # Targets: Triple Barrier labels if provided, otherwise log_return (col 0)
        if targets_override is not None:
            if len(targets_override) != len(matrix):
                raise ValueError(
                    f"targets_override length {len(targets_override)} does not "
                    f"match feature matrix length {len(matrix)}"
                )
            targets = np.asarray(targets_override, dtype=np.float64).copy()
            logger.info("[%s] Using Triple Barrier targets: +1=%d, -1=%d, 0=%d",
                        symbol,
                        int((targets > 0).sum()),
                        int((targets < 0).sum()),
                        int((targets == 0).sum()))
        else:
            targets = matrix[:, 0].copy()

        # Build ALL artifacts as LOCAL variables first (don't touch self.*)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)

        pca = None
        if pca_components is not None and pca_components < scaled.shape[1]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=pca_components)
            scaled = pca.fit_transform(scaled)
            logger.info("[%s] PCA: %d features → %d components (%.1f%% variance)",
                        symbol, matrix.shape[1], pca_components,
                        sum(pca.explained_variance_ratio_) * 100)

        input_size = scaled.shape[1]
        output_size = 3 if softmax else 1
        # Each override falls through to LSTMNetwork's constructor default
        # when None — keeps non-tune callers using the original defaults.
        net_kwargs = {"input_size": input_size, "output_size": output_size}
        if hidden_size_override is not None:
            net_kwargs["hidden_size"] = int(hidden_size_override)
        if num_layers_override is not None:
            net_kwargs["num_layers"] = int(num_layers_override)
        if dropout_override is not None:
            net_kwargs["dropout"] = float(dropout_override)
        model = LSTMNetwork(**net_kwargs).to(self.device)

        config = TrainingConfig()
        if learning_rate_override is not None:
            config.learning_rate = float(learning_rate_override)
        if batch_size_override is not None:
            config.batch_size = int(batch_size_override)
        if softmax:
            config.loss_type = "cross_entropy"
            config.use_focal_loss = bool(use_focal_loss)
            config.focal_gamma = float(focal_gamma)
            focal_status = (
                f"on(gamma={config.focal_gamma})"
                if config.use_focal_loss else "off"
            )
            logger.info(
                "[%s] Softmax head enabled: output_size=3, loss=cross_entropy, focal_loss=%s",
                symbol, focal_status,
            )
        trainer = ModelTrainer(model, config)
        result = trainer.fit(
            scaled, targets,
            sample_weights=sample_weights,
            targets_as_class=softmax,
            explicit_split=explicit_split,
        )

        # Audit H7: atomic swap of model + scaler + pca + manifest
        # under per-symbol lock. Main loop reads these via the same lock
        # so it never sees a partial state (new model + old manifest).
        #
        # Tune-mode artifacts (artifact_suffix != "") must NOT enter the
        # live self._models registry — only the production (unsuffixed)
        # path enters the live predictor. We still need access to
        # model/scaler/pca for the suffixed save below, so build the save
        # payload from the local variables instead.
        if artifact_suffix == "":
            update_lock = self._get_update_lock(symbol)
            with update_lock:
                self._models[symbol] = model
                self._scalers[symbol] = scaler
                self._pcas[symbol] = pca
                if feature_manifest is not None:
                    self._feature_manifests[symbol] = feature_manifest
            self.save(symbol)
        else:
            self._save_suffixed(
                symbol=symbol,
                model=model,
                scaler=scaler,
                pca=pca,
                feature_manifest=feature_manifest or [],
                suffix=artifact_suffix,
            )

        logger.info(
            "[%s] LSTM trained: %d bars, %d features, val_loss=%.6f, "
            "directional_acc=%.2f%%, epochs=%d",
            symbol, len(matrix), input_size,
            result.best_val_loss, result.directional_accuracy * 100,
            result.epochs_trained,
        )

        return result

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _save_suffixed(
        self, *,
        symbol: str,
        model: "LSTMNetwork",
        scaler,
        pca,
        feature_manifest: list[str],
        suffix: str,
    ) -> None:
        """Save a (model, scaler, pca) triple under a suffixed filename
        without touching ``self._models`` / ``self._scalers`` / ``self._pcas``.

        Used by Task 2.2b-2b's tune-mode and the bake-off artifacts
        (``lstm_{symbol}_default.pt`` / ``_tuned.pt`` / ``_trial_N.pt``).
        Mirrors ``save()`` but takes the artifacts as arguments so it
        doesn't read from the live registry — keeping bake-off runs
        independent of whatever the production model is at the moment.
        """
        model_path = Path(str(MODEL_PATH).format(symbol=symbol + suffix))
        scaler_path = Path(str(SCALER_PATH).format(symbol=symbol + suffix))
        model_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": model.state_dict(),
            "feature_manifest": list(feature_manifest),
        }
        torch.save(payload, model_path)
        logger.info(
            "[%s] LSTM weights saved to %s (suffixed bake-off artifact)",
            symbol, model_path,
        )

        if scaler is not None:
            import joblib
            joblib.dump(scaler, scaler_path)
            logger.info("[%s] LSTM scaler saved to %s", symbol, scaler_path)

        if pca is not None:
            import joblib
            pca_path = model_path.with_suffix(".pca.pkl")
            joblib.dump(pca, pca_path)
            logger.info("[%s] LSTM PCA saved to %s", symbol, pca_path)

    def save(self, symbol: str) -> None:
        """Save model weights and feature scaler to data/models/."""
        if symbol not in self._models:
            raise RuntimeError(f"No trained LSTM for {symbol}")

        model_path = Path(str(MODEL_PATH).format(symbol=symbol))
        scaler_path = Path(str(SCALER_PATH).format(symbol=symbol))
        model_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": self._models[symbol].state_dict(),
            "feature_manifest": self._feature_manifests.get(symbol, []),
        }
        torch.save(payload, model_path)
        logger.info("[%s] LSTM weights saved to %s", symbol, model_path)

        if symbol in self._scalers and self._scalers[symbol] is not None:
            import joblib
            joblib.dump(self._scalers[symbol], scaler_path)
            logger.info("[%s] LSTM scaler saved to %s", symbol, scaler_path)

        if symbol in self._pcas and self._pcas[symbol] is not None:
            import joblib
            pca_path = Path(str(MODEL_PATH).format(symbol=symbol)).with_suffix(".pca.pkl")
            joblib.dump(self._pcas[symbol], pca_path)
            logger.info("[%s] LSTM PCA saved to %s", symbol, pca_path)

    def load(self, symbol: str, suffix: str = "") -> bool:
        """Load weights and scaler. Returns False if files not found.

        Audit H7/C (HIGH-C): the atomic swap of model+scaler+pca+manifest
        must happen under the per-symbol update lock so ``predict()`` (which
        reads those four fields under the same lock) never observes a
        partially-swapped state during a monthly retrain reload.

        ``suffix`` (default ``""`` for backward compat with the live bot)
        addresses the suffixed bake-off artifacts saved by ``train(...)``
        with ``artifact_suffix=...``. ``"_default"`` / ``"_tuned"`` resolve to
        ``lstm_{symbol}_default.pt`` / ``lstm_{symbol}_tuned.pt`` so the
        the model bake-off verdict harness can evaluate either variant
        without disturbing the production unsuffixed file.
        """
        if suffix:
            model_path = Path(
                f"data/models/lstm_{symbol}{suffix}.pt"
            )
            scaler_path = Path(
                f"data/models/lstm_scaler_{symbol}{suffix}.pkl"
            )
        else:
            model_path = Path(str(MODEL_PATH).format(symbol=symbol))
            scaler_path = Path(str(SCALER_PATH).format(symbol=symbol))

        if not model_path.exists():
            logger.debug("[%s] No saved LSTM at %s", symbol, model_path)
            return False

        raw = torch.load(model_path, map_location=self.device, weights_only=False)

        # Support both formats: dict with manifest or plain state_dict
        if isinstance(raw, dict) and "state_dict" in raw:
            state_dict = raw["state_dict"]
            manifest = raw.get("feature_manifest", [])
        else:
            state_dict = raw
            manifest = []

        # Infer architecture from saved weights
        # lstm.weight_ih_l0 shape: (4*hidden_size, input_size)
        input_size = state_dict["lstm.weight_ih_l0"].shape[1]
        hidden_size = state_dict["lstm.weight_ih_l0"].shape[0] // 4
        num_layers = sum(1 for k in state_dict if k.startswith("lstm.weight_ih_l"))
        # Infer dropout from num_layers (only applied between layers)
        dropout = 0.3 if num_layers > 1 else 0.0
        # fc2.weight shape is (output_size, 64). Row count tells us whether
        # this is a 1-output regression model (legacy) or a 3-output softmax
        # classifier (Phase 18 class-balanced variant).
        output_size = 1
        if "fc2.weight" in state_dict:
            output_size = int(state_dict["fc2.weight"].shape[0])
        model = LSTMNetwork(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            output_size=output_size,
        ).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()

        # Build the new scaler/pca LOCALLY before acquiring the lock so
        # filesystem I/O happens without blocking the main loop.
        new_scaler = None
        if scaler_path.exists():
            import joblib
            new_scaler = joblib.load(scaler_path)

        new_pca = None
        pca_path = model_path.with_suffix(".pca.pkl")
        if pca_path.exists():
            import joblib
            new_pca = joblib.load(pca_path)
            logger.info("[%s] LSTM PCA loaded from %s", symbol, pca_path)

        # Atomic swap under the per-symbol lock (same lock used by predict()
        # and _train_on_matrix so readers never see a partial state).
        update_lock = self._get_update_lock(symbol)
        with update_lock:
            self._models[symbol] = model
            self._feature_manifests[symbol] = manifest
            if new_scaler is not None:
                self._scalers[symbol] = new_scaler
            # Always set pca (even to None) so a non-PCA model doesn't
            # inherit a stale PCA from a prior load of a different model.
            self._pcas[symbol] = new_pca

        logger.info("[%s] LSTM loaded from %s (input_size=%d)", symbol, model_path, input_size)
        return True

    async def save_version_record(
        self,
        symbol: str,
        trained_data_start,
        trained_data_end,
        val_loss: float,
        directional_accuracy: float,
        hyperparameters: dict,
    ) -> Optional[int]:
        """
        Write a ``model_versions`` row for a newly trained LSTM.
        Returns the new version integer, or None if no data_store configured.
        """
        if self.data_store is None:
            return None
        version = await self.data_store.save_model_version({
            "model_name": self.MODEL_NAME_FMT.format(symbol=symbol),
            "trained_data_start": trained_data_start,
            "trained_data_end": trained_data_end,
            "val_loss": float(val_loss),
            "directional_accuracy": float(directional_accuracy),
            "hyperparameters": hyperparameters,
        })
        self._versions[symbol] = version
        return version
