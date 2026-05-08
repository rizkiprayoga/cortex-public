"""
trainer.py — Deep Learning Training Pipeline

Handles the full training loop for LSTM and Transformer models:
    - Dataset creation (sliding window sequences, optional sample weights)
    - Train / validation / test split
    - Epoch loop with Adam optimizer and MSE loss (optionally weighted)
    - Early stopping on validation loss
    - Learning rate scheduling

Weighted training
-----------------
When ``sample_weights`` is passed to ``fit()``, each training sample's
squared error is multiplied by its weight before the batch mean. This is
used by the feedback loop to implement exponential decay: recent bars
get higher weights so the model adapts to recent prediction mistakes
without catastrophically forgetting older patterns.

Used by both ``LSTMPricePredictor.retrain()`` and
``scripts/train_deep_learning.py``.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.brain.deep_learning.focal_loss import FocalLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15           # Early stopping patience
    sequence_length: int = 60
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio = 1 - train_ratio - val_ratio
    # "mse" for regression (1-output head); "cross_entropy" for 3-class
    # softmax over {-1, 0, +1} Triple-Barrier labels. When cross_entropy,
    # targets are cast to long class indices via (target + 1).
    loss_type: str = "mse"
    use_focal_loss: bool = False
    focal_gamma: float = 2.0


@dataclass
class TrainingResult:
    best_val_loss: float
    epochs_trained: int
    train_losses: list[float]
    val_losses: list[float]
    directional_accuracy: float = 0.0   # test-set accuracy


class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for time-series models.

    Creates (sequence, target, weight) triples from a feature matrix:
        sequence: feature_matrix[i : i + sequence_length]  → (seq_len, n_features)
        target:   log_return at bar i + sequence_length     → scalar
        weight:   sample_weights[i + sequence_length]       → scalar (1.0 if unweighted)
    """

    def __init__(
        self,
        feature_matrix: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
        sample_weights: Optional[np.ndarray] = None,
        targets_as_class: bool = False,
    ):
        """
        Args:
            feature_matrix: Shape (n_bars, n_features)
            targets:        Shape (n_bars,) — log returns, OR {-1, 0, +1}
                            Triple-Barrier labels when ``targets_as_class``.
            sequence_length: Number of lookback bars per sample
            sample_weights: Optional shape (n_bars,) — per-bar importance weights.
                            If None, all samples are weighted equally (1.0).
            targets_as_class: When True, treats targets as integer class
                            labels and casts to torch.long. The {-1, 0, +1}
                            TB labels get shifted to {0, 1, 2} for CE loss.
        """
        self.feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
        self.targets_as_class = bool(targets_as_class)
        if self.targets_as_class:
            # Map {-1, 0, +1} → {0, 1, 2}
            shifted = np.rint(np.asarray(targets)).astype(np.int64) + 1
            shifted = np.clip(shifted, 0, 2)
            self.targets = torch.tensor(shifted, dtype=torch.long)
        else:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        self.sequence_length = sequence_length
        if sample_weights is not None:
            self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        else:
            self.sample_weights = torch.ones(len(targets), dtype=torch.float32)

    def __len__(self) -> int:
        return max(0, len(self.feature_matrix) - self.sequence_length)

    def __getitem__(self, idx: int):
        """Returns (sequence, target, weight) as torch tensors."""
        seq = self.feature_matrix[idx : idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length]
        weight = self.sample_weights[idx + self.sequence_length]
        return seq, target, weight


class ModelTrainer:
    """
    Generic trainer that works with any nn.Module returning a scalar prediction.

    Usage:
        trainer = ModelTrainer(model, config)
        result = trainer.fit(feature_matrix, targets)

        # Weighted training from FeedbackLoop:
        weights = FeedbackLoop.apply_exponential_weights(timestamps)
        result = trainer.fit(feature_matrix, targets, sample_weights=weights)
    """

    def __init__(self, model: nn.Module, config: Optional[TrainingConfig] = None):
        self.model = model
        self.config = config or TrainingConfig()
        self.device = next(model.parameters()).device

    def fit(
        self,
        feature_matrix: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        targets_as_class: bool = False,
        explicit_split: Optional[tuple[int, int]] = None,
    ) -> TrainingResult:
        """
        Run the full training loop.

        Args:
            feature_matrix: Shape (n_bars, n_features) — normalized features
            targets:        Shape (n_bars,) — log returns (MSE path) or
                            {-1, 0, +1} TB labels (cross-entropy path).
            sample_weights: Optional shape (n_bars,) — per-sample training
                            weights. When provided, each sample's squared
                            error is multiplied by its weight before the
                            batch mean. Used to implement exponential decay
                            for adaptive retraining (see FeedbackLoop).
            targets_as_class: True when this is a 3-class softmax run.
                            Must align with ``config.loss_type`` being
                            "cross_entropy".
            explicit_split: Optional ``(train_end_idx, val_end_idx)`` row
                            indices. When provided, the dataloaders are cut
                            at those exact boundaries and the proportional
                            ``train_ratio``/``val_ratio`` path is bypassed.
                            Used by the model bake-off (Task 2.2b-1) so the
                            train/val cut matches the calendar boundary
                            from the CLI flags rather than a hardcoded
                            70/15/15 ratio. The test slice ``[val_end:]``
                            may legally be empty (calendar-clipped data).

        Returns:
            TrainingResult with loss curves and best val loss.
        """
        train_loader, val_loader, test_loader = self._build_dataloaders(
            feature_matrix, targets, sample_weights,
            targets_as_class=targets_as_class,
            explicit_split=explicit_split,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(train_loader, optimizer)
            val_loss = self._validate_epoch(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d — train_loss=%.6f, val_loss=%.6f, lr=%.2e",
                    epoch + 1, self.config.epochs, train_loss, val_loss,
                    optimizer.param_groups[0]["lr"],
                )

            if patience_counter >= self.config.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Evaluate directional accuracy on test set
        directional_accuracy = self._evaluate_directional_accuracy(test_loader)

        return TrainingResult(
            best_val_loss=best_val_loss,
            epochs_trained=len(train_losses),
            train_losses=train_losses,
            val_losses=val_losses,
            directional_accuracy=directional_accuracy,
        )

    def _build_dataloaders(
        self,
        feature_matrix: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        targets_as_class: bool = False,
        explicit_split: Optional[tuple[int, int]] = None,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Chronologically split data and return (train_loader, val_loader, test_loader).

        Important: the split is temporal (no shuffling) so val/test come
        from AFTER the training period — simulates true out-of-sample testing.
        sample_weights are sliced alongside features/targets.

        When ``explicit_split=(train_end, val_end)`` is provided, those
        absolute row indices take precedence over the proportional
        ``train_ratio``/``val_ratio`` calculation. The test slice
        ``[val_end:]`` may be empty (e.g. when the caller has already
        clipped the matrix to ``[train_start, val_end_exclusive]`` so
        the test window never enters memory — Phase A invariant #14).
        """
        if explicit_split is not None:
            train_end, val_end = explicit_split
        else:
            n = len(feature_matrix)
            train_end = int(n * self.config.train_ratio)
            val_end = train_end + int(n * self.config.val_ratio)

        train_ds = TimeSeriesDataset(
            feature_matrix[:train_end], targets[:train_end],
            self.config.sequence_length,
            sample_weights[:train_end] if sample_weights is not None else None,
            targets_as_class=targets_as_class,
        )
        val_ds = TimeSeriesDataset(
            feature_matrix[train_end:val_end], targets[train_end:val_end],
            self.config.sequence_length,
            targets_as_class=targets_as_class,
        )
        test_ds = TimeSeriesDataset(
            feature_matrix[val_end:], targets[val_end:],
            self.config.sequence_length,
            targets_as_class=targets_as_class,
        )

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Run one training epoch using weighted MSE (regression) or
        weighted cross-entropy (3-class softmax). Branch on
        ``self.config.loss_type``.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        use_ce = self.config.loss_type == "cross_entropy"

        for sequences, targets, weights in loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            weights = weights.to(self.device)

            optimizer.zero_grad()
            predictions = self.model(sequences)
            if use_ce:
                loss = self.weighted_cross_entropy_loss(
                    predictions, targets, weights,
                    use_focal_loss=self.config.use_focal_loss,
                    focal_gamma=self.config.focal_gamma,
                )
            else:
                loss = self.weighted_mse_loss(predictions, targets, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate_epoch(self, loader: DataLoader) -> float:
        """
        Run one validation epoch. Returns mean loss. Unweighted by design
        so the val metric is comparable across runs.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        use_ce = self.config.loss_type == "cross_entropy"

        with torch.no_grad():
            for sequences, targets, _ in loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                predictions = self.model(sequences)
                if use_ce:
                    loss = nn.functional.cross_entropy(predictions, targets)
                else:
                    loss = nn.functional.mse_loss(predictions.squeeze(), targets)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _evaluate_directional_accuracy(self, loader: DataLoader) -> float:
        """
        Fraction of test samples where predicted direction matches actual.
        For softmax (3-class) mode: predicted direction = argmax mapped back
        to {-1, 0, +1} → sign() gives direction; skip timeout samples from
        both sides for a fairer directional metric.

        Phase A (Task 2.2b-1): when the caller passed an ``explicit_split``
        whose ``val_end == len(matrix)`` (calendar-clipped data), the test
        slice is empty and ``loader`` produces zero batches. Return
        ``float("nan")`` rather than 0.0 — NaN propagates clearly through
        MLflow as "no measurement" and is distinguishable from a
        legitimately-failed model that scored 0.0.
        """
        self.model.eval()
        correct = 0
        total = 0
        use_ce = self.config.loss_type == "cross_entropy"

        with torch.no_grad():
            for sequences, targets, _ in loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                predictions = self.model(sequences)
                if use_ce:
                    # Softmax path: compare argmax class to target class.
                    # (Stricter than sign-match — 3-way match counts as correct.)
                    pred_cls = predictions.argmax(dim=-1)
                    correct += (pred_cls == targets).sum().item()
                    total += len(targets)
                else:
                    preds = predictions.squeeze()
                    correct += ((preds > 0) == (targets > 0)).sum().item()
                    total += len(targets)

        if total == 0:
            logger.info(
                "test slice empty (calendar split) — "
                "directional_accuracy=NaN sentinel"
            )
            return float("nan")
        return correct / total

    @staticmethod
    def weighted_mse_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss for a batch.

            loss = sum(weights * (predictions - targets)²) / sum(weights)

        Uses weight-normalized denominator (not batch_size) so the scale
        is invariant to the total weight magnitude.
        """
        squared_errors = (predictions.squeeze() - targets) ** 2
        weighted_sum = (weights * squared_errors).sum()
        total_weight = weights.sum().clamp_min(1e-12)
        return weighted_sum / total_weight

    @staticmethod
    def weighted_cross_entropy_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Weighted cross-entropy for the 3-class softmax head.

            loss = sum(weights * CE(pred_i, target_i)) / sum(weights)

        ``predictions`` are raw logits (batch, 3); ``targets`` are long
        class indices in {0, 1, 2} (= {-1, 0, +1} TB labels shifted by 1).
        Sample weights come from the class-weighting path set up in
        ``scripts/train_deep_learning.py``.

        When ``use_focal_loss=True``, the per-sample CE is modulated by
        (1 - p_t)**focal_gamma where p_t = exp(-CE) is the predicted
        probability of the true class (Lin et al. 2017). This composes
        multiplicatively with the FeedbackLoop sample weights and
        addresses the EUR/JPY softmax collapse documented in
        memory/project_phase2a_pivot.md. Defaults preserve the original
        plain-CE behavior so existing callers (and ``test_trainer.py``)
        are unaffected.
        """
        if use_focal_loss:
            per_sample = FocalLoss(gamma=focal_gamma, reduction="none")(
                predictions, targets,
            )
        else:
            per_sample = nn.functional.cross_entropy(
                predictions, targets, reduction="none",
            )
        weighted_sum = (weights * per_sample).sum()
        total_weight = weights.sum().clamp_min(1e-12)
        return weighted_sum / total_weight
