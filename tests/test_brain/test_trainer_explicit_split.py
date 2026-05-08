"""
Phase A Task 2.2b-1: window-aware train/val/test split for ModelTrainer.
Spec invariant #14 — the split must match the calendar boundaries from
the CLI flags, not a proportional ratio.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _toy_matrix(n: int, n_features: int = 4):
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal(size=(n, n_features)).astype(np.float32)
    targets = matrix[:, 0].copy()
    return matrix, targets


def test_explicit_split_overrides_proportional():
    """When explicit_split=(train_end, val_end) is given, _build_dataloaders
    cuts at those exact row indices regardless of train_ratio/val_ratio."""
    from src.brain.deep_learning.trainer import (
        ModelTrainer, TrainingConfig,
    )

    matrix, targets = _toy_matrix(200, n_features=4)

    # Build a trivial model so ModelTrainer can be instantiated.
    model = torch.nn.Linear(4, 1)
    config = TrainingConfig(sequence_length=10, batch_size=8, epochs=1)
    trainer = ModelTrainer(model, config)

    train_loader, val_loader, test_loader = trainer._build_dataloaders(
        matrix, targets, explicit_split=(140, 200),  # 70% / 30% / 0%
    )

    # Sequence dataset shrinks each slice by sequence_length-1, but the
    # underlying split sizes are 140 / 60 / 0.
    assert len(train_loader.dataset) == 140 - config.sequence_length
    assert len(val_loader.dataset) == 60 - config.sequence_length
    assert len(test_loader.dataset) == 0


def test_explicit_split_propagates_through_fit():
    """fit(explicit_split=(t,v)) should make the resulting model train on
    exactly the first t rows and validate on rows [t, v]."""
    from src.brain.deep_learning.lstm_model import LSTMNetwork
    from src.brain.deep_learning.trainer import (
        ModelTrainer, TrainingConfig,
    )

    torch.manual_seed(0)
    matrix, targets = _toy_matrix(150, n_features=4)
    # Use LSTMNetwork (sequence-reducing) so fit() actually runs an epoch.
    # A bare nn.Linear would emit per-position outputs and break the loss
    # shape — we just need any model that maps (batch, seq, feat) → (batch, 1).
    model = LSTMNetwork(input_size=4, hidden_size=8, num_layers=1, dropout=0.0)
    config = TrainingConfig(sequence_length=10, batch_size=8, epochs=1)
    trainer = ModelTrainer(model, config)

    result = trainer.fit(matrix, targets, explicit_split=(100, 150))

    # No raised exceptions, val_losses populated for the val window only.
    assert result.epochs_trained >= 1
    # Test slice empty → directional_accuracy is NaN sentinel
    # (distinguishable in MLflow from a legitimately-failed 0.0 score).
    assert math.isnan(result.directional_accuracy)


def test_proportional_split_still_default():
    """Backwards compat: omitting explicit_split falls back to train_ratio/
    val_ratio. All existing callers (and existing tests) rely on this."""
    from src.brain.deep_learning.trainer import (
        ModelTrainer, TrainingConfig,
    )

    matrix, targets = _toy_matrix(200, n_features=4)
    model = torch.nn.Linear(4, 1)
    config = TrainingConfig(
        sequence_length=10, batch_size=8, epochs=1,
        train_ratio=0.7, val_ratio=0.15,
    )
    trainer = ModelTrainer(model, config)

    train_loader, val_loader, test_loader = trainer._build_dataloaders(
        matrix, targets,  # no explicit_split
    )

    # 70% / 15% / 15% of 200 = 140 / 30 / 30 (minus sequence_length-1 each).
    assert len(train_loader.dataset) == 140 - config.sequence_length
    assert len(val_loader.dataset) == 30 - config.sequence_length
    assert len(test_loader.dataset) == 30 - config.sequence_length
