"""Tests for TimeSeriesDataset and ModelTrainer."""

import numpy as np
import pytest
import torch

from src.brain.deep_learning.lstm_model import LSTMNetwork
from src.brain.deep_learning.trainer import (
    ModelTrainer,
    TimeSeriesDataset,
    TrainingConfig,
    TrainingResult,
)


class TestTimeSeriesDataset:

    def setup_method(self):
        np.random.seed(42)
        self.n_bars = 200
        self.n_features = 10
        self.seq_len = 60
        self.features = np.random.randn(self.n_bars, self.n_features).astype(np.float64)
        self.targets = np.random.randn(self.n_bars).astype(np.float64)

    def test_length(self):
        """Dataset length = n_bars - sequence_length."""
        ds = TimeSeriesDataset(self.features, self.targets, self.seq_len)
        assert len(ds) == self.n_bars - self.seq_len

    def test_getitem_shapes(self):
        """Each sample returns (seq, target, weight) with correct shapes."""
        ds = TimeSeriesDataset(self.features, self.targets, self.seq_len)
        seq, target, weight = ds[0]
        assert seq.shape == (self.seq_len, self.n_features)
        assert target.shape == ()
        assert weight.shape == ()

    def test_default_weights_are_ones(self):
        """Without sample_weights, all weights should be 1.0."""
        ds = TimeSeriesDataset(self.features, self.targets, self.seq_len)
        _, _, weight = ds[0]
        assert float(weight) == 1.0

    def test_custom_weights(self):
        """Custom sample_weights should be returned per sample."""
        weights = np.linspace(0.5, 2.0, self.n_bars)
        ds = TimeSeriesDataset(self.features, self.targets, self.seq_len, weights)
        _, _, w = ds[0]
        assert abs(float(w) - weights[self.seq_len]) < 1e-5

    def test_target_alignment(self):
        """Target at index i should correspond to bar i + seq_len."""
        ds = TimeSeriesDataset(self.features, self.targets, self.seq_len)
        _, target, _ = ds[5]
        expected = self.targets[5 + self.seq_len]
        assert abs(float(target) - expected) < 1e-5


class TestModelTrainer:

    def setup_method(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.n_bars = 500
        self.n_features = 10
        self.features = np.random.randn(self.n_bars, self.n_features).astype(np.float64)
        self.targets = np.random.randn(self.n_bars).astype(np.float64)
        self.model = LSTMNetwork(
            input_size=self.n_features, hidden_size=16,
            num_layers=1, dropout=0.0,
        )
        self.config = TrainingConfig(
            epochs=5, batch_size=32, sequence_length=20,
            patience=3, learning_rate=0.001,
        )

    def test_fit_returns_training_result(self):
        """fit() should return a TrainingResult with loss curves."""
        trainer = ModelTrainer(self.model, self.config)
        result = trainer.fit(self.features, self.targets)
        assert isinstance(result, TrainingResult)
        assert result.epochs_trained > 0
        assert len(result.train_losses) == result.epochs_trained
        assert len(result.val_losses) == result.epochs_trained
        assert result.best_val_loss >= 0

    def test_fit_with_sample_weights(self):
        """fit() should accept sample_weights without error."""
        weights = np.exp(np.linspace(-1, 0, self.n_bars))
        trainer = ModelTrainer(self.model, self.config)
        result = trainer.fit(self.features, self.targets, sample_weights=weights)
        assert isinstance(result, TrainingResult)
        assert result.epochs_trained > 0

    def test_directional_accuracy_in_range(self):
        """Directional accuracy should be between 0 and 1."""
        trainer = ModelTrainer(self.model, self.config)
        result = trainer.fit(self.features, self.targets)
        assert 0.0 <= result.directional_accuracy <= 1.0

    def test_weighted_mse_loss_unit_weights(self):
        """With unit weights, weighted MSE should equal standard MSE."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 1.9, 3.2])
        weights = torch.ones(3)
        wmse = ModelTrainer.weighted_mse_loss(preds, targets, weights)
        mse = torch.nn.functional.mse_loss(preds, targets)
        assert abs(wmse.item() - mse.item()) < 1e-5

    def test_weighted_mse_loss_zero_weight_ignored(self):
        """Samples with zero weight should not contribute to loss."""
        preds = torch.tensor([1.0, 100.0])  # second prediction is wildly off
        targets = torch.tensor([1.0, 0.0])
        weights = torch.tensor([1.0, 0.0])  # but its weight is zero
        wmse = ModelTrainer.weighted_mse_loss(preds, targets, weights)
        assert wmse.item() < 1e-5  # loss from first sample only: (1-1)^2 = 0

    def test_weighted_cross_entropy_focal_modulates_loss(self):
        """Focal modulation should down-weight a high-confidence-correct sample."""
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.5, 1.0, 0.5]])
        targets = torch.tensor([0, 1])
        weights = torch.tensor([1.0, 1.0])
        loss_plain = ModelTrainer.weighted_cross_entropy_loss(
            predictions=logits, targets=targets, weights=weights,
            use_focal_loss=False,
        )
        loss_focal = ModelTrainer.weighted_cross_entropy_loss(
            predictions=logits, targets=targets, weights=weights,
            use_focal_loss=True, focal_gamma=2.0,
        )
        assert loss_focal.item() < loss_plain.item()
        assert loss_focal.item() > 0.0

    def test_early_stopping(self):
        """Training should stop before max epochs if val loss plateaus."""
        config = TrainingConfig(
            epochs=100, batch_size=32, sequence_length=20,
            patience=2, learning_rate=0.0,  # zero lr = no improvement at all
        )
        trainer = ModelTrainer(self.model, config)
        result = trainer.fit(self.features, self.targets)
        # patience=2 means it should stop after epoch 3 (1 best + 2 no-improve)
        assert result.epochs_trained <= 5
