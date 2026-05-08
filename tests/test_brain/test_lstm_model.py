"""Tests for LSTMNetwork and LSTMPricePredictor."""

import numpy as np
import pytest
import torch

from src.brain.deep_learning.lstm_model import LSTMNetwork, LSTMPricePredictor


class TestLSTMNetwork:

    def setup_method(self):
        self.model = LSTMNetwork(input_size=20, hidden_size=32, num_layers=2, dropout=0.0)

    def test_forward_shape(self):
        """Output shape should be (batch, output_size)."""
        x = torch.randn(4, 60, 20)  # batch=4, seq_len=60, features=20
        out = self.model(x)
        assert out.shape == (4, 1)

    def test_forward_single_sample(self):
        """Works with batch_size=1."""
        x = torch.randn(1, 60, 20)
        out = self.model(x)
        assert out.shape == (1, 1)

    def test_forward_different_seq_len(self):
        """LSTM accepts variable sequence length."""
        x = torch.randn(2, 30, 20)
        out = self.model(x)
        assert out.shape == (2, 1)

    def test_output_is_finite(self):
        """Predictions should not be NaN or Inf."""
        x = torch.randn(4, 60, 20)
        out = self.model(x)
        assert torch.isfinite(out).all()


class TestLSTMPricePredictor:

    def setup_method(self):
        self.predictor = LSTMPricePredictor(device="cpu")
        # Manually register a model so we can test predict/save/load
        np.random.seed(42)
        self.input_size = 20
        model = LSTMNetwork(input_size=self.input_size, hidden_size=32, num_layers=2, dropout=0.0)
        model.eval()
        self.predictor._models["XAUUSD"] = model
        self.features = np.random.randn(60, self.input_size).astype(np.float64)

    def test_predict_returns_float(self):
        """predict() should return a Python float."""
        result = self.predictor.predict("XAUUSD", self.features)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_predict_raises_if_no_model(self):
        """predict() should raise RuntimeError for unknown symbol."""
        with pytest.raises(RuntimeError, match="No trained LSTM"):
            self.predictor.predict("UNKNOWN", self.features)

    def test_save_and_load(self, tmp_path, monkeypatch):
        """Saved model should reload and produce the same prediction."""
        monkeypatch.setattr(
            "src.brain.deep_learning.lstm_model.MODEL_PATH",
            tmp_path / "lstm_{symbol}.pt",
        )
        monkeypatch.setattr(
            "src.brain.deep_learning.lstm_model.SCALER_PATH",
            tmp_path / "lstm_scaler_{symbol}.pkl",
        )

        self.predictor.save("XAUUSD")

        predictor2 = LSTMPricePredictor(device="cpu")
        assert predictor2.load("XAUUSD")

        r1 = self.predictor.predict("XAUUSD", self.features)
        r2 = predictor2.predict("XAUUSD", self.features)
        assert abs(r1 - r2) < 1e-5

    def test_load_returns_false_when_no_file(self, tmp_path, monkeypatch):
        """load() should return False when no saved model exists."""
        monkeypatch.setattr(
            "src.brain.deep_learning.lstm_model.MODEL_PATH",
            tmp_path / "lstm_{symbol}.pt",
        )
        predictor2 = LSTMPricePredictor(device="cpu")
        assert predictor2.load("XAUUSD") is False

    def test_load_suffix_finds_bake_off_artifact(self, tmp_path, monkeypatch):
        """load(suffix='_tuned') resolves to lstm_{symbol}_tuned.pt without
        touching the unsuffixed production path. This is the contract the
        the model bake-off verdict harness depends on — it lets us evaluate
        ``_default`` vs ``_tuned`` cells while the live bot keeps loading
        the unsuffixed file."""
        # Point both default + bake-off paths at tmp_path so the unsuffixed
        # location is provably empty and the suffix-only load proves the
        # path resolution logic, not a stale fallback.
        monkeypatch.chdir(tmp_path)
        models_dir = tmp_path / "data" / "models"
        models_dir.mkdir(parents=True)

        # Save unsuffixed via the existing save() (uses MODEL_PATH/SCALER_PATH)
        monkeypatch.setattr(
            "src.brain.deep_learning.lstm_model.MODEL_PATH",
            models_dir / "lstm_{symbol}.pt",
        )
        monkeypatch.setattr(
            "src.brain.deep_learning.lstm_model.SCALER_PATH",
            models_dir / "lstm_scaler_{symbol}.pkl",
        )
        self.predictor.save("XAUUSD")

        # Move the unsuffixed artifact to the suffixed bake-off path.
        unsuffixed = models_dir / "lstm_XAUUSD.pt"
        suffixed = models_dir / "lstm_XAUUSD_tuned.pt"
        unsuffixed.rename(suffixed)
        # And make absolutely sure the unsuffixed path is gone — otherwise
        # an accidental fallthrough to the prod path could mask a bug.
        assert not unsuffixed.exists()

        # Suffix path is hard-coded to data/models/ in load(), and we
        # chdir'd to tmp_path, so this resolves to the renamed file above.
        predictor2 = LSTMPricePredictor(device="cpu")
        assert predictor2.load("XAUUSD", suffix="_tuned") is True
        # Round-trip parity: same prediction as the original predictor.
        r1 = self.predictor.predict("XAUUSD", self.features)
        r2 = predictor2.predict("XAUUSD", self.features)
        assert abs(r1 - r2) < 1e-5

    def test_load_suffix_returns_false_when_only_unsuffixed_present(
        self, tmp_path, monkeypatch,
    ):
        """suffix='_tuned' must NOT silently fall through to the production
        unsuffixed file. If the bake-off artifact is missing the verdict
        harness must fail loud, not score the prod model."""
        monkeypatch.chdir(tmp_path)
        models_dir = tmp_path / "data" / "models"
        models_dir.mkdir(parents=True)
        monkeypatch.setattr(
            "src.brain.deep_learning.lstm_model.MODEL_PATH",
            models_dir / "lstm_{symbol}.pt",
        )
        monkeypatch.setattr(
            "src.brain.deep_learning.lstm_model.SCALER_PATH",
            models_dir / "lstm_scaler_{symbol}.pkl",
        )
        self.predictor.save("XAUUSD")  # writes unsuffixed only
        assert (models_dir / "lstm_XAUUSD.pt").exists()
        assert not (models_dir / "lstm_XAUUSD_tuned.pt").exists()

        predictor2 = LSTMPricePredictor(device="cpu")
        assert predictor2.load("XAUUSD", suffix="_tuned") is False

    def test_predict_with_scaler(self, tmp_path, monkeypatch):
        """predict() should apply scaler when one is registered."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.random.randn(200, self.input_size))
        self.predictor._scalers["XAUUSD"] = scaler

        result = self.predictor.predict("XAUUSD", self.features)
        assert isinstance(result, float)
        assert np.isfinite(result)
