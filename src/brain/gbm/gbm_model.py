"""
GBMPredictor — LightGBM wrapper exposing the same .predict() interface
as ``LSTMPricePredictor`` so ``signal_combiner`` can route to either
without branching past load time.

Single head type per Phase A spec §1 anchor 7: 3-class multiclass
classifier on Triple-Barrier labels mapped {-1 → 0, 0 → 1, +1 → 2}.
Inference output is the directional score ``P(class +1) - P(class -1)``
in [-1.0, 1.0], matching the LSTM softmax-head convention.

Artifact format: pickle of
    {
      "booster_str":   <booster.model_to_string()>,
      "feature_names": [str, ...],
      "num_class":     3,
      "phase":         "phase_a",
      "created_at":    iso8601 UTC,
    }
``model_to_string`` is the LightGBM-blessed serialization for
non-PMML deployment — round-trips cleanly via ``Booster(model_str=...)``.
"""
from __future__ import annotations

import datetime
import pickle
from pathlib import Path
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd


class GBMPredictor:
    """Loaded GBM model with a numpy-friendly .predict() interface."""

    def __init__(
        self,
        booster: lgb.Booster,
        feature_names: list[str],
        num_class: int = 3,
    ):
        self.booster = booster
        self.feature_names = feature_names
        self.num_class = num_class

    @staticmethod
    def save(
        booster: lgb.Booster,
        feature_names: list[str],
        path: Union[str, Path],
        num_class: int = 3,
    ) -> None:
        """Persist booster + metadata to a single pickle file."""
        payload = {
            "booster_str": booster.model_to_string(),
            "feature_names": list(feature_names),
            "num_class": int(num_class),
            "phase": "phase_a",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Union[str, Path]) -> "GBMPredictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        booster = lgb.Booster(model_str=payload["booster_str"])
        return GBMPredictor(
            booster=booster,
            feature_names=payload["feature_names"],
            num_class=payload["num_class"],
        )

    def predict(
        self,
        symbol: str,
        features: Union[dict, "pd.Series", pd.DataFrame, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute directional score(s).

        Mirrors the ``LSTMPricePredictor.predict(symbol, x) -> float``
        contract so ``signal_combiner`` can route to either primary
        predictor without branching past load time. ``symbol`` is
        accepted for interface parity and ignored — GBM artifacts are
        per-file, not per-symbol within a single instance.

        Accepts:
            * ``dict[str, float]`` — single row keyed by feature name
            * ``pandas.Series`` — single row indexed by feature name
            * ``pandas.DataFrame`` — N rows, columns ``feature_names``
            * ``numpy.ndarray`` 1D ``(n_features,)`` — single row in the
              ``feature_names`` column order
            * ``numpy.ndarray`` 2D ``(N, n_features)`` — N rows in order

        Returns:
            float when input describes a single row; ndarray of length N
            for multi-row input. Score = ``P(class +1) - P(class -1)``,
            range [-1.0, 1.0], per the TB label convention
            (-1 → idx 0, 0 → idx 1, +1 → idx 2).
        """
        if isinstance(features, dict):
            arr = np.array(
                [[features[name] for name in self.feature_names]],
                dtype=np.float64,
            )
            is_single = True
        elif isinstance(features, pd.Series):
            arr = (
                features[self.feature_names]
                .to_numpy(dtype=np.float64)
                .reshape(1, -1)
            )
            is_single = True
        elif isinstance(features, pd.DataFrame):
            arr = features[self.feature_names].to_numpy(dtype=np.float64)
            is_single = arr.shape[0] == 1
        else:
            arr = np.asarray(features, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
                is_single = True
            else:
                is_single = arr.shape[0] == 1

        # LightGBM multiclass returns shape (N, num_class).
        probs = self.booster.predict(arr)
        # Directional score = P(+1) - P(-1) = probs[:, 2] - probs[:, 0]
        scores = probs[:, 2] - probs[:, 0]
        return float(scores[0]) if is_single else scores
