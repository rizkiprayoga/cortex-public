"""
Combinatorial Purged Cross-Validation algebra (A-6).

Implements the combo/purge/embargo logic from López de Prado,
*Advances in Financial Machine Learning*, chapter 7. The heavy
train-and-backtest orchestration lives in ``scripts/backtest_cpcv.py``.

Purge and embargo guard against two leak channels in time-series ML:

- **Purge**: labels look forward (Triple Barrier needs the NEXT N bars
  to determine TP/SL). A training bar whose label horizon overlaps a
  test bar would leak test-set information into training. Purge drops
  training bars within ``purge_bars`` before any test group.

- **Embargo**: even after the test window ends, auto-correlated features
  (e.g. rolling means) carry residual information for a few bars. Embargo
  drops training bars within ``embargo_bars`` after any test group.
"""
from __future__ import annotations

import itertools
import math
import statistics
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def enumerate_cpcv_combos(n_groups: int, k_test: int) -> list[tuple[int, ...]]:
    """All C(n_groups, k_test) sorted combinations of group indices.

    Each returned tuple is the set of group indices that will serve as
    the test set for one fold. Their complement forms the training set
    (before purge/embargo).
    """
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if k_test <= 0 or k_test >= n_groups:
        raise ValueError("k_test must satisfy 0 < k_test < n_groups")
    return [tuple(c) for c in itertools.combinations(range(n_groups), k_test)]


def split_index_into_groups(
    index: pd.Index | Sequence, n_groups: int,
) -> list[list[int]]:
    """Partition row indices into ``n_groups`` contiguous chronological
    chunks. Returns a list of lists of row positions (0-based ints).

    Uses ``np.array_split`` so groups differ by at most one element.
    """
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    positions = np.arange(len(index))
    chunks = np.array_split(positions, n_groups)
    return [list(map(int, chunk)) for chunk in chunks]


def build_fold_masks(
    n_rows: int,
    groups: list[list[int]],
    test_group_ids: Iterable[int],
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (train_mask, test_mask) boolean arrays of length ``n_rows``.

    - ``test_mask[i] = True`` iff row ``i`` belongs to any test group.
    - ``train_mask[i] = True`` iff row is NOT a test row AND NOT in the
      purge zone (``purge_bars`` rows before each test group's start)
      AND NOT in the embargo zone (``embargo_bars`` rows after each
      test group's end).
    """
    test_mask = np.zeros(n_rows, dtype=bool)
    excluded = np.zeros(n_rows, dtype=bool)   # test ∪ purge ∪ embargo

    for gid in test_group_ids:
        rows = groups[gid]
        if not rows:
            continue
        test_mask[rows] = True
        excluded[rows] = True
        start = min(rows)
        end = max(rows)
        if purge_bars > 0:
            purge_start = max(0, start - purge_bars)
            excluded[purge_start:start] = True
        if embargo_bars > 0:
            embargo_end = min(n_rows, end + 1 + embargo_bars)
            excluded[end + 1:embargo_end] = True

    train_mask = np.ones(n_rows, dtype=bool) & ~excluded
    return train_mask, test_mask


def aggregate_fold_metrics(fold_results: list[dict]) -> dict[str, dict[str, float]]:
    """Compute mean + std across folds for every numeric metric present.

    ``fold_results`` is a list of dicts, one per fold. All non-numeric
    values are ignored. Returns ``{metric: {"mean": ..., "std": ..., "n_folds": ...}}``.
    """
    if not fold_results:
        return {}
    keys: set[str] = set()
    for r in fold_results:
        for k, v in r.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                keys.add(k)

    out: dict[str, dict[str, float]] = {}
    for k in sorted(keys):
        vals = [float(r[k]) for r in fold_results
                if k in r and isinstance(r[k], (int, float))
                and math.isfinite(float(r[k]))]
        if not vals:
            continue
        out[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
            "n_folds": len(vals),
        }
    return out
