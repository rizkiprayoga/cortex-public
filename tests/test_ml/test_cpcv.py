"""Unit tests for src/ml/cpcv.py — Combinatorial Purged Cross-Validation logic.

These tests cover the combinatorial + purging + embargoing algebra only,
not the heavy train-and-backtest orchestration (that lives in
``scripts/backtest_cpcv.py`` and runs as an integration job).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


def test_enumerate_combos_returns_expected_count():
    """N=6, k=2 → C(6, 2) = 15 combinations (López de Prado standard)."""
    from src.ml.cpcv import enumerate_cpcv_combos

    combos = enumerate_cpcv_combos(n_groups=6, k_test=2)
    assert len(combos) == math.comb(6, 2)   # 15


def test_enumerate_combos_each_is_sorted_tuple_of_group_indices():
    """Every combo is a sorted tuple of k_test group indices in [0, n_groups)."""
    from src.ml.cpcv import enumerate_cpcv_combos

    combos = enumerate_cpcv_combos(n_groups=5, k_test=2)
    for combo in combos:
        assert len(combo) == 2
        assert list(combo) == sorted(combo)
        assert 0 <= combo[0] < combo[1] < 5


def test_enumerate_combos_no_duplicates():
    from src.ml.cpcv import enumerate_cpcv_combos
    combos = enumerate_cpcv_combos(n_groups=6, k_test=2)
    assert len(set(combos)) == len(combos)


def test_enumerate_combos_rejects_bad_params():
    from src.ml.cpcv import enumerate_cpcv_combos
    with pytest.raises(ValueError):
        enumerate_cpcv_combos(n_groups=0, k_test=1)
    with pytest.raises(ValueError):
        enumerate_cpcv_combos(n_groups=3, k_test=0)
    with pytest.raises(ValueError):
        enumerate_cpcv_combos(n_groups=3, k_test=3)   # k must be < n


def test_split_index_into_groups_equal_chunks():
    """Time index is split into N contiguous chronological groups."""
    from src.ml.cpcv import split_index_into_groups

    # 100 bars → 5 groups of 20 each
    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    groups = split_index_into_groups(idx, n_groups=5)

    assert len(groups) == 5
    # Every bar is in exactly one group
    all_ids = sorted([i for g in groups for i in g])
    assert all_ids == list(range(100))
    # Groups are contiguous and chronological
    for i, g in enumerate(groups):
        assert list(g) == sorted(g)
        if i > 0:
            assert max(groups[i - 1]) < min(g)


def test_split_index_handles_non_divisible_lengths():
    from src.ml.cpcv import split_index_into_groups
    # 103 bars / 4 groups: np.array_split produces 26, 26, 26, 25
    idx = pd.date_range("2024-01-01", periods=103, freq="h")
    groups = split_index_into_groups(idx, n_groups=4)
    sizes = [len(g) for g in groups]
    assert sum(sizes) == 103
    assert max(sizes) - min(sizes) <= 1   # as balanced as possible


def test_build_fold_masks_test_is_the_combo_groups():
    """test_mask selects rows whose group is in the combo's group list."""
    from src.ml.cpcv import build_fold_masks, split_index_into_groups

    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    groups = split_index_into_groups(idx, n_groups=5)
    # combo = (1, 3): test is groups 1 and 3
    train_mask, test_mask = build_fold_masks(
        n_rows=100, groups=groups, test_group_ids=(1, 3),
        purge_bars=0, embargo_bars=0,
    )
    # Test mask must exactly equal rows of groups 1 + 3
    expected_test = set(groups[1]) | set(groups[3])
    assert set(np.where(test_mask)[0]) == expected_test
    # Train + test are disjoint
    assert not np.any(train_mask & test_mask)


def test_build_fold_masks_purge_drops_bars_before_test():
    """Purge removes training bars within ``purge_bars`` BEFORE any test
    group, to prevent label-horizon leakage. Labels look forward, so the
    pre-test zone is where leakage happens."""
    from src.ml.cpcv import build_fold_masks, split_index_into_groups

    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    groups = split_index_into_groups(idx, n_groups=5)   # 20 bars each
    # Test = group 2 (rows 40..59). Purge 5 bars before.
    train_mask, test_mask = build_fold_masks(
        n_rows=100, groups=groups, test_group_ids=(2,),
        purge_bars=5, embargo_bars=0,
    )
    # Rows 35..39 should be excluded from train (purged) AND from test
    for i in range(35, 40):
        assert not train_mask[i], f"row {i} should be purged"
        assert not test_mask[i], f"row {i} should not be in test"


def test_build_fold_masks_embargo_drops_bars_after_test():
    """Embargo removes training bars within ``embargo_bars`` AFTER the test
    group ends, to prevent auto-correlation-based leakage."""
    from src.ml.cpcv import build_fold_masks, split_index_into_groups

    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    groups = split_index_into_groups(idx, n_groups=5)   # 20 bars each
    # Test = group 2 (rows 40..59). Embargo 5 bars after.
    train_mask, test_mask = build_fold_masks(
        n_rows=100, groups=groups, test_group_ids=(2,),
        purge_bars=0, embargo_bars=5,
    )
    # Rows 60..64 should be excluded from train (embargoed)
    for i in range(60, 65):
        assert not train_mask[i]
        assert not test_mask[i]
    # Row 65 is back in train
    assert train_mask[65]


def test_build_fold_masks_two_test_groups_handles_both_zones():
    """With multiple test groups, purge/embargo applies to each."""
    from src.ml.cpcv import build_fold_masks, split_index_into_groups

    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    groups = split_index_into_groups(idx, n_groups=5)
    # Test = groups 1 and 3 (rows 20..39 and 60..79)
    train_mask, test_mask = build_fold_masks(
        n_rows=100, groups=groups, test_group_ids=(1, 3),
        purge_bars=3, embargo_bars=3,
    )
    # Purge zone before group 1: rows 17..19
    # Embargo after group 1: rows 40..42
    # Purge before group 3: rows 57..59
    # Embargo after group 3: rows 80..82
    for i in (17, 18, 19, 40, 41, 42, 57, 58, 59, 80, 81, 82):
        assert not train_mask[i], f"row {i} should be purged/embargoed"


def test_aggregate_fold_metrics_returns_mean_std_across_folds():
    """Given per-fold metric dicts, aggregate returns mean + std per key."""
    from src.ml.cpcv import aggregate_fold_metrics

    fold_results = [
        {"profit_factor": 2.0, "max_drawdown_pct": 3.0, "total_trades": 20},
        {"profit_factor": 3.0, "max_drawdown_pct": 4.0, "total_trades": 25},
        {"profit_factor": 2.5, "max_drawdown_pct": 3.5, "total_trades": 22},
    ]
    agg = aggregate_fold_metrics(fold_results)
    assert agg["profit_factor"]["mean"] == pytest.approx(2.5)
    # std should be non-zero
    assert agg["profit_factor"]["std"] > 0
    assert agg["max_drawdown_pct"]["mean"] == pytest.approx(3.5)
    assert agg["total_trades"]["mean"] == pytest.approx(22.333, rel=1e-3)


def test_aggregate_fold_metrics_empty_input_returns_empty_dict():
    from src.ml.cpcv import aggregate_fold_metrics
    assert aggregate_fold_metrics([]) == {}
