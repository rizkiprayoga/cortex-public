"""
Phase A revert regression: the 4 historical-reader joins
(_join_macro_history, _join_yield_history, _join_curve_history,
_join_cot_history) MUST NOT be invoked from the LSTM input path
(transform_multi_timeframe_with_externals_async).

The helper methods themselves stay defined on FeatureEngineer because the
meta-labeler (Sprint 4) will reuse them — only the call sites inside
_inject_externals_async are removed. Spec §1 anchor 7.

Pre-revert: with a wired data_store, _inject_externals_async called all 4
helpers — this test would fail with "expected 0 calls, got 1" on each spy.
Post-revert: zero calls expected, test passes.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pandas as pd


def test_externals_async_does_not_call_historical_readers():
    """
    With data_store wired (non-None), transform_multi_timeframe_with_
    externals_async must not invoke any of the 4 feature_store-backed
    historical readers. Cross-asset and calendar are not asserted here
    (those legitimately stay in the LSTM input path per spec §1 anchor 7).
    """
    from src.data_pipeline.feature_engineering import FeatureEngineer

    # Sentinel data_store — non-None so the externals-async function takes
    # the _inject_externals_async branch. The helpers themselves are
    # patched to spies, so this object is never actually read from.
    data_store_sentinel = object()
    engineer = FeatureEngineer(data_store=data_store_sentinel)

    # Synthetic OHLCV with a real random walk so tech features (rolling
    # means, ATR, etc.) survive warmup-row drops. 800 H4 bars ≈ 4 months,
    # plenty for D1/W1 indicators.
    import numpy as np
    rng = np.random.default_rng(42)

    def _walk(n: int, freq: str, start: str = "2024-01-01") -> pd.DataFrame:
        idx = pd.date_range(start, periods=n, freq=freq)
        rets = rng.normal(0, 0.001, size=n)
        close = 1.10 * np.exp(np.cumsum(rets))
        spread = np.abs(rng.normal(0, 0.0005, size=n))
        return pd.DataFrame(
            {
                "open": close,
                "high": close + spread,
                "low": close - spread,
                "close": close,
                "volume": rng.integers(50, 500, size=n).astype(float),
            },
            index=idx,
        )

    ohlcv_by_tf = {
        "H4": _walk(800, "4h"),
        "H1": _walk(3200, "1h"),
        "D1": _walk(400, "1D"),
        "W1": _walk(80, "1W"),
    }

    # Patch the 4 historical-reader helpers as AsyncMock spies. The originals
    # mutate tech_df in place and return None, so the spies do the same.
    # Also stub CrossAssetFetcher.get_historical_cross_asset_features so the
    # test is hermetic (otherwise the real yfinance HTTP call fires).
    target = "src.data_pipeline.feature_engineering.FeatureEngineer"
    cross = "src.data_pipeline.market.cross_asset.CrossAssetFetcher"
    with (
        patch(f"{target}._join_macro_history", new_callable=AsyncMock) as macro,
        patch(f"{target}._join_yield_history", new_callable=AsyncMock) as yields,
        patch(f"{target}._join_curve_history", new_callable=AsyncMock) as curve,
        patch(f"{target}._join_cot_history", new_callable=AsyncMock) as cot,
        patch(
            f"{cross}.get_historical_cross_asset_features",
            return_value=pd.DataFrame(),
        ),
    ):
        spies = {
            "_join_macro_history": macro,
            "_join_yield_history": yields,
            "_join_curve_history": curve,
            "_join_cot_history": cot,
        }
        asyncio.run(
            engineer.transform_multi_timeframe_with_externals_async(
                ohlcv_by_tf, symbol="EURUSD", primary_tf="H4",
            )
        )

    for name, spy in spies.items():
        assert spy.await_count == 0, (
            f"Phase A revert incomplete: {name} was awaited "
            f"{spy.await_count} time(s); expected 0. The 4 historical "
            f"readers must not be invoked from the LSTM input path."
        )


def test_historical_reader_helpers_still_defined():
    """
    The helper methods themselves stay defined because the meta-labeler
    (Sprint 4) consumes them via read_feature_store_safe. Spec §1 anchor 7.
    """
    from src.data_pipeline.feature_engineering import FeatureEngineer

    for name in (
        "_join_macro_history",
        "_join_yield_history",
        "_join_curve_history",
        "_join_cot_history",
    ):
        assert hasattr(FeatureEngineer, name), (
            f"FeatureEngineer.{name} was removed — Sprint 4 meta-labeler "
            f"depends on it. Only the call sites should be removed."
        )
