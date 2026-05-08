"""
test_currency_exposure.py — Drift regression for the consolidated
currency-exposure source of truth.

Phase 2 prep (2026-04-25): five collector modules previously each owned
private ``_*_EXPOSURE`` frozensets. They were consolidated into
``src/data_pipeline/fundamental/_currency_exposure.py`` so adding a new
trading symbol is a one-file edit.

These tests fail loudly if someone:

1. Re-introduces a private frozenset literal in any collector module
   (the module's ``_*_EXPOSURE`` would no longer be the same object as
   the canonical one).
2. Adds a symbol to ``config/settings.yaml`` ``trading.symbols`` (or to
   the Phase-1C expansion list) without registering it in the canonical
   exposure sets.
3. Forgets the slash / no-slash spelling pair when adding a new symbol.
"""
from __future__ import annotations

import pytest

from src.data_pipeline.fundamental import _currency_exposure as canon
from src.data_pipeline.fundamental import cot_data
from src.data_pipeline.fundamental import macro_data
from src.data_pipeline.market import cross_asset
from src.data_pipeline.market import ecb_data
from src.data_pipeline.market import stooq_data


# ---------------------------------------------------------------------------
# 1. Identity — every module's private alias is the SAME OBJECT as canonical.
#    This is what catches a drift like "someone re-pasted a literal frozenset".
# ---------------------------------------------------------------------------

class TestExposureIdentityAcrossModules:
    """Every module's ``_*_EXPOSURE`` must ``is`` the canonical frozenset."""

    def test_macro_data_aliases_canonical(self):
        assert macro_data._EUR_EXPOSURE is canon.EUR_EXPOSURE
        assert macro_data._JPY_EXPOSURE is canon.JPY_EXPOSURE
        assert macro_data._GBP_EXPOSURE is canon.GBP_EXPOSURE
        assert macro_data._AUD_EXPOSURE is canon.AUD_EXPOSURE
        assert macro_data._CAD_EXPOSURE is canon.CAD_EXPOSURE
        assert macro_data._NZD_EXPOSURE is canon.NZD_EXPOSURE

    def test_cross_asset_aliases_canonical(self):
        assert cross_asset._EUR_EXPOSURE is canon.EUR_EXPOSURE
        assert cross_asset._JPY_EXPOSURE is canon.JPY_EXPOSURE
        assert cross_asset._GBP_EXPOSURE is canon.GBP_EXPOSURE
        assert cross_asset._AUD_EXPOSURE is canon.AUD_EXPOSURE
        assert cross_asset._NZD_EXPOSURE is canon.NZD_EXPOSURE

    def test_stooq_data_aliases_canonical(self):
        assert stooq_data._EUR_EXPOSURE is canon.EUR_EXPOSURE
        assert stooq_data._JPY_EXPOSURE is canon.JPY_EXPOSURE
        assert stooq_data._GBP_EXPOSURE is canon.GBP_EXPOSURE
        assert stooq_data._AUD_EXPOSURE is canon.AUD_EXPOSURE
        assert stooq_data._NZD_EXPOSURE is canon.NZD_EXPOSURE

    def test_ecb_data_aliases_canonical(self):
        assert ecb_data._EUR_EXPOSURE is canon.EUR_EXPOSURE

    def test_cot_data_aliases_canonical(self):
        assert cot_data._SYMBOL_CURRENCIES is canon.SYMBOL_CURRENCIES


# ---------------------------------------------------------------------------
# 2. Coverage — every live + expansion symbol must be classified correctly.
# ---------------------------------------------------------------------------

# Phase 1C expansion pairs (config-only scaffolding, not trading yet) +
# the 5 currently-live symbols. Drift here = a new pair was wired up
# somewhere without updating the canonical exposure sets.
_FX_SYMBOLS_LIVE_OR_EXPANSION: list[str] = [
    "EURUSD", "USDJPY", "USDCAD",                      # live FX
    "GBPUSD", "AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDNZD",  # expansion
]

# Symbols that are NOT FX (XAU, ETH) — must NOT appear in any per-currency
# set, since those sets gate FX-rate / yield-curve / COT-FX features.
_NON_FX_SYMBOLS: list[str] = ["XAUUSD", "ETHUSD"]


@pytest.mark.parametrize("sym", _FX_SYMBOLS_LIVE_OR_EXPANSION)
def test_every_fx_symbol_has_currencies(sym):
    """Every live/expansion FX symbol must resolve to ≥1 currency code."""
    assert canon.currencies_in(sym), (
        f"{sym} is not classified by canon.currencies_in() — "
        "missing from the exposure sets in _currency_exposure.py?"
    )


@pytest.mark.parametrize("sym", _NON_FX_SYMBOLS)
def test_non_fx_symbols_not_in_any_currency_set(sym):
    """XAU / ETH must stay out of the per-currency FX sets."""
    assert canon.currencies_in(sym) == set()
    assert sym not in canon.SYMBOL_CURRENCIES


@pytest.mark.parametrize("sym", _FX_SYMBOLS_LIVE_OR_EXPANSION)
def test_slash_and_slashless_spellings_both_present(sym):
    """If ``"EURUSD"`` is in EUR_EXPOSURE then ``"EUR/USD"`` must be too —
    both spellings appear in the codebase and we don't want callers to have
    to normalise."""
    slashless = sym
    slashed = f"{sym[:3]}/{sym[3:]}"

    for exposure in (
        canon.EUR_EXPOSURE,
        canon.JPY_EXPOSURE,
        canon.GBP_EXPOSURE,
        canon.AUD_EXPOSURE,
        canon.CAD_EXPOSURE,
        canon.NZD_EXPOSURE,
    ):
        assert (slashless in exposure) == (slashed in exposure), (
            f"{sym}: slash/no-slash spellings disagree in {exposure!r}"
        )


# ---------------------------------------------------------------------------
# 3. is_usd_axis() — used by Phase 2G coherence study and any future guard.
# ---------------------------------------------------------------------------

class TestUsdAxis:
    @pytest.mark.parametrize("sym", [
        "EURUSD", "USDJPY", "USDCAD", "GBPUSD", "AUDUSD",
        "EUR/USD", "USD/JPY", "USD/CAD", "GBP/USD", "AUD/USD",
    ])
    def test_usd_pairs_are_usd_axis(self, sym):
        assert canon.is_usd_axis(sym) is True

    @pytest.mark.parametrize("sym", [
        "EURGBP", "EURJPY", "GBPJPY", "AUDNZD",
        "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/NZD",
    ])
    def test_fx_crosses_are_not_usd_axis(self, sym):
        assert canon.is_usd_axis(sym) is False

    @pytest.mark.parametrize("sym", ["XAUUSD", "ETHUSD", "BTCUSD", "UNKNOWN"])
    def test_non_fx_symbols_are_not_usd_axis(self, sym):
        # XAU is technically quoted in USD but its drivers aren't a pure
        # DXY play — the coherence study treats it as USD-neutral.
        assert canon.is_usd_axis(sym) is False
