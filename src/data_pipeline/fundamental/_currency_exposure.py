"""
_currency_exposure.py — Single source of truth for per-currency symbol membership.

Maps each currency code (EUR/JPY/GBP/AUD/CAD/NZD) to the trading symbols whose
P&L is materially exposed to that currency. Used by every collector module that
emits currency-conditional features so a symbol's feature stack stays consistent
across macro / cross-asset / yield / rate-curve / COT pipelines.

Both spellings ("EURUSD" and "EUR/USD") appear in the project — the live MT5
broker side uses the slashless form, downstream pandas / yfinance code often
uses the slash form. Sets carry both so membership checks don't depend on
upstream string normalisation.

USD intentionally has no exposure set: every live FX symbol crosses USD except
the four EUR/GBP/JPY crosses (EURGBP, EURJPY, GBPJPY, AUDNZD). Use
``is_usd_axis(symbol)`` for the "this trade is on the USD axis" question
rather than reverse-engineering it from the per-currency sets.

Maintenance rule: any new pair added to ``config/settings.yaml`` must be added
to **this** module — nowhere else. The drift regression test in
``test_currency_exposure.py`` enforces that all collector modules import these
exact frozenset objects.
"""
from __future__ import annotations

# Per-currency exposure — slashless and slashed spellings both included.
EUR_EXPOSURE: frozenset[str] = frozenset({
    "EURUSD", "EUR/USD",
    "EURGBP", "EUR/GBP",
    "EURJPY", "EUR/JPY",
    "EURCHF", "EUR/CHF",
    "EURAUD", "EUR/AUD",
})

JPY_EXPOSURE: frozenset[str] = frozenset({
    "USDJPY", "USD/JPY",
    "EURJPY", "EUR/JPY",
    "GBPJPY", "GBP/JPY",
    "AUDJPY", "AUD/JPY",
    "NZDJPY", "NZD/JPY",
    "CADJPY", "CAD/JPY",
    "CHFJPY", "CHF/JPY",
})

GBP_EXPOSURE: frozenset[str] = frozenset({
    "GBPUSD", "GBP/USD",
    "EURGBP", "EUR/GBP",
    "GBPJPY", "GBP/JPY",
    "GBPCHF", "GBP/CHF",
    "GBPAUD", "GBP/AUD",
})

AUD_EXPOSURE: frozenset[str] = frozenset({
    "AUDUSD", "AUD/USD",
    "AUDNZD", "AUD/NZD",
    "EURAUD", "EUR/AUD",
    "AUDJPY", "AUD/JPY",
    "GBPAUD", "GBP/AUD",
})

CAD_EXPOSURE: frozenset[str] = frozenset({
    "USDCAD", "USD/CAD",
    "CADJPY", "CAD/JPY",
})

NZD_EXPOSURE: frozenset[str] = frozenset({
    "AUDNZD", "AUD/NZD",
    "NZDUSD", "NZD/USD",
    "NZDJPY", "NZD/JPY",
})

CHF_EXPOSURE: frozenset[str] = frozenset({
    "USDCHF", "USD/CHF",
    "EURCHF", "EUR/CHF",
    "CHFJPY", "CHF/JPY",
    "GBPCHF", "GBP/CHF",
})

# Inverse mapping — symbol → tuple of currency codes (in symbol-order, i.e.
# base currency first then quote currency). Used by COT data routing where
# we need to fetch positioning for both legs of a cross. XAUUSD/ETHUSD have
# no FX-currency interpretation and are intentionally absent.
SYMBOL_CURRENCIES: dict[str, tuple[str, ...]] = {
    "EURUSD":  ("EUR",),
    "EUR/USD": ("EUR",),
    "USDJPY":  ("JPY",),
    "USD/JPY": ("JPY",),
    "USDCAD":  ("CAD",),
    "USD/CAD": ("CAD",),
    "USDCHF":  ("CHF",),
    "USD/CHF": ("CHF",),
    "GBPUSD":  ("GBP",),
    "GBP/USD": ("GBP",),
    "AUDUSD":  ("AUD",),
    "AUD/USD": ("AUD",),
    "NZDUSD":  ("NZD",),
    "NZD/USD": ("NZD",),
    "EURGBP":  ("EUR", "GBP"),
    "EUR/GBP": ("EUR", "GBP"),
    "EURJPY":  ("EUR", "JPY"),
    "EUR/JPY": ("EUR", "JPY"),
    "EURCHF":  ("EUR", "CHF"),
    "EUR/CHF": ("EUR", "CHF"),
    "EURAUD":  ("EUR", "AUD"),
    "EUR/AUD": ("EUR", "AUD"),
    "GBPJPY":  ("GBP", "JPY"),
    "GBP/JPY": ("GBP", "JPY"),
    "GBPCHF":  ("GBP", "CHF"),
    "GBP/CHF": ("GBP", "CHF"),
    "GBPAUD":  ("GBP", "AUD"),
    "GBP/AUD": ("GBP", "AUD"),
    "AUDJPY":  ("AUD", "JPY"),
    "AUD/JPY": ("AUD", "JPY"),
    "AUDNZD":  ("AUD", "NZD"),
    "AUD/NZD": ("AUD", "NZD"),
    "NZDJPY":  ("NZD", "JPY"),
    "NZD/JPY": ("NZD", "JPY"),
    "CADJPY":  ("CAD", "JPY"),
    "CAD/JPY": ("CAD", "JPY"),
    "CHFJPY":  ("CHF", "JPY"),
    "CHF/JPY": ("CHF", "JPY"),
}


def currencies_in(symbol: str) -> set[str]:
    """Return the set of currency codes the given symbol is exposed to.

    Returns ``set()`` for non-FX symbols (XAUUSD, ETHUSD, …).
    """
    sym = symbol.upper()
    out: set[str] = set()
    if sym in EUR_EXPOSURE:
        out.add("EUR")
    if sym in JPY_EXPOSURE:
        out.add("JPY")
    if sym in GBP_EXPOSURE:
        out.add("GBP")
    if sym in AUD_EXPOSURE:
        out.add("AUD")
    if sym in CAD_EXPOSURE:
        out.add("CAD")
    if sym in NZD_EXPOSURE:
        out.add("NZD")
    if sym in CHF_EXPOSURE:
        out.add("CHF")
    return out


# Convenience set used by the few callers that just want "is this an FX
# pair we know about" — covers every symbol present in any per-currency set.
ALL_FX_SYMBOLS: frozenset[str] = (
    EUR_EXPOSURE
    | JPY_EXPOSURE
    | GBP_EXPOSURE
    | AUD_EXPOSURE
    | CAD_EXPOSURE
    | NZD_EXPOSURE
    | CHF_EXPOSURE
)


def is_usd_axis(symbol: str) -> bool:
    """True if the symbol crosses USD (one leg = USD).

    Used by the cross-symbol coherence checks (Phase 2G) and any future
    USD-direction guard. Returns False for FX crosses (EURGBP, EURJPY,
    GBPJPY, AUDNZD) and for non-FX symbols (XAUUSD, ETHUSD, …) — even
    though XAU is quoted in USD, its drivers aren't a pure DXY play.
    """
    sym = symbol.upper().replace("/", "")
    if sym not in {s.replace("/", "") for s in ALL_FX_SYMBOLS}:
        return False
    return "USD" in sym
