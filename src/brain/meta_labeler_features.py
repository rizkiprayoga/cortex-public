"""
Single source of truth for the meta-labeler feature schema.

Per the model bake-off spec §3 invariant #12 — both the training path
(``src/ml/meta_labeler.py``) and the inference path
(``src/brain/signal_combiner.py``) MUST import their feature-name list
from this module. The ``compute_schema_hash()`` helper produces a
deterministic identifier baked into every artifact; signal_combiner
refuses to load a bundle whose hash doesn't match the runtime contract.

Why this matters: "just one extra GBM-specific feature" creep would
defeat the modularity that makes the per-symbol bake-off comparison
clean. Ship a schema hash, lock the contract, force re-training when
the contract changes.

Schema breakdown (5 base + 17 fundamentals = 22 features):
  - Base (5): visible at signal time without external lookups
    combined_score, regime_index, direction_index, hour_of_day, day_of_week
  - fred_macro (5): dxy, vix, t10y, t2y, fed_funds
  - stooq_yields (3): us_10y_daily, us_2y_daily, us_slope_daily
  - ecb_yield_curve (2): eu_aaa_10y_daily, eu_aaa_2y_daily (read with symbol="_GLOBAL")
  - cot_disagg (2): net_spec, open_interest (XAUUSD only — NaN for FX)
  - cot_tff (2): net_spec, lev_long (FX only — column names currency-templated; NaN for XAU)
  - yfinance_cross_asset (3): vix_level, dxy_zscore, spx_zscore

Symbol-specific column-name resolution lives in ``COT_TFF_PREFIX_PER_SYMBOL``
so the user-facing schema name (``cot_tff__net_spec``) is unified across
EUR/JPY/CAD even though the underlying DB columns are eur_*/jpy_*/cad_*.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

# 5 base features — visible at signal time without external lookups.
# Names match the existing predict_proba primitives (kept for back-compat
# with shadow-mode logging and the existing test suite).
BASE_FEATURE_NAMES: tuple[str, ...] = (
    "combined_score",
    "regime_index",
    "direction_index",
    "hour_of_day",
    "day_of_week",
)

# 17 fundamental features — sourced from feature_store via
# read_feature_store_safe(as_of=trade_entry_ts) at training time. Live
# inference passes NaN until the bot wires fundamentals into the
# signal-time hot path (a separate post-Phase-A change).
FUNDAMENTAL_FEATURE_NAMES: tuple[str, ...] = (
    # fred_macro (per-symbol routing; columns are symbol-agnostic)
    "fred_macro__dxy",
    "fred_macro__vix",
    "fred_macro__t10y",
    "fred_macro__t2y",
    "fred_macro__fed_funds",
    # stooq_yields (per-symbol routing; us_* baseline columns)
    "stooq_yields__us_10y_daily",
    "stooq_yields__us_2y_daily",
    "stooq_yields__us_slope_daily",
    # ecb_yield_curve (read with symbol="_GLOBAL")
    "ecb_yield_curve__eu_aaa_10y_daily",
    "ecb_yield_curve__eu_aaa_2y_daily",
    # cot_disagg (XAUUSD only — NaN for FX)
    "cot_disagg__net_spec",
    "cot_disagg__open_interest",
    # cot_tff (FX only — NaN for XAU; underlying cols currency-templated)
    "cot_tff__net_spec",
    "cot_tff__lev_long",
    # yfinance_cross_asset (per-symbol routing; cols are symbol-agnostic)
    "yfinance_cross_asset__vix_level",
    "yfinance_cross_asset__dxy_zscore",
    "yfinance_cross_asset__spx_zscore",
)

# 4 execution-conditional features (Phase 2B Option 2 — 2026-04-27).
#
# Why exec features: fundamentals barely move across a 22-month OOS window,
# so a meta-labeler trained on (base + fundamentals) has very little
# fast-changing signal to discriminate between "now is a good moment to
# trade" vs "now is a bad moment". Exec features change every bar.
#
# What's intentionally NOT here:
#   - news/calendar proximity — already filtered upstream by news blackout
#   - primary win-rate over recent trades — feedback-loop risk under live
#     gating (meta-blocks → primary_wr drifts → meta gets more confident)
#   - time-of-day — already in BASE_FEATURE_NAMES via hour_of_day
EXEC_FEATURE_NAMES: tuple[str, ...] = (
    "exec__rv_short_20",   # 20-bar H4 close-return realized vol (~3.3 days)
    "exec__rv_long_60",    # 60-bar H4 close-return realized vol (~10 days)
    "exec__rv_ratio",      # rv_short / rv_long — regime-shift signal
    "exec__score_avg_20",  # rolling mean of combined_score over last 20 SIGNALS
                           # (signals not trades — gating-independent so no
                           # feedback loop between meta-labeler and feature)
)

EXPECTED_FEATURE_NAMES: tuple[str, ...] = (
    BASE_FEATURE_NAMES + FUNDAMENTAL_FEATURE_NAMES + EXEC_FEATURE_NAMES
)

# COT routing per symbol. XAUUSD uses the disaggregated COT report
# (commodity); FX pairs use the Traders-in-Financial-Futures (TFF)
# report keyed by the foreign currency. ETHUSD intentionally omitted —
# CFTC doesn't publish ETH; both COT cols stay NaN.
COT_GROUP_PER_SYMBOL: dict[str, str] = {
    "XAUUSD": "cot_disagg",
    "EURUSD": "cot_tff",
    "USDJPY": "cot_tff",
    "USDCAD": "cot_tff",
}

# Currency-prefix used in cot_tff column names. Maps unified schema
# columns (cot_tff__net_spec) to DB columns (eur_net_spec / jpy_net_spec
# / cad_net_spec).
COT_TFF_PREFIX_PER_SYMBOL: dict[str, str] = {
    "EURUSD": "eur",
    "USDJPY": "jpy",
    "USDCAD": "cad",
}


def compute_schema_hash(feature_names: Iterable[str]) -> str:
    """SHA-256 over sorted feature names; first 16 hex chars used as ID.

    Order-independent (sorts internally) so permutations of the same set
    produce the same hash. Truncated to 16 chars — collision risk over
    a small number of schemas is negligible and the shorter hash is
    easier to eyeball in logs / error messages.
    """
    canonical = "\n".join(sorted(feature_names)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


# Pre-computed for the canonical schema. Imported by signal_combiner so
# load-time validation doesn't recompute on every check.
EXPECTED_SCHEMA_HASH: str = compute_schema_hash(EXPECTED_FEATURE_NAMES)

# Phase 2A pivot follow-up (2026-04-27): a temporary 5-feature schema for
# the meta-labeler when fundamentals plumbing isn't wired to inference yet.
# Allows train/serve parity for the base-features-only path until the
# fundamentals are read from feature_store at signal time. signal_combiner
# accepts EITHER this or EXPECTED_SCHEMA_HASH so a bundle saved in
# base-only mode loads cleanly without the silent NaN-degradation that
# the full 22-feature inference suffers today.
BASE_ONLY_SCHEMA_HASH: str = compute_schema_hash(BASE_FEATURE_NAMES)

# Phase 2B Option 2 follow-up (2026-04-27): the prior 22-feature schema
# without exec features. Kept in ACCEPTED so legacy bundles still load
# cleanly during the rollout window, even though new training writes
# 26-feature bundles by default.
LEGACY_22_SCHEMA_HASH: str = compute_schema_hash(
    BASE_FEATURE_NAMES + FUNDAMENTAL_FEATURE_NAMES,
)
ACCEPTED_SCHEMA_HASHES: frozenset[str] = frozenset({
    EXPECTED_SCHEMA_HASH,
    BASE_ONLY_SCHEMA_HASH,
    LEGACY_22_SCHEMA_HASH,
})
