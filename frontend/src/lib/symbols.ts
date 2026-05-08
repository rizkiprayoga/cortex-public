/**
 * Single source of truth for the live trading symbol set.
 *
 * Updated 2026-04-29 (the trading universe) — symmetric MULTIPLIERS + :
 * XAU + 9 forex. Drops AUDUSD/GBPJPY (b3_u5 picks); adds NZDUSD (short-USD
 * diversifier) and EURJPY (cleaner JPY exposure than GBPJPY).
 *
 * Use this constant instead of hardcoding the symbol list in screens.
 * When the API returns dynamic per-symbol data (e.g. live signals),
 * prefer the API-provided keys; fall back to LIVE_SYMBOLS for layouts
 * that need to render slots even before data is available.
 *
 * Note: Currently this is the *target* production set (post Sprint 2
 * promotion). Until promotion lands in prod, the live API may return a
 * subset (the 5 currently-enabled live pairs). Layouts will render the
 * full 10-card grid with empty data for non-active pairs — that is the
 * expected behavior during the dev-preview window.
 */

export const LIVE_SYMBOLS = [
  "XAUUSD",
  "GBPUSD",
  "USDJPY",
  "USDCAD",
  "NZDUSD",
  "USDCHF",
  "GBPCHF",
  "EURAUD",
  "GBPAUD",
  "EURJPY",
] as const;

export type LiveSymbol = (typeof LIVE_SYMBOLS)[number];

/** History page filter dropdown — includes "All" prefix for the global view. */
export const SYMBOL_FILTERS = ["All", ...LIVE_SYMBOLS] as const;
