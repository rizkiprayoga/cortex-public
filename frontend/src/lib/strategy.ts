/* Mirror of src/strategy/orchestrator.py — vol-rank → strategy class.
 *
 * The orchestrator picks one of three strategy classes from the HMM regime's
 * expected_volatility, ranked among all_expected_vols.  We replicate the same
 * logic client-side so the dashboard can show the strategy that would fire
 * for the current signal, without needing a new backend field.
 *
 * If the backend cutoffs change, update LOW_VOL_CUTOFF / HIGH_VOL_CUTOFF.
 */

import type { RegimeData } from "@/lib/types";

// PLACEHOLDERS — tuned production values redacted from this public template.
// Match these to config/orchestrator vol-rank cutoffs in your private fork.
const LOW_VOL_CUTOFF = 0.0;
const HIGH_VOL_CUTOFF = 1.0;

export type StrategyName =
  | "LowVolAggressive"
  | "MidVolCautious"
  | "HighVolDefensive";

/**
 * Compute the vol rank of `expected` relative to `allVols` in [0, 1].
 * Returns 0.5 when the rank is undefined (cold start, single regime, etc.).
 * Uses np.searchsorted(side="left") semantics — ties go to the leftmost.
 */
export function volRank(
  expected: number,
  allVols: number[] | null | undefined,
): number {
  if (!allVols || allVols.length < 2) return 0.5;
  let min = Infinity;
  let max = -Infinity;
  for (const v of allVols) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (max - min < 1e-12) return 0.5;
  const sorted = [...allVols].sort((a, b) => a - b);
  let pos = 0;
  while (pos < sorted.length && sorted[pos] < expected) pos += 1;
  pos = Math.max(0, Math.min(pos, sorted.length - 1));
  return pos / (sorted.length - 1);
}

/**
 * Pick the strategy class for the given regime, mirroring
 * StrategyOrchestrator.select_strategy().
 */
export function deriveStrategyName(regime: RegimeData): StrategyName {
  const rank = volRank(regime.expected_volatility, regime.all_expected_vols);
  if (rank <= LOW_VOL_CUTOFF) return "LowVolAggressive";
  if (rank >= HIGH_VOL_CUTOFF) return "HighVolDefensive";
  return "MidVolCautious";
}

/** Short label for chips: "LowVol" / "MidVol" / "HighVol". */
export function strategyShortLabel(name: StrategyName): string {
  if (name === "LowVolAggressive") return "LowVol";
  if (name === "HighVolDefensive") return "HighVol";
  return "MidVol";
}

/** Strategy chip palette. Aligns with profit / muted / loss tokens. */
export function strategyColor(name: StrategyName): string {
  if (name === "LowVolAggressive") return "var(--color-profit)";
  if (name === "HighVolDefensive") return "var(--color-loss)";
  return "var(--color-text-muted)";
}

/**
 * Parse the canonical strategy name string emitted by the backend
 * (StrategyDecision.strategy_name).  Returns null for empty/unknown inputs
 * — typically legacy reconciled positions with no strategy attribution.
 */
export function parseStrategyName(name: string | null | undefined): StrategyName | null {
  if (!name) return null;
  if (name === "LowVolAggressive") return "LowVolAggressive";
  if (name === "MidVolCautious") return "MidVolCautious";
  if (name === "HighVolDefensive") return "HighVolDefensive";
  return null;
}

/** Default `allocation_pct` baked into each strategy class.  Mid-vol may be
 *  reduced to 0.60 at runtime when the signal direction is misaligned with
 *  EMA50, but those overrides aren't in the live payload — this is the
 *  class-default the dashboard surfaces alongside the chip. */
export function strategyDefaultAllocation(name: StrategyName): number {
  if (name === "LowVolAggressive") return 0.95;
  if (name === "HighVolDefensive") return 0.60;
  return 0.95; // MidVolCautious aligned default
}
