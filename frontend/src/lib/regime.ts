/**
 * Regime-duration helper.
 *
 * The HMM posterior saturates at 1.0 once a regime has been established
 * for enough bars, so "conf 100%" alone is ambiguous — the user can't tell
 * "just flipped, rounded up" from "held for 800 bars". We compute the
 * length of the current regime run by walking the signal-audit history
 * (newest-first) until we hit a different regime label.
 *
 * `bounded: true` means the history window ran out before we found a
 * different regime — duration is a lower bound, not exact. Call sites
 * render it as `≥ Xh` in that case.
 */

export interface RegimeHistoryRow {
  timestamp: string;
  regime: string | null;
}

interface RegimeRun {
  ms: number;
  bounded: boolean;
}

export function durationInRegime(
  history: RegimeHistoryRow[],
  currentRegime: string | null,
): RegimeRun {
  if (!currentRegime || history.length === 0) {
    return { ms: 0, bounded: true };
  }
  // API returns newest-first. Walk until regime changes.
  let oldestSameIdx = -1;
  for (let i = 0; i < history.length; i++) {
    if (history[i].regime === currentRegime) {
      oldestSameIdx = i;
    } else {
      break;
    }
  }
  if (oldestSameIdx < 0) return { ms: 0, bounded: false };
  const oldest = history[oldestSameIdx];
  const iso = oldest.timestamp;
  const t = Date.parse(
    iso.endsWith("Z") || /[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`,
  );
  if (!Number.isFinite(t)) return { ms: 0, bounded: true };
  const ms = Math.max(0, Date.now() - t);
  const bounded = oldestSameIdx === history.length - 1;
  return { ms, bounded };
}

export function fmtRegimeDuration(run: RegimeRun): string {
  if (run.ms <= 0) return "—";
  const prefix = run.bounded ? "≥" : "~";
  const mins = Math.floor(run.ms / 60_000);
  if (mins < 60) return `${prefix}${mins}m`;
  const hours = Math.floor(mins / 60);
  const remM = mins % 60;
  if (hours < 48) {
    return remM > 0 ? `${prefix}${hours}h ${remM}m` : `${prefix}${hours}h`;
  }
  const days = Math.floor(hours / 24);
  const remH = hours % 24;
  return remH > 0 ? `${prefix}${days}d ${remH}h` : `${prefix}${days}d`;
}
