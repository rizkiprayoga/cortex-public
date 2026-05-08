// Static fallback palette (dark-theme values). Legacy callers still import
// `colors.panel`, `colors.border`, etc. Components that must track the live
// theme should call `readThemeColors()` instead, which reads the CSS variables
// from `<html>` and returns the current resolved values.
export const colors = {
  bg: "#0a0e1a",
  panel: "#111827",
  panelHi: "#1a2237",
  border: "#1f2a3d",
  borderHi: "#2c3a54",
  text: "#f8fafc",
  textMuted: "#94a3b8",
  textDim: "#64748b",
  primary: "#06b6d4",
  profit: "#10b981",
  loss: "#f43f5e",
  warn: "#f59e0b",
} as const;

export type ThemeColors = {
  bg: string;
  panel: string;
  panelHi: string;
  border: string;
  borderHi: string;
  text: string;
  textMuted: string;
  textDim: string;
  primary: string;
  profit: string;
  loss: string;
  warn: string;
};

/** Snapshot the currently-applied theme colors from CSS custom properties. */
export function readThemeColors(): ThemeColors {
  if (typeof document === "undefined") return { ...colors };
  const root = document.documentElement;
  const cs = getComputedStyle(root);
  const read = (name: string, fallback: string) => {
    const v = cs.getPropertyValue(name).trim();
    return v || fallback;
  };
  return {
    bg: read("--color-bg", colors.bg),
    panel: read("--color-panel", colors.panel),
    panelHi: read("--color-panel-hi", colors.panelHi),
    border: read("--color-border", colors.border),
    borderHi: read("--color-border-hi", colors.borderHi),
    text: read("--color-text", colors.text),
    textMuted: read("--color-text-muted", colors.textMuted),
    textDim: read("--color-text-dim", colors.textDim),
    primary: read("--color-primary", colors.primary),
    profit: read("--color-profit", colors.profit),
    loss: read("--color-loss", colors.loss),
    warn: read("--color-warn", colors.warn),
  };
}

export const regimeColors = {
  Crash: "#7f1d1d",
  Bear: "#f43f5e",
  Neutral: "#94a3b8",
  Bull: "#10b981",
  Euphoria: "#8b5cf6",
} as const;

export type RegimeName = keyof typeof regimeColors;

export function regimeColor(regime: string | null | undefined): string {
  if (!regime) return regimeColors.Neutral;
  const key = (regime.charAt(0).toUpperCase() + regime.slice(1).toLowerCase()) as RegimeName;
  return regimeColors[key] ?? regimeColors.Neutral;
}
