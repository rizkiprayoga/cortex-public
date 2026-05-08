import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "dim" | "light" | "coffee";

export const THEMES: readonly Theme[] = ["dark", "dim", "light", "coffee"] as const;
const DEFAULT_THEME: Theme = "dark";
const THEME_STORAGE_KEY = "cortex-theme";

function isTheme(value: unknown): value is Theme {
  return value === "dark" || value === "dim" || value === "light" || value === "coffee";
}

export function readStoredTheme(): Theme {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (isTheme(stored)) return stored;
  } catch {
    /* localStorage disabled / SSR — fall through to default */
  }
  return DEFAULT_THEME;
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window === "undefined") return DEFAULT_THEME;
    const current = document.documentElement.dataset.theme;
    return isTheme(current) ? current : readStoredTheme();
  });

  useEffect(() => {
    applyTheme(theme);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      /* localStorage disabled — theme still applies for this session */
    }
  }, [theme]);

  const setTheme = useCallback((next: Theme) => setThemeState(next), []);

  return { theme, setTheme, themes: THEMES };
}
