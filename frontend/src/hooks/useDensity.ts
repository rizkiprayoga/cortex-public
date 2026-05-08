import { useCallback, useEffect, useState } from "react";

export type Density = "default" | "compact";

export const DENSITIES: readonly Density[] = ["default", "compact"] as const;
const DEFAULT_DENSITY: Density = "default";
const DENSITY_STORAGE_KEY = "cortex-density";

function isDensity(value: unknown): value is Density {
  return value === "default" || value === "compact";
}

export function readStoredDensity(): Density {
  try {
    const stored = window.localStorage.getItem(DENSITY_STORAGE_KEY);
    if (isDensity(stored)) return stored;
  } catch {
    /* localStorage disabled / SSR — fall through to default */
  }
  return DEFAULT_DENSITY;
}

export function applyDensity(density: Density): void {
  document.documentElement.dataset.density = density;
}

export function useDensity() {
  const [density, setDensityState] = useState<Density>(() => {
    if (typeof window === "undefined") return DEFAULT_DENSITY;
    const current = document.documentElement.dataset.density;
    return isDensity(current) ? current : readStoredDensity();
  });

  useEffect(() => {
    applyDensity(density);
    try {
      window.localStorage.setItem(DENSITY_STORAGE_KEY, density);
    } catch {
      /* localStorage disabled — density still applies for this session */
    }
  }, [density]);

  const setDensity = useCallback((next: Density) => setDensityState(next), []);

  return { density, setDensity, densities: DENSITIES };
}
