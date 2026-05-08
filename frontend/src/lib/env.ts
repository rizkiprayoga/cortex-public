/**
 * Runtime environment detection for the dashboard.
 *
 * Dev and prod run the same frontend bundle against isolated backends
 * on different ports (prod=8787, dev=8788). Port-based detection lets
 * us surface a "DEV" tag without coupling to build-time env vars or
 * requiring a separate build pipeline per environment.
 */
export type CortexEnv = "dev" | "prod";

const DEV_PORT = "8788";

export function getCortexEnv(): CortexEnv {
  if (typeof window === "undefined") return "prod";
  return window.location.port === DEV_PORT ? "dev" : "prod";
}

export function isDev(): boolean {
  return getCortexEnv() === "dev";
}
