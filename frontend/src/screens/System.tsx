import { useEffect, useState } from "react";
import { useSystemStatus, useBotStatus } from "@/hooks/useSystemStatus";
import { api } from "@/lib/api-client";
import { duration } from "@/lib/format";
import type { BotAction } from "@/lib/types";

// ─── Status strip cards ──────────────────────────────────────────────

function BotStatusCard({
  status,
  changedBy,
  changedAt,
}: {
  status: string;
  changedBy: string | null;
  changedAt: string | null;
}) {
  const meta: Record<
    string,
    { label: string; color: string; pulse: boolean }
  > = {
    running: { label: "RUNNING", color: "var(--chip-profit-fg)", pulse: true },
    paused: { label: "PAUSED", color: "var(--chip-warn-fg)", pulse: false },
    stopped: { label: "STOPPED", color: "var(--chip-loss-fg)", pulse: false },
  };
  const m = meta[status] ?? {
    label: status.toUpperCase(),
    color: "var(--color-text-muted)",
    pulse: false,
  };
  const ago = changedAt
    ? timeSince(new Date(changedAt).getTime())
    : null;
  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
      <p className="section-label">Bot status</p>
      <div className="mt-2 flex items-center gap-2">
        <span
          className={`w-2 h-2 rounded-full ${m.pulse ? "live-pulse" : ""}`}
          style={{ background: m.color }}
        />
        <span className="text-xl font-bold" style={{ color: m.color }}>
          {m.label}
        </span>
      </div>
      <p className="text-[11px] text-[var(--color-text-muted)] mt-1">
        {changedBy ? `changed by ${changedBy}` : "no changes yet"}
        {ago ? ` · ${ago} ago` : ""}
      </p>
    </div>
  );
}

function timeSince(ms: number): string {
  const secs = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ${m % 60}m`;
  return `${Math.floor(h / 24)}d`;
}

// ─── Log line colorization ──────────────────────────────────────────

function parseLogLine(line: string): {
  ts: string | null;
  level: "INFO" | "WARN" | "ERROR" | null;
  body: string;
} {
  // Typical loguru line: "2026-04-18 22:30:14 | INFO    | path:fn:ln - message"
  // Or already-trimmed: "22:30:14 INFO [..] message"
  const tsMatch = line.match(/^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2})/);
  const ts = tsMatch ? tsMatch[1].split(/\s+/).pop() ?? null : null;
  const lvlMatch = line.match(/\b(INFO|WARN|WARNING|ERROR|CRITICAL)\b/);
  let level: "INFO" | "WARN" | "ERROR" | null = null;
  if (lvlMatch) {
    if (lvlMatch[1] === "INFO") level = "INFO";
    else if (lvlMatch[1] === "ERROR" || lvlMatch[1] === "CRITICAL") level = "ERROR";
    else level = "WARN";
  }
  return { ts, level, body: line };
}

function LogTail({ lines }: { lines: string[] }) {
  const levelColor = (l: string | null): string => {
    if (l === "INFO") return "var(--color-profit)";
    if (l === "WARN") return "var(--color-warn)";
    if (l === "ERROR") return "var(--color-loss)";
    return "var(--color-text-muted)";
  };
  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ background: "var(--color-bg)", border: "1px solid var(--color-border)" }}
    >
      <div
        className="px-4 py-2 border-b flex items-center gap-3 text-[11px]"
        style={{
          borderColor: "var(--color-border)",
          background: "var(--color-panel)",
        }}
      >
        <span className="inline-flex items-center gap-1">
          <span className="w-2 h-2 rounded-full" style={{ background: "var(--color-profit)" }} />
          INFO
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="w-2 h-2 rounded-full" style={{ background: "var(--color-warn)" }} />
          WARN
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="w-2 h-2 rounded-full" style={{ background: "var(--color-loss)" }} />
          ERROR
        </span>
        <span className="ml-auto text-[var(--color-text-dim)]">
          {lines.length} lines
        </span>
      </div>
      <pre
        className="mono text-[11px] p-4 leading-relaxed overflow-x-auto max-h-96"
        style={{ color: "var(--color-text-muted)" }}
      >
        {lines.length === 0
          ? "(empty)"
          : [...lines].reverse().map((raw, i) => {
              const p = parseLogLine(raw);
              return (
                <div key={i} className="whitespace-pre">
                  {p.ts && (
                    <span style={{ color: "var(--color-text-dim)" }}>{p.ts} </span>
                  )}
                  {p.level && (
                    <span style={{ color: levelColor(p.level) }}>
                      {p.level.padEnd(5)}{" "}
                    </span>
                  )}
                  <span>{raw.replace(/^.*?(INFO|WARN|WARNING|ERROR|CRITICAL)\s*[|:]?\s*/, "")}</span>
                </div>
              );
            })}
      </pre>
    </div>
  );
}

// ─── Screen ──────────────────────────────────────────────────────────

export function System() {
  const { data: sysStatus, dataUpdatedAt } = useSystemStatus();
  const { data: botStatus, refetch: refetchBot } = useBotStatus();

  const [, setNow] = useState(Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const [actionLoading, setActionLoading] = useState(false);
  const [confirmStop, setConfirmStop] = useState(false);
  const [stopInput, setStopInput] = useState("");
  const [confirmRestart, setConfirmRestart] = useState(false);
  const [restartLoading, setRestartLoading] = useState(false);
  const [restartMsg, setRestartMsg] = useState("");
  const [error, setError] = useState("");

  const currentStatus = (botStatus?.status ?? "unknown").toLowerCase();

  const handleBotAction = async (action: BotAction) => {
    if (action === "stop" && !confirmStop) {
      setConfirmStop(true);
      return;
    }
    setActionLoading(true);
    setError("");
    try {
      const body: Record<string, string> = { action };
      if (action === "stop") body.confirmation = stopInput;
      await api.post("/api/bot/control", body);
      await refetchBot();
      setConfirmStop(false);
      setStopInput("");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Action failed");
    } finally {
      setActionLoading(false);
    }
  };

  const handleRestart = async () => {
    if (!confirmRestart) {
      setConfirmRestart(true);
      return;
    }
    setRestartLoading(true);
    setError("");
    setRestartMsg("");
    try {
      const resp = await api.post<{ message: string; pid?: number }>(
        "/api/system/restart",
        {},
      );
      setRestartMsg(resp.message ?? "Restart scheduled. Refresh the dashboard in ~10s.");
      setConfirmRestart(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Restart failed");
    } finally {
      setRestartLoading(false);
    }
  };

  const updatedSecAgo = dataUpdatedAt
    ? Math.max(0, Math.round((Date.now() - dataUpdatedAt) / 1000))
    : null;

  const heartbeatAge = sysStatus?.heartbeat_age_seconds;
  const hbColor =
    heartbeatAge == null
      ? "var(--color-text-muted)"
      : heartbeatAge < 60
        ? "var(--color-profit)"
        : heartbeatAge < 300
          ? "var(--color-warn)"
          : "var(--color-loss)";

  const errorCount = sysStatus?.recent_errors.length ?? 0;

  return (
    <div className="space-y-4">
      <header className="flex items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[var(--color-text)]">System</h1>
          <p className="text-xs text-[var(--color-text-dim)] mt-0.5">
            Bot control · heartbeat · log tail
            {updatedSecAgo != null ? ` · refreshed ${updatedSecAgo}s ago` : ""}
          </p>
        </div>
      </header>

      {/* Status strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <BotStatusCard
          status={currentStatus}
          changedBy={botStatus?.changed_by ?? null}
          changedAt={botStatus?.last_change ?? null}
        />
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
          <p className="section-label">Uptime</p>
          <p className="tnum text-xl font-bold mt-2">
            {sysStatus ? duration(sysStatus.uptime_seconds) : "—"}
          </p>
          <p className="mono text-[11px] text-[var(--color-text-muted)] mt-1">
            api :{sysStatus?.api_port ?? 8787}
          </p>
        </div>
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
          <p className="section-label">Heartbeat</p>
          <p className="text-xl font-bold mt-2" style={{ color: hbColor }}>
            {heartbeatAge == null ? "—" : `${Math.round(heartbeatAge)}s ago`}
          </p>
          <p className="text-[11px] text-[var(--color-text-muted)] mt-1">
            equity ${(sysStatus?.heartbeat_equity ?? 0).toFixed(2)} ·{" "}
            {sysStatus?.heartbeat_open_positions ?? 0} pos
          </p>
        </div>
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
          <p className="section-label">Errors · 10min</p>
          <p
            className="text-xl font-bold mt-2"
            style={{
              color: errorCount === 0 ? "var(--color-profit)" : "var(--color-loss)",
            }}
          >
            {errorCount === 0 ? "none" : errorCount}
          </p>
          <p className="text-[11px] text-[var(--color-text-muted)] mt-1">
            {sysStatus?.dashboard_locked ? "dashboard LOCKED" : "noise-filtered"}
          </p>
        </div>
      </div>

      {/* Control card */}
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold text-[var(--color-text)]">Bot control</h3>
          <span className="text-[11px] text-[var(--color-text-dim)]">
            Actions logged to audit trail.
          </span>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => handleBotAction("start")}
            disabled={actionLoading || currentStatus === "running"}
            className="px-4 py-2.5 rounded-lg text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              background: "rgba(16,185,129,0.12)",
              color: "var(--color-profit)",
              border: "1px solid rgba(16,185,129,0.28)",
            }}
          >
            ▶ Start
          </button>
          <button
            onClick={() => handleBotAction("pause")}
            disabled={actionLoading || currentStatus === "paused"}
            className="px-4 py-2.5 rounded-lg text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              background: "rgba(245,158,11,0.12)",
              color: "var(--color-warn)",
              border: "1px solid rgba(245,158,11,0.28)",
            }}
          >
            ⏸ Pause
          </button>
          <button
            onClick={() => handleBotAction("stop")}
            disabled={actionLoading || currentStatus === "stopped"}
            className="px-4 py-2.5 rounded-lg text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              background: "rgba(244,63,94,0.12)",
              color: "var(--color-loss)",
              border: "1px solid rgba(244,63,94,0.28)",
            }}
          >
            ■ Stop
          </button>
          <button
            onClick={handleRestart}
            disabled={restartLoading}
            className="ml-auto px-4 py-2.5 rounded-lg text-white text-sm font-semibold bg-brand-gradient hover:brightness-110 disabled:opacity-50"
            title="Kills and re-launches the bot via Task Scheduler"
          >
            ↻ {confirmRestart ? "Click again to confirm" : "Restart"}
          </button>
        </div>
        {restartMsg && (
          <p className="mt-3 text-xs" style={{ color: "var(--color-primary)" }}>
            {restartMsg}
          </p>
        )}
        {confirmRestart && !restartLoading && !restartMsg && (
          <p className="mt-2 text-xs" style={{ color: "var(--color-warn)" }}>
            Restart will close the current bot and start a fresh detached instance. This
            briefly interrupts trading.
          </p>
        )}
        {error && (
          <p className="mt-2 text-xs" style={{ color: "var(--color-loss)" }}>
            {error}
          </p>
        )}
        {confirmStop && (
          <div
            className="mt-4 p-4 rounded-lg"
            style={{
              background: "rgba(244,63,94,0.06)",
              border: "1px solid rgba(244,63,94,0.28)",
            }}
          >
            <p className="text-sm mb-2" style={{ color: "var(--color-loss)" }}>
              This will HALT the trading loop and flatten all positions.
            </p>
            <p className="text-xs text-[var(--color-text-muted)] mb-3">
              Type <span className="mono font-bold">STOP</span> to confirm:
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={stopInput}
                onChange={(e) => setStopInput(e.target.value)}
                placeholder="STOP"
                className="flex-1 px-3 py-2 rounded bg-[var(--color-panel-hi)] border border-[var(--color-border)] text-sm focus:outline-none"
                style={{ color: "var(--color-text)" }}
                autoFocus
              />
              <button
                onClick={() => handleBotAction("stop")}
                disabled={stopInput !== "STOP" || actionLoading}
                className="px-4 py-2 rounded text-white text-sm font-medium disabled:opacity-40"
                style={{ background: "var(--color-loss)" }}
              >
                Confirm
              </button>
              <button
                onClick={() => {
                  setConfirmStop(false);
                  setStopInput("");
                }}
                className="px-4 py-2 rounded bg-[var(--color-panel-hi)] text-[var(--color-text-muted)] text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Recent errors (if any) */}
      {errorCount > 0 && sysStatus && (
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-6">
          <h3 className="text-base font-semibold text-[var(--color-text)] mb-3">
            Errors · last 10 min
          </h3>
          <pre
            className="mono text-[11px] p-3 rounded overflow-x-auto max-h-40 whitespace-pre-wrap"
            style={{ background: "var(--color-bg)", color: "var(--color-loss)" }}
          >
            {sysStatus.recent_errors.join("\n")}
          </pre>
        </div>
      )}

      {/* Log tail */}
      {sysStatus && (
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-base font-semibold text-[var(--color-text)]">
              Log tail · newest first
            </h3>
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
              style={{ background: "var(--color-panel-hi)", color: "var(--color-text-muted)" }}
            >
              auto-refresh · 5s
            </span>
          </div>
          <LogTail lines={sysStatus.log_tail} />
        </div>
      )}
    </div>
  );
}
