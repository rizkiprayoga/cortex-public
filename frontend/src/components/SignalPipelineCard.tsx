import { useMemo, useState } from "react";
import { useSignalAudit } from "@/hooks/useSignalAudit";
import { useTradeHistory } from "@/hooks/useHistory";
import type { SignalAuditItem, TradeHistoryItem } from "@/lib/types";

type Window = "7d" | "30d" | "90d";
const WINDOW_DAYS: Record<Window, number> = { "7d": 7, "30d": 30, "90d": 90 };
const THRESHOLD = 0.45;

function isThresholdBlock(r: string | null): boolean {
  return !!r && /threshold/i.test(r);
}
function isFlickerBlock(r: string | null): boolean {
  return !!r && /flicker/i.test(r);
}

export function SignalPipelineCard({ symbol }: { symbol: string }) {
  const [win, setWin] = useState<Window>("30d");
  // Backend filters by symbol (case-insensitive). page_size=1500 covers
  // ~90d of per-symbol audit rows for any active pair (audit writes ~15-25
  // rows/day per symbol). Backend cap is 2000 so this has headroom.
  const { data: audit, isLoading: auditLoading } = useSignalAudit({ symbol, pageSize: 1500 });
  const { data: trades, isLoading: tradesLoading } = useTradeHistory(1, 500, symbol);

  const stats = useMemo(() => {
    const empty = {
      evaluated: 0, aboveThreshold: 0, passedFlicker: 0,
      passedOther: 0, executed: 0, won: 0, lost: 0, open: 0,
      pnl: 0,
    };
    const items: SignalAuditItem[] = audit?.items ?? [];
    const tradeRows: TradeHistoryItem[] = trades?.trades ?? [];
    if (items.length === 0) return empty;

    const cutoff = Date.now() - WINDOW_DAYS[win] * 86_400_000;

    // Filter audit rows to window (symbol already filtered server-side).
    const windowed = items.filter((i) => {
      const t = Date.parse(i.timestamp);
      return !Number.isNaN(t) && t >= cutoff;
    });

    const evaluated = windowed.length;

    // Above threshold: abs(score) >= THRESHOLD (regardless of executed / block)
    const aboveThreshold = windowed.filter(
      (i) => i.combined_score != null && Math.abs(i.combined_score) >= THRESHOLD,
    ).length;

    // Passed flicker: above threshold AND not blocked by flicker
    const passedFlicker = windowed.filter(
      (i) =>
        i.combined_score != null &&
        Math.abs(i.combined_score) >= THRESHOLD &&
        !isFlickerBlock(i.block_reason ?? null),
    ).length;

    // Passed other gates: above threshold AND not blocked by flicker AND not blocked by
    // any other gate. Practically: either executed=true OR block_reason is empty/null.
    const passedOther = windowed.filter(
      (i) =>
        i.combined_score != null &&
        Math.abs(i.combined_score) >= THRESHOLD &&
        (i.executed || !i.block_reason) &&
        !isFlickerBlock(i.block_reason ?? null) &&
        !isThresholdBlock(i.block_reason ?? null),
    ).length;

    // Trade outcomes in window for this symbol (account-scoped — trades
    // endpoint filters by the currently active mt5_account).
    const tradesInWindow = tradeRows.filter((t) => {
      const opened = t.timestamp_open;
      const tOpen = opened ? Date.parse(opened) : NaN;
      return !Number.isNaN(tOpen) && tOpen >= cutoff;
    });

    // Executed = actual trade count (account-scoped), not signal_audit's
    // executed=True count. The audit is GLOBAL across accounts and
    // account-blind — showing audit's executed-count on a demo account
    // that has zero real trades was misleading (caught 2026-04-19).
    const executed = tradesInWindow.length;

    let won = 0, lost = 0, open = 0, pnl = 0;
    for (const t of tradesInWindow) {
      const closed = t.timestamp_close;
      if (!closed) { open += 1; continue; }
      const p = t.pnl_usd ?? 0;
      pnl += p;
      if (p > 0) won += 1;
      else if (p < 0) lost += 1;
    }

    return { evaluated, aboveThreshold, passedFlicker, passedOther, executed, won, lost, open, pnl };
  }, [audit, trades, win]);

  const isLoading = auditLoading || tradesLoading;
  const pct = (n: number) => stats.evaluated > 0 ? (n / stats.evaluated) * 100 : 0;
  const fmtPct = (n: number) => stats.evaluated > 0 ? `${((n / stats.evaluated) * 100).toFixed(0)}%` : "—";
  const fmtStepConv = (num: number, den: number) => den > 0 ? `${((num / den) * 100).toFixed(1)}%` : "—";
  const closedTrades = stats.won + stats.lost;
  const winRate = closedTrades > 0 ? (stats.won / closedTrades) * 100 : 0;

  // Width of each bar as % of evaluated — same denominator for visual comparability.
  type Step = { label: string; count: number; gate?: string; fill: string };
  const steps: Step[] = [
    { label: "Evaluated (H4 bars)", count: stats.evaluated, fill: "var(--color-primary)" },
    { label: "Above threshold", count: stats.aboveThreshold, fill: "var(--color-primary)", gate: `↓ signal threshold ${THRESHOLD}` },
    { label: "Passed flicker", count: stats.passedFlicker, fill: "var(--color-primary)", gate: "↓ flicker (2-bar agreement)" },
    { label: "Other gates passed", count: stats.passedOther, fill: "var(--color-primary)", gate: "↓ news + CB + direction" },
    { label: "Executed", count: stats.executed, fill: "var(--color-profit)", gate: "↓ sizing + broker acceptance" },
  ];
  const outcomeSteps: Step[] = [
    { label: "Won", count: stats.won, fill: "var(--color-profit)" },
    { label: "Lost", count: stats.lost, fill: "var(--color-loss)" },
  ];

  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
      <div className="flex items-start justify-between gap-3 mb-4 flex-wrap">
        <div>
          <p className="section-label">Signal pipeline · {symbol}</p>
          <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
            Every H4 bar → trade outcome · win/loss from closed trades in window
          </p>
        </div>
        <div className="flex items-center gap-1 bg-[var(--color-panel-hi)] rounded-lg p-0.5 text-[11px]">
          {(["7d", "30d", "90d"] as const).map((w) => {
            const active = w === win;
            return (
              <button
                key={w}
                onClick={() => setWin(w)}
                className="px-2.5 py-1 rounded"
                style={active ? { background: "var(--color-panel)", color: "var(--color-primary)", fontWeight: 600 } : { color: "var(--color-text-muted)" }}
              >
                {w}
              </button>
            );
          })}
        </div>
      </div>

      {isLoading && stats.evaluated === 0 ? (
        <div className="py-10 text-center text-xs text-[var(--color-text-dim)]">Loading…</div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-[1.8fr_1fr] gap-6 items-start">
          {/* LEFT — funnel steps. Always render all 7 rows (even when
              data is sparse) so the card shape is stable. */}
          <div className="flex flex-col gap-2">
            {steps.map((s, i) => (
              <div key={s.label}>
                <div
                  className="grid items-center gap-3 text-[11px]"
                  style={{ gridTemplateColumns: "160px 1fr 70px 60px" }}
                >
                  <span className="text-[var(--color-text)]">{s.label}</span>
                  <div className="h-[32px] rounded bg-black/5 relative overflow-hidden">
                    <div
                      className="absolute inset-y-0 left-0"
                      style={{ width: `${pct(s.count)}%`, background: s.fill, opacity: 0.72 }}
                    />
                  </div>
                  <span className="mono text-right text-[var(--color-primary)] font-semibold">
                    {s.count}
                  </span>
                  <span className="mono text-right text-[var(--color-text-muted)]">
                    {fmtPct(s.count)}
                  </span>
                </div>
                {steps[i + 1]?.gate && (
                  <div className="text-[10px] text-[var(--color-text-dim)] ml-7 mt-0.5 mb-0.5">
                    {steps[i + 1].gate}
                  </div>
                )}
              </div>
            ))}
            {stats.open > 0 && (
              <div className="text-[10px] text-[var(--color-text-dim)] ml-7 mt-0.5">
                ↓ closed ({stats.open} still open)
              </div>
            )}
            {outcomeSteps.map((s) => (
              <div
                key={s.label}
                className="grid items-center gap-3 text-[11px]"
                style={{ gridTemplateColumns: "160px 1fr 70px 60px" }}
              >
                <span className="text-[var(--color-text)]">{s.label}</span>
                <div className="h-[22px] rounded bg-black/5 relative overflow-hidden">
                  <div
                    className="absolute inset-y-0 left-0"
                    style={{ width: `${pct(s.count)}%`, background: s.fill, opacity: 0.82 }}
                  />
                </div>
                <span className="mono text-right text-[var(--color-primary)] font-semibold">
                  {s.count}
                </span>
                <span className="mono text-right text-[var(--color-text-muted)]">
                  {s.label === "Won" && closedTrades > 0
                    ? `${winRate.toFixed(0)}% WR`
                    : s.label === "Lost"
                      ? `${stats.pnl >= 0 ? "+" : "−"}$${Math.abs(stats.pnl).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                      : fmtPct(s.count)}
                </span>
              </div>
            ))}
          </div>

          {/* RIGHT — conversion summary */}
          <div className="rounded-lg bg-[var(--color-panel-hi)] border border-[var(--color-border)] p-4">
            <p className="section-label">Conversion rates</p>
            <div className="flex flex-col gap-1.5 mt-3 text-[12px]">
              <ConvRow k="Eval → threshold" v={fmtStepConv(stats.aboveThreshold, stats.evaluated)} />
              <ConvRow k="Threshold → flicker" v={fmtStepConv(stats.passedFlicker, stats.aboveThreshold)} />
              <ConvRow k="Flicker → other gates" v={fmtStepConv(stats.passedOther, stats.passedFlicker)} />
              <ConvRow k="Gates → executed" v={fmtStepConv(stats.executed, stats.passedOther)} />
              <ConvRow
                k="End-to-end"
                v={fmtStepConv(stats.executed, stats.evaluated)}
                accent
              />
            </div>
            {closedTrades > 0 && (
              <div className="mt-4 pt-3 border-t border-[var(--color-border)] text-[11px] text-[var(--color-text-muted)]">
                <b className="text-[var(--color-text)]">Outcomes:</b> {stats.won}W / {stats.lost}L{" "}
                {stats.open > 0 ? `/ ${stats.open}O ` : ""}·{" "}
                <span style={{ color: stats.pnl >= 0 ? "var(--color-profit)" : "var(--color-loss)" }}>
                  {stats.pnl >= 0 ? "+" : "−"}${Math.abs(stats.pnl).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
                {" "}over {closedTrades + stats.open} trade{closedTrades + stats.open === 1 ? "" : "s"} in {win}.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function ConvRow({ k, v, accent }: { k: string; v: string; accent?: boolean }) {
  return (
    <div className="flex items-baseline justify-between py-1 border-b border-dashed border-[var(--color-border)] last:border-none">
      <span className="text-[var(--color-text-muted)]">{k}</span>
      <span
        className="mono font-semibold tabular-nums"
        style={accent ? { color: "var(--color-primary)" } : { color: "var(--color-text)" }}
      >
        {v}
      </span>
    </div>
  );
}
