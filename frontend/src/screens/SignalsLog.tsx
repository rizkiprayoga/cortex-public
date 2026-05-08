import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useSignalAudit } from "@/hooks/useSignalAudit";
import { shortDate } from "@/lib/format";
import { regimeColor } from "@/lib/tokens";
import type { SignalAuditItem } from "@/lib/types";
import { SkeletonTableRow } from "@/components/states/Skeleton";
import { SYMBOL_FILTERS as SYMBOLS } from "@/lib/symbols";

const OUTCOMES: Array<{ label: string; value?: string; kind?: "all" | "executed" }> = [
  { label: "All", kind: "all" },
  { label: "✓ Approved", kind: "executed" },
  { label: "Combiner", value: "combiner_rejected" },
  { label: "News", value: "news_blackout" },
  { label: "Sizing", value: "sizing" },
  { label: "Broker", value: "broker_reject" },
  { label: "Trade disabled", value: "trade_disabled" },
];

const PAGE_SIZE = 50;

function FilterPill({
  label,
  active,
  onClick,
  tone = "neutral",
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  tone?: "neutral" | "success";
}) {
  const activeBg =
    tone === "success"
      ? { background: "rgba(16,185,129,0.12)", color: "var(--color-profit)" }
      : { background: "rgba(6,182,212,0.15)", color: "var(--color-primary)" };
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] transition-colors"
      style={
        active
          ? activeBg
          : { background: "var(--color-panel-hi)", color: "var(--color-text-muted)" }
      }
    >
      {label}
    </button>
  );
}

function OutcomeChip({ row }: { row: SignalAuditItem }) {
  if (row.executed) {
    return (
      <span
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
        style={{ background: "rgba(16,185,129,0.15)", color: "var(--color-profit)" }}
      >
        ✓ {row.direction?.toUpperCase() ?? "BUY"}
      </span>
    );
  }
  const reason = row.block_reason ?? "";
  if (/news_blackout|blackout|news/i.test(reason)) {
    return (
      <span
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
        style={{ background: "rgba(245,158,11,0.15)", color: "var(--color-warn)" }}
      >
        ⚠ blocked
      </span>
    );
  }
  if (/^broker_reject|^broker\b/i.test(reason)) {
    return (
      <span
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
        style={{ background: "rgba(244,63,94,0.15)", color: "var(--color-loss)" }}
      >
        ✗ broker
      </span>
    );
  }
  if (/sizing|volume|margin|exposure/i.test(reason)) {
    return (
      <span
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
        style={{ background: "rgba(139,92,246,0.15)", color: "var(--violet)" }}
      >
        ✗ sizing
      </span>
    );
  }
  if (/trade_disabled|paused|halt|breaker/i.test(reason)) {
    return (
      <span
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
        style={{ background: "rgba(244,63,94,0.15)", color: "var(--color-loss)" }}
      >
        ⏸ halt
      </span>
    );
  }
  if (/combiner|threshold|regime|long_only|direction/i.test(reason)) {
    return (
      <span
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
        style={{ background: "rgba(148,163,184,0.15)", color: "var(--color-text-muted)" }}
      >
        · flat
      </span>
    );
  }
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
      style={{ background: "rgba(148,163,184,0.15)", color: "var(--color-text-muted)" }}
    >
      · flat
    </span>
  );
}

export function SignalsLog() {
  const [symbol, setSymbol] = useState<string | undefined>(undefined);
  const [outcomeKey, setOutcomeKey] = useState<string>("all"); // "all" | "executed" | block_reason value
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);

  const executedFilter = outcomeKey === "executed" ? true : undefined;
  const blockReasonFilter =
    outcomeKey === "all" || outcomeKey === "executed" ? undefined : outcomeKey;

  const { data, isLoading, error } = useSignalAudit({
    symbol,
    executed: executedFilter,
    blockReason: blockReasonFilter,
    page,
    pageSize: PAGE_SIZE,
  });

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  const visibleRows = useMemo(() => {
    const rows = data?.items ?? [];
    if (!search.trim()) return rows;
    const q = search.trim().toLowerCase();
    return rows.filter(
      (r) =>
        (r.reasoning ?? "").toLowerCase().includes(q) ||
        (r.block_reason ?? "").toLowerCase().includes(q),
    );
  }, [data, search]);

  return (
    <div className="space-y-4">
      <header className="flex items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[var(--color-text)]">Signals Log</h1>
          <p className="text-xs text-[var(--color-text-dim)] mt-0.5">
            Every signal decision, approved or blocked — reason-coded audit trail
          </p>
        </div>
      </header>

      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] overflow-hidden">
        <div className="px-5 py-4 border-b border-[var(--color-border)]">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div>
              <h3 className="text-base font-semibold text-[var(--color-text)]">
                {data ? `${data.total.toLocaleString()} decisions` : "loading…"} · newest first
              </h3>
              <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                Source: <span className="mono">signal_audit.csv</span>
              </p>
            </div>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search reasoning…"
              className="px-3 py-1.5 text-xs rounded-lg bg-[var(--color-panel-hi)] border border-[var(--color-border)] focus:outline-none focus:border-[var(--color-primary)] w-64"
            />
          </div>
          <div className="flex items-center gap-2 mt-4 flex-wrap">
            <span className="section-label mr-2">Symbol</span>
            {SYMBOLS.map((s) => (
              <FilterPill
                key={s}
                label={s}
                active={(s === "All" && !symbol) || s === symbol}
                onClick={() => {
                  setSymbol(s === "All" ? undefined : s);
                  setPage(1);
                }}
              />
            ))}
            <span className="w-px h-4 bg-[var(--color-border)] mx-2" />
            <span className="section-label mr-2">Outcome</span>
            {OUTCOMES.map((o) => {
              const key = o.kind ?? o.value ?? o.label;
              return (
                <FilterPill
                  key={o.label}
                  label={o.label}
                  active={outcomeKey === key}
                  tone={o.kind === "executed" ? "success" : "neutral"}
                  onClick={() => {
                    setOutcomeKey(key);
                    setPage(1);
                  }}
                />
              );
            })}
          </div>
        </div>

        {isLoading ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <tbody>
                {Array.from({ length: 8 }).map((_, i) => (
                  <SkeletonTableRow key={i} cols={7} />
                ))}
              </tbody>
            </table>
          </div>
        ) : error ? (
          <p className="text-center text-sm text-[var(--color-loss)] py-10">
            Failed to load signal audit.
          </p>
        ) : visibleRows.length === 0 ? (
          <p className="text-center text-sm text-[var(--color-text-dim)] py-10">
            No signal audit rows match the current filter.
          </p>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-[10px] uppercase tracking-[0.14em] text-[var(--color-text-dim)] border-b border-[var(--color-border)]">
                    <th className="text-left px-5 py-3 font-semibold">Time</th>
                    <th className="text-left px-2 py-3 font-semibold">Symbol</th>
                    <th className="text-left px-2 py-3 font-semibold">Outcome</th>
                    <th className="text-right px-2 py-3 font-semibold">Score</th>
                    <th className="text-left px-2 py-3 font-semibold">Regime</th>
                    <th className="text-left px-2 py-3 font-semibold w-1/3">Reasoning</th>
                    <th className="text-left px-5 py-3 font-semibold">Block reason</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleRows.map((row, i) => {
                    const score = row.combined_score ?? 0;
                    const scoreCol =
                      score > 0.1
                        ? "var(--color-profit)"
                        : score < -0.1
                          ? "var(--color-loss)"
                          : "var(--color-text-muted)";
                    const regimeCol = row.regime
                      ? regimeColor(row.regime)
                      : "var(--color-text-muted)";
                    return (
                      <tr
                        key={`${row.timestamp}-${row.symbol}-${i}`}
                        className="border-b border-[var(--color-border)] hover:bg-[var(--color-panel-hi)] transition-colors"
                      >
                        <td className="px-5 py-3 mono text-xs text-[var(--color-text-muted)]">
                          {shortDate(row.timestamp)}
                        </td>
                        <td className="px-2 py-3 mono font-semibold">
                          <Link
                            to={`/ui/signals/${row.symbol}`}
                            className="hover:text-[var(--color-primary)]"
                          >
                            {row.symbol}
                          </Link>
                        </td>
                        <td className="px-2 py-3">
                          <OutcomeChip row={row} />
                        </td>
                        <td className="px-2 py-3 text-right mono" style={{ color: scoreCol }}>
                          {row.combined_score != null
                            ? `${row.combined_score > 0 ? "+" : row.combined_score < 0 ? "−" : ""}${Math.abs(
                                row.combined_score,
                              ).toFixed(2)}`
                            : "—"}
                        </td>
                        <td className="px-2 py-3 text-xs" style={{ color: regimeCol }}>
                          {row.regime ?? "—"}
                          {row.regime_prob != null && (
                            <span className="text-[var(--color-text-dim)] ml-1 mono">
                              {(row.regime_prob * 100).toFixed(0)}%
                            </span>
                          )}
                        </td>
                        <td className="px-2 py-3 text-xs text-[var(--color-text-muted)]" title={row.reasoning ?? ""}>
                          {row.reasoning
                            ? row.reasoning.length > 120
                              ? row.reasoning.slice(0, 117) + "…"
                              : row.reasoning
                            : "—"}
                        </td>
                        <td className="px-5 py-3 mono text-[11px]">
                          <span
                            style={{
                              color: row.executed
                                ? "var(--color-profit)"
                                : /news|blackout/i.test(row.block_reason ?? "")
                                  ? "var(--color-warn)"
                                  : /^broker_reject|^broker\b/i.test(row.block_reason ?? "")
                                    ? "var(--color-loss)"
                                    : "var(--color-text-dim)",
                            }}
                          >
                            {row.block_reason ?? (row.executed ? "—" : "—")}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="px-5 py-3 border-t border-[var(--color-border)] flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">
                Showing {visibleRows.length} of {data?.total ?? 0}
                {search ? ` · filtered by "${search}"` : ""}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  className="px-2.5 py-1 rounded bg-[var(--color-panel-hi)] text-[var(--color-text-muted)] disabled:opacity-40"
                >
                  ←
                </button>
                <span className="mono">
                  Page {page} of {totalPages}
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                  className="px-2.5 py-1 rounded bg-[var(--color-panel-hi)] text-[var(--color-text-muted)] disabled:opacity-40"
                >
                  →
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

