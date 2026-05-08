import { useEffect, useMemo, useState } from "react";
import { X, Plus } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { EquityChart, type EquityPoint as ChartPoint } from "@/components/EquityChart";
import { LIVE_SYMBOLS } from "@/lib/symbols";
import {
  useBacktestRuns,
  useBacktestStatus,
  useSubmitBacktest,
} from "@/hooks/useBacktest";
import { useBacktestDetail } from "@/hooks/useBacktestDetail";
import { usd, shortDate, num } from "@/lib/format";
import { colors } from "@/lib/tokens";
import type { BacktestRunSummary } from "@/lib/types";

// ─── Helpers ─────────────────────────────────────────────────────────

function pfColor(pf: number): string {
  if (pf >= 3) return "var(--color-profit)";
  if (pf >= 2) return "var(--color-warn)";
  return "var(--color-loss)";
}

function winColor(wr: number): string {
  // wr is 0..1
  if (wr >= 0.55) return "var(--color-profit)";
  if (wr >= 0.45) return "var(--color-warn)";
  return "var(--color-loss)";
}

function sparkPath(up: boolean, strength: number): string {
  // Cheap synthetic sparkline based on direction + magnitude (0..1 clamp).
  // Up = rising with wiggle; flat = slight oscillation; down = falling.
  const s = Math.max(0, Math.min(1, strength));
  if (up) {
    const end = 18 - 14 * s; // stronger = lower Y (higher on chart)
    return `M0 16 L20 ${16 - 2 - s * 3} L40 ${14 - s * 3} L60 ${12 - s * 4} L80 ${end.toFixed(1)}`;
  }
  const end = 4 + 10 * s;
  return `M0 4 L20 ${4 + s * 2} L40 ${8 + s * 2} L60 ${10 + s * 3} L80 ${end.toFixed(1)}`;
}

function StatusChip({ status }: { status: string }) {
  const meta: Record<string, { bg: string; color: string; dot: string }> = {
    done: {
      bg: "rgba(16,185,129,0.12)",
      color: "var(--chip-profit-fg)",
      dot: "var(--color-profit)",
    },
    running: {
      bg: "rgba(6,182,212,0.12)",
      color: "var(--color-primary)",
      dot: "var(--color-primary)",
    },
    pending: {
      bg: "rgba(245,158,11,0.15)",
      color: "var(--color-warn)",
      dot: "var(--color-warn)",
    },
    failed: {
      bg: "rgba(244,63,94,0.12)",
      color: "var(--chip-loss-fg)",
      dot: "var(--color-loss)",
    },
  };
  const m = meta[status] ?? {
    bg: "var(--color-panel-hi)",
    color: "var(--color-text-muted)",
    dot: "var(--color-text-dim)",
  };
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
      style={{ background: m.bg, color: m.color }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{ background: m.dot }}
      />
      {status}
    </span>
  );
}

// ─── Detail drawer ───────────────────────────────────────────────────

function RunDrawer({ runId, onClose }: { runId: string; onClose: () => void }) {
  const { data: detail, isLoading } = useBacktestDetail(runId);
  const [section, setSection] = useState<"summary" | "equity" | "trades">("equity");

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  const chartData: ChartPoint[] = useMemo(() => {
    if (!detail) return [];
    return detail.equity_curve.map((p) => ({
      t: p.bar_timestamp,
      equity: p.equity,
      drawdown: p.drawdown_pct / 100,
    }));
  }, [detail]);

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} aria-hidden />
      <aside
        className="relative w-full max-w-3xl bg-[var(--color-panel)] border-l border-[var(--color-border-hi)] overflow-y-auto p-0 shadow-2xl"
        style={{ boxShadow: "-20px 0 60px rgba(0,0,0,0.5)" }}
      >
        <div className="px-6 py-4 border-b border-[var(--color-border)] flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={onClose}
              className="text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
            >
              ← Backtest
            </button>
            <span className="text-[var(--color-text-dim)]">/</span>
            <span className="mono text-[var(--color-primary)] font-semibold">
              #{runId.slice(0, 8)}
            </span>
            {detail && (
              <>
                <StatusChip status={detail.summary.status ?? "done"} />
                <span className="mono font-semibold">{detail.summary.symbol}</span>
                <span className="text-xs text-[var(--color-text-muted)]">
                  {detail.summary.start_date?.slice(0, 10)} →{" "}
                  {detail.summary.end_date?.slice(0, 10)} · {detail.summary.timeframe}
                </span>
              </>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="flex gap-1">
              {(["summary", "equity", "trades"] as const).map((s) => {
                const active = s === section;
                return (
                  <button
                    key={s}
                    onClick={() => setSection(s)}
                    className="px-3 py-1.5 text-xs rounded capitalize"
                    style={
                      active
                        ? { background: "rgba(6,182,212,0.15)", color: "var(--color-primary)" }
                        : { background: "var(--color-panel-hi)", color: "var(--color-text-muted)" }
                    }
                  >
                    {s}
                  </button>
                );
              })}
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-md hover:bg-[var(--color-panel-hi)] text-[var(--color-text-muted)]"
              aria-label="Close"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {isLoading && (
          <p className="text-sm text-[var(--color-text-muted)] text-center py-12">
            Loading run…
          </p>
        )}
        {detail && (
          <div className="p-6 space-y-6">
            {/* Stat grid (always visible) */}
            <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
              <div>
                <p className="section-label">PnL</p>
                <p
                  className="tnum text-xl font-bold mt-1"
                  style={{
                    color:
                      detail.summary.net_pnl >= 0
                        ? "var(--color-profit)"
                        : "var(--color-loss)",
                  }}
                >
                  {detail.summary.net_pnl >= 0 ? "+" : "−"}
                  ${Math.abs(detail.summary.net_pnl).toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
                </p>
              </div>
              <div>
                <p className="section-label">Trades</p>
                <p className="tnum text-xl font-bold mt-1">{detail.summary.total_trades}</p>
              </div>
              <div>
                <p className="section-label">Win rate</p>
                <p
                  className="tnum text-xl font-bold mt-1"
                  style={{ color: winColor(detail.summary.win_rate) }}
                >
                  {(detail.summary.win_rate * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="section-label">PF</p>
                <p
                  className="tnum text-xl font-bold mt-1"
                  style={{ color: pfColor(detail.summary.profit_factor) }}
                >
                  {detail.summary.profit_factor.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="section-label">Max DD</p>
                <p className="tnum text-xl font-bold mt-1" style={{ color: "var(--color-warn)" }}>
                  −{detail.summary.max_drawdown_pct.toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="section-label">Sharpe</p>
                <p className="tnum text-xl font-bold mt-1">{num(detail.summary.sharpe_ratio)}</p>
              </div>
              <div>
                <p className="section-label">Calmar</p>
                <p
                  className="tnum text-xl font-bold mt-1"
                  title="CAGR / |Max DD| — risk-adjusted return. Empty (—) when DD < 0.5%"
                >
                  {detail.summary.calmar_ratio > 0
                    ? detail.summary.calmar_ratio.toFixed(2)
                    : "—"}
                </p>
              </div>
            </div>

            {section === "summary" && (
              <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel-hi)] p-4 text-sm space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[var(--color-text-muted)]">Model</span>
                  <span className="mono">
                    {detail.summary.model_name ?? `lstm_${detail.summary.symbol}`}
                    {detail.summary.model_version ? ` v${detail.summary.model_version}` : ""}
                  </span>
                </div>
                {detail.summary.model_trained_at && (
                  <div className="flex items-center justify-between">
                    <span className="text-[var(--color-text-muted)]">Model trained</span>
                    <span className="mono">{shortDate(detail.summary.model_trained_at)}</span>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <span className="text-[var(--color-text-muted)]">Mode</span>
                  <span className="mono">{detail.summary.mode ?? "simple"}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[var(--color-text-muted)]">Initial equity</span>
                  <span className="mono">
                    {usd(detail.summary.initial_equity ?? 10000)}
                  </span>
                </div>
              </div>
            )}

            {section === "equity" && chartData.length > 1 && (
              <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel-hi)] p-4">
                <p className="section-label mb-3">Equity curve</p>
                <EquityChart data={chartData} height={260} />
                <p className="text-[11px] text-[var(--color-text-dim)] mt-2">
                  Cyan line · equity · Red line · drawdown from peak.
                </p>
              </div>
            )}

            {section === "trades" && (
              <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel-hi)] overflow-hidden">
                <div className="px-4 py-3 border-b border-[var(--color-border)]">
                  <p className="section-label">Trades · {detail.trades.length}</p>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-[10px] uppercase tracking-[0.14em] text-[var(--color-text-dim)] border-b border-[var(--color-border)]">
                        <th className="text-left px-4 py-2 font-semibold">Exit time</th>
                        <th className="text-left px-2 py-2 font-semibold">Dir</th>
                        <th className="text-right px-2 py-2 font-semibold">Entry → Exit</th>
                        <th className="text-right px-2 py-2 font-semibold">PnL</th>
                        <th className="text-right px-2 py-2 font-semibold">R</th>
                        <th className="text-left px-2 py-2 font-semibold">Reason</th>
                        <th className="text-left px-4 py-2 font-semibold">Regime</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.trades.slice(0, 500).map((t, i) => (
                        <tr
                          key={i}
                          className="border-b border-[var(--color-border)] hover:bg-[var(--color-panel)]"
                        >
                          <td className="px-4 py-2 mono text-[var(--color-text-muted)]">
                            {shortDate(t.exit_time)}
                          </td>
                          <td className="px-2 py-2">
                            <span
                              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px]"
                              style={{
                                background:
                                  t.direction === "buy"
                                    ? "rgba(16,185,129,0.15)"
                                    : "rgba(244,63,94,0.15)",
                                color:
                                  t.direction === "buy"
                                    ? "var(--chip-profit-fg)"
                                    : "var(--chip-loss-fg)",
                              }}
                            >
                              {t.direction === "buy" ? "▲" : "▼"} {t.direction?.toUpperCase()}
                            </span>
                          </td>
                          <td className="px-2 py-2 text-right mono">
                            {num(t.entry_price)}{" "}
                            <span className="text-[var(--color-text-dim)]">→</span>{" "}
                            {num(t.exit_price)}
                          </td>
                          <td
                            className="px-2 py-2 text-right mono font-semibold"
                            style={{
                              color: t.pnl >= 0 ? "var(--color-profit)" : "var(--color-loss)",
                            }}
                          >
                            {t.pnl >= 0 ? "+" : "−"}${Math.abs(t.pnl).toFixed(0)}
                          </td>
                          <td
                            className="px-2 py-2 text-right mono"
                            style={{
                              color: t.r_multiple >= 0 ? "var(--color-profit)" : "var(--color-loss)",
                            }}
                          >
                            {t.r_multiple >= 0 ? "+" : ""}{t.r_multiple.toFixed(2)}R
                          </td>
                          <td className="px-2 py-2 mono text-[var(--color-text-muted)]">
                            {t.exit_reason}
                          </td>
                          <td className="px-4 py-2 text-[var(--color-text-muted)]">
                            {t.regime_label ?? "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {detail.trades.length > 500 && (
                  <p className="px-4 py-2 text-[11px] text-[var(--color-text-dim)]">
                    Showing first 500 of {detail.trades.length}
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

// ─── Compare view ────────────────────────────────────────────────────

const COMPARE_COLORS = [colors.primary, colors.profit, colors.warn];

function CompareView({
  runIds,
  runs,
  onClear,
}: {
  runIds: string[];
  runs: BacktestRunSummary[];
  onClear: () => void;
}) {
  const d1 = useBacktestDetail(runIds[0] ?? null);
  const d2 = useBacktestDetail(runIds[1] ?? null);
  const d3 = useBacktestDetail(runIds[2] ?? null);
  const details = [d1, d2, d3];

  const merged = useMemo(() => {
    const map = new Map<string, Record<string, number | string>>();
    details.forEach((q, idx) => {
      if (!q.data) return;
      for (const p of q.data.equity_curve) {
        const key = p.bar_timestamp;
        const row = map.get(key) ?? { t: key };
        row[`run${idx}`] = p.equity;
        map.set(key, row);
      }
    });
    return Array.from(map.values()).sort((a, b) =>
      String(a.t).localeCompare(String(b.t)),
    );
  }, [details]);

  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="section-label">Compare</p>
          <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
            {runIds.length} of 3 selected
          </p>
        </div>
        <button
          onClick={onClear}
          className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
        >
          Clear selection
        </button>
      </div>

      {merged.length > 1 && (
        <div className="h-[260px] mb-4">
          <ResponsiveContainer>
            <LineChart data={merged} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid stroke={colors.border} strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="t" stroke={colors.textDim} tick={{ fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={48} />
              <YAxis stroke={colors.textDim} tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={52} />
              <Tooltip
                contentStyle={{ background: colors.panelHi, border: `1px solid ${colors.border}`, borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: colors.textMuted }}
                itemStyle={{ color: colors.text }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {runIds.map((id, i) => (
                <Line
                  key={id}
                  type="monotone"
                  dataKey={`run${i}`}
                  name={runs.find((r) => r.id === id)?.symbol ?? id.slice(0, 8)}
                  stroke={COMPARE_COLORS[i]}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-[10px] uppercase tracking-[0.14em] text-[var(--color-text-dim)] border-b border-[var(--color-border)]">
              <th className="text-left px-2 py-2 font-semibold">Metric</th>
              {runIds.map((id, i) => {
                const run = runs.find((r) => r.id === id);
                return (
                  <th
                    key={id}
                    className="text-right px-2 py-2 font-semibold"
                    style={{ color: COMPARE_COLORS[i] }}
                  >
                    {run?.symbol ?? id.slice(0, 8)}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {METRIC_ROWS.map(({ label, accessor, fmt }) => (
              <tr key={label} className="border-b border-[var(--color-border)]">
                <td className="px-2 py-2 text-[var(--color-text-muted)]">{label}</td>
                {runIds.map((id) => {
                  const run = runs.find((r) => r.id === id);
                  if (!run) return <td key={id} />;
                  return (
                    <td key={id} className="px-2 py-2 text-right mono">
                      {fmt(accessor(run))}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const METRIC_ROWS: {
  label: string;
  accessor: (r: BacktestRunSummary) => number;
  fmt: (v: number) => string;
}[] = [
  { label: "Net PnL", accessor: (r) => r.net_pnl, fmt: (v) => usd(v) },
  { label: "Win rate", accessor: (r) => r.win_rate, fmt: (v) => `${(v * 100).toFixed(1)}%` },
  { label: "Profit factor", accessor: (r) => r.profit_factor, fmt: (v) => v.toFixed(2) },
  { label: "Sharpe", accessor: (r) => r.sharpe_ratio, fmt: (v) => num(v, 3) },
  { label: "Calmar", accessor: (r) => r.calmar_ratio, fmt: (v) => (v > 0 ? v.toFixed(2) : "—") },
  { label: "Max DD", accessor: (r) => r.max_drawdown_pct, fmt: (v) => `${num(v)}%` },
  { label: "Trades", accessor: (r) => r.total_trades, fmt: (v) => String(v) },
  {
    label: "Deflated Sharpe",
    accessor: (r) => (r.deflated_sharpe == null ? NaN : r.deflated_sharpe),
    fmt: (v) => (Number.isNaN(v) ? "—" : v.toFixed(3)),
  },
  {
    label: "Sharpe stability",
    accessor: (r) => (r.sharpe_stability == null ? NaN : r.sharpe_stability),
    fmt: (v) => (Number.isNaN(v) ? "—" : `${(v * 100).toFixed(0)}%`),
  },
];

// ─── Submit form (collapsible "+ New run" panel) ─────────────────────

function NewRunForm({
  onSubmitted,
  onCancel,
}: {
  onSubmitted: (runId: string) => void;
  onCancel: () => void;
}) {
  const submit = useSubmitBacktest();
  const [symbol, setSymbol] = useState("XAUUSD");
  const [timeframe, setTimeframe] = useState("H4");
  const [startDate, setStartDate] = useState("2025-01-01");
  const [endDate, setEndDate] = useState("2025-12-31");
  const [equity, setEquity] = useState(10000);

  function handle(e: React.FormEvent) {
    e.preventDefault();
    submit.mutate(
      { symbol, timeframe, start_date: startDate, end_date: endDate, initial_equity: equity },
      { onSuccess: (resp) => onSubmitted(resp.run_id) },
    );
  }

  return (
    <form
      onSubmit={handle}
      className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel-hi)] p-4 flex flex-wrap gap-3 items-end"
    >
      <Field label="Symbol">
        <select
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          className="bg-[var(--color-panel)] border border-[var(--color-border)] rounded px-3 py-1.5 text-sm"
        >
          {LIVE_SYMBOLS.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
      </Field>
      <Field label="Timeframe">
        <select
          value={timeframe}
          onChange={(e) => setTimeframe(e.target.value)}
          className="bg-[var(--color-panel)] border border-[var(--color-border)] rounded px-3 py-1.5 text-sm"
        >
          {["H1", "H4", "D1"].map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
      </Field>
      <Field label="Start">
        <input
          type="date"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          className="bg-[var(--color-panel)] border border-[var(--color-border)] rounded px-3 py-1.5 text-sm"
        />
      </Field>
      <Field label="End">
        <input
          type="date"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          className="bg-[var(--color-panel)] border border-[var(--color-border)] rounded px-3 py-1.5 text-sm"
        />
      </Field>
      <Field label="Equity ($)">
        <input
          type="number"
          value={equity}
          onChange={(e) => setEquity(Number(e.target.value))}
          min={1000}
          step={1000}
          className="bg-[var(--color-panel)] border border-[var(--color-border)] rounded px-3 py-1.5 text-sm w-28 mono"
        />
      </Field>
      <button
        type="submit"
        disabled={submit.isPending}
        className="px-4 py-1.5 text-sm rounded-lg text-white font-semibold bg-brand-gradient hover:brightness-110 disabled:opacity-50"
      >
        {submit.isPending ? "Submitting…" : "Run"}
      </button>
      <button
        type="button"
        onClick={onCancel}
        className="px-3 py-1.5 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
      >
        Cancel
      </button>
      {submit.isError && (
        <p className="w-full text-xs text-[var(--color-loss)]">
          {submit.error.message}
        </p>
      )}
    </form>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[11px] text-[var(--color-text-muted)] mb-1">{label}</label>
      {children}
    </div>
  );
}

// ─── Main screen ─────────────────────────────────────────────────────

export function Backtest() {
  const { data: runs, isLoading: runsLoading } = useBacktestRuns();
  const [filter, setFilter] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [drawerId, setDrawerId] = useState<string | null>(null);
  const [submittedId, setSubmittedId] = useState<string | null>(null);
  const { data: submittedStatus } = useBacktestStatus(submittedId);
  const [compareIds, setCompareIds] = useState<string[]>([]);

  const filtered = useMemo(() => {
    if (!runs) return [] as BacktestRunSummary[];
    const q = filter.trim().toUpperCase();
    if (!q) return runs.runs;
    return runs.runs.filter((r) => r.symbol.toUpperCase().includes(q));
  }, [runs, filter]);

  const toggleCompare = (id: string) => {
    setCompareIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 3) return prev;
      return [...prev, id];
    });
  };

  return (
    <div className="space-y-4">
      <header className="flex items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[var(--color-text)]">Backtest</h1>
          <p className="text-xs text-[var(--color-text-dim)] mt-0.5">
            Submit runs · drill in · compare up to 3
          </p>
        </div>
      </header>

      {showForm && (
        <NewRunForm
          onSubmitted={(id) => {
            setSubmittedId(id);
            setShowForm(false);
          }}
          onCancel={() => setShowForm(false)}
        />
      )}

      {submittedStatus && submittedStatus.run.status !== "done" && (
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-4 flex items-center gap-3">
          <StatusChip status={submittedStatus.run.status} />
          <span className="text-sm text-[var(--color-text-muted)] mono">
            #{submittedStatus.run.id.slice(0, 8)}
          </span>
          {submittedStatus.run.status === "failed" && submittedStatus.error_message && (
            <span className="text-xs text-[var(--color-loss)]">
              {submittedStatus.error_message}
            </span>
          )}
          {(submittedStatus.run.status === "pending" ||
            submittedStatus.run.status === "running") && (
            <div className="h-3 w-3 border-2 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
          )}
        </div>
      )}

      {compareIds.length > 0 && runs && (
        <CompareView
          runIds={compareIds}
          runs={runs.runs}
          onClear={() => setCompareIds([])}
        />
      )}

      {/* Runs table card */}
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] overflow-hidden">
        <div className="px-5 py-4 border-b border-[var(--color-border)] flex items-center justify-between gap-3 flex-wrap">
          <div>
            <h3 className="text-base font-semibold">
              Recent runs · {runs?.runs.length ?? 0} total
            </h3>
            <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
              Check up to 3 boxes to compare · click any ID to open detail
            </p>
          </div>
          <div className="flex items-center gap-2">
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter by symbol…"
              className="px-3 py-1.5 text-xs rounded-lg bg-[var(--color-panel-hi)] border border-[var(--color-border)] focus:outline-none focus:border-[var(--color-primary)]"
            />
            <button
              onClick={() => setShowForm((v) => !v)}
              className="inline-flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg text-white font-medium bg-brand-gradient hover:brightness-110"
            >
              <Plus size={12} /> New run
            </button>
          </div>
        </div>

        {runsLoading ? (
          <p className="text-center text-sm text-[var(--color-text-muted)] py-10">
            Loading runs…
          </p>
        ) : filtered.length === 0 ? (
          <p className="text-center text-sm text-[var(--color-text-dim)] py-10">
            {filter ? `No runs match "${filter}"` : "No backtest runs yet"}
          </p>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-[10px] uppercase tracking-[0.14em] text-[var(--color-text-dim)] border-b border-[var(--color-border)]">
                    <th className="text-left px-5 py-3 font-semibold"></th>
                    <th className="text-left px-2 py-3 font-semibold">ID</th>
                    <th className="text-left px-2 py-3 font-semibold">Status</th>
                    <th className="text-left px-2 py-3 font-semibold">Symbol</th>
                    <th className="text-left px-2 py-3 font-semibold">Period</th>
                    <th className="text-right px-2 py-3 font-semibold">Trades</th>
                    <th className="text-right px-2 py-3 font-semibold">PnL</th>
                    <th className="text-right px-2 py-3 font-semibold">Win rate</th>
                    <th className="text-right px-2 py-3 font-semibold">PF</th>
                    <th className="text-right px-2 py-3 font-semibold">Calmar</th>
                    <th className="text-right px-2 py-3 font-semibold">Equity</th>
                    <th className="text-right px-5 py-3 font-semibold">Created</th>
                  </tr>
                </thead>
                <tbody className="mono">
                  {filtered.map((r) => {
                    const checked = compareIds.includes(r.id);
                    const atLimit = !checked && compareIds.length >= 3;
                    const canOpen = r.status === "done";
                    const sparkUp = r.net_pnl > 0;
                    const strength = Math.max(
                      0.15,
                      Math.min(1, Math.abs(r.net_pnl) / 5000),
                    );
                    return (
                      <tr
                        key={r.id}
                        className="border-b border-[var(--color-border)] hover:bg-[var(--color-panel-hi)] transition-colors"
                      >
                        <td className="px-5 py-3">
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={atLimit || r.status !== "done"}
                            onChange={() => toggleCompare(r.id)}
                            className="accent-[var(--color-primary)] cursor-pointer disabled:cursor-not-allowed"
                          />
                        </td>
                        <td
                          onClick={() => canOpen && setDrawerId(r.id)}
                          className={`px-2 py-3 ${
                            canOpen
                              ? "text-[var(--color-primary)] hover:brightness-125 cursor-pointer"
                              : "text-[var(--color-text-dim)]"
                          }`}
                        >
                          {r.id.slice(0, 8)}
                        </td>
                        <td className="px-2 py-3">
                          <StatusChip status={r.status} />
                        </td>
                        <td className="px-2 py-3 font-semibold">{r.symbol}</td>
                        <td className="px-2 py-3 text-xs text-[var(--color-text-muted)]">
                          {r.start_date?.slice(0, 10)} → {r.end_date?.slice(0, 10)}
                        </td>
                        <td className="px-2 py-3 text-right">{r.total_trades}</td>
                        <td
                          className="px-2 py-3 text-right font-semibold"
                          style={{
                            color:
                              r.net_pnl >= 0
                                ? "var(--color-profit)"
                                : "var(--color-loss)",
                          }}
                        >
                          {r.net_pnl >= 0 ? "+" : "−"}${Math.abs(r.net_pnl).toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
                        </td>
                        <td className="px-2 py-3 text-right">
                          <span style={{ color: winColor(r.win_rate) }}>
                            {(r.win_rate * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-2 py-3 text-right">
                          <span style={{ color: pfColor(r.profit_factor), fontWeight: 600 }}>
                            {r.profit_factor.toFixed(2)}
                          </span>
                        </td>
                        <td className="px-2 py-3 text-right">
                          {r.calmar_ratio > 0 ? (
                            <span style={{ color: pfColor(r.calmar_ratio), fontWeight: 600 }}>
                              {r.calmar_ratio.toFixed(2)}
                            </span>
                          ) : (
                            <span className="text-[var(--color-text-dim)]">—</span>
                          )}
                        </td>
                        <td className="px-2 py-3 text-right">
                          <svg width="80" height="20" viewBox="0 0 80 20" className="inline-block">
                            <path
                              d={sparkPath(sparkUp, strength)}
                              fill="none"
                              stroke={
                                sparkUp ? "var(--color-profit)" : "var(--color-loss)"
                              }
                              strokeWidth={1.6}
                              strokeLinecap="round"
                            />
                          </svg>
                        </td>
                        <td className="px-5 py-3 text-right text-xs text-[var(--color-text-muted)]">
                          {shortDate(r.created_at)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="px-5 py-3 border-t border-[var(--color-border)] flex items-center justify-between text-xs flex-wrap gap-2">
              <span className="text-[var(--color-text-muted)]">
                Showing {filtered.length} of {runs?.runs.length ?? 0}
              </span>
              <p className="text-[10px] text-[var(--color-text-muted)]">
                Win rate ·{" "}
                <span style={{ color: "var(--color-profit)" }}>≥55% green</span> ·{" "}
                <span style={{ color: "var(--color-warn)" }}>45–55% amber</span> ·{" "}
                <span style={{ color: "var(--color-loss)" }}>&lt;45% red</span>
                &nbsp;·&nbsp;PF ·{" "}
                <span style={{ color: "var(--color-profit)" }}>≥3 green</span> ·{" "}
                <span style={{ color: "var(--color-warn)" }}>2–3 amber</span> ·{" "}
                <span style={{ color: "var(--color-loss)" }}>&lt;2 red</span>
              </p>
            </div>
          </>
        )}
      </div>

      {drawerId && <RunDrawer runId={drawerId} onClose={() => setDrawerId(null)} />}
    </div>
  );
}
