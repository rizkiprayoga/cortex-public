import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  useAccuracyTimeSeries,
  useModelSummary,
  useModelVersionHistory,
} from "@/hooks/useModels";
import { shortDate, num } from "@/lib/format";
import { colors } from "@/lib/tokens";
import type { ModelSummaryRow } from "@/lib/types";
import { LIVE_SYMBOLS } from "@/lib/symbols";

// Distinct colors per the universe sweep sprint winner pair so the directional-accuracy
// chart can be visually decoded. Existing live-pair colors preserved.
const SERIES_COLORS: Record<string, string> = {
  XAUUSD: "var(--color-primary)",   // gold/cyan
  EURUSD: "var(--color-profit)",    // green (legacy — still in routes/news)
  USDJPY: "var(--color-warn)",      // yellow
  USDCAD: "var(--color-loss)",      // pink
  ETHUSD: "#8b5cf6",                // purple (legacy)
  // the universe sweep sprint the trading universe pair set
  GBPUSD: "#3b82f6",                // blue
  NZDUSD: "#10b981",                // emerald (the trading universe — replaces AUDUSD)
  USDCHF: "#ef4444",                // red
  GBPCHF: "#f97316",                // orange
  EURAUD: "#a855f7",                // violet
  GBPAUD: "#14b8a6",                // teal
  EURJPY: "#6366f1",                // indigo (the trading universe — replaces GBPJPY)
  // Legacy mappings retained so historical signals/trades still color-code
  AUDUSD: "#84cc16",                // lime
  GBPJPY: "#ec4899",                // pink
};
const MIN_PREDICTIONS = 50;
const HEALTHY = 0.52;
const WARN = 0.48;

function daysAgo(iso: string | null): string {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const hours = (Date.now() - then) / 3_600_000;
  if (hours < 1) return "just now";
  if (hours < 24) return `${hours.toFixed(0)}h ago`;
  const days = hours / 24;
  if (days < 30) return `${days.toFixed(0)}d ago`;
  return shortDate(iso);
}

function statusMeta(dirAcc: number | null, warming: boolean) {
  if (warming) {
    return {
      label: "warming",
      bg: "var(--color-panel-hi)",
      color: "var(--color-text-muted)",
      accColor: "var(--color-text-dim)",
    };
  }
  if (dirAcc == null) {
    return {
      label: "stale",
      bg: "rgba(245,158,11,0.16)",
      color: "var(--chip-warn-fg)",
      accColor: "var(--color-text-dim)",
    };
  }
  if (dirAcc >= HEALTHY) {
    return {
      label: "live",
      bg: "rgba(16,185,129,0.16)",
      color: "var(--chip-profit-fg)",
      accColor: "var(--color-profit)",
    };
  }
  if (dirAcc >= WARN) {
    return {
      label: "stale",
      bg: "rgba(245,158,11,0.16)",
      color: "var(--chip-warn-fg)",
      accColor: "var(--color-warn)",
    };
  }
  return {
    label: "dead",
    bg: "rgba(244,63,94,0.16)",
    color: "var(--chip-loss-fg)",
    accColor: "var(--color-loss)",
  };
}

// ─── Per-symbol card ─────────────────────────────────────────────────

function SymbolCard({ row }: { row: ModelSummaryRow }) {
  const dirAcc = row.live_dir_acc;
  const warming = row.n_predictions < MIN_PREDICTIONS;
  const mtime = row.lstm_trained_at ?? row.lstm_file_mtime;
  const meta = statusMeta(dirAcc, warming);
  const head = row.lstm_head ?? "regression";
  const mae = row.live_mae != null ? num(row.live_mae, 3) : "—";
  // Color the "retrained …" line by staleness: >45d red, >30d amber,
  // else dim. Monthly cron → anything older than 30d suggests a miss.
  const ageDays = mtime
    ? (Date.now() - new Date(mtime).getTime()) / 86_400_000
    : null;
  const staleColor =
    ageDays == null
      ? "var(--color-text-dim)"
      : ageDays > 45
        ? "var(--color-loss)"
        : ageDays > 30
          ? "var(--color-warn)"
          : "var(--color-text-dim)";

  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
      <div className="flex items-center justify-between">
        <span className="mono text-[11px] font-semibold">{row.symbol}</span>
        <span
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
          style={{ background: meta.bg, color: meta.color }}
        >
          ● {meta.label}
        </span>
      </div>
      <p className="text-[10px] mt-0.5" style={{ color: staleColor }}>
        {row.lstm_version != null ? `v${row.lstm_version} · ` : ""}retrained {daysAgo(mtime)}
      </p>
      {row.drift_status != null && (
        <p
          className="text-[10px] mt-0.5"
          title={
            row.drift_psi_max != null
              ? `PSI ${row.drift_psi_max.toFixed(3)}${row.drift_worst_feature ? ` · worst ${row.drift_worst_feature}` : ""}`
              : "drift check pending"
          }
          style={{
            color:
              row.drift_status === "alert"
                ? "var(--color-loss)"
                : row.drift_status === "warn"
                ? "var(--color-warn)"
                : "var(--color-text-dim)",
          }}
        >
          drift {row.drift_status}
          {row.drift_psi_max != null ? ` · PSI ${row.drift_psi_max.toFixed(2)}` : ""}
        </p>
      )}
      <p
        className="tnum text-4xl font-bold mt-2"
        style={{ color: meta.accColor }}
      >
        {warming || dirAcc == null ? "—" : `${(dirAcc * 100).toFixed(1)}%`}
      </p>
      <p className="text-[11px] text-[var(--color-text-muted)] mt-1">
        {warming ? `warming up (${row.n_predictions}/${MIN_PREDICTIONS})` : `dir acc · ${row.n_predictions} preds`}
      </p>
      <div className="mt-3 pt-3 border-t border-[var(--color-border)] space-y-0.5">
        <p className="mono text-[10px] text-[var(--color-text-dim)]">
          head: {head}
          {!warming && row.live_mae != null ? ` · MAE ${mae}` : ""}
        </p>
        <p className="mono text-[10px] text-[var(--color-text-dim)]">
          train val_loss {row.lstm_val_loss != null ? num(row.lstm_val_loss, 4) : "—"}
          {row.lstm_train_dir_acc != null
            ? ` · train dir ${(row.lstm_train_dir_acc * 100).toFixed(1)}%`
            : ""}
        </p>
      </div>
    </div>
  );
}

// ─── Accuracy overlay (30-day, per-symbol) ───────────────────────────

function AccuracyOverlay({ symbols }: { symbols: readonly string[] }) {
  const queries = [
    useAccuracyTimeSeries(symbols[0] ?? "", 30),
    useAccuracyTimeSeries(symbols[1] ?? "", 30),
    useAccuracyTimeSeries(symbols[2] ?? "", 30),
    useAccuracyTimeSeries(symbols[3] ?? "", 30),
    useAccuracyTimeSeries(symbols[4] ?? "", 30),
  ];

  const merged = useMemo(() => {
    const map = new Map<string, Record<string, number | string>>();
    queries.forEach((q, i) => {
      const sym = symbols[i];
      if (!q.data || !sym) return;
      for (const pt of q.data.points) {
        const row = map.get(pt.date) ?? { date: pt.date };
        row[sym] = Math.round(pt.directional_accuracy * 1000) / 1000;
        map.set(pt.date, row);
      }
    });
    return Array.from(map.values()).sort((a, b) =>
      String(a.date).localeCompare(String(b.date)),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queries.map((q) => q.dataUpdatedAt).join("|")]);

  if (merged.length === 0) {
    return (
      <div className="h-[260px] flex items-center justify-center text-sm text-[var(--color-text-dim)]">
        No accuracy data yet — bot needs more predictions with confirmed outcomes.
      </div>
    );
  }

  return (
    <div className="h-[260px]">
      <ResponsiveContainer>
        <LineChart data={merged} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={colors.border} strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="date"
            stroke={colors.textDim}
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            minTickGap={32}
          />
          <YAxis
            domain={[0.4, 0.65]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            stroke={colors.textDim}
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={44}
          />
          <Tooltip
            contentStyle={{
              background: colors.panelHi,
              border: `1px solid ${colors.border}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: colors.textMuted }}
            itemStyle={{ color: colors.text }}
            formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
          />
          <ReferenceLine
            y={0.52}
            stroke="var(--color-profit)"
            strokeDasharray="4 4"
            strokeOpacity={0.6}
            label={{ value: "52% healthy", position: "insideTopLeft", fill: "var(--color-profit)", fontSize: 9 }}
          />
          <ReferenceLine
            y={0.48}
            stroke="var(--color-warn)"
            strokeDasharray="4 4"
            strokeOpacity={0.6}
            label={{ value: "48% warn", position: "insideBottomLeft", fill: "var(--color-warn)", fontSize: 9 }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {symbols.map((sym) => (
            <Line
              key={sym}
              type="monotone"
              dataKey={sym}
              stroke={SERIES_COLORS[sym] ?? colors.primary}
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Retrain history (compact list) ──────────────────────────────────

function RetrainHistory() {
  const [model, setModel] = useState<string>("lstm_XAUUSD");
  const { data } = useModelVersionHistory(model, 10);

  const options = useMemo(() => {
    const names: string[] = [];
    for (const sym of LIVE_SYMBOLS) {
      names.push(`lstm_${sym}`);
      names.push(`hmm_${sym}`);
    }
    return names;
  }, []);

  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="section-label">Retrain history · last 10</p>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="bg-[var(--color-panel-hi)] border border-[var(--color-border)] rounded px-2 py-1 text-xs mono"
        >
          {options.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </div>
      {!data || data.versions.length === 0 ? (
        <p className="text-sm text-[var(--color-text-dim)] text-center py-6">
          No version history yet. Training scripts will populate this on next retrain.
        </p>
      ) : (
        <div className="space-y-1">
          {data.versions.map((v, idx) => {
            const active = idx === 0;
            const valLoss = v.val_loss;
            const valLossCol =
              valLoss == null
                ? "var(--color-text-dim)"
                : valLoss < 0.005
                  ? "var(--color-profit)"
                  : valLoss < 0.007
                    ? "var(--color-warn)"
                    : "var(--color-loss)";
            return (
              <div
                key={v.version}
                className={`flex items-center gap-3 p-2 rounded hover:bg-[var(--color-panel-hi)] transition-colors ${
                  active ? "bg-[var(--color-panel-hi)]/50" : ""
                }`}
              >
                <span
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
                  style={{ background: "rgba(99,102,241,0.12)", color: "var(--chip-info-fg)" }}
                >
                  v{v.version}
                </span>
                <span className="mono text-xs text-[var(--color-text-muted)]">
                  {shortDate(v.trained_at)}
                </span>
                <span className="text-xs flex-1">
                  {model}
                  {v.trained_data_start && v.trained_data_end ? (
                    <>
                      {" "}
                      ·{" "}
                      <span className="mono text-[var(--color-text-muted)]">
                        {String(v.trained_data_start).slice(0, 10)} →{" "}
                        {String(v.trained_data_end).slice(0, 10)}
                      </span>
                    </>
                  ) : null}{" "}
                  · val_loss{" "}
                  <b className="mono" style={{ color: valLossCol }}>
                    {valLoss != null ? num(valLoss, 4) : "—"}
                  </b>
                  {v.directional_accuracy != null && (
                    <>
                      {" "}
                      · dir <b className="mono">{(v.directional_accuracy * 100).toFixed(1)}%</b>
                    </>
                  )}
                </span>
                {active && (
                  <span
                    className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px]"
                    style={{ background: "rgba(6,182,212,0.15)", color: "var(--color-primary)" }}
                  >
                    active
                  </span>
                )}
                <span className="text-[10px] text-[var(--color-text-dim)]">
                  {daysAgo(v.trained_at)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── Screen ──────────────────────────────────────────────────────────

export function Models() {
  const { data, isLoading } = useModelSummary();
  const nextRetrain = data?.symbols[0]?.next_retrain_due ?? null;

  return (
    <div className="space-y-4">
      <header className="flex items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[var(--color-text)]">Models</h1>
          <p className="text-xs text-[var(--color-text-dim)] mt-0.5">
            LSTM / HMM health · directional accuracy · retrain cadence
          </p>
        </div>
        {nextRetrain && (
          <span className="text-[11px] text-[var(--color-text-dim)] mono">
            next retrain {shortDate(nextRetrain)}
          </span>
        )}
      </header>

      {isLoading && (
        <p className="text-center text-sm text-[var(--color-text-muted)] py-10">
          Loading model summary…
        </p>
      )}

      {data && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-5 gap-3">
            {data.symbols.map((row) => (
              <SymbolCard key={row.symbol} row={row} />
            ))}
          </div>

          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
            <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
              <div>
                <p className="section-label">30-day directional accuracy · all symbols</p>
                <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                  Rolling 5-day window · healthy ≥52% · warn 48%
                </p>
              </div>
              <div className="flex items-center gap-3 text-[11px]">
                {LIVE_SYMBOLS.map((s) => (
                  <span key={s} className="inline-flex items-center gap-1">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ background: SERIES_COLORS[s] ?? "var(--color-primary)" }}
                    />
                    {s}
                  </span>
                ))}
              </div>
            </div>
            <AccuracyOverlay symbols={LIVE_SYMBOLS} />
          </div>

          <RetrainHistory />
        </>
      )}
    </div>
  );
}
