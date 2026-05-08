import { useEffect, useMemo, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { useSignal } from "@/hooks/useSignal";
import { useLiveState } from "@/hooks/useLiveState";
import { useCandles, useLatestPrice } from "@/hooks/useCandles";
import { useSignalHistory } from "@/hooks/useSignalHistory";
import { useSignalAudit } from "@/hooks/useSignalAudit";
import type { SignalData, ChartTimeframe } from "@/lib/types";
import { PriceChart, type ChartType, type SignalMarker } from "@/components/PriceChart";
import { regimeColor } from "@/lib/tokens";
import { useNewsBlackoutForSymbol } from "@/hooks/useNewsBlackouts";
import { durationInRegime, fmtRegimeDuration } from "@/lib/regime";
import {
  deriveStrategyName,
  strategyColor,
  strategyShortLabel,
} from "@/lib/strategy";
import { SignalTensionCard } from "@/components/SignalTensionCard";
import { SignalPipelineCard } from "@/components/SignalPipelineCard";

import { LIVE_SYMBOLS as DEFAULT_SYMBOLS } from "@/lib/symbols";

const TF_OPTIONS: ChartTimeframe[] = ["H1", "H4", "D1"];

const TF_SECONDS: Record<ChartTimeframe, number> = {
  M15: 15 * 60,
  H1: 60 * 60,
  H4: 4 * 60 * 60,
  D1: 24 * 60 * 60,
  W1: 7 * 24 * 60 * 60,
};

function parseToSeconds(iso: string): number {
  const s = iso.endsWith("Z") || /[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`;
  const t = Date.parse(s);
  return Number.isFinite(t) ? Math.floor(t / 1000) : 0;
}

/**
 * Bucket direction-flip markers to the display-bar resolution.
 * Rule (mockup Signal density): signals are binned to the bar-open timestamp.
 * If 2+ flips fall in the same bar, emit a single cluster marker with the
 * count instead of stacking arrows on top of each other.
 */
function bucketMarkersByBar(
  markers: SignalMarker[],
  tfSeconds: number,
): SignalMarker[] {
  const buckets = new Map<number, SignalMarker[]>();
  for (const m of markers) {
    const t = parseToSeconds(m.time);
    if (!t) continue;
    const bucket = Math.floor(t / tfSeconds) * tfSeconds;
    const arr = buckets.get(bucket) ?? [];
    arr.push(m);
    buckets.set(bucket, arr);
  }
  const out: SignalMarker[] = [];
  const bucketTimes = [...buckets.keys()].sort((a, b) => a - b);
  for (const bucket of bucketTimes) {
    const group = buckets.get(bucket)!;
    const iso = new Date(bucket * 1000).toISOString();
    if (group.length === 1) {
      out.push({ ...group[0], time: iso });
    } else {
      const last = group[group.length - 1];
      out.push({
        time: iso,
        direction: last.direction,
        executed: false,
        score: last.score,
        cluster: group.length,
      });
    }
  }
  return out;
}

// ─── Helpers ─────────────────────────────────────────────────────────

function fmtPct(v: number): string {
  const sign = v > 0 ? "+" : v < 0 ? "−" : "";
  return `${sign}${Math.abs(v).toFixed(2)}%`;
}

function fmtScore(v: number | null): string {
  if (v == null) return "—";
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(2)}`;
}

function formatPrice(price: number | null, symbol: string): string {
  if (price == null || !Number.isFinite(price)) return "—";
  if (symbol.includes("JPY")) return price.toFixed(3);
  if (symbol === "XAUUSD" || symbol === "ETHUSD" || symbol === "BTCUSD") {
    return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return price.toFixed(5);
}

function fmtClock(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || /[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hour12: false });
}

// Short date+time for Recent decisions list. Today → "14:08", else "Apr 18 14:08".
function fmtClockDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || /[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return "—";
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  const hm = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hour12: false });
  if (sameDay) return hm;
  const md = d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return `${md} ${hm}`;
}

function reasonLineFor(signal: SignalData): string {
  if (signal.should_trade) {
    return `Approved · ${signal.direction?.toUpperCase()}`;
  }
  // Try to pull a friendly reason from the reasoning trail (last entry).
  const last = signal.reasoning[signal.reasoning.length - 1];
  if (last) return last.length > 64 ? last.slice(0, 61) + "…" : last;
  const absScore = Math.abs(signal.combined_score);
  if (absScore < 0.45) return `Flat · |score| ${absScore.toFixed(2)} below threshold`;
  return "Flat · regime/direction mismatch";
}

// ─── Signals Index ───────────────────────────────────────────────────

interface AggregatedApproval {
  symbol: string;
  approved: number;
  total: number;
  pct: number;
}

export function SignalsIndex() {
  const { data } = useLiveState();
  const { data: audit } = useSignalAudit({ pageSize: 200 });

  const liveSymbols = data ? Object.keys(data.signals) : [];
  const symbols = liveSymbols.length > 0 ? liveSymbols : [...DEFAULT_SYMBOLS];

  const [sortKey, setSortKey] = useState<"score" | "symbol">("score");
  const [activityFilter, setActivityFilter] = useState<"all" | "approved" | "blocked">("all");
  const [approvalRange, setApprovalRange] = useState<"24h" | "7d" | "30d">("24h");
  const [distributionRange, setDistributionRange] = useState<"24h" | "7d" | "30d">("7d");

  // Aggregate audit rows for a given window. Returns per-symbol approval
  // counters AND 10-bucket score histogram so the Approval Rate and
  // Score Distribution cards can use independent windows — the two
  // answer different questions (short-term gate bias vs long-run score
  // shape), so they shouldn't be locked together.
  const aggregate = useMemo(() => {
    type RecItem = {
      timestamp: string;
      symbol: string;
      executed: boolean;
      combined_score: number | null;
      block_reason: string | null;
      direction: string | null;
      should_trade: boolean;
    };
    const rows: RecItem[] = (audit?.items ?? []) as RecItem[];
    return (range: "24h" | "7d" | "30d") => {
      const hours = range === "24h" ? 24 : range === "7d" ? 24 * 7 : 24 * 30;
      const cutoff = Date.now() - hours * 3_600_000;
      const windowed = rows.filter((r) => {
        const t = Date.parse(r.timestamp);
        return Number.isFinite(t) && t >= cutoff;
      });
      const approvalBySymbol: Record<string, AggregatedApproval> = {};
      for (const sym of symbols) {
        approvalBySymbol[sym] = { symbol: sym, approved: 0, total: 0, pct: 0 };
      }
      const scoreBuckets = new Array(10).fill(0) as number[];
      let totalApproved = 0;
      for (const r of windowed) {
        const appr = approvalBySymbol[r.symbol] ?? (approvalBySymbol[r.symbol] = {
          symbol: r.symbol,
          approved: 0,
          total: 0,
          pct: 0,
        });
        appr.total += 1;
        if (r.executed) {
          appr.approved += 1;
          totalApproved += 1;
        }
        const s = r.combined_score;
        if (s != null && Number.isFinite(s)) {
          const clamped = Math.max(-1, Math.min(1, s));
          const idx = Math.min(9, Math.max(0, Math.floor((clamped + 1) / 0.2)));
          scoreBuckets[idx] += 1;
        }
      }
      for (const k of Object.keys(approvalBySymbol)) {
        const a = approvalBySymbol[k];
        a.pct = a.total > 0 ? (a.approved / a.total) * 100 : 0;
      }
      return {
        approvalBySymbol,
        scoreBuckets,
        totalEvaluated: windowed.length,
        totalApproved,
      };
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audit, symbols.join(",")]);

  const approvalAgg = useMemo(() => aggregate(approvalRange), [aggregate, approvalRange]);
  const distributionAgg = useMemo(() => aggregate(distributionRange), [aggregate, distributionRange]);
  const { approvalBySymbol, totalEvaluated, totalApproved } = approvalAgg;
  const { scoreBuckets, totalEvaluated: distributionTotal } = distributionAgg;

  // Sorted symbols for top grid.
  const sortedSymbols = useMemo(() => {
    const arr = [...symbols];
    if (sortKey === "score") {
      arr.sort(
        (a, b) =>
          Math.abs(data?.signals[b]?.combined_score ?? 0) -
          Math.abs(data?.signals[a]?.combined_score ?? 0),
      );
    } else {
      arr.sort();
    }
    return arr;
  }, [symbols, sortKey, data]);

  const approvedLastHour = useMemo(() => {
    const cutoff = Date.now() - 3_600_000;
    return (audit?.items ?? []).filter((r) => {
      const t = new Date(r.timestamp.endsWith("Z") ? r.timestamp : `${r.timestamp}Z`).getTime();
      return Number.isFinite(t) && t >= cutoff && r.executed;
    }).length;
  }, [audit]);

  const recentActivity = useMemo(() => {
    const rows = audit?.items ?? [];
    const filtered = rows.filter((r) => {
      if (activityFilter === "all") return true;
      if (activityFilter === "approved") return r.executed;
      return !r.executed;
    });
    return filtered.slice(0, 12);
  }, [audit, activityFilter]);

  // For title under "Decisions · last hour" — recalculate live totals
  const lastHour = useMemo(() => {
    const cutoff = Date.now() - 3_600_000;
    const rows = (audit?.items ?? []).filter((r) => {
      const t = new Date(r.timestamp.endsWith("Z") ? r.timestamp : `${r.timestamp}Z`).getTime();
      return Number.isFinite(t) && t >= cutoff;
    });
    const approved = rows.filter((r) => r.executed).length;
    return { total: rows.length, approved, flat: rows.length - approved };
  }, [audit]);

  return (
    <div className="space-y-4">
      <header className="flex items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[var(--color-text)]">Signals</h1>
          <p className="text-xs text-[var(--color-text-dim)] mt-0.5">
            Per-symbol decisions · score fusion · approval audit
          </p>
        </div>
        <Link
          to="/ui/signals-log"
          className="inline-flex items-center px-3 py-1.5 rounded-lg text-xs font-medium text-white bg-brand-gradient hover:brightness-110"
        >
          Full audit log →
        </Link>
      </header>

      {/* Top card: Decisions last hour + 5 compact per-symbol cards */}
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-[var(--color-text)]">
              Decisions · last hour
            </h3>
            <p className="text-xs text-[var(--color-text-muted)]">
              {lastHour.total} evaluated · {lastHour.approved} approved · {lastHour.flat} flat
              {approvedLastHour > 0 ? ` · ${approvedLastHour} filled` : ""}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setSortKey(sortKey === "score" ? "symbol" : "score")}
              className="px-3 py-1.5 text-xs rounded-lg bg-[var(--color-panel-hi)] border border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
            >
              Sort: {sortKey === "score" ? "Score" : "Symbol"}
            </button>
          </div>
        </div>
        {symbols.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-5 gap-3">
            {sortedSymbols.map((sym) => {
              const signal = data?.signals[sym];
              if (!signal) return <MissingSignalCard key={sym} symbol={sym} />;
              return <CompactSignalCard key={sym} symbol={sym} signal={signal} />;
            })}
          </div>
        ) : (
          <p className="text-center text-sm text-[var(--color-text-dim)] py-8">
            No signals available yet
          </p>
        )}
        <p className="mt-4 text-xs text-[var(--color-text-dim)]">
          Click any card to drill into the symbol — chart, regime evolution, news blackout, recent decisions.
        </p>
      </div>

      {/* Secondary row: approval rate + activity feed + score distribution */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
        {/* Approval rate 24h */}
        <div className="xl:col-span-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="section-label">Approval rate · {approvalRange}</p>
              <p className="text-xs text-[var(--color-text-muted)] mt-0.5">Signals passing all gates</p>
            </div>
            <div className="flex gap-1">
              {(["24h", "7d", "30d"] as const).map((r) => {
                const active = r === approvalRange;
                return (
                  <button
                    key={r}
                    onClick={() => setApprovalRange(r)}
                    className="px-2 py-0.5 text-[10px] rounded"
                    style={
                      active
                        ? { background: "var(--color-panel-hi)", color: "var(--color-primary)" }
                        : { color: "var(--color-text-muted)" }
                    }
                  >
                    {r}
                  </button>
                );
              })}
            </div>
          </div>
          <div className="space-y-3">
            {symbols.map((sym) => {
              const a = approvalBySymbol[sym] ?? { approved: 0, total: 0, pct: 0 };
              // Show full 6-char symbol — stripping the USD suffix gave
              // inconsistent labels (XAU/GBP/NZD vs USDJPY/USDCAD/USDCHF).
              return (
                <div key={sym} className="flex items-center gap-3">
                  <span className="mono w-14 text-[11px] font-medium">{sym}</span>
                  <div className="flex-1 h-1.5 rounded-full bg-[var(--color-panel-hi)] overflow-hidden">
                    <div
                      className="h-full"
                      style={{ width: `${Math.min(a.pct, 100).toFixed(1)}%`, background: "var(--color-primary)" }}
                    />
                  </div>
                  <span className="mono w-10 text-right text-[12px] font-semibold">
                    {a.pct.toFixed(0)}%
                  </span>
                  <span className="mono w-14 text-right text-[10px] text-[var(--color-text-dim)]">
                    {a.approved} / {a.total}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="mt-4 pt-3 border-t border-[var(--color-border)] text-[10px] text-[var(--color-text-dim)] flex items-center justify-between">
            <span>Across {totalEvaluated} evaluations · {totalApproved} approved</span>
            <Link to="/ui/signals-log" className="text-[var(--color-primary)] hover:brightness-125">
              Full audit →
            </Link>
          </div>
        </div>

        {/* Recent activity feed */}
        <div className="xl:col-span-5 rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="section-label">Recent activity</p>
              <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                Last {recentActivity.length} decisions
              </p>
            </div>
            <div className="flex gap-1 text-[10px]">
              {(["all", "approved", "blocked"] as const).map((f) => {
                const active = f === activityFilter;
                return (
                  <button
                    key={f}
                    onClick={() => setActivityFilter(f)}
                    className="px-2 py-0.5 rounded capitalize"
                    style={
                      active
                        ? { background: "var(--color-panel-hi)", color: "var(--color-primary)" }
                        : { color: "var(--color-text-muted)" }
                    }
                  >
                    {f}
                  </button>
                );
              })}
            </div>
          </div>
          <ul className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
            {recentActivity.length === 0 && (
              <li className="text-xs text-[var(--color-text-dim)] text-center py-8">
                No recent decisions match this filter
              </li>
            )}
            {recentActivity.map((r, i) => {
              const score = r.combined_score ?? 0;
              const scoreCol =
                score > 0 ? "var(--color-profit)" : score < 0 ? "var(--color-loss)" : "var(--color-text-muted)";
              let chip: { text: string; bg: string; color: string };
              if (r.executed) {
                chip = {
                  text: `✓ ${r.direction ?? "buy"}`,
                  bg: "rgba(16,185,129,0.15)",
                  color: "var(--color-profit)",
                };
              } else if (r.block_reason && /blackout|news/i.test(r.block_reason)) {
                chip = {
                  text: "⚠ block",
                  bg: "rgba(245,158,11,0.15)",
                  color: "var(--color-warn)",
                };
              } else if (r.block_reason && /^broker_reject|^broker\b/i.test(r.block_reason)) {
                chip = {
                  text: "✗ broker",
                  bg: "rgba(244,63,94,0.15)",
                  color: "var(--color-loss)",
                };
              } else {
                chip = {
                  text: "·",
                  bg: "rgba(148,163,184,0.15)",
                  color: "var(--color-text-muted)",
                };
              }
              const rowBg = r.executed
                ? "rgba(16,185,129,0.05)"
                : r.block_reason && /^broker_reject|^broker\b/i.test(r.block_reason)
                  ? "rgba(244,63,94,0.05)"
                  : undefined;
              return (
                <li
                  key={`${r.timestamp}-${r.symbol}-${i}`}
                  className="flex items-center gap-3 p-2 rounded hover:bg-[var(--color-panel-hi)] transition-colors"
                  style={rowBg ? { background: rowBg } : undefined}
                >
                  <span
                    className="mono text-[10px] text-[var(--color-text-dim)] w-[96px] shrink-0 whitespace-nowrap"
                    title={new Date(
                      r.timestamp.endsWith("Z") || /[+-]\d\d:?\d\d$/.test(r.timestamp)
                        ? r.timestamp
                        : `${r.timestamp}Z`,
                    ).toLocaleString()}
                  >
                    {fmtClockDate(r.timestamp)}
                  </span>
                  <span
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] shrink-0"
                    style={{ background: chip.bg, color: chip.color }}
                  >
                    {chip.text}
                  </span>
                  <Link
                    to={`/ui/signals/${r.symbol}`}
                    className="mono text-xs font-semibold w-16 shrink-0 hover:text-[var(--color-primary)]"
                  >
                    {r.symbol}
                  </Link>
                  <span className="text-xs text-[var(--color-text-muted)] flex-1 truncate">
                    {r.block_reason ??
                      (r.executed
                        ? `Approved · ${r.direction?.toUpperCase() ?? ""}`
                        : "Flat")}
                  </span>
                  <span className="mono text-xs shrink-0" style={{ color: scoreCol }}>
                    {fmtScore(r.combined_score)}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>

        {/* Score distribution */}
        <div className="xl:col-span-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="section-label">Score distribution · {distributionRange}</p>
              <p className="text-xs text-[var(--color-text-muted)] mt-0.5">Where scores cluster</p>
            </div>
            <div className="flex gap-1">
              {(["24h", "7d", "30d"] as const).map((r) => {
                const active = r === distributionRange;
                return (
                  <button
                    key={r}
                    onClick={() => setDistributionRange(r)}
                    className="px-2 py-0.5 text-[10px] rounded"
                    style={
                      active
                        ? { background: "var(--color-panel-hi)", color: "var(--color-primary)" }
                        : { color: "var(--color-text-muted)" }
                    }
                  >
                    {r}
                  </button>
                );
              })}
            </div>
          </div>
          <ScoreDistribution buckets={scoreBuckets} total={distributionTotal} />
          <p className="mt-3 pt-3 border-t border-[var(--color-border)] text-[10px] leading-relaxed text-[var(--color-text-dim)]">
            Threshold ±0.45 marks the trade-gate. Scores clustering near 0 = bot staying flat is working correctly.
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── Compact signal card (Signals index row 1) ───────────────────────

function CompactSignalCard({ symbol, signal }: { symbol: string; signal: SignalData }) {
  const { price, changePct } = useLatestPrice(symbol);
  const regime = signal.regime.regime_label;
  const regimeCol = regimeColor(regime);
  const stratName = deriveStrategyName(signal.regime);
  const stratLabel = strategyShortLabel(stratName);
  const stratCol = strategyColor(stratName);
  const score = signal.combined_score;
  const scoreCol =
    score > 0.1 ? "var(--color-profit)" : score < -0.1 ? "var(--color-loss)" : "var(--color-text-muted)";
  const priceDeltaCol =
    changePct == null ? "var(--color-text-muted)" : changePct >= 0 ? "var(--color-profit)" : "var(--color-loss)";
  const box = signal.should_trade
    ? { bg: "rgba(16,185,129,0.08)", border: "rgba(16,185,129,0.2)" }
    : score < -0.45
      ? { bg: "rgba(244,63,94,0.08)", border: "rgba(244,63,94,0.2)" }
      : { bg: "rgba(100,116,139,0.08)", border: "transparent" };

  return (
    <Link
      to={`/ui/signals/${symbol}`}
      className="block rounded-xl border border-[color:var(--color-border-hi)] bg-[var(--color-panel-hi)] p-4 transition-all hover:-translate-y-px hover:border-[color:rgba(99,102,241,0.35)]"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="mono text-[11px] font-semibold text-[var(--color-text)]">{symbol}</span>
        <div className="flex items-center gap-1">
          <span
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
            style={{ background: `${regimeCol}26`, color: regimeCol }}
          >
            {regime}
          </span>
          <span
            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px]"
            style={{ background: `${stratCol}1f`, color: stratCol }}
            title={`Strategy class: ${stratName}`}
          >
            {stratLabel}
          </span>
        </div>
      </div>
      <p className="tnum text-xl font-bold mt-1 text-[var(--color-text)]">
        {formatPrice(price, symbol)}
      </p>
      <p className="tnum text-xs" style={{ color: priceDeltaCol }}>
        {changePct == null ? "—" : `${changePct >= 0 ? "▲" : "▼"} ${fmtPct(changePct)}`}
      </p>
      <div
        className="mt-3 p-2 rounded"
        style={{ background: box.bg, border: `1px solid ${box.border}` }}
      >
        <p className="text-[10px] text-[var(--color-text-muted)]">Score</p>
        <p className="tnum text-base font-bold" style={{ color: scoreCol }}>
          {fmtScore(score)}
        </p>
        <p className="text-[10px] text-[var(--color-text-dim)] mt-1">
          {reasonLineFor(signal)}
        </p>
      </div>
    </Link>
  );
}

function MissingSignalCard({ symbol }: { symbol: string }) {
  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel-hi)] p-4 opacity-60">
      <span className="mono text-[11px] font-semibold">{symbol}</span>
      <p className="text-xs text-[var(--color-text-dim)] mt-3">No signal yet</p>
    </div>
  );
}

// ─── Score distribution (10 buckets, ±0.45 guide lines) ──────────────

function ScoreDistribution({ buckets, total }: { buckets: number[]; total: number }) {
  if (total === 0) {
    return (
      <div className="h-32 flex items-center justify-center text-xs text-[var(--color-text-dim)] text-center px-2">
        No signal evaluations in this window yet.
        <br />
        Bot evaluates at each H4 bar close (6×/day).
      </div>
    );
  }
  const maxBucket = Math.max(1, ...buckets);
  return (
    <>
      <div className="flex items-end justify-between gap-1 h-32">
        {buckets.map((n, i) => {
          const h = (n / maxBucket) * 100;
          const bucketStart = -1 + i * 0.2;
          const bucketEnd = bucketStart + 0.2;
          const color =
            bucketStart < -0.2
              ? "var(--color-loss)"
              : bucketStart >= 0.2
                ? "var(--color-profit)"
                : "var(--color-text-muted)";
          const opacity = total > 0 ? Math.max(0.35, Math.min(1, n / maxBucket + 0.3)) : 0.35;
          return (
            <div
              key={i}
              className="flex-1 rounded-t"
              style={{ height: `${Math.max(h, n > 0 ? 4 : 0)}%`, background: color, opacity }}
              title={`${bucketStart.toFixed(1)} to ${bucketEnd.toFixed(1)} · ${n}`}
            />
          );
        })}
      </div>
      <div className="relative h-4 mt-1">
        <div
          className="absolute top-0 bottom-0 w-px"
          style={{ left: "27.5%", background: "var(--color-warn)", opacity: 0.6 }}
        />
        <div
          className="absolute top-0 bottom-0 w-px"
          style={{ left: "72.5%", background: "var(--color-warn)", opacity: 0.6 }}
        />
        <span
          className="absolute mono text-[9px]"
          style={{ left: "14%", top: "2px", color: "var(--color-warn)" }}
        >
          −0.45
        </span>
        <span
          className="absolute mono text-[9px]"
          style={{ left: "73%", top: "2px", color: "var(--color-warn)" }}
        >
          +0.45
        </span>
      </div>
      <div className="flex justify-between mono text-[9px] text-[var(--color-text-dim)] mt-1">
        <span>−1</span>
        <span>0</span>
        <span>+1</span>
      </div>
    </>
  );
}

// ─── Signal Detail (SignalPage) ──────────────────────────────────────

type SignalDensity = "flips" | "all" | "approved";

const CHART_TYPE_STORAGE = "cortex-chart-type";
function readChartType(): ChartType {
  try {
    const v = window.localStorage.getItem(CHART_TYPE_STORAGE);
    if (v === "candles" || v === "line" || v === "area") return v;
  } catch {
    /* noop */
  }
  return "candles";
}

const CHART_TYPE_OPTIONS: ChartType[] = ["candles", "line", "area"];
const CHART_TYPE_LABEL: Record<ChartType, string> = {
  candles: "Candles",
  line: "Line",
  area: "Area",
};

function SignalDetail({ symbol }: { symbol: string }) {
  const { data: signal, isLoading, error } = useSignal(symbol);
  const { data: live } = useLiveState();
  const [tf, setTf] = useState<ChartTimeframe>("H1");
  const [density, setDensity] = useState<SignalDensity>("flips");
  const [chartType, setChartType] = useState<ChartType>(() => readChartType());
  useEffect(() => {
    try {
      window.localStorage.setItem(CHART_TYPE_STORAGE, chartType);
    } catch {
      /* noop */
    }
  }, [chartType]);
  const { data: candles, isLoading: candlesLoading } = useCandles({ symbol, timeframe: tf, limit: 1000 });
  const { data: signalHistory } = useSignalHistory(symbol, 200);
  const { entry: news } = useNewsBlackoutForSymbol(symbol);
  const { price: headerPrice, changePct: headerChangePct } = useLatestPrice(symbol);

  const signalMarkers: SignalMarker[] = useMemo(() => {
    const rows = (signalHistory ?? []).filter(
      (s) => s.direction === "buy" || s.direction === "sell",
    );
    const chronological = [...rows].reverse();
    let raw: SignalMarker[] = [];
    if (density === "flips") {
      let prev: string | null = null;
      for (const s of chronological) {
        if (s.direction !== prev) {
          raw.push({
            time: s.timestamp,
            direction: s.direction as "buy" | "sell",
            executed: s.executed,
            score: s.combined_score ?? undefined,
          });
          prev = s.direction;
        }
      }
    } else if (density === "approved") {
      raw = chronological
        .filter((s) => s.executed)
        .map((s) => ({
          time: s.timestamp,
          direction: s.direction as "buy" | "sell",
          executed: true,
          score: s.combined_score ?? undefined,
        }));
    } else {
      // "all"
      raw = chronological.map((s) => ({
        time: s.timestamp,
        direction: s.direction as "buy" | "sell",
        executed: s.executed,
        score: s.combined_score ?? undefined,
      }));
    }
    // Bucket to the display-bar resolution so HTF (H4/D1) don't stack.
    return bucketMarkersByBar(raw, TF_SECONDS[tf]);
  }, [signalHistory, density, tf]);

  // NOTE: this memo MUST live above the loading/error early-returns.
  // Rules of hooks — hook count must be identical across every render.
  const heldRun = useMemo(
    () => durationInRegime(signalHistory ?? [], signal?.regime.regime_label ?? null),
    [signalHistory, signal],
  );
  const heldText = fmtRegimeDuration(heldRun);

  if (isLoading) {
    return <p className="text-sm text-[var(--color-text-muted)]">Loading signal for {symbol}…</p>;
  }
  if (error || !signal) {
    return (
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-8 text-center">
        <p className="text-sm text-[var(--color-text-dim)]">No signal available for {symbol}</p>
      </div>
    );
  }

  const regime = signal.regime;
  const regCol = regimeColor(regime.regime_label);
  const confPct = (regime.state_probability * 100).toFixed(2);
  const headerPriceDeltaCol =
    headerChangePct == null
      ? "var(--color-text-muted)"
      : headerChangePct >= 0
        ? "var(--color-profit)"
        : "var(--color-loss)";

  const liveSymbols = live ? Object.keys(live.signals) : [...DEFAULT_SYMBOLS];

  // XAU flipped to bidirectional 2026-04-27 (Cell C A/B verdict). Source of
  // truth lives in config/settings.yaml::strategy.long_only_symbols, which
  // currently contains only ETHUSD (and ETH isn't in trading.symbols, so the
  // chip stays dormant in practice — kept for forward-compat).
  const isLongOnly = symbol === "ETHUSD";
  const recentForSymbol = (signalHistory ?? []).slice(0, 12);

  // Derive HMM / LSTM partial scores for the breakdown panel.
  const hmmContrib = regime.position_multiplier; // 0..2
  const lstmContrib = signal.lstm_prediction;

  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] overflow-hidden">
      {/* Breadcrumb + header */}
      <div className="px-6 py-4 border-b border-[var(--color-border)] flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3 flex-wrap">
          <Link to="/ui/signals" className="text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text)]">
            ← Signals
          </Link>
          <span className="text-[var(--color-text-dim)]">/</span>
          <span className="mono text-xl font-bold text-[var(--color-text)]">{symbol}</span>
          <span
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
            style={{
              background: `${regCol}26`,
              color: regCol,
              border: `1px solid ${regCol}44`,
            }}
          >
            {regime.regime_label} · conf {confPct}%
            {heldText !== "—" && (
              <span className="ml-1 text-[var(--color-text-dim)]">· held {heldText}</span>
            )}
          </span>
          {(() => {
            const stratName = deriveStrategyName(regime);
            const stratCol = strategyColor(stratName);
            return (
              <span
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
                style={{
                  background: `${stratCol}1f`,
                  color: stratCol,
                  border: `1px solid ${stratCol}44`,
                }}
                title={`Strategy class for current vol rank: ${stratName}. Set by StrategyOrchestrator from regime.expected_volatility.`}
              >
                {strategyShortLabel(stratName)}
              </span>
            );
          })()}
          {isLongOnly && (
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
              style={{ background: "rgba(234,179,8,0.12)", color: "var(--chip-warn-fg)" }}
            >
              Long-only
            </span>
          )}
          {headerPrice != null && (
            <>
              <span className="tnum text-lg font-semibold text-[var(--color-text)]">
                {formatPrice(headerPrice, symbol)}
              </span>
              {headerChangePct != null && (
                <span className="tnum text-xs" style={{ color: headerPriceDeltaCol }}>
                  {headerChangePct >= 0 ? "▲" : "▼"} {fmtPct(headerChangePct)}
                </span>
              )}
            </>
          )}
        </div>
        <div className="flex items-center gap-4 flex-wrap">
          {/* Signal density filter */}
          <div className="flex items-center gap-1 text-[11px] text-[var(--color-text-dim)]">
            <span>Signals:</span>
            {(["flips", "all", "approved"] as const).map((d) => {
              const active = density === d;
              return (
                <button
                  key={d}
                  onClick={() => setDensity(d)}
                  className="px-2 py-0.5 rounded capitalize"
                  style={
                    active
                      ? { background: "rgba(6,182,212,0.15)", color: "var(--color-primary)" }
                      : { color: "var(--color-text-muted)" }
                  }
                >
                  {d}
                </button>
              );
            })}
          </div>
          <div className="w-px h-4 bg-[var(--color-border)]" />
          <div className="flex gap-1" role="tablist" aria-label="Timeframe">
            {TF_OPTIONS.map((t) => {
              const active = t === tf;
              return (
                <button
                  key={t}
                  onClick={() => setTf(t)}
                  className="px-2.5 py-1 text-[11px] rounded-md transition-colors"
                  style={
                    active
                      ? { background: "var(--color-panel-hi)", color: "var(--color-primary)", fontWeight: 500 }
                      : { color: "var(--color-text-muted)" }
                  }
                >
                  {t}
                </button>
              );
            })}
          </div>
          <div className="w-px h-4 bg-[var(--color-border)]" />
          <div
            className="inline-flex rounded-md border border-[var(--color-border)] overflow-hidden"
            role="tablist"
            aria-label="Chart style"
          >
            {CHART_TYPE_OPTIONS.map((ct) => {
              const active = ct === chartType;
              return (
                <button
                  key={ct}
                  onClick={() => setChartType(ct)}
                  aria-pressed={active}
                  className="px-2 py-1 text-[11px] transition-colors"
                  style={
                    active
                      ? { background: "var(--color-panel-hi)", color: "var(--color-primary)", fontWeight: 500 }
                      : { color: "var(--color-text-muted)" }
                  }
                >
                  {CHART_TYPE_LABEL[ct]}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Symbol switcher strip */}
      <div className="px-6 py-3 border-b border-[var(--color-border)] flex items-center gap-2 overflow-x-auto">
        <span className="text-[10px] uppercase tracking-[0.14em] text-[var(--color-text-dim)] mr-2 shrink-0">
          Switch:
        </span>
        {liveSymbols.map((sym) => (
          <SymbolSwitcherChip key={sym} symbol={sym} active={sym === symbol} />
        ))}
      </div>

      {/* Chart + rail grid. items-stretch (default) makes both columns
          the same height; main col is flex so its last card (regime evo)
          can stretch to close the bottom gap vs the rail's stretched
          Recent decisions. */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 p-6">
        <div className="xl:col-span-8 flex flex-col gap-4">
          {/* Chart card — hero price + legend + chart */}
          <div className="rounded-xl bg-[var(--color-panel-hi)] border border-[var(--color-border)] p-4">
            <div className="flex items-end justify-between mb-3 flex-wrap gap-3">
              <div>
                <p className="text-xs text-[var(--color-text-muted)]">Last price</p>
                <p className="tnum text-4xl font-bold">
                  {formatPrice(headerPrice, symbol)}{" "}
                  {headerChangePct != null && (
                    <span className="text-base" style={{ color: headerPriceDeltaCol }}>
                      {headerChangePct >= 0 ? "▲" : "▼"} {fmtPct(headerChangePct)}
                    </span>
                  )}
                </p>
              </div>
              <div className="flex items-center gap-4 text-[11px] text-[var(--color-text-muted)] flex-wrap">
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-3 h-0.5" style={{ background: "var(--color-primary)" }} />
                  Price
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ background: "var(--color-profit)" }}
                  />
                  Buy flip
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ background: "var(--color-loss)" }}
                  />
                  Sell flip
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span
                    className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full mono font-bold text-[8px]"
                    style={{ background: "var(--color-warn)", color: "var(--color-bg)" }}
                  >
                    3
                  </span>
                  Cluster (n flips in bar)
                </span>
              </div>
            </div>
            {candlesLoading && !candles ? (
              <div className="h-[420px] flex items-center justify-center text-sm text-[var(--color-text-dim)]">
                Loading candles…
              </div>
            ) : candles && candles.bars.length > 0 ? (
              <PriceChart
                bars={candles.bars}
                height={420}
                symbol={symbol}
                markers={signalMarkers}
                chartType={chartType}
              />
            ) : (
              <div className="h-[420px] flex items-center justify-center text-sm text-[var(--color-text-dim)]">
                No candle data for {symbol} / {tf}
              </div>
            )}
          </div>

          {/* Signal tension: score trajectory vs threshold + block-reason
              bars for this symbol. Replaced the static SignalDensityCard
              (2026-04-19) — same slot, real audit data. */}
          <SignalTensionCard symbol={symbol} />

          {/* HMM regime evolution · 48h — flex:1 stretches to match rail height */}
          <div className="flex-1 flex flex-col min-h-0">
            <RegimeEvolutionCard history={signalHistory ?? []} />
          </div>
        </div>

        <div className="xl:col-span-4 flex flex-col gap-4">
        {/* Score breakdown */}
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel-hi)] p-5">
          <p className="section-label mb-3">Score breakdown</p>
          <div className="space-y-3">
            <BreakdownRow
              label="HMM regime · 30%"
              value={hmmContrib}
              rangeMin={0}
              rangeMax={2}
              fmt={(v) => `×${v.toFixed(2)}`}
            />
            <BreakdownRow
              label="LSTM pred · 70%"
              value={lstmContrib}
              rangeMin={-1}
              rangeMax={1}
              fmt={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(3)}`}
            />
            <div className="pt-3 border-t border-[var(--color-border)]">
              <div className="flex items-center justify-between">
                <span className="text-xs text-[var(--color-text-muted)]">Combined</span>
                <span
                  className="mono text-lg font-bold"
                  style={{
                    color:
                      signal.combined_score > 0.1
                        ? "var(--color-profit)"
                        : signal.combined_score < -0.1
                          ? "var(--color-loss)"
                          : "var(--color-text-muted)",
                  }}
                >
                  {fmtScore(signal.combined_score)}
                </span>
              </div>
              <p className="text-[10px] text-[var(--color-text-dim)] mt-1">
                Trade gate: |score| ≥ 0.45 (EUR/JPY ≥ 0.55)
              </p>
            </div>
            <div className="flex items-center justify-between pt-2">
              <span className="text-xs text-[var(--color-text-muted)]">Confidence</span>
              <span className="mono text-sm">{(signal.confidence * 100).toFixed(2)}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-[var(--color-text-muted)]">Uncertainty mode</span>
              <span
                className="text-xs"
                style={{
                  color: signal.uncertainty_mode ? "var(--color-warn)" : "var(--color-text-dim)",
                }}
              >
                {signal.uncertainty_mode ? "ON" : "OFF"}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-[var(--color-text-muted)]">Size discount</span>
              <span className="mono text-sm">{signal.size_discount.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* News blackout — standalone card per mockup */}
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel-hi)] p-4">
          <p className="section-label mb-2">News blackout</p>
          {news?.exempt ? (
            <>
              <span
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
                style={{ background: "rgba(234,179,8,0.12)", color: "var(--chip-warn-fg)" }}
              >
                ⚠ {symbol} exempt
              </span>
              <p className="text-xs text-[var(--color-text-muted)] mt-3">
                {symbol === "XAUUSD"
                  ? "Gold is FOMC-exempt per policy."
                  : "Crypto trades 24/7 — no central-bank blackouts."}
              </p>
            </>
          ) : news?.state === "blackout" && news.active_event ? (
            <div
              className="text-[11px] rounded px-2 py-1.5"
              style={{ background: "rgba(244,63,94,0.12)", color: "var(--color-loss)" }}
            >
              ⚠ {news.active_event.cb} BLOCKED · reopens{" "}
              {fmtClock(news.active_event.blackout_end_utc)}
            </div>
          ) : news?.next_event ? (
            <>
              <p className="text-sm font-semibold text-[var(--color-text)]">
                {news.next_event.cb}
              </p>
              <p className="text-xs text-[var(--color-text-muted)] mt-0.5 mono">
                {new Date(news.next_event.event_utc).toLocaleString(undefined, {
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                })}{" "}
                · T−24h blackout
              </p>
            </>
          ) : (
            <p className="text-xs text-[var(--color-text-dim)]">
              No upcoming blackouts in feed.
            </p>
          )}
        </div>

        {/* Current regime probabilities (5-bar) */}
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel-hi)] p-4">
          <p className="section-label mb-3">Regime probabilities · now</p>
          <div className="flex items-center gap-2 mb-3">
            <span
              className="text-sm font-semibold px-2 py-0.5 rounded-full"
              style={{ background: `${regCol}26`, color: regCol }}
            >
              {regime.regime_label}
            </span>
            <span className="mono text-xs text-[var(--color-text-dim)]">
              pos_mult {regime.position_multiplier.toFixed(2)}×
            </span>
          </div>
          <RegimeBarsInline probs={regime.all_probabilities} currentIdx={regime.regime_index} />
        </div>

        {/* Recent decisions table — flex-1 absorbs remaining rail height */}
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel-hi)] p-5 flex-1 flex flex-col min-h-0">
          <p className="section-label mb-3">Recent decisions · {symbol}</p>
          <ul className="space-y-1.5 flex-1 overflow-y-auto pr-1">
            {recentForSymbol.length === 0 && (
              <li className="text-xs text-[var(--color-text-dim)] text-center py-4">
                No signal history yet.
              </li>
            )}
            {recentForSymbol.map((r, i) => {
              const score = r.combined_score ?? 0;
              const scoreCol =
                score > 0 ? "var(--color-profit)" : score < 0 ? "var(--color-loss)" : "var(--color-text-muted)";
              const dot = r.executed
                ? { bg: "rgba(16,185,129,0.16)", color: "var(--color-profit)", label: "✓" }
                : r.block_reason && /blackout|news/i.test(r.block_reason)
                  ? { bg: "rgba(245,158,11,0.16)", color: "var(--color-warn)", label: "⚠" }
                  : { bg: "rgba(148,163,184,0.15)", color: "var(--color-text-muted)", label: "·" };
              return (
                <li
                  key={`${r.timestamp}-${i}`}
                  className="flex items-center gap-2 py-1 px-2 rounded hover:bg-[var(--color-panel)] transition-colors"
                  title={r.block_reason ?? (r.executed ? "Approved" : "Flat")}
                >
                  <span
                    className="mono text-[10px] text-[var(--color-text-dim)] w-[96px] shrink-0 whitespace-nowrap"
                    title={new Date(
                      r.timestamp.endsWith("Z") || /[+-]\d\d:?\d\d$/.test(r.timestamp)
                        ? r.timestamp
                        : `${r.timestamp}Z`,
                    ).toLocaleString()}
                  >
                    {fmtClockDate(r.timestamp)}
                  </span>
                  <span
                    className="inline-flex items-center justify-center w-5 h-5 rounded-full text-[10px] shrink-0"
                    style={{ background: dot.bg, color: dot.color }}
                  >
                    {dot.label}
                  </span>
                  <span className="text-[11px] text-[var(--color-text-muted)] flex-1 truncate">
                    {r.block_reason ??
                      (r.executed
                        ? `Approved · ${r.direction?.toUpperCase() ?? ""}`
                        : "Flat")}
                  </span>
                  <span className="mono text-[11px] shrink-0" style={{ color: scoreCol }}>
                    {fmtScore(r.combined_score)}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
        </div>

        {/* Pipeline funnel as a third grid row spanning all 12 columns so
            its left/right edges align exactly with main+rail content. */}
        <div className="xl:col-span-12">
          <SignalPipelineCard symbol={symbol} />
        </div>
      </div>
    </div>
  );
}

// (removed SignalDensityCard — replaced by SignalTensionCard 2026-04-19)

// ─── HMM regime evolution · 48h ──────────────────────────────────────

type HistoryRow = {
  timestamp: string;
  regime: string | null;
  regime_prob: number | null;
};

function RegimeEvolutionCard({ history }: { history: HistoryRow[] }) {
  const BUCKETS = 12;
  const WINDOW_MS = 48 * 3600_000;

  const bars = useMemo(() => {
    const now = Date.now();
    const bucketMs = WINDOW_MS / BUCKETS;
    const buckets: Array<{
      regime: string | null;
      prob: number;
      count: number;
    }> = Array.from({ length: BUCKETS }, () => ({ regime: null, prob: 0, count: 0 }));
    for (const h of history) {
      if (!h.regime) continue;
      const t = new Date(
        h.timestamp.endsWith("Z") || /[+-]\d\d:?\d\d$/.test(h.timestamp)
          ? h.timestamp
          : `${h.timestamp}Z`,
      ).getTime();
      if (!Number.isFinite(t)) continue;
      const age = now - t;
      if (age < 0 || age > WINDOW_MS) continue;
      const idx = BUCKETS - 1 - Math.floor(age / bucketMs);
      if (idx < 0 || idx >= BUCKETS) continue;
      const cur = buckets[idx];
      cur.count += 1;
      cur.prob = Math.max(cur.prob, h.regime_prob ?? 0);
      cur.regime = h.regime; // keep the most recent regime seen in bucket
    }
    return buckets;
  }, [history]);

  const regimeBg = (r: string | null) => {
    if (!r) return "var(--color-panel-hi)";
    return regimeColor(r);
  };

  const hasData = bars.some((b) => b.count > 0);

  return (
    <div className="rounded-xl bg-[var(--color-panel-hi)] border border-[var(--color-border)] p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <p className="section-label">HMM regime evolution · 48h</p>
        <span className="text-[11px] text-[var(--color-text-dim)]">
          hover for bucket detail
        </span>
      </div>
      {hasData ? (
        <>
          {/* flex-1 bars — heatmap grows to fill stretched card height so
              there's no whitespace at the bottom when main col is forced
              taller by the rail's flex-1 Recent decisions. */}
          <div className="grid grid-cols-12 gap-1 flex-1 min-h-[56px]">
            {bars.map((b, i) => {
              const isNow = i === BUCKETS - 1;
              return (
                <div
                  key={i}
                  className="rounded-sm"
                  title={
                    b.count === 0
                      ? "no data"
                      : `${b.regime} · conf ${(b.prob * 100).toFixed(0)}% · ${b.count} bar${b.count === 1 ? "" : "s"}`
                  }
                  style={{
                    background: regimeBg(b.regime),
                    opacity: b.count === 0 ? 0.15 : 0.35 + 0.65 * Math.min(b.prob, 1),
                    boxShadow: isNow ? "0 0 0 1px var(--color-primary)" : undefined,
                    border: isNow ? "1px solid var(--color-primary)" : undefined,
                  }}
                />
              );
            })}
          </div>
          <div className="flex justify-between mt-2 mono text-[10px] text-[var(--color-text-dim)]">
            <span>48h ago</span>
            <span>36h</span>
            <span>24h</span>
            <span>12h</span>
            <span>now</span>
          </div>
          <p className="mt-3 text-xs text-[var(--color-text-muted)]">
            {describeRegimeRun(bars)}
          </p>
        </>
      ) : (
        <p className="text-center text-xs text-[var(--color-text-dim)] py-6">
          No regime history in the last 48h.
        </p>
      )}
    </div>
  );
}

function describeRegimeRun(
  bars: Array<{ regime: string | null; prob: number; count: number }>,
): string {
  const recent = bars.slice().reverse().find((b) => b.regime);
  if (!recent) return "No regime data in window.";
  let runHours = 0;
  for (let i = bars.length - 1; i >= 0; i--) {
    if (bars[i].regime === recent.regime) runHours += 48 / bars.length;
    else break;
  }
  return `Regime ${recent.regime} for the past ~${Math.round(runHours)}h · confidence ${(recent.prob * 100).toFixed(0)}%.`;
}

function SymbolSwitcherChip({ symbol, active }: { symbol: string; active: boolean }) {
  const { price, changePct } = useLatestPrice(symbol);
  const { data: live } = useLiveState();
  const signal = live?.signals[symbol];
  const regime = signal?.regime.regime_label;
  const regimeCol = regime ? regimeColor(regime) : "var(--color-text-dim)";
  const priceDeltaCol =
    changePct == null ? "var(--color-text-muted)" : changePct >= 0 ? "var(--color-profit)" : "var(--color-loss)";
  return (
    <Link
      to={`/ui/signals/${symbol}`}
      className="shrink-0 flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-colors"
      style={
        active
          ? {
              background: "rgba(6,182,212,0.15)",
              color: "var(--color-primary)",
              border: "1px solid rgba(6,182,212,0.35)",
              fontWeight: 600,
            }
          : {
              background: "var(--color-panel-hi)",
              color: "var(--color-text-muted)",
            }
      }
    >
      <span className="mono">{symbol}</span>
      <span className="tnum">{formatPrice(price, symbol)}</span>
      {changePct != null && (
        <span className="tnum text-[10px]" style={{ color: priceDeltaCol }}>
          {fmtPct(changePct)}
        </span>
      )}
      {regime && (
        <span
          className="text-[9px] px-1.5 py-0.5 rounded-full"
          style={{ background: `${regimeCol}26`, color: regimeCol }}
        >
          {regime}
        </span>
      )}
    </Link>
  );
}

function BreakdownRow({
  label,
  value,
  rangeMin,
  rangeMax,
  fmt,
}: {
  label: string;
  value: number;
  rangeMin: number;
  rangeMax: number;
  fmt: (v: number) => string;
}) {
  const span = rangeMax - rangeMin;
  const pct = span > 0 ? ((value - rangeMin) / span) * 100 : 0;
  return (
    <div>
      <div className="flex items-center justify-between text-xs mb-1">
        <span className="text-[var(--color-text-muted)]">{label}</span>
        <span className="mono">{fmt(value)}</span>
      </div>
      <div className="h-1.5 rounded-full bg-[var(--color-panel)] overflow-hidden">
        <div
          className="h-full rounded-full bg-brand-gradient"
          style={{ width: `${Math.min(Math.max(pct, 0), 100).toFixed(1)}%` }}
        />
      </div>
    </div>
  );
}

const REGIME_ORDER = ["Crash", "Bear", "Neutral", "Bull", "Euphoria"] as const;
function RegimeBarsInline({ probs, currentIdx }: { probs: number[]; currentIdx: number }) {
  if (!probs || probs.length !== 5) return null;
  const max = Math.max(...probs);
  const colors = [
    "var(--regime-crash, #7f1d1d)",
    "var(--regime-bear, #f43f5e)",
    "var(--regime-neutral, #94a3b8)",
    "var(--regime-bull, #10b981)",
    "var(--regime-euphoria, #8b5cf6)",
  ];
  return (
    <div className="space-y-2">
      {REGIME_ORDER.map((name, i) => {
        const p = probs[i] ?? 0;
        const pct = p * 100;
        const isCurrent = i === currentIdx;
        const color = colors[i];
        return (
          <div key={name} className="flex items-center gap-2">
            <span
              className="w-16 text-[11px]"
              style={{
                color: isCurrent ? "var(--color-text)" : "var(--color-text-dim)",
                fontWeight: isCurrent ? 600 : 400,
              }}
            >
              {name}
            </span>
            <div className="flex-1 h-1.5 rounded-full bg-[var(--color-panel)] overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.min(pct, 100).toFixed(1)}%`,
                  background: color,
                  opacity: p === max && max > 0 ? 1 : 0.6,
                }}
              />
            </div>
            <span className="mono w-10 text-right text-[11px]">{pct.toFixed(0)}%</span>
          </div>
        );
      })}
    </div>
  );
}

export function SignalPage() {
  const { symbol } = useParams<{ symbol: string }>();
  if (!symbol) return <SignalsIndex />;
  return <SignalDetail symbol={symbol} />;
}
