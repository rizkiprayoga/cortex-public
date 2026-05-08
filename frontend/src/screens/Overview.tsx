import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Download, RefreshCcw, X } from "lucide-react";
import { useLiveState } from "@/hooks/useLiveState";
import { useEquityCurve } from "@/hooks/useHistory";
import { useCandles, useLatestPrice } from "@/hooks/useCandles";
import { usePositions } from "@/hooks/usePositions";
import { useModelSummary } from "@/hooks/useModels";
import { useInvariants } from "@/hooks/useInvariants";
import { useNewsBlackouts, type NewsSymbolEntry } from "@/hooks/useNewsBlackouts";
import { useCurrentAccount } from "@/hooks/useAccount";
import { useTradeHistory } from "@/hooks/useHistory";
import { useRiskConfig } from "@/hooks/useConfig";
import { useSignalAudit } from "@/hooks/useSignalAudit";
import { durationInRegime, fmtRegimeDuration, type RegimeHistoryRow } from "@/lib/regime";
import { EquityChart, type EquityPoint as ChartPoint } from "@/components/EquityChart";
import { PriceChart, type ChartType } from "@/components/PriceChart";
import { LiveDot } from "@/components/LiveDot";
import { Sparkline } from "@/components/Sparkline";
import { useNewsBlackoutForSymbol } from "@/hooks/useNewsBlackouts";
import { regimeColor, regimeColors } from "@/lib/tokens";
import { usd } from "@/lib/format";
import type { ChartTimeframe, SignalData } from "@/lib/types";
import { SkeletonCard, SkeletonChart } from "@/components/states/Skeleton";
import { useFlashOnChange } from "@/hooks/useFlashOnChange";

import { LIVE_SYMBOLS as DEFAULT_SYMBOLS } from "@/lib/symbols";

const MIN_PREDICTIONS = 50;
const TF_OPTIONS: ChartTimeframe[] = ["H1", "H4", "D1"];
const CHART_TYPE_OPTIONS: ChartType[] = ["candles", "line", "area"];
const CHART_TYPE_STORAGE = "cortex-chart-type";
const CHART_SYMBOL_STORAGE = "cortex-chart-symbol";
const CHART_TYPE_LABEL: Record<ChartType, string> = {
  candles: "Candles",
  line: "Line",
  area: "Area",
};

function readChartType(): ChartType {
  try {
    const v = window.localStorage.getItem(CHART_TYPE_STORAGE);
    if (v === "candles" || v === "line" || v === "area") return v;
  } catch {
    /* localStorage disabled */
  }
  return "candles";
}

function readChartSymbol(): string {
  try {
    const v = window.localStorage.getItem(CHART_SYMBOL_STORAGE);
    if (v && (DEFAULT_SYMBOLS as readonly string[]).includes(v)) return v;
  } catch {
    /* localStorage disabled */
  }
  return "XAUUSD";
}

function modelDotStatus(dirAcc: number | null): "live" | "stale" | "dead" {
  if (dirAcc == null) return "stale";
  if (dirAcc >= 0.52) return "live";
  if (dirAcc >= 0.48) return "stale";
  return "dead";
}

function fmtPct(v: number): string {
  const sign = v > 0 ? "+" : v < 0 ? "−" : "";
  return `${sign}${Math.abs(v).toFixed(2)}%`;
}

function fmtShortDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function relHours(iso: string): string {
  const ms = new Date(iso).getTime() - Date.now();
  const abs = Math.abs(ms);
  const h = Math.floor(abs / 3_600_000);
  if (h >= 48) return `${Math.floor(h / 24)}d`;
  if (h > 0) return `${h}h`;
  const m = Math.floor(abs / 60_000);
  return `${m}m`;
}

// ─── Hero KPI ─────────────────────────────────────────────────────────

interface KpiCardProps {
  label: string;
  value: string;
  href: string;
  accent?: "profit" | "loss" | "neutral";
  topAccent?: string; // CSS color for border-top stripe
  chip?: React.ReactNode;
  sub?: React.ReactNode;
  spark?: { data: number[]; color: string };
  /** Numeric value to watch for tick-flash (green up, red down). Omit to disable. */
  flashValue?: number | null;
}

function KpiCard({ label, value, href, accent = "neutral", topAccent, chip, sub, spark, flashValue }: KpiCardProps) {
  const valueColor =
    accent === "profit" ? "var(--color-profit)" : accent === "loss" ? "var(--color-loss)" : undefined;
  const flashClass = useFlashOnChange(flashValue);
  return (
    <Link
      to={href}
      className="group relative block rounded-xl border bg-[var(--color-panel)] border-[var(--color-border)] p-5 transition-all hover:-translate-y-px hover:border-[color:rgba(99,102,241,0.35)] hover:shadow-[0_10px_28px_-14px_rgba(99,102,241,0.35)]"
      style={topAccent ? { borderTop: `2px solid ${topAccent}` } : undefined}
    >
      <div className="flex items-center justify-between">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-text-dim)] transition-colors group-hover:text-[var(--chip-info-fg,theme(colors.indigo.200))]">
          {label}
          <ArrowRight
            size={11}
            className="ml-1 inline opacity-0 -translate-x-0.5 transition-all group-hover:opacity-100 group-hover:translate-x-0"
            style={{ color: "var(--indigo)" }}
          />
        </p>
        {chip}
      </div>
      <p
        className={`num mt-3 text-[44px] leading-none font-bold tracking-[-0.025em] inline-block ${flashClass}`}
        style={{
          color: valueColor,
          ...(accent === "neutral" && !valueColor
            ? {
                background:
                  "linear-gradient(135deg, var(--hero-grad-from) 0%, var(--hero-grad-to) 100%)",
                WebkitBackgroundClip: "text",
                backgroundClip: "text",
                WebkitTextFillColor: "transparent",
                color: "transparent",
              }
            : {}),
        }}
      >
        {value}
      </p>
      <div className="mt-2 flex items-center justify-between">
        <span className="text-xs text-[var(--color-text-muted)]">{sub}</span>
        {spark && spark.data.length >= 2 ? (
          <Sparkline data={spark.data} width={80} height={24} color={spark.color} />
        ) : (
          <svg width="80" height="24" viewBox="0 0 80 24" aria-hidden>
            <line x1="0" y1="12" x2="80" y2="12" stroke="var(--color-text-dim)" strokeDasharray="2 2" strokeWidth="1" opacity="0.5" />
          </svg>
        )}
      </div>
    </Link>
  );
}

// Daily R card — routes to Signals Log (today's decisions)
function DailyRCard({
  r,
  count,
  cap,
  capPct,
}: {
  r: number;
  count: number;
  cap: number;
  capPct: number;
}) {
  const rColor =
    r > 0.1 ? "var(--color-profit)" : r < -0.1 ? "var(--color-loss)" : undefined;
  return (
    <Link
      to="/ui/signals-log"
      className="group relative block rounded-xl border bg-[var(--color-panel)] border-[var(--color-border)] p-5 transition-all hover:-translate-y-px hover:border-[color:rgba(99,102,241,0.35)] hover:shadow-[0_10px_28px_-14px_rgba(99,102,241,0.35)]"
    >
      <div className="flex items-center justify-between">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-text-dim)] transition-colors group-hover:text-[var(--chip-info-fg,theme(colors.indigo.200))]">
          Daily R
          <ArrowRight
            size={11}
            className="ml-1 inline opacity-0 -translate-x-0.5 transition-all group-hover:opacity-100 group-hover:translate-x-0"
            style={{ color: "var(--indigo)" }}
          />
        </p>
        <span className="mono text-[10px] text-[var(--color-text-dim)]">
          {count}/{cap} trades
        </span>
      </div>
      <p
        className="tnum mt-3 text-[44px] leading-none font-bold tracking-[-0.025em]"
        style={{
          color: rColor,
          ...(rColor
            ? {}
            : {
                background:
                  "linear-gradient(135deg, var(--hero-grad-from) 0%, var(--hero-grad-to) 100%)",
                WebkitBackgroundClip: "text",
                backgroundClip: "text",
                WebkitTextFillColor: "transparent",
                color: "transparent",
              }),
        }}
      >
        {r >= 0 ? "+" : "−"}
        {Math.abs(r).toFixed(1)}
        <span className="text-2xl text-[var(--color-text-muted)] ml-1.5">R</span>
      </p>
      <div className="mt-3">
        <div className="h-1.5 rounded-full bg-[var(--color-panel-hi)] overflow-hidden">
          <div
            className="h-full rounded-full bg-brand-gradient"
            style={{ width: `${capPct.toFixed(1)}%` }}
          />
        </div>
        <p className="text-[10px] text-[var(--color-text-dim)] mt-1">
          Rolling 24h cap
        </p>
      </div>
    </Link>
  );
}

// ─── Rail: compact Health + Next-news ────────────────────────────────

function HealthRailCard() {
  const { data } = useInvariants(20);
  const findings = data?.findings ?? [];
  const lastFiring = findings.find((f) => !f.passed);
  const allClear = !lastFiring;
  // "since" = how long since we *last* saw a firing event (or bot start if none).
  // Use the oldest passing finding's timestamp as a proxy for "watching since".
  const since = useMemo(() => {
    if (findings.length === 0) return null;
    const oldest = findings[findings.length - 1].ts;
    const ms = Date.now() - new Date(oldest).getTime();
    if (!Number.isFinite(ms) || ms < 0) return null;
    const h = Math.floor(ms / 3_600_000);
    if (h >= 24) return `${Math.floor(h / 24)}d ${h % 24}h`;
    if (h > 0) return `${h}h`;
    const m = Math.floor(ms / 60_000);
    return `${m}m`;
  }, [findings]);
  return (
    <Link
      to="/ui/system"
      className="group block rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5 transition-all hover:-translate-y-px hover:border-[color:rgba(99,102,241,0.35)]"
    >
      <div className="flex items-center justify-between mb-3">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-text-dim)]">
          Health
        </p>
        <span className="mono text-[10px] text-[var(--color-text-dim)]">
          {findings.length} invariants
        </span>
      </div>
      <div className="flex items-center gap-3">
        <div
          className="w-12 h-12 shrink-0 rounded-full flex items-center justify-center text-2xl leading-none"
          style={{
            background: allClear ? "rgba(16,185,129,0.12)" : "rgba(244,63,94,0.12)",
            border: `1px solid ${allClear ? "rgba(16,185,129,0.28)" : "rgba(244,63,94,0.28)"}`,
            color: allClear ? "var(--color-profit)" : "var(--color-loss)",
          }}
          aria-hidden
        >
          {allClear ? "✓" : "⚠"}
        </div>
        <div className="min-w-0">
          <p className="text-sm font-semibold text-[var(--color-text)] truncate">
            {allClear ? "All systems nominal" : lastFiring.invariant}
          </p>
          <p className="text-xs text-[var(--color-text-muted)] truncate">
            {allClear
              ? since
                ? `Last fired: none · since ${since}`
                : "No invariants recorded yet"
              : lastFiring.message}
          </p>
        </div>
      </div>
    </Link>
  );
}

interface NewsCardEvent {
  cb: string;
  event_utc: string;
  blackout_start_utc: string;
  blackout_end_utc: string;
  affected: string[];
  state: "active" | "upcoming";
}

const MAX_NEWS_EVENTS = 3;

// ─── News event row + drawer ──────────────────────────────────────────
//
// Shared row renderer used by both the rail card (3 visible) and the
// "View all" drawer (no cap). Same visual treatment in both places so
// nothing surprises the operator when they expand.

function NewsEventRow({ ev }: { ev: NewsCardEvent }) {
  const isActive = ev.state === "active";
  const countdownLabel = isActive
    ? `ends in ${relHours(ev.blackout_end_utc)}`
    : `blackout in ${relHours(ev.blackout_start_utc)}`;
  const countdownColor = isActive ? "var(--color-loss)" : "var(--color-warn)";
  return (
    <li
      key={`${ev.cb}|${ev.event_utc}`}
      className="rounded-lg border border-[var(--color-border)] bg-[var(--color-panel-hi)]/40 p-2.5"
    >
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-semibold text-[var(--color-text)] truncate">
          {ev.cb}
        </p>
        <span className="mono text-[10px] shrink-0" style={{ color: countdownColor }}>
          {countdownLabel}
        </span>
      </div>
      <div className="flex items-center justify-between gap-2 mt-1">
        <p className="text-[11px] text-[var(--color-text-muted)]">
          {fmtShortDate(ev.event_utc)} · {isActive ? "in blackout" : "T−24h blackout"}
        </p>
        <div className="flex flex-wrap gap-1 justify-end">
          {ev.affected.map((sym) => (
            <Link
              key={sym}
              to={`/ui/signals/${sym}`}
              className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] bg-[var(--color-panel-hi)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
            >
              {sym}
            </Link>
          ))}
        </div>
      </div>
    </li>
  );
}

function NewsEventsDrawer({
  events,
  onClose,
}: {
  events: NewsCardEvent[];
  onClose: () => void;
}) {
  // Local Escape-key handler — mirrors PositionDrawer's pattern. Inlined
  // here rather than extracted to avoid a new shared-hook file for one use.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);
  const activeCount = events.filter((e) => e.state === "active").length;
  const upcomingCount = events.length - activeCount;
  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} aria-hidden />
      <aside
        className="relative w-full max-w-md bg-[var(--color-panel)] border-l border-[var(--color-border-hi)] overflow-y-auto p-6 shadow-2xl"
        style={{ boxShadow: "-20px 0 60px rgba(0,0,0,0.5)" }}
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-[var(--color-text)]">
              News events
            </h2>
            <p className="text-xs text-[var(--color-text-muted)]">
              {activeCount} active · {upcomingCount} upcoming
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md hover:bg-[var(--color-panel-hi)] text-[var(--color-text-muted)]"
            aria-label="Close"
          >
            <X size={16} />
          </button>
        </div>
        <ul className="flex flex-col gap-2.5">
          {events.map((ev) => (
            <NewsEventRow key={`${ev.cb}|${ev.event_utc}`} ev={ev} />
          ))}
        </ul>
        <p className="mt-5 text-[10px] text-[var(--color-text-dim)]">
          Blackout: T−24h to T+2h · bot skips entries on these symbols during the window.
        </p>
      </aside>
    </div>
  );
}

function NextNewsRailCard() {
  const { data } = useNewsBlackouts();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const events: NewsCardEvent[] = useMemo(() => {
    if (!data) return [];
    // Collect every distinct event across non-exempt symbols. The same event
    // (e.g. FOMC) commonly applies to several symbols — dedupe by name+time
    // and accumulate the affected list. Includes both currently-active
    // blackouts and the next upcoming one per symbol, so an overlapping pair
    // (yesterday's BoJ post-news + today's BoC) shows both rows.
    const acc = new Map<string, NewsCardEvent>();
    const push = (
      ev: NonNullable<NewsSymbolEntry["next_event"] | NewsSymbolEntry["active_event"]>,
      symbol: string,
      state: "active" | "upcoming",
    ) => {
      const key = `${ev.cb}|${ev.event_utc}`;
      const existing = acc.get(key);
      if (existing) {
        if (!existing.affected.includes(symbol)) existing.affected.push(symbol);
        if (state === "active") existing.state = "active";
      } else {
        acc.set(key, {
          cb: ev.cb,
          event_utc: ev.event_utc,
          blackout_start_utc: ev.blackout_start_utc,
          blackout_end_utc: ev.blackout_end_utc,
          affected: [symbol],
          state,
        });
      }
    };
    for (const s of data.symbols) {
      if (s.exempt) continue;
      if (s.active_event) push(s.active_event, s.symbol, "active");
      if (s.next_event) push(s.next_event, s.symbol, "upcoming");
    }
    // 2-tier sort (operator priority): currently-blocking trades come first,
    // upcoming blocks come after. The earlier "imminent <15min upcoming bumps
    // to top" rule was confusing — a soon-to-start blackout would jump above
    // currently-active blackouts that block trades RIGHT NOW. Operator's
    // mental model is "what is currently affecting trading, then what's next".
    //   Tier 1: active blackouts — sorted by ends-soonest
    //   Tier 2: upcoming blackouts — sorted by starts-soonest
    return Array.from(acc.values()).sort((a, b) => {
      if (a.state !== b.state) return a.state === "active" ? -1 : 1;
      if (a.state === "active") {
        return new Date(a.blackout_end_utc).getTime() - new Date(b.blackout_end_utc).getTime();
      }
      return new Date(a.blackout_start_utc).getTime() - new Date(b.blackout_start_utc).getTime();
    });
  }, [data]);

  if (events.length === 0) {
    return (
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5 flex-1 flex flex-col">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-text-dim)] mb-2">
          Next news events
        </p>
        <p className="text-sm text-[var(--color-text-muted)]">No upcoming blackouts in feed.</p>
      </div>
    );
  }

  const visible = events.slice(0, MAX_NEWS_EVENTS);
  const overflow = events.length - visible.length;

  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5 flex-1 flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-text-dim)]">
          Next news events
        </p>
        <span className="mono text-[10px] text-[var(--color-text-dim)]">
          {events.length} upcoming
        </span>
      </div>
      <ul className="flex flex-col gap-2.5">
        {visible.map((ev) => (
          <NewsEventRow key={`${ev.cb}|${ev.event_utc}`} ev={ev} />
        ))}
      </ul>
      <div className="mt-2 flex justify-end">
        {overflow > 0 ? (
          <button
            onClick={() => setDrawerOpen(true)}
            className="text-[11px] text-[var(--color-primary)] hover:brightness-125"
          >
            View all {events.length} →
          </button>
        ) : (
          // Even when nothing overflows, expose the drawer so operator can
          // always inspect the same data without having to wait for >3 events.
          <button
            onClick={() => setDrawerOpen(true)}
            className="text-[11px] text-[var(--color-text-muted)] hover:text-[var(--color-primary)]"
          >
            Details →
          </button>
        )}
      </div>
      <div className="flex-1" />
      <p className="mt-3 text-[10px] text-[var(--color-text-dim)]">
        Blackout: T−24h to T+2h · bot skips entries on these symbols during the window.
      </p>
      {drawerOpen && (
        <NewsEventsDrawer events={events} onClose={() => setDrawerOpen(false)} />
      )}
    </div>
  );
}

// ─── Per-symbol regime cell (matches mockup regime card exactly) ──────

const REGIME_ORDER = ["Crash", "Bear", "Neutral", "Bull", "Euphoria"] as const;
const REGIME_AXIS_LABELS = ["C", "B", "N", "B", "E"] as const;

function formatPrice(price: number | null, symbol: string): string {
  if (price == null || !Number.isFinite(price)) return "—";
  if (symbol.includes("JPY")) return price.toFixed(3);
  if (symbol === "XAUUSD" || symbol === "ETHUSD" || symbol === "BTCUSD") {
    return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return price.toFixed(5);
}

function RegimeCell({
  symbol,
  signal,
  history,
}: {
  symbol: string;
  signal: SignalData;
  history: RegimeHistoryRow[];
}) {
  const { price, changePct } = useLatestPrice(symbol);
  const { entry: news } = useNewsBlackoutForSymbol(symbol);
  const regime = signal.regime.regime_label;
  const regimeCol = regimeColor(regime);
  const stateConfPct = (signal.regime.state_probability * 100).toFixed(2);
  const heldRun = durationInRegime(history, regime);
  const heldText = fmtRegimeDuration(heldRun);
  const probs = signal.regime.all_probabilities ?? [];
  const maxProb = probs.length ? Math.max(...probs) : 0;

  const score = signal.combined_score;
  const scoreCol =
    score > 0.1 ? "var(--color-profit)" : score < -0.1 ? "var(--color-loss)" : "var(--color-text-muted)";
  const priceDeltaCol =
    changePct == null ? "var(--color-text-muted)" : changePct >= 0 ? "var(--color-profit)" : "var(--color-loss)";

  const topAccent =
    regime === "Bull" || regime === "Euphoria"
      ? "var(--color-profit)"
      : regime === "Bear" || regime === "Crash"
        ? "var(--color-loss)"
        : "var(--color-text-dim)";
  const glowColor =
    regime === "Bull" || regime === "Euphoria"
      ? "rgba(16,185,129,0.22)"
      : regime === "Bear" || regime === "Crash"
        ? "rgba(244,63,94,0.22)"
        : "rgba(148,163,184,0.18)";

  // Footer note: surface the actual block reason from the signal_combiner's
  // reasoning trail, NOT a guess based on score magnitude. The trail joins
  // strings with " | " and the block reason is the last segment.
  // Patterns we recognize (signal_combiner.py emits these):
  //   "flickering: only 1/2 bars in history, waiting for stability"
  //   "confluence_fail: <dir> contradicts <regime>, HMM=<x>"
  //   "below_threshold: |score|=0.32 < 0.45"
  //   "long_only_mode: short blocked"
  // Anything else falls back to the raw last segment (truncated).
  const footerNote = (() => {
    if (signal.should_trade) return signal.direction?.toUpperCase() ?? "—";
    const trail = signal.reasoning ?? [];
    const last = trail.length > 0 ? trail[trail.length - 1] : "";
    const lastSegment = last.split(" | ").pop() ?? "";
    if (lastSegment.includes("flickering")) return "Flat · waiting for stability";
    if (lastSegment.includes("confluence_fail")) return "Flat · regime/direction mismatch";
    if (lastSegment.includes("below_threshold")) return `Flat · |score| ${Math.abs(score).toFixed(2)} below threshold`;
    if (lastSegment.includes("long_only")) return "Flat · long-only blocks short";
    if (lastSegment.includes("direction_conflict")) return "Flat · waiting for prior exit";
    if (!lastSegment) return "Flat";
    // Fallback for unrecognized reasons — show first 40 chars
    return `Flat · ${lastSegment.length > 40 ? lastSegment.slice(0, 37) + "…" : lastSegment}`;
  })();
  // News/blackout status — prefer ACTIVE blackout over next-upcoming
  // event so the card shows the reason trades are currently suppressed.
  const newsBlackoutActive = news?.state === "blackout" && !!news.active_event;
  const newsPostNews = news?.state === "post_news" && !!news.active_event;
  let footerNews = "";
  let footerNewsColor: string | undefined;
  if (newsBlackoutActive && news?.active_event) {
    footerNews = `🔒 ${news.active_event.cb} · ends ${relHours(news.active_event.blackout_end_utc)}`;
    footerNewsColor = "var(--color-loss)";
  } else if (newsPostNews && news?.active_event) {
    footerNews = `${news.active_event.cb} · T+2h cooldown`;
    footerNewsColor = "var(--color-warn)";
  } else if (news?.next_event) {
    footerNews = `${news.next_event.cb} ${relHours(news.next_event.blackout_start_utc)}`;
  } else if (symbol === "ETHUSD") {
    footerNews = "crypto 24/7";
  } else if (symbol === "XAUUSD") {
    footerNews = "XAU exempt";
  }

  return (
    <Link
      to={`/ui/signals/${symbol}`}
      className="group relative overflow-hidden block rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-4 transition-all hover:-translate-y-px hover:border-[color:rgba(99,102,241,0.35)] hover:shadow-[0_10px_28px_-14px_rgba(99,102,241,0.35)]"
      style={{ borderTop: `2px solid ${topAccent}` }}
    >
      {/* corner glow */}
      <div
        aria-hidden
        className="absolute inset-0 opacity-30 pointer-events-none"
        style={{ background: `radial-gradient(circle at 100% 0%, ${glowColor}, transparent 60%)` }}
      />
      <div className="relative">
        {/* Header: symbol + regime chip */}
        <div className="flex items-center justify-between">
          <span className="mono text-[11px] font-semibold tracking-wide text-[var(--color-text)]">
            {symbol}
          </span>
          <span
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
            style={{ background: `${regimeCol}26`, color: regimeCol }}
            title={`HMM regime confidence: ${stateConfPct}%`}
          >
            {regime}
          </span>
        </div>

        {/* Price + change */}
        <p className="tnum text-2xl font-bold mt-1 text-[var(--color-text)]">
          {formatPrice(price, symbol)}
        </p>
        <p className="tnum text-xs" style={{ color: priceDeltaCol }}>
          {changePct == null
            ? "—"
            : `${changePct >= 0 ? "▲" : "▼"} ${fmtPct(changePct)}`}
        </p>

        {/* Regime histogram (C · B · N · B · E) */}
        {probs.length === 5 && (
          <div className="mt-3">
            <div className="flex justify-between text-[9px] font-semibold uppercase tracking-wider text-[var(--color-text-dim)] mb-1">
              <span>Regime probs</span>
              <span title={`Held in ${regime} for ${heldText} · posterior ${stateConfPct}%`}>
                conf {stateConfPct}%
                {heldText !== "—" && (
                  <span className="ml-1 normal-case tracking-normal">· {heldText}</span>
                )}
              </span>
            </div>
            <div className="flex items-end gap-1 h-8">
              {REGIME_ORDER.map((name, i) => {
                const p = probs[i] ?? 0;
                const h = maxProb > 0 ? Math.max((p / maxProb) * 100, 4) : 4;
                const isMax = p === maxProb && maxProb > 0;
                return (
                  <div
                    key={name}
                    className="flex-1 rounded-t"
                    style={{
                      height: `${h}%`,
                      background: regimeColors[name],
                      opacity: isMax ? 1 : 0.5,
                    }}
                    title={`${name}: ${(p * 100).toFixed(0)}%`}
                  />
                );
              })}
            </div>
            <div className="flex gap-1 mt-1 mono text-[9px] text-[var(--color-text-dim)]">
              {REGIME_AXIS_LABELS.map((l, i) => (
                <span key={i} className="flex-1 text-center">{l}</span>
              ))}
            </div>
          </div>
        )}

        {/* Score + mini-sparkline */}
        <div className="mt-3 flex items-center justify-between">
          <div>
            <p className="text-[9px] font-semibold uppercase tracking-wider text-[var(--color-text-dim)]">
              Score
            </p>
            <p className="tnum text-lg font-bold" style={{ color: scoreCol }}>
              {score >= 0 ? "+" : ""}{score.toFixed(2)}
            </p>
          </div>
          <Sparkline
            data={[0, score * 0.3, score * 0.55, score * 0.8, score]}
            width={56}
            height={24}
            color={scoreCol}
          />
        </div>

        {/* Footer row */}
        <div
          className="mt-2 pt-2 border-t flex justify-between text-[10px] text-[var(--color-text-dim)]"
          style={{ borderColor: "var(--color-border)" }}
        >
          <span>{footerNote}</span>
          {footerNews && (
            <span
              className="mono"
              style={footerNewsColor ? { color: footerNewsColor } : undefined}
              title={
                newsBlackoutActive && news?.active_event
                  ? `Blackout ends ${news.active_event.blackout_end_utc}`
                  : undefined
              }
            >
              {footerNews}
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}

// ─── Model Health row (mini bar list) ────────────────────────────────

function ModelHealthPanel() {
  const { data: modelSummary } = useModelSummary();
  if (!modelSummary || modelSummary.symbols.length === 0) return null;

  // Max staleness across all per-symbol LSTM artifacts. Monthly retrain
  // cadence — so >30 days means the 1st-of-month cron may have missed a
  // run (caught a silent retrain failure on 2026-04-18 per memory).
  const now = Date.now();
  const ages = modelSummary.symbols
    .map((m) => m.lstm_trained_at ?? m.lstm_file_mtime)
    .filter((s): s is string => !!s)
    .map((iso) => (now - new Date(iso).getTime()) / 86_400_000);
  const maxAgeDays = ages.length > 0 ? Math.max(...ages) : null;
  const stalenessColor =
    maxAgeDays == null
      ? "var(--color-text-dim)"
      : maxAgeDays > 45
        ? "var(--color-loss)"
        : maxAgeDays > 30
          ? "var(--color-warn)"
          : "var(--color-profit)";

  return (
    <div className="h-full rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-text-dim)]">
          Model health · directional accuracy
        </p>
        <div className="flex items-center gap-2">
          {maxAgeDays != null && (
            <span
              className="mono text-[10px] px-2 py-0.5 rounded-full"
              style={{
                background: `color-mix(in oklab, ${stalenessColor} 14%, transparent)`,
                color: stalenessColor,
              }}
              title={`Oldest LSTM artifact age — retrain cron runs monthly. >30d = cron miss suspected; >45d = overdue.`}
            >
              oldest {Math.floor(maxAgeDays)}d
            </span>
          )}
          <Link to="/ui/models" className="text-[11px] text-[var(--color-primary)] hover:brightness-125">
            details →
          </Link>
        </div>
      </div>
      <div className="space-y-2.5">
        {modelSummary.symbols.map((m) => {
          const warmingUp = m.n_predictions < MIN_PREDICTIONS;
          const dirAcc = m.live_dir_acc;
          const pct = dirAcc != null ? dirAcc * 100 : 0;
          const barColor =
            warmingUp
              ? "var(--color-text-dim)"
              : dirAcc == null
                ? "var(--color-text-dim)"
                : dirAcc >= 0.52
                  ? "var(--color-profit)"
                  : dirAcc >= 0.48
                    ? "var(--color-warn)"
                    : "var(--color-loss)";
          const label = m.symbol;  // Show full pair name (was truncated to "GBP" etc., ambiguous when multiple GBP-pairs in universe)
          return (
            <div key={m.symbol} className="flex items-center gap-3">
              <span className="mono w-20 text-[11px] font-medium text-[var(--color-text)]">
                {label}
              </span>
              <div className="flex-1 h-1.5 rounded-full bg-[var(--color-panel-hi)] overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${Math.min(Math.max(pct, 0), 100)}%`, background: barColor }}
                />
              </div>
              <span
                className="mono w-14 text-right text-[11px] font-semibold"
                style={{ color: barColor }}
                title={`${m.n_predictions} predictions`}
              >
                {warmingUp || dirAcc == null ? "warming" : `${pct.toFixed(1)}%`}
              </span>
              <LiveDot status={modelDotStatus(dirAcc)} size={6} />
            </div>
          );
        })}
      </div>
      <p className="mt-4 text-[10px] text-[var(--color-text-dim)]">
        Green ≥52% · Amber 48–52% · Dim warming-up
      </p>
    </div>
  );
}

// ─── Screen ──────────────────────────────────────────────────────────

export function Overview() {
  const { data, isLoading, error, dataUpdatedAt } = useLiveState();
  const { data: equityCurve } = useEquityCurve(500);
  const { data: account } = useCurrentAccount();
  const { data: riskConfig } = useRiskConfig();
  const { data: tradesToday } = useTradeHistory(1, 100);
  // Pooled signal-audit history — one request feeds all 5 RegimeCells so we
  // can display "held Xh" without spawning a hook per card. Backend caps
  // page_size at 200; that gives ~40 rows/symbol (~10h of M15 history).
  const { data: pooledAudit } = useSignalAudit({ pageSize: 200 });
  const historyBySymbol = useMemo(() => {
    const out: Record<string, RegimeHistoryRow[]> = {};
    for (const item of pooledAudit?.items ?? []) {
      const arr = out[item.symbol] ?? (out[item.symbol] = []);
      arr.push({ timestamp: item.timestamp, regime: item.regime });
    }
    return out;
  }, [pooledAudit]);
  const queryClient = useQueryClient();

  // "Updated Xs ago" subtitle — re-render every second.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  const updatedAgoText = useMemo(() => {
    if (!dataUpdatedAt) return "connecting…";
    const secs = Math.max(0, Math.floor((now - dataUpdatedAt) / 1000));
    if (secs < 60) return `updated ${secs}s ago`;
    const m = Math.floor(secs / 60);
    if (m < 60) return `updated ${m}m ago`;
    const h = Math.floor(m / 60);
    return `updated ${h}h ago`;
  }, [dataUpdatedAt, now]);

  const refreshAll = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["live-state"] });
    queryClient.invalidateQueries({ queryKey: ["equity-curve"] });
    queryClient.invalidateQueries({ queryKey: ["invariants-recent"] });
    queryClient.invalidateQueries({ queryKey: ["positions"] });
    queryClient.invalidateQueries({ queryKey: ["model-summary"] });
    queryClient.invalidateQueries({ queryKey: ["news-blackouts"] });
  }, [queryClient]);

  const exportEquityCsv = useCallback(() => {
    if (!equityCurve || equityCurve.points.length === 0) return;
    const header = "timestamp,equity,balance,floating_pnl\n";
    const rows = equityCurve.points
      .map((p) => `${p.timestamp},${p.equity},${p.balance ?? ""},${p.floating_pnl ?? ""}`)
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cortex-equity-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [equityCurve]);

  const [chartSymbol, setChartSymbol] = useState<string>(() => readChartSymbol());
  const [chartTf, setChartTf] = useState<ChartTimeframe>("H1");
  const [chartType, setChartType] = useState<ChartType>(() => readChartType());
  const [searchParams, setSearchParams] = useSearchParams();

  // Persist chart-symbol choice across reloads + screen swaps so coming
  // back to Overview restores the last viewed pair instead of always
  // snapping to XAUUSD.
  useEffect(() => {
    try {
      window.localStorage.setItem(CHART_SYMBOL_STORAGE, chartSymbol);
    } catch {
      /* localStorage disabled */
    }
  }, [chartSymbol]);

  // Honor `?symbol=XYZ` deep-links from other screens (e.g. Position card
  // pill on /positions). Validate against the live universe so a stale
  // bookmark can't pin the chart to a delisted symbol. Scroll the chart
  // into view so the user lands on what they clicked.
  useEffect(() => {
    const requested = searchParams.get("symbol");
    if (!requested) return;
    if ((DEFAULT_SYMBOLS as readonly string[]).includes(requested)) {
      setChartSymbol(requested);
      // Strip the param so a refresh doesn't override later manual swaps.
      const next = new URLSearchParams(searchParams);
      next.delete("symbol");
      setSearchParams(next, { replace: true });
      // Land at the top of Overview so the KPI / equity / news rail is
      // visible. The chart updates in place; user can scroll to it.
      requestAnimationFrame(() => {
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    }
    // searchParams is stable per react-router, but disable to prevent loops
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist chart-type choice across reloads.
  useEffect(() => {
    try {
      window.localStorage.setItem(CHART_TYPE_STORAGE, chartType);
    } catch {
      /* localStorage disabled */
    }
  }, [chartType]);
  const { data: candles, isFetching: candlesFetching } = useCandles({
    symbol: chartSymbol,
    timeframe: chartTf,
    limit: 1000,
  });
  const { data: openPositions } = usePositions();
  const { price: chartPrice, changePct: chartChangePct } = useLatestPrice(chartSymbol);

  const chartPositions = useMemo(
    () => (openPositions ?? []).filter((p) => p.symbol === chartSymbol),
    [openPositions, chartSymbol],
  );

  const [equityRange, setEquityRange] = useState<"1D" | "7D" | "30D" | "All">("All");
  const equityPoints: ChartPoint[] = useMemo(() => {
    if (!equityCurve || equityCurve.points.length === 0) return [];
    const all = equityCurve.points.map((p) => ({ t: p.timestamp, equity: p.equity }));
    if (equityRange === "All") return all;
    const hours = equityRange === "1D" ? 24 : equityRange === "7D" ? 24 * 7 : 24 * 30;
    const cutoff = Date.now() - hours * 3_600_000;
    const filtered = all.filter((p) => {
      const t = new Date(p.t.endsWith("Z") || /[+-]\d\d:?\d\d$/.test(p.t) ? p.t : `${p.t}Z`).getTime();
      return Number.isFinite(t) ? t >= cutoff : true;
    });
    return filtered.length > 0 ? filtered : all;
  }, [equityCurve, equityRange]);

  const equitySpark = useMemo(
    () => equityPoints.slice(-40).map((p) => p.equity),
    [equityPoints],
  );
  const floatSpark = useMemo(() => {
    if (!equityCurve) return [];
    return equityCurve.points
      .slice(-40)
      .map((p) => p.floating_pnl ?? 0);
  }, [equityCurve]);

  // Daily R — sum of r-multiple for trades closed today (UTC).
  // NOTE: must live above loading/error early-returns (rules of hooks).
  const dailyR = useMemo(() => {
    if (!tradesToday) return { r: 0, count: 0 };
    const todayUtc = new Date().toISOString().slice(0, 10);
    let r = 0;
    let count = 0;
    for (const t of tradesToday.trades) {
      if (!t.timestamp_close) continue;
      const day = t.timestamp_close.slice(0, 10);
      if (day !== todayUtc) continue;
      count += 1;
      if (t.r_multiple_at_exit != null && Number.isFinite(t.r_multiple_at_exit)) {
        r += t.r_multiple_at_exit;
      }
    }
    return { r, count };
  }, [tradesToday]);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
          <div className="xl:col-span-8">
            <SkeletonChart height={320} />
          </div>
          <div className="xl:col-span-4 space-y-4">
            <SkeletonCard />
            <SkeletonCard />
          </div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-[var(--color-loss)]">
          Failed to load: {error instanceof Error ? error.message : "Unknown"}
        </p>
      </div>
    );
  }

  const acct = data.account;
  const breaker = data.breaker;
  const availableSymbols = Object.keys(data.signals).length > 0
    ? Object.keys(data.signals)
    : [...DEFAULT_SYMBOLS];

  const floatingPct = acct && acct.balance ? (acct.floating_pnl / acct.balance) * 100 : 0;
  const equityDelta = acct && acct.balance ? ((acct.equity - acct.balance) / acct.balance) * 100 : 0;
  const chartSignal = data.signals[chartSymbol];
  const chartRegime = chartSignal?.regime.regime_label ?? null;
  const chartRegimeColor = chartRegime ? regimeColor(chartRegime) : "var(--color-text-dim)";
  const chartRegimeConf = chartSignal ? (chartSignal.regime.state_probability * 100).toFixed(2) : null;

  const acctIsDemo = account?.is_demo ?? true;
  const acctBalanceText = account ? usd(account.balance) : "—";
  const dailyCap = riskConfig?.max_daily_trades ?? 12;
  const dailyCapPct = Math.min(100, (dailyR.count / Math.max(dailyCap, 1)) * 100);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[var(--color-text)]">Overview</h1>
          <p className="text-xs text-[var(--color-text-dim)] mt-0.5">
            Live · {updatedAgoText}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={exportEquityCsv}
            disabled={!equityCurve || equityCurve.points.length === 0}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-[var(--color-panel)] border border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:border-[var(--color-border-hi)] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            title="Export equity curve as CSV"
          >
            <Download size={12} /> Export
          </button>
          <button
            onClick={refreshAll}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-[var(--color-panel)] border border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:border-[var(--color-border-hi)] transition-colors"
            title="Force refresh all live data"
          >
            <RefreshCcw size={12} /> Refresh
          </button>
          <span
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium border"
            style={
              acctIsDemo
                ? {
                    background: "rgba(234,179,8,0.12)",
                    color: "var(--chip-warn-fg)",
                    borderColor: "rgba(234,179,8,0.28)",
                  }
                : {
                    background: "rgba(16,185,129,0.12)",
                    color: "var(--chip-profit-fg)",
                    borderColor: "rgba(16,185,129,0.28)",
                  }
            }
          >
            {acctIsDemo ? "DEMO" : "LIVE"} · {acctBalanceText}
          </span>
        </div>
      </header>

      {/* HERO KPI STRIP — 4 click-through cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard
          label="Equity"
          value={acct ? usd(acct.equity) : "—"}
          flashValue={acct?.equity ?? null}
          href="/ui/history"
          chip={
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
              style={{ background: "rgba(16,185,129,0.12)", color: "var(--chip-profit-fg)" }}
            >
              ● LIVE
            </span>
          }
          sub={
            acct ? (
              <span>
                <span style={{ color: equityDelta >= 0 ? "var(--color-profit)" : "var(--color-loss)" }}>
                  {equityDelta >= 0 ? "▲" : "▼"} {fmtPct(equityDelta)}
                </span>{" "}
                today
              </span>
            ) : null
          }
          spark={
            equitySpark.length >= 2
              ? { data: equitySpark, color: equityDelta >= 0 ? "var(--color-profit)" : "var(--color-loss)" }
              : undefined
          }
        />
        <KpiCard
          label="Balance"
          value={acct ? usd(acct.balance) : "—"}
          href="/ui/history?tab=account"
          sub={<span>Peak {usd(data.peak_equity)}</span>}
        />
        <KpiCard
          label="Float P/L"
          value={acct ? usd(acct.floating_pnl) : "—"}
          flashValue={acct?.floating_pnl ?? null}
          href="/ui/positions"
          accent={acct && acct.floating_pnl >= 0 ? "profit" : acct ? "loss" : "neutral"}
          topAccent={
            acct && acct.floating_pnl >= 0 ? "var(--color-profit)" : acct ? "var(--color-loss)" : undefined
          }
          chip={
            acct ? (
              <span
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
                style={{
                  background: floatingPct >= 0 ? "rgba(16,185,129,0.12)" : "rgba(244,63,94,0.12)",
                  color: floatingPct >= 0 ? "var(--chip-profit-fg)" : "var(--chip-loss-fg)",
                }}
              >
                {floatingPct >= 0 ? "▲" : "▼"} {fmtPct(floatingPct)}
              </span>
            ) : null
          }
          sub={<span>{data.positions_count} open</span>}
          spark={
            floatSpark.length >= 2
              ? { data: floatSpark, color: floatingPct >= 0 ? "var(--color-profit)" : "var(--color-loss)" }
              : undefined
          }
        />
        <DailyRCard
          r={dailyR.r}
          count={dailyR.count}
          cap={dailyCap}
          capPct={dailyCapPct}
        />
      </div>

      {/* CHART + RAIL — 12-col grid */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        {/* Price chart, 8 cols */}
        <div className="xl:col-span-8 rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] overflow-hidden flex flex-col">
          <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4 border-b border-[var(--color-border)]">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="mono text-xl font-bold text-[var(--color-text)]">{chartSymbol}</span>
              {chartPrice != null && (
                <span className="tnum text-2xl font-semibold text-[var(--color-text)]">
                  {chartPrice.toLocaleString(undefined, { maximumFractionDigits: 5 })}
                </span>
              )}
              {chartChangePct != null && (
                <span
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
                  style={{
                    background: chartChangePct >= 0 ? "rgba(16,185,129,0.12)" : "rgba(244,63,94,0.12)",
                    color: chartChangePct >= 0 ? "var(--chip-profit-fg)" : "var(--chip-loss-fg)",
                  }}
                >
                  {chartChangePct >= 0 ? "▲" : "▼"} {fmtPct(chartChangePct)}
                </span>
              )}
              {chartRegime && (
                <span
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
                  style={{
                    background: `${chartRegimeColor}22`,
                    color: chartRegimeColor,
                    border: `1px solid ${chartRegimeColor}44`,
                  }}
                >
                  ● {chartRegime} regime{chartRegimeConf ? ` · ${chartRegimeConf}%` : ""}
                </span>
              )}
              <LiveDot status={candles && !candlesFetching ? "live" : "stale"} size={6} />
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1" role="tablist" aria-label="Timeframe">
                {TF_OPTIONS.map((tf) => {
                  const active = tf === chartTf;
                  return (
                    <button
                      key={tf}
                      onClick={() => setChartTf(tf)}
                      className={`px-2.5 py-1 text-[11px] rounded-md transition-colors num ${
                        active
                          ? "bg-[var(--color-panel-hi)] font-medium"
                          : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
                      }`}
                      style={active ? { color: "var(--color-primary)" } : undefined}
                    >
                      {tf}
                    </button>
                  );
                })}
              </div>
              <div className="w-px h-4 bg-[var(--color-border)]" aria-hidden />
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
                      className={`px-2 py-1 text-[11px] transition-colors ${
                        active
                          ? "bg-[var(--color-panel-hi)] font-medium"
                          : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
                      }`}
                      style={active ? { color: "var(--color-primary)" } : undefined}
                      title={`Chart style: ${CHART_TYPE_LABEL[ct]}`}
                    >
                      {CHART_TYPE_LABEL[ct]}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
          {/* flex-1 lets the chart absorb any extra row height the side
              rail might force (e.g. when NextNewsRailCard stacks 3 events).
              min-h-[380px] is the prior fixed height so the chart never
              shrinks below the original baseline. */}
          <div className="flex-1 min-h-[380px]">
            {candles && candles.bars.length > 0 ? (
              <PriceChart
                bars={candles.bars}
                height="100%"
                positions={chartPositions}
                symbol={chartSymbol}
                chartType={chartType}
              />
            ) : (
              <div className="h-full flex items-center justify-center text-sm text-[var(--color-text-dim)]">
                {candlesFetching ? "Loading candles..." : `No candle data for ${chartSymbol} / ${chartTf}`}
              </div>
            )}
          </div>
          {/* Bottom symbol tabs strip */}
          <div className="flex items-center gap-1 px-5 py-3 border-t border-[var(--color-border)] overflow-x-auto">
            {availableSymbols.map((sym) => (
              <SymbolTab
                key={sym}
                symbol={sym}
                active={sym === chartSymbol}
                onClick={() => setChartSymbol(sym)}
              />
            ))}
          </div>
        </div>

        {/* Side rail, 4 cols */}
        <div className="xl:col-span-4 flex flex-col gap-4">
          {breaker && (
            <Link
              to="/ui/config"
              className="block rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5 transition-all hover:-translate-y-px hover:border-[color:rgba(99,102,241,0.35)]"
            >
              <div className="flex items-center justify-between mb-3">
                <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-text-dim)]">
                  Risk · Circuit breakers
                </p>
                <span
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
                  style={{
                    background:
                      breaker.active_breakers.length > 0
                        ? "rgba(244,63,94,0.12)"
                        : "rgba(16,185,129,0.12)",
                    color:
                      breaker.active_breakers.length > 0
                        ? "var(--chip-loss-fg)"
                        : "var(--chip-profit-fg)",
                  }}
                >
                  ● {breaker.active_breakers.length > 0 ? "Active" : "All clear"}
                </span>
              </div>
              <RailGauge label="Daily DD" value={breaker.daily_dd_pct} threshold={3.0} />
              <RailGauge label="Weekly DD" value={breaker.weekly_dd_pct} threshold={5.0} />
              <RailGauge label="Peak DD" value={breaker.peak_dd_pct} threshold={10.0} />
              <RailConsecLoss
                value={breaker.consecutive_losses}
                limit={breaker.consecutive_loss_limit || 4}
              />
            </Link>
          )}
          <HealthRailCard />
          <NextNewsRailCard />
        </div>
      </div>

      {/* PER-SYMBOL REGIME ROW */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-text-dim)]">
            Per-symbol regime · HMM + LSTM combined score
          </p>
          <Link to="/ui/signals" className="text-xs text-[var(--color-primary)] hover:brightness-125">
            View all signals →
          </Link>
        </div>
        {Object.keys(data.signals).length === 0 ? (
          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-8 text-center">
            <p className="text-sm text-[var(--color-text-dim)]">No signals available yet</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-5 gap-3">
            {Object.entries(data.signals).map(([symbol, signal]) => (
              <RegimeCell
                key={symbol}
                symbol={symbol}
                signal={signal}
                history={historyBySymbol[symbol] ?? []}
              />
            ))}
          </div>
        )}
      </div>

      {/* EQUITY CURVE + MODEL HEALTH */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div className="xl:col-span-8 rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-text-dim)]">
                Equity curve
              </p>
              <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                {equityPoints.length > 1
                  ? `Last ${equityPoints.length} points · ${equityRange}`
                  : "Fresh account"}
              </p>
            </div>
            <div className="flex items-center gap-1">
              {(["1D", "7D", "30D", "All"] as const).map((rng) => {
                const active = equityRange === rng;
                return (
                  <button
                    key={rng}
                    onClick={() => setEquityRange(rng)}
                    className={`px-2.5 py-1 text-[11px] rounded transition-colors ${
                      active
                        ? "bg-[var(--color-panel-hi)] font-medium"
                        : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
                    }`}
                    style={active ? { color: "var(--color-primary)" } : undefined}
                  >
                    {rng}
                  </button>
                );
              })}
            </div>
          </div>
          {equityPoints.length > 1 ? (
            <EquityChart data={equityPoints} height={220} />
          ) : (
            <div className="relative h-[220px] flex items-center justify-center">
              <div className="absolute inset-0 flex items-center px-2">
                <svg viewBox="0 0 800 160" preserveAspectRatio="none" width="100%" height="100%" aria-hidden>
                  <defs>
                    <linearGradient id="flatArea" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0" stopColor="var(--color-primary)" stopOpacity="0.25" />
                      <stop offset="1" stopColor="var(--color-primary)" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <line x1="0" y1="80" x2="800" y2="80" stroke="var(--color-primary)" strokeWidth="1.2" opacity="0.7" />
                  <path d="M0 80 L800 80 L800 160 L0 160 Z" fill="url(#flatArea)" />
                </svg>
              </div>
              <div className="relative text-center">
                <p className="text-sm text-[var(--color-text-muted)]">No equity movement yet</p>
                <p className="text-xs text-[var(--color-text-dim)] mt-1">
                  Chart appears after first closed trade
                </p>
              </div>
            </div>
          )}
        </div>
        <div className="xl:col-span-4">
          <ModelHealthPanel />
        </div>
      </div>
    </div>
  );
}

// ─── Small helpers ──────────────────────────────────────────────────

function SymbolTab({
  symbol,
  active,
  onClick,
}: {
  symbol: string;
  active: boolean;
  onClick: () => void;
}) {
  const { changePct } = useLatestPrice(symbol);
  const delta =
    changePct == null
      ? null
      : changePct >= 0
        ? { text: `+${changePct.toFixed(2)}%`, color: "var(--color-profit)" }
        : { text: `${changePct.toFixed(2)}%`, color: "var(--color-loss)" };
  return (
    <button
      onClick={onClick}
      className={`shrink-0 px-3 py-1.5 text-[11px] rounded-lg transition-colors mono ${
        active
          ? "font-semibold"
          : "text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-panel-hi)]"
      }`}
      style={active ? { background: "rgba(6,182,212,0.12)", color: "var(--color-primary)" } : undefined}
    >
      {symbol}
      {delta && (
        <span className="ml-1.5" style={active ? undefined : { color: delta.color }}>
          · {delta.text}
        </span>
      )}
    </button>
  );
}

function RailConsecLoss({ value, limit }: { value: number; limit: number }) {
  const safeLimit = Math.max(limit, 1);
  const lit = Math.min(Math.max(value, 0), safeLimit);
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex items-center justify-between text-xs mb-1">
        <span className="text-[var(--color-text-muted)]">Consec SL</span>
        <span className="mono text-[var(--color-text)]">
          {lit} / {safeLimit}
        </span>
      </div>
      <div className="flex gap-1">
        {Array.from({ length: safeLimit }).map((_, i) => {
          const active = i < lit;
          const color =
            !active
              ? "var(--color-panel-hi)"
              : lit >= safeLimit
                ? "var(--color-loss)"
                : lit >= Math.ceil(safeLimit / 2)
                  ? "var(--color-warn)"
                  : "var(--color-profit)";
          return (
            <span
              key={i}
              className="flex-1 h-1.5 rounded"
              style={{ background: color }}
            />
          );
        })}
      </div>
    </div>
  );
}

function RailGauge({ label, value, threshold }: { label: string; value: number; threshold: number }) {
  const abs = Math.abs(value);
  const pct = threshold > 0 ? Math.min((abs / threshold) * 100, 100) : 0;
  const color =
    pct >= 90 ? "var(--color-loss)" : pct >= 50 ? "var(--color-warn)" : "var(--color-profit)";
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex items-center justify-between text-xs mb-1">
        <span className="text-[var(--color-text-muted)]">{label}</span>
        <span className="mono text-[var(--color-text)]">
          {abs.toFixed(1)}% / {threshold.toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-[var(--color-panel-hi)] overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct.toFixed(1)}%`, background: color }}
        />
      </div>
    </div>
  );
}

