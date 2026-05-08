import { useEffect, useMemo, useRef, useState } from "react";
import { useSignalAudit } from "@/hooks/useSignalAudit";
import { shortDate } from "@/lib/format";
import type { SignalAuditItem } from "@/lib/types";

type Window = "48h" | "7d" | "30d";

const WINDOW_HOURS: Record<Window, number> = { "48h": 48, "7d": 168, "30d": 720 };
const THRESHOLD = 0.45;

interface BlockBucket {
  reason: string;
  count: number;
  color: string;
}

// Categories color-code the bars. The canonical bucket names here mirror
// what main.py + signal_combiner.py write into block_reason; unknown
// strings fall through to "var(--color-primary)" so the user still sees
// a bar with the raw label.
const RISK_REASONS = new Set(["cb_blocked", "breaker", "risk_monitor", "consec_sl"]);
const NEWS_REASONS = new Set(["news_blackout", "news", "news_exempt"]);
const GATE_REASONS = new Set([
  "long_only", "flicker", "direction_conflict", "confluence",
  "sizing", "combiner_rejected", "score", "threshold", "regime",
]);

function classifyReason(reason: string): string {
  if (RISK_REASONS.has(reason)) return "var(--color-loss)";
  if (NEWS_REASONS.has(reason)) return "var(--color-warn)";
  if (GATE_REASONS.has(reason)) return "var(--chip-info-fg)";
  return "var(--color-primary)";
}

function normReason(raw: string | null): string {
  if (!raw) return "unknown";
  // Lowercase, take first token before " · " or ":" so the long-form
  // reasoning strings ("combiner_rejected · score 0.22") bucket cleanly.
  const first = raw.toLowerCase().split(/[·:|\n]/)[0].trim();
  if (/threshold|below threshold/.test(first)) return "threshold";
  if (/confluence/.test(first)) return "confluence";
  if (/flicker/.test(first)) return "flicker";
  if (/blackout|news/.test(first)) return "news_blackout";
  if (/long[_\s-]?only/.test(first)) return "long_only";
  if (/combiner/.test(first)) return "combiner_rejected";
  if (/\bcb\b|breaker|halt|consec/.test(first)) return "cb_blocked";
  if (/direction|conflict/.test(first)) return "direction_conflict";
  if (/sizing|volume|lot/.test(first)) return "sizing";
  if (/regime/.test(first)) return "regime";
  // Pass through the raw first token (truncated) so unknown reasons
  // still get their own bar — never silently dump into "other".
  return first.replace(/\s+/g, "_").slice(0, 22) || "unknown";
}

// Chart dimensions — SVG uses actual pixel coordinates (not a stretched
// viewBox) so lines and text stay sharp at any container width.
const CHART_HEIGHT = 180;
const CHART_LEFT_PAD = 30;
const CHART_RIGHT_PAD = 10;
const CHART_TOP_PAD = 20;
const CHART_BOTTOM_PAD = 20;

export function SignalTensionCard({ symbol }: { symbol: string }) {
  const [win, setWin] = useState<Window>("48h");
  // Backend filters by symbol (case-insensitive). 800 covers 30d+ per
  // symbol comfortably (~15-25 audit rows/day/symbol).
  const { data: audit, isLoading } = useSignalAudit({ symbol, pageSize: 800 });

  // Measure container width so the SVG renders at actual pixel density.
  // Without this, preserveAspectRatio="none" stretches the viewBox and
  // blurs strokes + text on wide containers.
  const chartBoxRef = useRef<HTMLDivElement>(null);
  const [chartWidth, setChartWidth] = useState(460);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  useEffect(() => {
    const el = chartBoxRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = Math.max(200, Math.floor(entries[0].contentRect.width));
      setChartWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const { trajectory, buckets, crossings, summary } = useMemo(() => {
    type TrajPoint = {
      t: number;
      score: number;
      signed: number;
      direction: string | null;
      executed: boolean;
      regime: string | null;
      blockReason: string | null;
      iso: string;
    };
    const empty = { trajectory: [] as TrajPoint[], buckets: [] as BlockBucket[], crossings: [] as Array<{ t: number; score: number; kind: "up" | "down" | "near" }>, summary: { avg: 0, near: 0, crosses: 0, last: null as number | null } };
    const items: SignalAuditItem[] = audit?.items ?? [];
    if (items.length === 0) return empty;

    const cutoff = Date.now() - WINDOW_HOURS[win] * 3_600_000;
    const windowed = items
      .filter((i) => {
        const t = Date.parse(i.timestamp);
        return !Number.isNaN(t) && t >= cutoff;
      })
      .sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));

    // Trajectory — combined_score vs timestamp (keeps signed score +
    // direction + executed flag for the hover tooltip).
    const trajectory: TrajPoint[] = windowed
      .filter((i) => i.combined_score != null)
      .map((i) => ({
        t: Date.parse(i.timestamp),
        score: Math.abs(i.combined_score as number),
        signed: i.combined_score as number,
        direction: i.direction ?? null,
        executed: Boolean(i.executed),
        regime: i.regime ?? null,
        blockReason: i.block_reason ?? null,
        iso: i.timestamp,
      }));

    // Crossings — where abs(score) crosses the threshold between consecutive points
    const crossings: Array<{ t: number; score: number; kind: "up" | "down" | "near" }> = [];
    for (let i = 1; i < trajectory.length; i++) {
      const prev = trajectory[i - 1].score;
      const cur = trajectory[i].score;
      if (prev < THRESHOLD && cur >= THRESHOLD) {
        crossings.push({ t: trajectory[i].t, score: cur, kind: "up" });
      } else if (prev >= THRESHOLD && cur < THRESHOLD) {
        crossings.push({ t: trajectory[i].t, score: cur, kind: "down" });
      } else if (prev < THRESHOLD && cur < THRESHOLD && cur >= THRESHOLD - 0.05 && cur > prev) {
        crossings.push({ t: trajectory[i].t, score: cur, kind: "near" });
      }
    }

    // Block-reason buckets from blocked rows only
    const reasonCounts = new Map<string, number>();
    for (const it of windowed) {
      if (it.executed) continue;
      const r = normReason(it.block_reason ?? null);
      reasonCounts.set(r, (reasonCounts.get(r) ?? 0) + 1);
    }
    const buckets: BlockBucket[] = Array.from(reasonCounts.entries())
      .map(([reason, count]) => ({ reason, count, color: classifyReason(reason) }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 6);

    // Summary line
    const avg = trajectory.length > 0
      ? trajectory.reduce((s, p) => s + p.score, 0) / trajectory.length
      : 0;
    const nearCount = crossings.filter((c) => c.kind === "near").length;
    const crossCount = crossings.filter((c) => c.kind === "up" || c.kind === "down").length;
    const last = trajectory.length > 0 ? trajectory[trajectory.length - 1].score : null;

    return {
      trajectory,
      buckets,
      crossings,
      summary: { avg, near: nearCount, crosses: crossCount, last },
    };
  }, [audit, win]);

  // SVG trajectory — uses ACTUAL pixel coordinates (chartWidth × CHART_HEIGHT).
  // Time maps to [LEFT_PAD, width - RIGHT_PAD]; score maps to [H - BOTTOM_PAD, TOP_PAD].
  // Identity scalers when trajectory is empty so the SVG renders cleanly.
  const svgPath = useMemo(() => {
    const plotLeft = CHART_LEFT_PAD;
    const plotRight = chartWidth - CHART_RIGHT_PAD;
    const plotTop = CHART_TOP_PAD;
    const plotBottom = CHART_HEIGHT - CHART_BOTTOM_PAD;
    const plotW = Math.max(1, plotRight - plotLeft);
    const plotH = Math.max(1, plotBottom - plotTop);
    if (trajectory.length < 2) {
      const noop = () => 0;
      return { line: "", area: "", thresholdY: plotTop + plotH / 2, x: noop, y: noop, plotLeft, plotRight, plotTop, plotBottom };
    }
    const tMin = trajectory[0].t;
    const tMax = trajectory[trajectory.length - 1].t;
    const dt = Math.max(tMax - tMin, 1);
    const x = (t: number) => plotLeft + ((t - tMin) / dt) * plotW;
    const y = (s: number) => plotBottom - Math.min(Math.max(s, 0), 1) * plotH;
    const segs = trajectory.map((p, i) => `${i === 0 ? "M" : "L"}${x(p.t).toFixed(1)} ${y(p.score).toFixed(1)}`);
    const line = segs.join(" ");
    const area = `${line} L${x(tMax).toFixed(1)} ${plotBottom} L${x(tMin).toFixed(1)} ${plotBottom} Z`;
    return { line, area, thresholdY: y(THRESHOLD), x, y, plotLeft, plotRight, plotTop, plotBottom };
  }, [trajectory, chartWidth]);

  const maxBucket = buckets.reduce((m, b) => Math.max(m, b.count), 0);

  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
      <div className="flex items-start justify-between gap-3 mb-4 flex-wrap">
        <div>
          <p className="section-label">Signal tension · {symbol}</p>
          <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
            Score trajectory vs threshold · block-reason breakdown
          </p>
        </div>
        <div className="flex items-center gap-1 bg-[var(--color-panel-hi)] rounded-lg p-0.5 text-[11px]">
          {(["48h", "7d", "30d"] as const).map((w) => {
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

      <div className="grid grid-cols-1 md:grid-cols-[3fr_2fr] gap-4">
        {/* LEFT — Score trajectory */}
        <div className="rounded-lg bg-[var(--color-panel-hi)] border border-[var(--color-border)] p-4">
          <p className="section-label mb-2">Score · threshold {THRESHOLD}</p>
          <div ref={chartBoxRef} className="w-full">
            {isLoading ? (
              <div className="h-[180px] flex items-center justify-center text-xs text-[var(--color-text-dim)]">Loading…</div>
            ) : trajectory.length < 2 ? (
              <div className="h-[180px] flex items-center justify-center text-xs text-[var(--color-text-dim)]">
                Not enough audit rows in the last {win} to render.
              </div>
            ) : (
              <>
                <div className="relative" style={{ width: chartWidth, height: CHART_HEIGHT }}>
                  <svg
                    width={chartWidth}
                    height={CHART_HEIGHT}
                    viewBox={`0 0 ${chartWidth} ${CHART_HEIGHT}`}
                    style={{ display: "block", touchAction: "none" }}
                    onPointerMove={(e) => {
                      if (trajectory.length < 2) return;
                      const rect = e.currentTarget.getBoundingClientRect();
                      const px = e.clientX - rect.left;
                      let best = 0;
                      let bestDist = Infinity;
                      for (let i = 0; i < trajectory.length; i++) {
                        const d = Math.abs(svgPath.x(trajectory[i].t) - px);
                        if (d < bestDist) {
                          bestDist = d;
                          best = i;
                        }
                      }
                      setHoverIdx(best);
                    }}
                    onPointerLeave={() => setHoverIdx(null)}
                  >
                    {/* Horizontal grid lines at 0.25 / 0.5 / 0.75 score. */}
                    <g stroke="rgba(0,0,0,0.06)" strokeWidth="1" shapeRendering="crispEdges">
                      {[0.25, 0.5, 0.75].map((s) => (
                        <line
                          key={s}
                          x1={svgPath.plotLeft}
                          y1={svgPath.y(s)}
                          x2={svgPath.plotRight}
                          y2={svgPath.y(s)}
                        />
                      ))}
                    </g>
                    {/* Threshold dashed line */}
                    <line
                      x1={svgPath.plotLeft}
                      y1={svgPath.thresholdY}
                      x2={svgPath.plotRight}
                      y2={svgPath.thresholdY}
                      stroke="var(--color-primary)"
                      strokeWidth="1.2"
                      strokeDasharray="4 4"
                      opacity="0.85"
                      shapeRendering="geometricPrecision"
                    />
                    <text
                      x={svgPath.plotRight}
                      y={svgPath.thresholdY - 4}
                      fill="var(--color-primary)"
                      fontSize="10"
                      fontWeight="600"
                      textAnchor="end"
                      fontFamily="ui-monospace, monospace"
                    >
                      {THRESHOLD.toFixed(2)}
                    </text>
                    <path d={svgPath.area} fill="var(--color-primary)" opacity="0.08" />
                    <path
                      d={svgPath.line}
                      fill="none"
                      stroke="var(--color-primary)"
                      strokeWidth="1.8"
                      shapeRendering="geometricPrecision"
                    />
                    {crossings.map((c, i) => {
                      const fill =
                        c.kind === "up"
                          ? "var(--color-profit)"
                          : c.kind === "down"
                            ? "var(--color-loss)"
                            : "var(--color-warn)";
                      return (
                        <circle
                          key={`${c.t}-${i}`}
                          cx={svgPath.x(c.t)}
                          cy={svgPath.y(c.score)}
                          r="3.5"
                          fill={fill}
                          stroke="var(--color-panel)"
                          strokeWidth="1.2"
                        />
                      );
                    })}
                    {/* Hover crosshair + point marker */}
                    {hoverIdx != null && trajectory[hoverIdx] && (
                      <g pointerEvents="none">
                        <line
                          x1={svgPath.x(trajectory[hoverIdx].t)}
                          y1={svgPath.plotTop}
                          x2={svgPath.x(trajectory[hoverIdx].t)}
                          y2={svgPath.plotBottom}
                          stroke="var(--color-text-muted)"
                          strokeWidth="1"
                          strokeDasharray="3 3"
                          opacity="0.5"
                          shapeRendering="crispEdges"
                        />
                        <circle
                          cx={svgPath.x(trajectory[hoverIdx].t)}
                          cy={svgPath.y(trajectory[hoverIdx].score)}
                          r="4.5"
                          fill="var(--color-panel)"
                          stroke="var(--color-primary)"
                          strokeWidth="1.8"
                        />
                      </g>
                    )}
                  </svg>
                  {/* Hover tooltip — HTML overlay positioned relative to the chart box */}
                  {hoverIdx != null && trajectory[hoverIdx] && (() => {
                    const p = trajectory[hoverIdx];
                    const px = svgPath.x(p.t);
                    const py = svgPath.y(p.score);
                    const TT_W = 180;
                    const TT_H = 88;
                    const gap = 10;
                    // Flip left when close to the right edge
                    const leftPx = px + TT_W + gap > chartWidth
                      ? Math.max(0, px - TT_W - gap)
                      : px + gap;
                    const topPx = Math.min(
                      Math.max(0, py - TT_H / 2),
                      CHART_HEIGHT - TT_H,
                    );
                    const signStr = p.signed > 0 ? "+" : p.signed < 0 ? "−" : "";
                    const signedFmt = `${signStr}${Math.abs(p.signed).toFixed(3)}`;
                    const scoreCol = p.score >= THRESHOLD
                      ? "var(--color-profit)"
                      : p.score >= THRESHOLD - 0.05
                        ? "var(--color-warn)"
                        : "var(--color-text-muted)";
                    return (
                      <div
                        className="absolute pointer-events-none rounded-lg border shadow-lg text-[11px]"
                        style={{
                          left: leftPx,
                          top: topPx,
                          width: TT_W,
                          background: "var(--color-panel)",
                          borderColor: "var(--color-border)",
                          padding: "8px 10px",
                        }}
                      >
                        <div className="mono text-[10px] text-[var(--color-text-muted)]">
                          {shortDate(p.iso)}
                        </div>
                        <div className="flex items-baseline justify-between mt-1">
                          <span className="text-[var(--color-text-muted)]">score</span>
                          <span className="mono font-semibold" style={{ color: scoreCol }}>
                            {signedFmt}
                          </span>
                        </div>
                        <div className="flex items-baseline justify-between">
                          <span className="text-[var(--color-text-muted)]">regime</span>
                          <span className="mono">{p.regime ?? "—"}</span>
                        </div>
                        <div className="flex items-baseline justify-between">
                          <span className="text-[var(--color-text-muted)]">
                            {p.executed ? "executed" : "blocked"}
                          </span>
                          <span
                            className="mono truncate ml-2"
                            style={{
                              color: p.executed
                                ? "var(--color-profit)"
                                : "var(--color-text)",
                              maxWidth: 100,
                            }}
                            title={p.executed ? (p.direction ?? "—") : (p.blockReason ?? "—")}
                          >
                            {p.executed
                              ? (p.direction ?? "—")
                              : (p.blockReason ?? "—")}
                          </span>
                        </div>
                      </div>
                    );
                  })()}
                </div>
                <p className="text-[10px] text-[var(--color-text-muted)] mt-1">
                  {summary.crosses} cross{summary.crosses === 1 ? "" : "es"} ·{" "}
                  {summary.near} near-miss{summary.near === 1 ? "" : "es"} · avg{" "}
                  {summary.avg.toFixed(2)}
                  {summary.last != null ? ` · now ${summary.last.toFixed(2)}` : ""}
                  <span className="hidden md:inline text-[var(--color-text-dim)]"> · hover for detail</span>
                </p>
              </>
            )}
          </div>
        </div>

        {/* RIGHT — Block-reason bars */}
        <div className="rounded-lg bg-[var(--color-panel-hi)] border border-[var(--color-border)] p-4">
          <p className="section-label mb-3">Why signals died · {win}</p>
          {isLoading ? (
            <div className="text-xs text-[var(--color-text-dim)]">Loading…</div>
          ) : buckets.length === 0 ? (
            <div className="text-xs text-[var(--color-text-dim)]">
              No blocked signals in this window.
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {buckets.map((b) => {
                const pct = maxBucket > 0 ? (b.count / maxBucket) * 100 : 0;
                return (
                  <div
                    key={b.reason}
                    className="grid items-center gap-2 text-[11px]"
                    style={{ gridTemplateColumns: "90px 1fr 32px" }}
                  >
                    <span className="text-[var(--color-text)] truncate" title={b.reason}>
                      {b.reason}
                    </span>
                    <div className="h-[10px] rounded-full bg-black/5 overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${pct}%`, background: b.color }}
                      />
                    </div>
                    <span className="mono text-right text-[var(--color-text-muted)]">
                      {b.count}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
