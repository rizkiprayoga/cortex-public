import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { X } from "lucide-react";
import { usePositions } from "@/hooks/usePositions";
import { TradeTimeline } from "@/components/TradeTimeline";
import type { PositionData } from "@/lib/types";
import {
  parseStrategyName,
  strategyColor,
  strategyShortLabel,
} from "@/lib/strategy";

function useEscapeKey(active: boolean, onClose: () => void) {
  useEffect(() => {
    if (!active) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [active, onClose]);
}

function formatPrice(price: number | null | undefined, symbol: string): string {
  if (price == null || !Number.isFinite(price)) return "—";
  if (symbol.includes("JPY")) return price.toFixed(3);
  if (symbol === "XAUUSD" || symbol === "ETHUSD" || symbol === "BTCUSD") {
    return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return price.toFixed(5);
}

function formatOpenedAgo(iso: string | null): string {
  if (!iso) return "—";
  const t = new Date(iso.endsWith("Z") || /[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`).getTime();
  if (!Number.isFinite(t)) return "—";
  const diffSec = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (diffSec < 60) return `${diffSec}s ago`;
  const m = Math.floor(diffSec / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  const remM = m % 60;
  if (h < 48) return `${h}h ${remM}m ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function computeRMetrics(pos: PositionData) {
  const risk = Math.abs(pos.entry_price - pos.initial_stop);
  if (risk < 1e-9) return { risk: 0, mfe: 0, mae: 0, stopR: 0, currR: 0 };
  const sign = pos.direction === "buy" ? 1 : -1;
  const mfe = pos.max_price != null ? (sign * (pos.max_price - pos.entry_price)) / risk : 0;
  const mae = pos.min_price != null ? (sign * (pos.min_price - pos.entry_price)) / risk : 0;
  const stopR = (sign * (pos.current_stop - pos.entry_price)) / risk;
  const currR =
    pos.current_price != null ? (sign * (pos.current_price - pos.entry_price)) / risk : 0;
  return { risk, mfe, mae, stopR, currR };
}

// ─── R progress bar: -1R to +3R ──────────────────────────────────────

function formatDollar(n: number): string {
  // Compact $ display for the R-bar tickmarks. Always show sign for non-zero;
  // zero shows bare "$0" so the entry tick reads cleanly.
  if (n === 0) return "$0";
  const sign = n >= 0 ? "+" : "−";
  return `${sign}$${Math.abs(n).toFixed(0)}`;
}

function RProgressBar({ pos }: { pos: PositionData }) {
  const { mfe, mae, stopR, currR } = computeRMetrics(pos);
  const MIN_R = -1;
  const MAX_R = 3;
  const toPct = (r: number) =>
    Math.min(100, Math.max(0, ((r - MIN_R) / (MAX_R - MIN_R)) * 100));
  const entryPct = toPct(0);
  const stopPct = toPct(Math.max(MIN_R, Math.min(MAX_R, stopR)));
  const mfePct = toPct(Math.max(MIN_R, Math.min(MAX_R, mfe)));
  const maePct = toPct(Math.max(MIN_R, Math.min(MAX_R, mae)));
  const currPct = toPct(Math.max(MIN_R, Math.min(MAX_R, currR)));
  const isWinning = currR >= 0;
  // 1R in account currency — null when MT5 risk calc was unavailable
  // (legacy positions, MT5 connection blip). Tickmarks hide when null.
  const oneR = pos.risk_dollars;
  return (
    <div>
      <div className="relative h-3 rounded-full bg-[var(--color-panel-hi)] overflow-hidden">
        {/* MAE → MFE excursion band */}
        <div
          className="absolute inset-y-0"
          style={{
            left: `${Math.min(maePct, mfePct)}%`,
            width: `${Math.abs(mfePct - maePct)}%`,
            background: isWinning ? "rgba(6,182,212,0.25)" : "rgba(244,63,94,0.30)",
          }}
        />
        {/* Entry line */}
        <div
          className="absolute inset-y-0 w-0.5"
          style={{ left: `${entryPct}%`, background: "var(--color-text)" }}
          title="Entry"
        />
        {/* Stop line */}
        <div
          className="absolute inset-y-0 w-0.5"
          style={{ left: `${stopPct}%`, background: "var(--color-warn)" }}
          title={`Stop ${stopR >= 0 ? "+" : ""}${stopR.toFixed(2)}R`}
        />
        {/* Current price marker */}
        <div
          className="absolute inset-y-0 w-1 rounded"
          style={{
            left: `${currPct}%`,
            background: isWinning ? "var(--color-profit)" : "var(--color-loss)",
          }}
          title={`Current ${currR >= 0 ? "+" : ""}${currR.toFixed(2)}R`}
        />
      </div>
      <div className="flex justify-between mono text-[9px] text-[var(--color-text-dim)] mt-1">
        <span>−1R</span>
        <span>entry</span>
        <span>+1R</span>
        <span>+2R</span>
        <span>+3R</span>
      </div>
      {oneR != null && oneR > 0 && (
        <div className="flex justify-between mono text-[9px] mt-0.5">
          <span style={{ color: "var(--color-loss)" }}>{formatDollar(-oneR)}</span>
          <span style={{ color: "var(--color-text-dim)" }}>{formatDollar(0)}</span>
          <span style={{ color: "var(--color-profit)" }}>{formatDollar(oneR)}</span>
          <span style={{ color: "var(--color-profit)" }}>{formatDollar(2 * oneR)}</span>
          <span style={{ color: "var(--color-profit)" }}>{formatDollar(3 * oneR)}</span>
        </div>
      )}
    </div>
  );
}

// ─── Time-exit countdown formatter ────────────────────────────────────

function formatTimeExit(remainingSec: number | null): {
  text: string;
  color: string;
} {
  if (remainingSec == null) {
    return { text: "", color: "var(--color-text-dim)" };
  }
  if (remainingSec <= 0) {
    return { text: "due now", color: "var(--color-loss)" };
  }
  const h = Math.floor(remainingSec / 3600);
  const m = Math.floor((remainingSec % 3600) / 60);
  const text = h >= 1 ? `${h}h ${m}m` : `${m}m`;
  // Color thresholds: <4h red, 4-12h amber, >12h muted (no urgency cue).
  let color = "var(--color-text-dim)";
  if (remainingSec < 4 * 3600) color = "var(--color-loss)";
  else if (remainingSec < 12 * 3600) color = "var(--color-warn)";
  return { text, color };
}

function TierPill({ label, done }: { label: string; done: boolean }) {
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
      style={{
        background: done ? "rgba(16,185,129,0.15)" : "var(--color-panel-hi)",
        color: done ? "var(--color-profit)" : "var(--color-text-dim)",
      }}
    >
      {label} {done ? "✓" : "·"}
    </span>
  );
}

// ─── Single position card ────────────────────────────────────────────

function PositionCard({
  pos,
  onOpen,
}: {
  pos: PositionData;
  onOpen: () => void;
}) {
  const navigate = useNavigate();
  const { mfe, currR } = computeRMetrics(pos);
  const isBuy = pos.direction === "buy";
  const floating = pos.floating_pnl ?? 0;
  const pnlPositive = floating >= 0;
  const topAccent = pnlPositive ? "var(--color-profit)" : "var(--color-loss)";
  const rValue = Number.isFinite(currR) ? currR : mfe;
  const goToChart = () =>
    navigate(`/ui?symbol=${encodeURIComponent(pos.symbol)}`);
  return (
    <div
      className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5"
      style={{ borderTop: `2px solid ${topAccent}` }}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              onClick={goToChart}
              title={`Open ${pos.symbol} chart on Overview`}
              className="mono text-lg font-bold text-[var(--color-text)] hover:text-[var(--color-primary)] transition-colors cursor-pointer"
            >
              {pos.symbol}
            </button>
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
              style={{
                background: isBuy ? "rgba(16,185,129,0.16)" : "rgba(244,63,94,0.16)",
                color: isBuy ? "var(--chip-profit-fg)" : "var(--chip-loss-fg)",
              }}
            >
              {isBuy ? "▲ BUY" : "▼ SELL"}
            </span>
            {(() => {
              const strat = parseStrategyName(pos.strategy_name);
              if (!strat) return null;
              const stratCol = strategyColor(strat);
              return (
                <span
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
                  style={{
                    background: `${stratCol}1f`,
                    color: stratCol,
                    border: `1px solid ${stratCol}44`,
                  }}
                  title={`Strategy at entry: ${strat}`}
                >
                  {strategyShortLabel(strat)}
                </span>
              );
            })()}
          </div>
          <p className="mono text-[11px] text-[var(--color-text-dim)] mt-0.5">
            #{pos.ticket} · opened {formatOpenedAgo(pos.opened_at)}
            {(() => {
              const te = formatTimeExit(pos.time_exit_remaining_sec);
              if (!te.text) return null;
              return (
                <>
                  {" · "}
                  <span style={{ color: te.color }}>⏱ time exit in {te.text}</span>
                </>
              );
            })()}
          </p>
        </div>
        <div className="text-right shrink-0">
          <p
            className="tnum text-2xl font-bold"
            style={{ color: pnlPositive ? "var(--color-profit)" : "var(--color-loss)" }}
          >
            {pnlPositive ? "+" : "−"}${Math.abs(floating).toFixed(2)}
          </p>
          <p
            className="text-xs"
            style={{ color: rValue >= 0 ? "var(--color-profit)" : "var(--color-loss)" }}
          >
            {rValue >= 0 ? "+" : ""}{rValue.toFixed(2)}R
          </p>
        </div>
      </div>
      {/* 4-col mini grid */}
      <div className="grid grid-cols-4 gap-3 mb-3 text-xs">
        <div>
          <p className="text-[10px] text-[var(--color-text-dim)]">Entry</p>
          <p className="mono font-semibold">{formatPrice(pos.entry_price, pos.symbol)}</p>
        </div>
        <div>
          <p className="text-[10px] text-[var(--color-text-dim)]">Current</p>
          <p className="mono font-semibold">{formatPrice(pos.current_price, pos.symbol)}</p>
        </div>
        <div>
          <p className="text-[10px] text-[var(--color-text-dim)]">Stop</p>
          <p
            className="mono font-semibold"
            style={{
              color:
                pos.current_stop !== pos.initial_stop && pos.tier_1_done
                  ? "var(--color-profit)"
                  : "var(--color-text)",
            }}
          >
            {formatPrice(pos.current_stop, pos.symbol)}
          </p>
        </div>
        <div>
          <p className="text-[10px] text-[var(--color-text-dim)]">Volume</p>
          <p className="mono font-semibold">
            {pos.volume.toFixed(2)}
            {pos.volume !== pos.initial_volume && (
              <span className="text-[9px] text-[var(--color-text-dim)]">
                {" "}
                / {pos.initial_volume.toFixed(2)}
              </span>
            )}
          </p>
        </div>
      </div>
      <RProgressBar pos={pos} />
      <div className="flex items-center justify-between mt-4 pt-3 border-t border-[var(--color-border)]">
        <div className="flex gap-1.5">
          <TierPill label="T1" done={pos.tier_1_done} />
          <TierPill label="T2" done={pos.tier_2_done} />
          <TierPill label="Run" done={pos.tier_1_done && pos.tier_2_done} />
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onOpen();
          }}
          className="text-xs text-[var(--color-primary)] hover:brightness-125"
        >
          Details →
        </button>
      </div>
    </div>
  );
}

// ─── Side drawer for detail ──────────────────────────────────────────

function PositionDrawer({ pos, onClose }: { pos: PositionData; onClose: () => void }) {
  useEscapeKey(true, onClose);
  const { risk, mfe, mae, stopR, currR } = computeRMetrics(pos);
  const isBuy = pos.direction === "buy";
  const floating = pos.floating_pnl ?? 0;
  const pnlPositive = floating >= 0;
  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} aria-hidden />
      <aside
        className="relative w-full max-w-md bg-[var(--color-panel)] border-l border-[var(--color-border-hi)] overflow-y-auto p-6 shadow-2xl"
        style={{ boxShadow: "-20px 0 60px rgba(0,0,0,0.5)" }}
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-lg font-semibold text-[var(--color-text)]">
              <span className="mono">{pos.symbol}</span>{" "}
              <span className="text-[var(--color-text-dim)]">#{pos.ticket}</span>
            </h2>
            <p className="text-xs text-[var(--color-text-muted)]">
              {pos.strategy_name} · opened {formatOpenedAgo(pos.opened_at)}
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

        <div
          className="rounded-xl p-4 mb-5"
          style={{
            background: pnlPositive ? "rgba(16,185,129,0.08)" : "rgba(244,63,94,0.08)",
            border: `1px solid ${pnlPositive ? "rgba(16,185,129,0.22)" : "rgba(244,63,94,0.22)"}`,
          }}
        >
          <div className="flex items-center justify-between">
            <span className="section-label">Floating P/L</span>
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
              style={{
                background: isBuy ? "rgba(16,185,129,0.16)" : "rgba(244,63,94,0.16)",
                color: isBuy ? "var(--chip-profit-fg)" : "var(--chip-loss-fg)",
              }}
            >
              {isBuy ? "▲ BUY" : "▼ SELL"}
            </span>
          </div>
          <p
            className="tnum text-3xl font-bold mt-1"
            style={{ color: pnlPositive ? "var(--color-profit)" : "var(--color-loss)" }}
          >
            {pnlPositive ? "+" : "−"}${Math.abs(floating).toFixed(2)}
          </p>
          <p className="mt-1 text-xs" style={{ color: pnlPositive ? "var(--color-profit)" : "var(--color-loss)" }}>
            {currR >= 0 ? "+" : ""}{currR.toFixed(2)}R
            {pos.risk_dollars != null && pos.risk_dollars > 0
              ? ` · 1R = $${pos.risk_dollars.toFixed(2)}`
              : ` · risk Δ ${risk.toFixed(5)}`}
          </p>
        </div>

        <div className="space-y-3 text-sm">
          <DrawerRow label="Entry" value={<span className="mono">{formatPrice(pos.entry_price, pos.symbol)}</span>} />
          <DrawerRow
            label="Current"
            value={<span className="mono">{formatPrice(pos.current_price, pos.symbol)}</span>}
          />
          <DrawerRow
            label="Current stop"
            value={
              <span className="mono">
                {formatPrice(pos.current_stop, pos.symbol)}{" "}
                <span className="text-[var(--color-text-dim)]">
                  (init {formatPrice(pos.initial_stop, pos.symbol)})
                </span>
              </span>
            }
          />
          {pos.take_profit != null && pos.take_profit > 0 && (
            <DrawerRow
              label="Take-profit"
              value={<span className="mono">{formatPrice(pos.take_profit, pos.symbol)}</span>}
            />
          )}
          <DrawerRow
            label="Volume"
            value={
              <span className="mono">
                {pos.volume.toFixed(2)} / {pos.initial_volume.toFixed(2)}
              </span>
            }
          />
          <DrawerRow
            label="ATR trail ×"
            value={<span className="mono">{pos.atr_trail_mult.toFixed(1)}</span>}
          />
          {pos.time_exit_bars != null && pos.time_exit_bars > 0 && (
            <DrawerRow
              label="Time exit"
              value={(() => {
                const te = formatTimeExit(pos.time_exit_remaining_sec);
                if (!te.text) {
                  return (
                    <span className="mono text-[var(--color-text-dim)]">
                      —
                    </span>
                  );
                }
                return (
                  <span className="mono" style={{ color: te.color }}>
                    {te.text}{" "}
                    <span className="text-[var(--color-text-dim)]">
                      ({pos.time_exit_bars}h limit)
                    </span>
                  </span>
                );
              })()}
            />
          )}
          <div className="pt-2">
            <p className="section-label mb-2">R-multiple progress</p>
            <RProgressBar pos={pos} />
            <div className="flex justify-between mono text-[11px] mt-2">
              <span style={{ color: "var(--color-loss)" }}>MAE {mae.toFixed(2)}R</span>
              <span style={{ color: "var(--color-warn)" }}>stop {stopR.toFixed(2)}R</span>
              <span style={{ color: "var(--color-profit)" }}>MFE +{mfe.toFixed(2)}R</span>
            </div>
          </div>
          <div className="flex gap-2 pt-2">
            <TierPill label="T1" done={pos.tier_1_done} />
            <TierPill label="T2" done={pos.tier_2_done} />
            <TierPill label="Runner" done={pos.tier_1_done && pos.tier_2_done} />
          </div>
        </div>

        <div className="pt-5 mt-5 border-t border-[var(--color-border)]">
          <p className="section-label mb-3">Activity</p>
          <TradeTimeline ticket={pos.ticket} />
        </div>
      </aside>
    </div>
  );
}

function DrawerRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between items-baseline">
      <span className="text-[var(--color-text-muted)]">{label}</span>
      <span className="text-[var(--color-text)]">{value}</span>
    </div>
  );
}

// ─── Screen ──────────────────────────────────────────────────────────

export function Positions() {
  const { data: positions, isLoading } = usePositions();
  const [selected, setSelected] = useState<PositionData | null>(null);
  const count = positions?.length ?? 0;
  const totalPnl = (positions ?? []).reduce((sum, p) => sum + (p.floating_pnl ?? 0), 0);
  const totalPnlPositive = totalPnl >= 0;

  return (
    <div className="space-y-4">
      <header className="flex items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[var(--color-text)]">Open positions</h1>
          <p className="text-xs text-[var(--color-text-dim)] mt-0.5">
            {isLoading
              ? "loading…"
              : `${count} active · floating ${totalPnlPositive ? "+" : "−"}$${Math.abs(totalPnl).toFixed(2)}`}
          </p>
        </div>
        {count > 0 && (
          <span
            className="tnum text-xl font-bold"
            style={{ color: totalPnlPositive ? "var(--color-profit)" : "var(--color-loss)" }}
          >
            {totalPnlPositive ? "+" : "−"}${Math.abs(totalPnl).toFixed(2)}
          </span>
        )}
      </header>

      {isLoading && (
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-10 text-center">
          <p className="text-sm text-[var(--color-text-muted)]">Loading positions…</p>
        </div>
      )}

      {!isLoading && count === 0 && (
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-10 flex flex-col items-center text-center">
          <div
            className="w-16 h-16 rounded-full flex items-center justify-center mb-4"
            style={{
              background: "rgba(99,102,241,0.12)",
              border: "1px solid rgba(99,102,241,0.28)",
            }}
          >
            <span className="text-3xl text-[var(--color-text-muted)]">◌</span>
          </div>
          <p className="text-base font-semibold text-[var(--color-text)]">No open positions</p>
          <p className="text-xs text-[var(--color-text-muted)] mt-1 max-w-md">
            Bot is scanning every M15 bar. Approved signals open here.
          </p>
        </div>
      )}

      {!isLoading && count > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {positions!.map((pos) => (
            <PositionCard key={pos.ticket} pos={pos} onOpen={() => setSelected(pos)} />
          ))}
        </div>
      )}

      {selected && <PositionDrawer pos={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
