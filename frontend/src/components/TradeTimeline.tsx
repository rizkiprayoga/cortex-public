import { useTradeTimeline } from "@/hooks/useHistory";
import { shortDate, num, usd } from "@/lib/format";
import type { TradeEventItem, SignalAuditItem } from "@/lib/types";

/**
 * Per-ticket activity stream — renders events (entry, modify, partial
 * close, exit) from trade_events.csv plus the signals that fired around
 * the trade's open time from signal_audit.csv. Read-only, no mutations.
 */
export function TradeTimeline({ ticket }: { ticket: number }) {
  const { data, isLoading, error } = useTradeTimeline(ticket);

  if (isLoading) {
    return (
      <div className="text-xs text-[var(--color-text-dim)] py-3 px-2">
        Loading activity…
      </div>
    );
  }
  if (error) {
    return (
      <div className="text-xs text-[var(--color-loss)] py-3 px-2">
        Failed to load activity: {(error as Error).message}
      </div>
    );
  }
  if (!data || (data.events.length === 0 && data.signals.length === 0)) {
    return (
      <div className="text-xs text-[var(--color-text-dim)] py-3 px-2">
        No recorded activity for this ticket.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 py-3 px-2">
      {data.signals.length > 0 && <SignalPanel signals={data.signals} />}
      {data.events.length > 0 && <EventPanel events={data.events} />}
    </div>
  );
}

function eventBadge(ev: string): { label: string; color: string } {
  const map: Record<string, { label: string; color: string }> = {
    entry: { label: "ENTRY", color: "bg-[var(--color-primary)]/20 text-[var(--color-primary)]" },
    modify: { label: "MODIFY", color: "bg-[var(--color-warn)]/20 text-[var(--color-warn)]" },
    partial_close: { label: "PARTIAL", color: "bg-[var(--color-warn)]/20 text-[var(--color-warn)]" },
    exit: { label: "EXIT", color: "bg-[var(--color-loss)]/20 text-[var(--color-loss)]" },
    full_close_rejected: { label: "REJECTED", color: "bg-[var(--color-loss)]/30 text-[var(--color-loss)]" },
  };
  return map[ev] ?? { label: ev.toUpperCase(), color: "bg-[var(--color-panel-hi)] text-[var(--color-text-muted)]" };
}

function EventPanel({ events }: { events: TradeEventItem[] }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] mb-2">
        Bot actions
      </p>
      <div className="flex flex-col gap-1.5">
        {events.map((ev, i) => {
          const badge = eventBadge(ev.event);
          return (
            <div
              key={`${ev.timestamp}-${i}`}
              className="flex items-start gap-3 px-2 py-1.5 rounded-md bg-[var(--color-panel-hi)]/40 text-xs"
            >
              <span
                className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold ${badge.color}`}
              >
                {badge.label}
              </span>
              <span className="num shrink-0 text-[var(--color-text-muted)] w-32">
                {shortDate(ev.timestamp)}
              </span>
              <span className="flex-1 text-[var(--color-text)]">
                <EventDetail ev={ev} />
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function EventDetail({ ev }: { ev: TradeEventItem }) {
  const parts: string[] = [];
  if (ev.event === "entry") {
    if (ev.direction) parts.push(ev.direction.toUpperCase());
    if (ev.lot_size != null) parts.push(`${num(ev.lot_size, 2)} lots`);
    if (ev.entry_price != null) parts.push(`@ ${num(ev.entry_price, 5)}`);
    if (ev.sl_price != null) parts.push(`SL ${num(ev.sl_price, 5)}`);
    if (ev.regime_at_entry) parts.push(`regime=${ev.regime_at_entry}`);
    if (ev.combined_score_at_entry != null)
      parts.push(`score=${num(ev.combined_score_at_entry, 2)}`);
  } else if (ev.event === "modify") {
    if (ev.sl_price != null) parts.push(`SL → ${num(ev.sl_price, 5)}`);
    if (ev.be_locked) parts.push("BE locked");
    if (ev.exit_reason) parts.push(`(${ev.exit_reason})`);
  } else if (ev.event === "partial_close") {
    if (ev.lot_size != null) parts.push(`closed ${num(ev.lot_size, 2)} lots`);
    if (ev.current_price != null) parts.push(`@ ${num(ev.current_price, 5)}`);
    if (ev.pnl_usd != null) parts.push(`PnL ${usd(ev.pnl_usd)}`);
    if (ev.r_multiple != null) parts.push(`${num(ev.r_multiple, 2)}R`);
    if (ev.exit_reason) parts.push(`(${ev.exit_reason})`);
  } else if (ev.event === "exit") {
    if (ev.current_price != null) parts.push(`@ ${num(ev.current_price, 5)}`);
    if (ev.pnl_usd != null) parts.push(`PnL ${usd(ev.pnl_usd)}`);
    if (ev.r_multiple != null) parts.push(`${num(ev.r_multiple, 2)}R`);
    if (ev.bars_held != null) parts.push(`${ev.bars_held} bars`);
    if (ev.exit_reason) parts.push(`— ${ev.exit_reason}`);
  } else if (ev.event === "full_close_rejected") {
    parts.push("broker rejected close order");
    if (ev.exit_reason) parts.push(`(${ev.exit_reason})`);
  } else {
    if (ev.exit_reason) parts.push(ev.exit_reason);
  }
  return <>{parts.join(" · ") || "—"}</>;
}

function SignalPanel({ signals }: { signals: SignalAuditItem[] }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] mb-2">
        Signals around entry (±30 min)
      </p>
      <div className="flex flex-col gap-1.5">
        {signals.map((s, i) => (
          <div
            key={`${s.timestamp}-${i}`}
            className="flex items-start gap-3 px-2 py-1.5 rounded-md bg-[var(--color-panel-hi)]/40 text-xs"
          >
            <span
              className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                s.executed
                  ? "bg-[var(--color-profit)]/20 text-[var(--color-profit)]"
                  : s.should_trade
                  ? "bg-[var(--color-warn)]/20 text-[var(--color-warn)]"
                  : "bg-[var(--color-panel-hi)] text-[var(--color-text-muted)]"
              }`}
            >
              {s.executed ? "EXECUTED" : s.should_trade ? "BLOCKED" : "SIGNAL"}
            </span>
            <span className="num shrink-0 text-[var(--color-text-muted)] w-32">
              {shortDate(s.timestamp)}
            </span>
            <span className="flex-1 text-[var(--color-text)]">
              {s.direction && (
                <span
                  className={
                    s.direction === "buy"
                      ? "text-[var(--color-primary)]"
                      : "text-[var(--color-loss)]"
                  }
                >
                  {s.direction.toUpperCase()}
                </span>
              )}
              {s.regime ? ` · ${s.regime}` : ""}
              {s.combined_score != null
                ? ` · score=${num(s.combined_score, 2)}`
                : ""}
              {s.block_reason ? ` · blocked: ${s.block_reason}` : ""}
              {s.reasoning ? (
                <div className="text-[var(--color-text-muted)] text-[11px] mt-0.5">
                  {s.reasoning}
                </div>
              ) : null}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
