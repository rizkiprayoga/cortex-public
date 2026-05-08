import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ChevronDown, ChevronRight } from "lucide-react";
import { EquityChart, type EquityPoint as ChartPoint } from "@/components/EquityChart";
import { TradeTimeline } from "@/components/TradeTimeline";
import {
  useAccountLedger,
  useTradeHistory,
  useTradingMetrics,
  useEquityCurve,
} from "@/hooks/useHistory";
import { usd, num, shortDate } from "@/lib/format";
import type { BalanceOperation, TradeHistoryItem } from "@/lib/types";
import { SYMBOL_FILTERS } from "@/lib/symbols";


type HistoryTab = "trades" | "account";

// ─── Reason chip ──────────────────────────────────────────────────────

const REASON_META: Record<
  string,
  { label: string; fg: string; bg: string }
> = {
  take_profit: { label: "TP", fg: "var(--color-profit)", bg: "rgba(16,185,129,0.15)" },
  stop_loss: { label: "SL", fg: "var(--color-loss)", bg: "rgba(244,63,94,0.15)" },
  time_exit: { label: "TIME", fg: "var(--color-text-muted)", bg: "rgba(148,163,184,0.15)" },
  reversal_hard_exit: { label: "REVERSAL", fg: "var(--color-warn)", bg: "rgba(245,158,11,0.15)" },
  manual: { label: "MANUAL", fg: "var(--color-primary)", bg: "rgba(6,182,212,0.15)" },
  breaker_emergency: { label: "BREAKER", fg: "var(--color-loss)", bg: "rgba(244,63,94,0.20)" },
  unknown: { label: "?", fg: "var(--color-text-dim)", bg: "var(--color-panel-hi)" },
};

function ReasonChip({ code, full }: { code: string | null; full: string | null }) {
  if (!code) return <span className="text-[var(--color-text-dim)]">—</span>;
  const meta = REASON_META[code] ?? {
    label: code.toUpperCase().slice(0, 8),
    fg: "var(--color-text-dim)",
    bg: "var(--color-panel-hi)",
  };
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold mono"
      style={{ color: meta.fg, background: meta.bg }}
      title={full ?? code}
    >
      {meta.label}
    </span>
  );
}

// ─── KPI strip card ──────────────────────────────────────────────────

function HKPI({
  label,
  value,
  color,
}: {
  label: string;
  value: React.ReactNode;
  color?: string;
}) {
  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-4">
      <p className="section-label">{label}</p>
      <p className="tnum text-xl font-bold mt-1" style={color ? { color } : undefined}>
        {value}
      </p>
    </div>
  );
}

// ─── Screen ──────────────────────────────────────────────────────────

export function History() {
  const [searchParams, setSearchParams] = useSearchParams();
  // Tab is a pure projection of the URL — no local mirror state. Back/forward
  // and external deeplinks reflect immediately without a syncing effect.
  const tab: HistoryTab = searchParams.get("tab") === "account" ? "account" : "trades";
  const [symbol, setSymbol] = useState<string | undefined>(undefined);
  const [page, setPage] = useState(1);
  const [expandedTicket, setExpandedTicket] = useState<number | null>(null);
  const pageSize = 20;

  const switchTab = (next: HistoryTab) => {
    const nextParams = new URLSearchParams(searchParams);
    if (next === "account") nextParams.set("tab", "account");
    else nextParams.delete("tab");
    setSearchParams(nextParams, { replace: true });
  };

  const { data: trades, isLoading } = useTradeHistory(page, pageSize, symbol);
  const { data: metrics } = useTradingMetrics(symbol);
  const { data: equityCurve } = useEquityCurve(500);
  const { data: ledger, isLoading: ledgerLoading } = useAccountLedger(365);

  const totalPages = trades ? Math.max(1, Math.ceil(trades.total / pageSize)) : 1;

  const chartData: ChartPoint[] = useMemo(() => {
    if (!equityCurve) return [];
    return equityCurve.points.map((p) => ({ t: p.timestamp, equity: p.equity }));
  }, [equityCurve]);

  const tradeCount = trades?.total ?? 0;
  const accountCount = ledger?.operations.length ?? 0;

  return (
    <div className="space-y-4">
      <header className="flex items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[var(--color-text)]">History</h1>
          <p className="text-xs text-[var(--color-text-dim)] mt-0.5">
            Trade history &amp; account ledger
          </p>
        </div>
      </header>

      {/* 8-col KPI strip (2×4 at lg, 1×8 at xl) */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3">
          <HKPI
            label="Net PnL"
            value={`${metrics.net_pnl >= 0 ? "+" : "−"}${usd(Math.abs(metrics.net_pnl))}`}
            color={metrics.net_pnl >= 0 ? "var(--color-profit)" : "var(--color-loss)"}
          />
          <HKPI label="Trades" value={metrics.total_trades} />
          <HKPI
            label="Win rate"
            value={`${(metrics.win_rate * 100).toFixed(1)}%`}
            color={metrics.win_rate >= 0.5 ? "var(--color-profit)" : "var(--color-warn)"}
          />
          <HKPI
            label="Profit factor"
            value={num(metrics.profit_factor)}
            color={metrics.profit_factor >= 1 ? "var(--color-profit)" : "var(--color-loss)"}
          />
          <HKPI
            label="Avg R"
            value={`${(metrics.total_r / Math.max(metrics.total_trades, 1)) >= 0 ? "+" : ""}${(
              metrics.total_r / Math.max(metrics.total_trades, 1)
            ).toFixed(2)}`}
            color={
              metrics.total_r >= 0 ? "var(--color-profit)" : "var(--color-loss)"
            }
          />
          <HKPI
            label="Max DD"
            value={`−${metrics.max_drawdown_pct.toFixed(1)}%`}
            color="var(--color-warn)"
          />
          <HKPI label="Sharpe" value={num(metrics.sharpe_daily)} />
          <HKPI
            label="Calmar 90d"
            value={metrics.calmar_90d > 0 ? metrics.calmar_90d.toFixed(2) : "—"}
          />
        </div>
      )}

      {/* Equity chart */}
      {chartData.length > 1 && (
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className="section-label">Equity curve</p>
              <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                Last {chartData.length} points
              </p>
            </div>
          </div>
          <EquityChart data={chartData} height={200} />
        </div>
      )}

      {/* Main card: tabs + filters + table */}
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] overflow-hidden">
        <div className="flex items-center border-b border-[var(--color-border)] flex-wrap gap-2">
          <button
            onClick={() => switchTab("trades")}
            className="px-4 py-3 text-sm font-semibold border-b-2 -mb-px"
            style={{
              borderColor: tab === "trades" ? "var(--indigo)" : "transparent",
              color: tab === "trades" ? "var(--chip-info-fg)" : "var(--color-text-muted)",
            }}
          >
            Trades{" "}
            <span
              className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] ml-1 mono"
              style={{ background: "var(--color-panel-hi)", color: "var(--color-text-muted)" }}
            >
              {tradeCount}
            </span>
          </button>
          <button
            onClick={() => switchTab("account")}
            className="px-4 py-3 text-sm border-b-2 -mb-px"
            style={{
              borderColor: tab === "account" ? "var(--indigo)" : "transparent",
              color: tab === "account" ? "var(--chip-info-fg)" : "var(--color-text-muted)",
              fontWeight: tab === "account" ? 600 : 400,
            }}
          >
            Account ledger{" "}
            <span
              className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] ml-1 mono"
              style={{ background: "var(--color-panel-hi)", color: "var(--color-text-muted)" }}
            >
              {accountCount}
            </span>
          </button>
          {tab === "trades" && (
            <div className="ml-auto px-4 flex items-center gap-1 flex-wrap py-1.5">
              {SYMBOL_FILTERS.map((s) => {
                const active = (s === "All" && !symbol) || s === symbol;
                return (
                  <button
                    key={s}
                    onClick={() => {
                      setSymbol(s === "All" ? undefined : s);
                      setPage(1);
                    }}
                    className="px-2.5 py-1 text-[11px] rounded transition-colors mono"
                    style={
                      active
                        ? { background: "var(--color-panel-hi)", color: "var(--color-primary)" }
                        : { color: "var(--color-text-muted)" }
                    }
                  >
                    {s}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {tab === "trades" ? (
          <TradesTable
            trades={trades?.trades ?? []}
            isLoading={isLoading}
            total={tradeCount}
            page={page}
            totalPages={totalPages}
            onPage={setPage}
            expandedTicket={expandedTicket}
            onToggleExpand={setExpandedTicket}
          />
        ) : (
          <AccountTab loading={ledgerLoading} operations={ledger?.operations ?? []} />
        )}
      </div>
    </div>
  );
}

// ─── Trades table ────────────────────────────────────────────────────

function DirChip({ direction }: { direction: string | null }) {
  if (!direction) return <span className="text-[var(--color-text-dim)]">—</span>;
  const isBuy = direction === "buy";
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]"
      style={{
        background: isBuy ? "rgba(16,185,129,0.15)" : "rgba(244,63,94,0.15)",
        color: isBuy ? "var(--chip-profit-fg)" : "var(--chip-loss-fg)",
      }}
    >
      {isBuy ? "▲ BUY" : "▼ SELL"}
    </span>
  );
}

function TradesTable({
  trades,
  isLoading,
  total,
  page,
  totalPages,
  onPage,
  expandedTicket,
  onToggleExpand,
}: {
  trades: TradeHistoryItem[];
  isLoading: boolean;
  total: number;
  page: number;
  totalPages: number;
  onPage: (p: number) => void;
  expandedTicket: number | null;
  onToggleExpand: (n: number | null) => void;
}) {
  if (isLoading) {
    return (
      <p className="text-center text-sm text-[var(--color-text-muted)] py-10">
        Loading trades…
      </p>
    );
  }
  if (trades.length === 0) {
    return (
      <p className="text-center text-sm text-[var(--color-text-dim)] py-10">
        No trades found.
      </p>
    );
  }

  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-[10px] uppercase tracking-[0.14em] text-[var(--color-text-dim)] border-b border-[var(--color-border)]">
              <th className="text-left px-5 py-3 font-semibold">When</th>
              <th className="text-left px-2 py-3 font-semibold">Symbol</th>
              <th className="text-left px-2 py-3 font-semibold">Dir</th>
              <th className="text-right px-2 py-3 font-semibold">Entry → Exit</th>
              <th className="text-right px-2 py-3 font-semibold">PnL</th>
              <th className="text-right px-2 py-3 font-semibold">R</th>
              <th className="text-left px-2 py-3 font-semibold">Reason</th>
              <th className="text-right px-2 py-3 font-semibold">Bars</th>
              <th className="text-right px-5 py-3 font-semibold"></th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => {
              const canExpand = t.ticket != null && t.ticket > 0;
              const isExpanded = canExpand && expandedTicket === t.ticket;
              const pnl = t.pnl_usd ?? 0;
              const pnlCol = pnl >= 0 ? "var(--color-profit)" : "var(--color-loss)";
              const r = t.r_multiple_at_exit;
              const rCol =
                r == null
                  ? "var(--color-text-dim)"
                  : r >= 0
                    ? "var(--color-profit)"
                    : "var(--color-loss)";
              return (
                <RowFragment key={t.id}>
                  <tr
                    className={`border-b border-[var(--color-border)] hover:bg-[var(--color-panel-hi)] transition-colors ${
                      canExpand ? "cursor-pointer" : ""
                    }`}
                    onClick={() => {
                      if (!canExpand || t.ticket == null) return;
                      onToggleExpand(isExpanded ? null : t.ticket);
                    }}
                  >
                    <td className="px-5 py-3 mono text-xs text-[var(--color-text-muted)]">
                      <span className="inline-flex items-center gap-1">
                        {canExpand ? (
                          isExpanded ? (
                            <ChevronDown size={12} />
                          ) : (
                            <ChevronRight size={12} />
                          )
                        ) : (
                          <span className="w-3" />
                        )}
                        {shortDate(t.timestamp_close)}
                      </span>
                    </td>
                    <td className="px-2 py-3 mono font-semibold">{t.symbol}</td>
                    <td className="px-2 py-3">
                      <DirChip direction={t.direction} />
                    </td>
                    <td className="px-2 py-3 text-right mono text-xs">
                      {num(t.entry_price)}{" "}
                      <span className="text-[var(--color-text-dim)]">→</span>{" "}
                      {num(t.exit_price)}
                    </td>
                    <td
                      className="px-2 py-3 text-right mono font-semibold"
                      style={{ color: pnlCol }}
                    >
                      {pnl >= 0 ? "+" : "−"}${Math.abs(pnl).toFixed(2)}
                    </td>
                    <td className="px-2 py-3 text-right mono" style={{ color: rCol }}>
                      {r != null ? `${r >= 0 ? "+" : ""}${r.toFixed(2)}R` : "—"}
                    </td>
                    <td className="px-2 py-3">
                      <ReasonChip code={t.close_reason_code} full={t.close_reason} />
                    </td>
                    <td className="px-2 py-3 text-right mono text-xs text-[var(--color-text-muted)]">
                      {t.bars_held ?? "—"}
                    </td>
                    <td className="px-5 py-3 text-right">
                      {canExpand && (
                        <span className="text-[11px] text-[var(--color-primary)]">
                          {isExpanded ? "Hide ↑" : "View →"}
                        </span>
                      )}
                    </td>
                  </tr>
                  {isExpanded && t.ticket != null && (
                    <tr className="bg-[var(--color-panel-hi)]/30">
                      <td colSpan={9} className="px-5 py-4">
                        <TradeTimeline ticket={t.ticket} />
                      </td>
                    </tr>
                  )}
                </RowFragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="px-5 py-3 border-t border-[var(--color-border)] flex items-center justify-between text-xs text-[var(--color-text-muted)]">
        <span>
          Showing {trades.length} of {total}
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPage(Math.max(1, page - 1))}
            disabled={page <= 1}
            className="px-2.5 py-1 rounded bg-[var(--color-panel-hi)] disabled:opacity-40"
          >
            ←
          </button>
          <span className="mono">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => onPage(Math.min(totalPages, page + 1))}
            disabled={page >= totalPages}
            className="px-2.5 py-1 rounded bg-[var(--color-panel-hi)] disabled:opacity-40"
          >
            →
          </button>
        </div>
      </div>
    </>
  );
}

// React requires a keyed parent — use a fragment via a helper.
function RowFragment({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

// ─── Account ledger tab ──────────────────────────────────────────────

function AccountTab({
  loading,
  operations,
}: {
  loading: boolean;
  operations: BalanceOperation[];
}) {
  const enriched = useMemo(() => {
    if (operations.length === 0)
      return [] as Array<BalanceOperation & { running: number }>;
    const oldestFirst = [...operations].sort((a, b) =>
      String(a.time).localeCompare(String(b.time)),
    );
    let running = 0;
    return oldestFirst
      .map((op) => {
        running += op.amount;
        return { ...op, running };
      })
      .reverse();
  }, [operations]);

  if (loading) {
    return (
      <p className="text-center text-sm text-[var(--color-text-muted)] py-10">
        Loading account ledger…
      </p>
    );
  }
  if (enriched.length === 0) {
    return (
      <p className="text-center text-sm text-[var(--color-text-dim)] py-10">
        No account events in the last 365 days.
      </p>
    );
  }

  const typeLabel = (t: string): string =>
    t === "deposit"
      ? "DEPOSIT"
      : t === "withdrawal"
        ? "WITHDRAWAL"
        : t === "credit"
          ? "CREDIT"
          : t === "trade"
            ? "TRADE"
            : t.toUpperCase();

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[10px] uppercase tracking-[0.14em] text-[var(--color-text-dim)] border-b border-[var(--color-border)]">
            <th className="text-left px-5 py-3 font-semibold">Time</th>
            <th className="text-left px-2 py-3 font-semibold">Type</th>
            <th className="text-left px-2 py-3 font-semibold">Detail</th>
            <th className="text-right px-2 py-3 font-semibold">Amount</th>
            <th className="text-right px-2 py-3 font-semibold">Balance after</th>
            <th className="text-right px-5 py-3 font-semibold">Ticket</th>
          </tr>
        </thead>
        <tbody>
          {enriched.map((op, i) => (
            <tr
              key={`${op.ticket}-${i}`}
              className="border-b border-[var(--color-border)] hover:bg-[var(--color-panel-hi)] transition-colors"
            >
              <td className="px-5 py-3 mono text-xs text-[var(--color-text-muted)]">
                {shortDate(op.time)}
              </td>
              <td className="px-2 py-3">
                <span
                  className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold mono"
                  style={{
                    background:
                      op.type === "deposit"
                        ? "rgba(16,185,129,0.15)"
                        : op.type === "withdrawal"
                          ? "rgba(244,63,94,0.15)"
                          : op.type === "credit"
                            ? "rgba(245,158,11,0.15)"
                            : "rgba(6,182,212,0.15)",
                    color:
                      op.type === "deposit"
                        ? "var(--color-profit)"
                        : op.type === "withdrawal"
                          ? "var(--color-loss)"
                          : op.type === "credit"
                            ? "var(--color-warn)"
                            : "var(--color-primary)",
                  }}
                >
                  {typeLabel(op.type)}
                </span>
              </td>
              <td className="px-2 py-3 text-xs text-[var(--color-text-muted)]">
                {op.comment || "—"}
              </td>
              <td
                className="px-2 py-3 text-right mono font-semibold"
                style={{
                  color: op.amount >= 0 ? "var(--color-profit)" : "var(--color-loss)",
                }}
              >
                {op.amount >= 0 ? "+" : "−"}${Math.abs(op.amount).toFixed(2)}
              </td>
              <td className="px-2 py-3 text-right mono text-[var(--color-text-muted)]">
                {usd(op.running)}
              </td>
              <td className="px-5 py-3 text-right mono text-xs text-[var(--color-text-muted)]">
                {op.ticket || "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
