"""
manager.py — Central Alert Manager

Coordinates Telegram and Email notifications for key trading events.
All methods are fire-and-forget — failures are logged but never raised.
The trading loop must never crash because of a notification failure.

Alert triggers
--------------
    1. Circuit breaker trip   (critical — immediate)
    2. Trade execution        (info — entry placed or position closed)
    3. Emergency close        (critical — all positions flattened)
    4. Daily P&L summary      (scheduled — end-of-day digest)
    5. System status           (startup, shutdown, reconnect events)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from src.alerts.telegram import TelegramNotifier
from src.alerts.email import EmailNotifier

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


class AlertManager:
    """
    Unified alert dispatcher for all notification channels.

    Usage
    -----
        alerts = AlertManager()
        alerts.notify_breaker_trip(snapshot, equity=12345.0)
        alerts.notify_trade_entry(symbol="XAUUSD", direction="buy", ...)
    """

    def __init__(
        self,
        telegram: Optional[TelegramNotifier] = None,
        email: Optional[EmailNotifier] = None,
    ):
        self.telegram = telegram or TelegramNotifier()
        self.email = email or EmailNotifier()
        # Per-instance throttle map — see notify_trade_blocked.
        self._block_alert_last_fired: dict[tuple[str, str, str], float] = {}
        # Email-scope env flags (Telegram is always unaffected):
        #   EMAIL_WEEKLY_ONLY=1 — strict. Only the weekly summary reaches
        #     email. Daily summary, trades, system events, breakers:
        #     Telegram-only. Use when you want one pristine inbox report.
        #   EMAIL_DIGEST_ONLY=1 — looser. Daily + weekly summaries on
        #     email; everything else Telegram-only. Deprecated in favor
        #     of EMAIL_WEEKLY_ONLY.
        # If both are set, WEEKLY_ONLY wins (stricter).
        self.email_weekly_only = _env_bool("EMAIL_WEEKLY_ONLY", default=False)
        self.email_digest_only = _env_bool("EMAIL_DIGEST_ONLY", default=False)
        channels = []
        if self.telegram.enabled:
            channels.append("Telegram")
        if self.email.enabled:
            if self.email_weekly_only:
                suffix = " (weekly-only)"
            elif self.email_digest_only:
                suffix = " (digest-only)"
            else:
                suffix = ""
            channels.append(f"Email{suffix}")
        if channels:
            logger.info("AlertManager active: %s", ", ".join(channels))
        else:
            logger.info("AlertManager: no channels configured — alerts disabled")

    @property
    def enabled(self) -> bool:
        return self.telegram.enabled or self.email.enabled

    # ------------------------------------------------------------------
    # Low-level dispatch
    # ------------------------------------------------------------------

    def _send(
        self,
        text: str,
        subject: str,
        html: Optional[str] = None,
        channels: tuple[str, ...] = ("telegram", "email"),
        is_digest: bool = False,
        is_weekly: bool = False,
    ) -> None:
        """Send via the specified channels. Never raises.

        ``channels`` — which channels may receive this alert. Callers can
        pass ``("email",)`` for email-only (e.g. weekly summary) or
        ``("telegram",)`` for Telegram-only. Default broadcasts to both.

        ``is_digest`` — True marks the message as a periodic summary
        (daily + weekly both qualify). Used by ``EMAIL_DIGEST_ONLY``.

        ``is_weekly`` — True marks the message as the Sunday weekly
        summary specifically. A stricter subset of digest. Used by
        ``EMAIL_WEEKLY_ONLY``.

        Email gating (Telegram is always unaffected):
          - EMAIL_WEEKLY_ONLY=1 → skip email unless is_weekly is True
          - EMAIL_DIGEST_ONLY=1 → skip email unless is_digest is True
        Both flags leave Telegram delivery untouched.
        """
        if "telegram" in channels:
            try:
                self.telegram.send(text)
            except Exception as exc:
                logger.warning("Telegram dispatch error: %s", exc)

        email_wanted = "email" in channels
        if email_wanted and self.email_weekly_only and not is_weekly:
            email_wanted = False  # stricter: weekly-only mode
        elif email_wanted and self.email_digest_only and not is_digest:
            email_wanted = False  # looser: digest-only mode
        if email_wanted:
            try:
                self.email.send(subject, html or f"<pre>{_escape_html(text)}</pre>")
            except Exception as exc:
                logger.warning("Email dispatch error: %s", exc)

    # ------------------------------------------------------------------
    # Alert: Circuit Breaker Trip
    # ------------------------------------------------------------------

    def notify_breaker_trip(
        self,
        active_breakers: list[str],
        daily_dd_pct: float,
        weekly_dd_pct: float,
        peak_dd_pct: float,
        equity: float,
        requires_flat: bool = True,
    ) -> None:
        """Alert on circuit breaker activation."""
        ts = _now_str()
        breakers_str = ", ".join(active_breakers) or "none"
        action = "EMERGENCY CLOSE FIRED" if requires_flat else "Size halved"

        text = (
            f"CIRCUIT BREAKER TRIPPED\n"
            f"Time: {ts}\n"
            f"Active: {breakers_str}\n"
            f"Daily DD: {daily_dd_pct:.2f}%\n"
            f"Weekly DD: {weekly_dd_pct:.2f}%\n"
            f"Peak DD: {peak_dd_pct:.2f}%\n"
            f"Equity: ${equity:,.2f}\n"
            f"Action: {action}"
        )

        html = (
            f"<h3 style='color:red;'>CIRCUIT BREAKER TRIPPED</h3>"
            f"<table>"
            f"<tr><td><b>Time</b></td><td>{ts}</td></tr>"
            f"<tr><td><b>Active</b></td><td>{breakers_str}</td></tr>"
            f"<tr><td><b>Daily DD</b></td><td>{daily_dd_pct:.2f}%</td></tr>"
            f"<tr><td><b>Weekly DD</b></td><td>{weekly_dd_pct:.2f}%</td></tr>"
            f"<tr><td><b>Peak DD</b></td><td>{peak_dd_pct:.2f}%</td></tr>"
            f"<tr><td><b>Equity</b></td><td>${equity:,.2f}</td></tr>"
            f"<tr><td><b>Action</b></td><td style='color:red;'>{action}</td></tr>"
            f"</table>"
        )

        self._send(text, "CIRCUIT BREAKER TRIPPED", html)

    # ------------------------------------------------------------------
    # Alert: Emergency Close Result
    # ------------------------------------------------------------------

    def notify_emergency_close(
        self,
        closed_tickets: list[int],
        failed_tickets: list[int],
    ) -> None:
        """Alert after EmergencyClose sweep completes."""
        ts = _now_str()
        text = (
            f"EMERGENCY CLOSE COMPLETE\n"
            f"Time: {ts}\n"
            f"Closed: {len(closed_tickets)} positions {closed_tickets}\n"
            f"Failed: {len(failed_tickets)} positions {failed_tickets}"
        )
        if failed_tickets:
            text += "\nMANUAL INTERVENTION REQUIRED for failed tickets!"

        self._send(text, "EMERGENCY CLOSE COMPLETE")

    # ------------------------------------------------------------------
    # Alert: Trade Entry
    # ------------------------------------------------------------------

    def notify_trade_entry(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        entry_price: float,
        stop_loss: float,
        ticket: int,
        strategy: str = "",
        # --- Brain-level context (all optional; omitted if None) ---
        regime: Optional[str] = None,
        regime_prob: Optional[float] = None,
        lstm_prediction: Optional[float] = None,
        combined_score: Optional[float] = None,
        confidence: Optional[float] = None,
        reasoning: Optional[list[str]] = None,
    ) -> None:
        """
        Alert on new trade execution, with the Brain's decision context.

        ``reasoning`` is the raw list from ``SignalResult.reasoning`` /
        ``StrategyDecision.reasoning`` — we dump all lines so the operator
        can audit why the bot entered.
        """
        ts = _now_str()
        risk_pts = abs(entry_price - stop_loss)

        brain_lines = []
        if regime is not None:
            prob = f" ({regime_prob*100:.0f}%)" if regime_prob is not None else ""
            brain_lines.append(f"Regime: {regime}{prob}")
        if lstm_prediction is not None:
            brain_lines.append(f"LSTM: {lstm_prediction:+.4f}")
        if combined_score is not None:
            brain_lines.append(f"Combined score: {combined_score:+.3f}")
        if confidence is not None:
            brain_lines.append(f"Confidence: {confidence:.2f}")

        rationale_block = ""
        if reasoning:
            lines = [f"  • {r}" for r in reasoning[:15]]  # cap at 15 lines
            rationale_block = "\nRationale:\n" + "\n".join(lines)

        text = (
            f"NEW TRADE OPENED\n"
            f"Time: {ts}\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction.upper()}\n"
            f"Lots: {lot_size}\n"
            f"Entry: {entry_price:.5g}\n"
            f"SL: {stop_loss:.5g} ({risk_pts:.2f} pts)\n"
            f"Ticket: {ticket}\n"
            f"Strategy: {strategy}"
            + ("\n" + "\n".join(brain_lines) if brain_lines else "")
            + rationale_block
        )

        self._send(text, f"Trade Opened: {symbol} {direction.upper()}")

    # ------------------------------------------------------------------
    # Alert: Trade Close
    # ------------------------------------------------------------------

    def notify_trade_close(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        lot_size: float,
        pnl: float,
        ticket: int,
        reason: str = "",
        # --- Brain-level context (optional, matches notify_trade_entry) ---
        regime: Optional[str] = None,
        regime_prob: Optional[float] = None,
        lstm_prediction: Optional[float] = None,
        combined_score: Optional[float] = None,
        bars_held: Optional[int] = None,
        r_multiple: Optional[float] = None,
        # --- Extended fields (user request: entry time, duration, SL, strategy) ---
        entry_time: Optional[str] = None,
        duration: Optional[str] = None,
        initial_stop: Optional[float] = None,
        strategy_name: Optional[str] = None,
    ) -> None:
        """Alert on position close with optional performance context."""
        ts = _now_str()
        pnl_color = "green" if pnl >= 0 else "red"
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

        # --- Trade detail lines ---
        detail_lines = []
        if entry_time is not None:
            detail_lines.append(f"Entry time: {entry_time}")
        if duration is not None:
            detail_lines.append(f"Duration: {duration}")
        if initial_stop is not None:
            sl_dist = abs(entry_price - initial_stop)
            detail_lines.append(f"Initial SL: {initial_stop:.5g} ({sl_dist:.5g} pts)")
        if strategy_name:
            detail_lines.append(f"Strategy: {strategy_name}")

        # --- Brain context lines ---
        brain_lines = []
        if regime is not None:
            prob = f" ({regime_prob*100:.0f}%)" if regime_prob is not None else ""
            brain_lines.append(f"Regime at close: {regime}{prob}")
        if lstm_prediction is not None:
            brain_lines.append(f"LSTM at close: {lstm_prediction:+.4f}")
        if combined_score is not None:
            brain_lines.append(f"Score at close: {combined_score:+.3f}")
        if bars_held is not None:
            detail_lines.append(f"Bars held: {bars_held}")
        if r_multiple is not None:
            detail_lines.append(f"R-multiple: {r_multiple:+.2f}R")

        text = (
            f"TRADE CLOSED\n"
            f"Closed: {ts}\n"
            f"Symbol: {symbol} {direction.upper()}\n"
            f"Entry: {entry_price:.5g} -> Exit: {exit_price:.5g}\n"
            f"Lots: {lot_size}\n"
            f"PnL: {pnl_str}\n"
            f"Ticket: {ticket}\n"
            f"Reason: {reason}"
            + ("\n" + "\n".join(detail_lines) if detail_lines else "")
            + ("\n" + "\n".join(brain_lines) if brain_lines else "")
        )

        all_info_lines = detail_lines + brain_lines
        info_rows = "".join(
            f"<tr><td><b>{_escape_html(line.split(':',1)[0])}</b></td>"
            f"<td>{_escape_html(line.split(':',1)[1].strip())}</td></tr>"
            for line in all_info_lines if ":" in line
        )
        html = (
            f"<h3>Trade Closed: {_escape_html(symbol)}</h3>"
            f"<table>"
            f"<tr><td><b>Direction</b></td><td>{_escape_html(direction.upper())}</td></tr>"
            f"<tr><td><b>Entry</b></td><td>{entry_price:.5g}</td></tr>"
            f"<tr><td><b>Exit</b></td><td>{exit_price:.5g}</td></tr>"
            f"<tr><td><b>Lots</b></td><td>{lot_size}</td></tr>"
            f"<tr><td><b>PnL</b></td><td style='color:{pnl_color};'>"
            f"{_escape_html(pnl_str)}</td></tr>"
            f"<tr><td><b>Reason</b></td><td>{_escape_html(str(reason))}</td></tr>"
            + info_rows +
            f"</table>"
        )

        self._send(text, f"Trade Closed: {symbol} {pnl_str}", html)

    # ------------------------------------------------------------------
    # Alert: Daily Summary
    # ------------------------------------------------------------------

    def notify_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        open_positions: int,
        trades_today: int,
        win_rate: Optional[float] = None,
        breaker_status: str = "clear",
        # --- Extended fields (all optional for backward compat) ---
        balance: Optional[float] = None,
        floating_pnl: Optional[float] = None,
        realized_pnl: Optional[float] = None,
        weekly_pnl: Optional[float] = None,
        daily_dd_pct: Optional[float] = None,
        weekly_dd_pct: Optional[float] = None,
        peak_dd_pct: Optional[float] = None,
        margin_used: Optional[float] = None,
        free_margin: Optional[float] = None,
        per_symbol_pnl: Optional[dict[str, dict]] = None,
        open_position_details: Optional[list[dict]] = None,
        regime_summary: Optional[dict[str, str]] = None,
    ) -> None:
        """End-of-day digest with P&L, account state, and per-symbol breakdown."""
        ts = _now_str()
        pnl_sign = "+" if daily_pnl >= 0 else ""
        wr_str = f"{win_rate * 100:.1f}%" if win_rate is not None else "N/A"

        # --- Account section ---
        lines = [
            f"DAILY SUMMARY — {ts[:10]}",
            "",
            f"Equity: ${equity:,.2f}",
        ]
        if balance is not None:
            lines.append(f"Balance: ${balance:,.2f}")
        daily_pnl_str = f"+${daily_pnl:,.2f}" if daily_pnl >= 0 else f"-${abs(daily_pnl):,.2f}"
        lines.append(f"Daily PnL: {daily_pnl_str}")
        if realized_pnl is not None:
            r_str = f"+${realized_pnl:,.2f}" if realized_pnl >= 0 else f"-${abs(realized_pnl):,.2f}"
            lines.append(f"  Realized: {r_str}")
        if floating_pnl is not None:
            f_str = f"+${floating_pnl:,.2f}" if floating_pnl >= 0 else f"-${abs(floating_pnl):,.2f}"
            lines.append(f"  Floating: {f_str}")
        if weekly_pnl is not None:
            w_str = f"+${weekly_pnl:,.2f}" if weekly_pnl >= 0 else f"-${abs(weekly_pnl):,.2f}"
            lines.append(f"Weekly PnL: {w_str}")

        # --- Drawdown section ---
        dd_parts = []
        if daily_dd_pct is not None:
            dd_parts.append(f"D:{daily_dd_pct:.1f}%")
        if weekly_dd_pct is not None:
            dd_parts.append(f"W:{weekly_dd_pct:.1f}%")
        if peak_dd_pct is not None:
            dd_parts.append(f"Peak:{peak_dd_pct:.1f}%")
        if dd_parts:
            lines.append(f"Drawdown: {' | '.join(dd_parts)}")

        # --- Margin section ---
        if margin_used is not None and free_margin is not None:
            lines.append(f"Margin: ${margin_used:,.0f} used / ${free_margin:,.0f} free")

        # --- Trade stats ---
        lines.append("")
        lines.append(f"Trades Today: {trades_today}  |  Win Rate: {wr_str}")
        lines.append(f"Open Positions: {open_positions}")
        lines.append(f"Breaker Status: {breaker_status}")

        # --- Per-symbol P&L breakdown ---
        if per_symbol_pnl:
            lines.append("")
            lines.append("Per-Symbol P&L:")
            for sym, info in sorted(per_symbol_pnl.items()):
                cnt = info.get("count", 0)
                pnl = info.get("pnl", 0.0)
                wins = info.get("wins", 0)
                pnl_fmt = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
                t_word = "trade" if cnt == 1 else "trades"
                lines.append(f"  {sym}: {pnl_fmt} ({cnt} {t_word}, {wins}W)")

        # --- Open positions ---
        if open_position_details:
            lines.append("")
            lines.append("Open Positions:")
            for p in open_position_details:
                sym = p.get("symbol", "?")
                d = p.get("direction", "?").upper()
                entry = p.get("entry_price", 0)
                fl_pnl = p.get("floating_pnl", 0)
                be = "BE" if p.get("be_locked") else ""
                fl_fmt = f"+${fl_pnl:,.2f}" if fl_pnl >= 0 else f"-${abs(fl_pnl):,.2f}"
                lines.append(f"  {sym} {d} @ {entry:.5g}  {fl_fmt} {be}")

        # --- Regime summary ---
        if regime_summary:
            lines.append("")
            lines.append("Regimes: " + " | ".join(
                f"{sym}={reg}" for sym, reg in sorted(regime_summary.items())
            ))

        text = "\n".join(lines)

        pnl_color = "green" if daily_pnl >= 0 else "red"
        html = (
            f"<h3>Cortex Daily Summary — {ts[:10]}</h3>"
            f"<table>"
            f"<tr><td><b>Equity</b></td><td>${equity:,.2f}</td></tr>"
            + (f"<tr><td><b>Balance</b></td><td>${balance:,.2f}</td></tr>" if balance is not None else "")
            + f"<tr><td><b>Daily PnL</b></td><td style='color:{pnl_color};'>"
            f"{pnl_sign}${daily_pnl:,.2f}</td></tr>"
            f"<tr><td><b>Trades Today</b></td><td>{trades_today}</td></tr>"
            f"<tr><td><b>Win Rate</b></td><td>{wr_str}</td></tr>"
            f"<tr><td><b>Open Positions</b></td><td>{open_positions}</td></tr>"
            f"<tr><td><b>Breaker Status</b></td><td>{breaker_status}</td></tr>"
            + ("".join(
                f"<tr><td><b>DD</b></td><td>{' | '.join(dd_parts)}</td></tr>"
            ) if dd_parts else "")
            + f"</table>"
        )

        self._send(
            text,
            f"Daily Summary: {pnl_sign}${daily_pnl:,.2f}",
            html,
            is_digest=True,
        )

    # ------------------------------------------------------------------
    # Alert: Weekly Summary (email-only by design)
    # ------------------------------------------------------------------

    def notify_weekly_summary(
        self,
        week_start: str,       # ISO date of the week's Monday (UTC)
        week_end: str,         # ISO date of the week's Sunday (UTC)
        starting_equity: float,
        ending_equity: float,
        net_pnl: float,
        trade_count: int,
        wins: int,
        losses: int,
        max_dd_pct: float,
        per_symbol: Optional[dict[str, dict]] = None,  # {sym: {pnl, count, wins, ...}}
        best_trade: Optional[dict] = None,             # {symbol, pnl, entry_time}
        worst_trade: Optional[dict] = None,
        breaker_events: Optional[list[str]] = None,    # e.g. ["2026-04-19 daily_soft"]
        stale_models: Optional[list[str]] = None,      # symbols with LSTM older than 30d
    ) -> None:
        """Weekly digest — rendered to email only.

        Even if both channels are configured, this alert is routed strictly
        to email. Telegram is not touched. Designed to give operators a
        Sunday-evening review mail without any mid-week per-trade noise.
        """
        win_rate = (wins / max(wins + losses, 1)) * 100 if (wins + losses) > 0 else 0.0
        # Derive % from the same `net_pnl` we display so the two never
        # disagree. `(ending - starting) / starting` would also fold in
        # floating P/L from positions open at the window boundaries —
        # accurate as a return number, but confusing when shown next to
        # a Net PnL that's strictly realized-pnl from closed trades.
        net_pct = (net_pnl / starting_equity * 100) if starting_equity > 0 else 0.0
        pnl_sign = "+" if net_pnl >= 0 else "-"
        pnl_abs = f"${abs(net_pnl):,.2f}"
        net_pct_sign = "+" if net_pct >= 0 else ""
        pnl_color = "#16a34a" if net_pnl >= 0 else "#dc2626"

        # --- Plain-text body (also the email fallback) ---
        lines = [
            f"WEEKLY SUMMARY · {week_start} → {week_end}",
            "",
            f"Starting equity: ${starting_equity:,.2f}",
            f"Ending equity:   ${ending_equity:,.2f}",
            f"Net PnL:         {pnl_sign}{pnl_abs} ({net_pct_sign}{net_pct:.2f}%)",
            f"Max drawdown:    {max_dd_pct:.2f}%",
            "",
            f"Trades: {trade_count}  |  Wins: {wins}  |  Losses: {losses}  |  WR: {win_rate:.1f}%",
        ]

        if per_symbol:
            lines.append("")
            lines.append("Per-symbol:")
            for sym, info in sorted(per_symbol.items()):
                cnt = info.get("count", 0)
                pnl = info.get("pnl", 0.0)
                w = info.get("wins", 0)
                pnl_fmt = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
                sym_wr = (w / max(cnt, 1)) * 100 if cnt > 0 else 0.0
                lines.append(f"  {sym}: {pnl_fmt} · {cnt} trade{'s' if cnt != 1 else ''} · {sym_wr:.0f}% WR")

        if best_trade or worst_trade:
            lines.append("")
            if best_trade:
                bt_pnl = best_trade.get("pnl", 0.0)
                lines.append(f"Best trade:  {best_trade.get('symbol', '?')} +${bt_pnl:,.2f}  ({best_trade.get('entry_time', '')[:10]})")
            if worst_trade:
                wt_pnl = worst_trade.get("pnl", 0.0)
                lines.append(f"Worst trade: {worst_trade.get('symbol', '?')} -${abs(wt_pnl):,.2f}  ({worst_trade.get('entry_time', '')[:10]})")

        if breaker_events:
            lines.append("")
            lines.append("Breakers fired this week:")
            for ev in breaker_events:
                lines.append(f"  · {ev}")

        if stale_models:
            lines.append("")
            lines.append(f"⚠ Stale models (>30d since retrain): {', '.join(stale_models)}")

        text = "\n".join(lines)

        # --- HTML body ---
        sym_rows = ""
        if per_symbol:
            sym_rows = "".join(
                f"<tr>"
                f"<td style='padding:4px 12px;'>{_escape_html(str(sym))}</td>"
                f"<td style='padding:4px 12px; text-align:right; color:{'#16a34a' if info.get('pnl', 0) >= 0 else '#dc2626'};'>"
                f"{'+' if info.get('pnl', 0) >= 0 else '-'}${abs(info.get('pnl', 0)):,.2f}</td>"
                f"<td style='padding:4px 12px; text-align:right;'>{info.get('count', 0)}</td>"
                f"<td style='padding:4px 12px; text-align:right;'>"
                f"{(info.get('wins', 0) / max(info.get('count', 1), 1)) * 100:.0f}%</td>"
                f"</tr>"
                for sym, info in sorted(per_symbol.items())
            )

        breakers_html = ""
        if breaker_events:
            items = "".join(f"<li>{_escape_html(ev)}</li>" for ev in breaker_events)
            breakers_html = f"<h4 style='margin:18px 0 6px 0;'>Breakers fired</h4><ul>{items}</ul>"

        stale_html = ""
        if stale_models:
            stale_html = (
                f"<p style='color:#c47a1e; margin-top:16px;'>"
                f"⚠ <b>Stale models</b> (>30d since retrain): {_escape_html(', '.join(stale_models))}"
                f"</p>"
            )

        best_worst_html = ""
        if best_trade or worst_trade:
            best_worst_html = "<h4 style='margin:18px 0 6px 0;'>Best / worst trade</h4><ul>"
            if best_trade:
                best_worst_html += (
                    f"<li>Best: <b>{_escape_html(str(best_trade.get('symbol', '?')))}</b> "
                    f"<span style='color:#16a34a;'>+${best_trade.get('pnl', 0):,.2f}</span> "
                    f"<span style='color:#666;'>({_escape_html(str(best_trade.get('entry_time', ''))[:10])})</span></li>"
                )
            if worst_trade:
                best_worst_html += (
                    f"<li>Worst: <b>{_escape_html(str(worst_trade.get('symbol', '?')))}</b> "
                    f"<span style='color:#dc2626;'>-${abs(worst_trade.get('pnl', 0)):,.2f}</span> "
                    f"<span style='color:#666;'>({_escape_html(str(worst_trade.get('entry_time', ''))[:10])})</span></li>"
                )
            best_worst_html += "</ul>"

        html = (
            f"<div style='font-family: system-ui, Segoe UI, Arial; max-width: 620px; color:#1f2937;'>"
            f"<h2 style='margin:0 0 4px 0;'>Cortex Weekly Summary</h2>"
            f"<p style='color:#6b7280; margin:0 0 16px 0;'>{week_start} → {week_end} (UTC)</p>"
            f"<table style='border-collapse:collapse; margin-bottom:12px;'>"
            f"<tr><td style='padding:4px 12px; color:#6b7280;'>Starting equity</td>"
            f"<td style='padding:4px 12px; font-family:ui-monospace,monospace;'>${starting_equity:,.2f}</td></tr>"
            f"<tr><td style='padding:4px 12px; color:#6b7280;'>Ending equity</td>"
            f"<td style='padding:4px 12px; font-family:ui-monospace,monospace;'>${ending_equity:,.2f}</td></tr>"
            f"<tr><td style='padding:4px 12px; color:#6b7280;'>Net PnL</td>"
            f"<td style='padding:4px 12px; font-family:ui-monospace,monospace; color:{pnl_color}; font-weight:600;'>"
            f"{pnl_sign}{pnl_abs} ({net_pct_sign}{net_pct:.2f}%)</td></tr>"
            f"<tr><td style='padding:4px 12px; color:#6b7280;'>Max drawdown</td>"
            f"<td style='padding:4px 12px; font-family:ui-monospace,monospace;'>{max_dd_pct:.2f}%</td></tr>"
            f"<tr><td style='padding:4px 12px; color:#6b7280;'>Trades</td>"
            f"<td style='padding:4px 12px; font-family:ui-monospace,monospace;'>"
            f"{trade_count} ({wins}W / {losses}L · {win_rate:.1f}% WR)</td></tr>"
            f"</table>"
            + (
                f"<h4 style='margin:18px 0 6px 0;'>Per-symbol</h4>"
                f"<table style='border-collapse:collapse; border-top:1px solid #e5e7eb; border-bottom:1px solid #e5e7eb;'>"
                f"<thead><tr style='background:#f9fafb; color:#6b7280; font-size:12px;'>"
                f"<th style='padding:6px 12px; text-align:left;'>Symbol</th>"
                f"<th style='padding:6px 12px; text-align:right;'>PnL</th>"
                f"<th style='padding:6px 12px; text-align:right;'>Trades</th>"
                f"<th style='padding:6px 12px; text-align:right;'>WR</th>"
                f"</tr></thead>"
                f"<tbody style='font-family:ui-monospace,monospace; font-size:13px;'>{sym_rows}</tbody>"
                f"</table>"
                if sym_rows else ""
            )
            + best_worst_html
            + breakers_html
            + stale_html
            + f"<p style='margin-top:28px; color:#9ca3af; font-size:11px;'>"
            f"Generated Sunday 23:55 UTC · demo account only · don't act on this alone</p>"
            f"</div>"
        )

        subject = f"Cortex · Weekly {week_start[-5:]}-{week_end[-5:]}: {pnl_sign}{pnl_abs}"
        self._send(
            text,
            subject,
            html,
            channels=("email",),     # strictly email — no Telegram spam
            is_digest=True,
            is_weekly=True,          # passes EMAIL_WEEKLY_ONLY gate
        )

    # ------------------------------------------------------------------
    # Alert: Trade Blocked
    # ------------------------------------------------------------------

    # Throttle identical block alerts — key is (symbol, category, reason),
    # value is the last-fired monotonic timestamp. Prevents spamming the
    # operator when e.g. USDCAD fails `Invalid stops` every 15-min cycle
    # until the price moves. See `_BLOCK_ALERT_THROTTLE_SEC` below.
    # Moved from class-level to instance-level 2026-04-19 so tests + the
    # weekly-sample script don't share throttle state with production.
    _BLOCK_ALERT_THROTTLE_SEC: float = 3600.0  # 1 hour between repeats

    def notify_trade_blocked(
        self,
        symbol: str,
        reason: str,
        retcode: Optional[int] = None,
        category: str = "broker_reject",
    ) -> None:
        """
        Alert when a signal that was approved by the strategy gets blocked
        downstream (broker reject, sizing failure, AutoTrading off, etc.).

        Categories:
        - "broker_reject"  — mt5.order_send / order_check returned a non-DONE retcode
        - "sizing"         — PortfolioManager refused (cap, margin, BE, etc.)
        - "trade_disabled" — MT5 account_info.trade_allowed/trade_expert is False
        - "self_check"     — startup or cycle-level health probe failed

        Throttled to at most one alert per (symbol, category, reason) every
        ``_BLOCK_ALERT_THROTTLE_SEC`` (1 hour). The trading_bot.log still
        captures every reject at WARNING level so nothing is hidden — the
        throttle only suppresses repetitive Telegram / email.
        """
        import time as _time
        key = (symbol, category, reason[:200])
        now_mono = _time.monotonic()
        last = self._block_alert_last_fired.get(key, 0.0)
        if now_mono - last < self._BLOCK_ALERT_THROTTLE_SEC:
            logger.debug(
                "notify_trade_blocked suppressed (throttle): %s %s %s",
                symbol, category, reason[:80],
            )
            return
        self._block_alert_last_fired[key] = now_mono

        ts = _now_str()
        rc_part = f" (retcode={retcode})" if retcode is not None else ""
        text = (
            f"TRADE BLOCKED — {category}\n"
            f"Time: {ts}\n"
            f"Symbol: {symbol}\n"
            f"Reason: {reason}{rc_part}"
        )
        html = (
            f"<h3 style='color:#f59e0b;'>Trade Blocked: {_escape_html(symbol)}</h3>"
            f"<table>"
            f"<tr><td><b>Category</b></td><td>{_escape_html(category)}</td></tr>"
            f"<tr><td><b>Reason</b></td><td>{_escape_html(str(reason))}</td></tr>"
            + (f"<tr><td><b>Retcode</b></td><td>{retcode}</td></tr>" if retcode is not None else "")
            + f"<tr><td><b>Time</b></td><td>{ts}</td></tr>"
            f"</table>"
        )
        self._send(text, f"BLOCKED: {symbol} ({category})", html)

    # ------------------------------------------------------------------
    # Alert: System Status
    # ------------------------------------------------------------------

    def notify_system(self, event: str, details: str = "") -> None:
        """System lifecycle events: startup, shutdown, reconnect, errors."""
        ts = _now_str()
        text = f"SYSTEM: {event}\nTime: {ts}"
        if details:
            text += f"\n{details}"
        self._send(text, f"System: {event}")

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_all(self) -> dict[str, bool]:
        """
        Send a test message on every enabled channel.

        Returns:
            {"telegram": True/False, "email": True/False}
        """
        results = {
            "telegram": self.telegram.test_connection() if self.telegram.enabled else False,
            "email": self.email.test_connection() if self.email.enabled else False,
        }
        logger.info("Alert test results: %s", results)
        return results


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _now_str() -> str:
    """Format 'now' in the operator's local timezone for Telegram/email.
    Defaults to Asia/Jakarta (WIB, UTC+7) — override with ALERT_DISPLAY_TZ
    env var (any zoneinfo name like 'Europe/Berlin' or 'UTC')."""
    import os
    from zoneinfo import ZoneInfo
    tzname = os.getenv("ALERT_DISPLAY_TZ", "Asia/Jakarta")
    try:
        tz = ZoneInfo(tzname)
    except Exception:
        tz = timezone.utc
        tzname = "UTC"
    return datetime.now(tz=tz).strftime(f"%Y-%m-%d %H:%M:%S {tzname}")


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
