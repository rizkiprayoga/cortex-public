"""
account_monitor.py — MT5 Account State

Provides real-time account information: balance, equity, margin,
open positions, floating P&L, and balance operations (deposits,
withdrawals, credits).

Key MT5 functions used:
    mt5.account_info()     — Balance, equity, margin level
    mt5.positions_get()    — All open positions
    mt5.history_deals_get  — Historical deals incl. balance operations
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import MetaTrader5 as mt5

from src.broker.mt5_connector import MT5Connector
from src.data_pipeline.mt5_feed import _broker_ts_to_utc  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class AccountSnapshot:
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float       # equity / margin * 100
    floating_pnl: float
    open_positions: int


@dataclass
class BalanceOp:
    """One MT5 balance-operation deal — deposit, withdrawal, or credit."""
    time: str            # ISO-8601 UTC
    type: str            # "deposit" | "withdrawal" | "credit"
    amount: float
    comment: str
    ticket: int

    @classmethod
    def from_mt5(cls, deal) -> "BalanceOp":
        # mt5.DEAL_TYPE_BALANCE = 2, mt5.DEAL_TYPE_CREDIT = 3
        # A positive profit on a BALANCE deal is a deposit; negative is withdrawal.
        amt = float(deal.profit or 0.0)
        if getattr(deal, "type", None) == mt5.DEAL_TYPE_CREDIT:
            kind = "credit"
        elif amt >= 0:
            kind = "deposit"
        else:
            kind = "withdrawal"
        return cls(
            time=_broker_ts_to_utc(int(deal.time)).replace(tzinfo=timezone.utc).isoformat(),
            type=kind,
            amount=amt,
            comment=str(getattr(deal, "comment", "") or ""),
            ticket=int(getattr(deal, "ticket", 0) or 0),
        )


class AccountMonitor:
    """Reads live account state from MetaTrader 5."""

    def __init__(self, connector: MT5Connector):
        self.connector = connector
        self._peak_equity: float = 0.0
        self._session_start_balance: Optional[float] = None
        self._session_start_date: Optional[datetime] = None

    def get_info(self) -> AccountSnapshot:
        """
        Fetch current account snapshot from MT5.

        Returns:
            AccountSnapshot with balance, equity, margin, positions.
        """
        info = mt5.account_info()
        if info is None:
            err = mt5.last_error()
            raise RuntimeError(f"mt5.account_info() returned None: {err}")

        positions = mt5.positions_get() or []
        margin = float(info.margin or 0.0)
        equity = float(info.equity or 0.0)
        margin_level = (equity / margin * 100.0) if margin > 0 else 0.0

        snapshot = AccountSnapshot(
            balance=float(info.balance or 0.0),
            equity=equity,
            margin=margin,
            free_margin=float(info.margin_free or 0.0),
            margin_level=margin_level,
            floating_pnl=equity - float(info.balance or 0.0),
            open_positions=len(positions),
        )

        # Track peak equity for drawdown calc
        if snapshot.equity > self._peak_equity:
            self._peak_equity = snapshot.equity

        # Capture session-opening balance once per UTC day
        today = datetime.now(tz=timezone.utc).date()
        if (self._session_start_date is None
                or self._session_start_date != today):
            self._session_start_balance = snapshot.balance
            self._session_start_date = today

        return snapshot

    def get_open_positions(self, symbol: Optional[str] = None) -> list:
        """
        Return list of open MT5 position objects.

        Args:
            symbol: Filter by symbol (None = all positions).
        """
        if symbol is None:
            return list(mt5.positions_get() or [])
        return list(mt5.positions_get(symbol=symbol) or [])

    def get_daily_pnl(self) -> float:
        """
        Approximate today's P&L = current equity - session start balance.
        """
        if self._session_start_balance is None:
            # Force session-start capture by reading account info once
            try:
                self.get_info()
            except Exception:
                return 0.0
        info = mt5.account_info()
        if info is None or self._session_start_balance is None:
            return 0.0
        return float(info.equity or 0.0) - self._session_start_balance

    def get_peak_equity(self) -> float:
        """
        Return the peak equity recorded this session.
        """
        return self._peak_equity

    def trading_enabled(self) -> tuple[bool, str]:
        """
        Inspect MT5 account flags that block automated order submission.

        Catches the silent failure modes that have bitten us in production:
        - User toggled "Algo Trading" off in the MT5 terminal
          → ``trade_allowed = False`` → every order_send returns
          retcode 10027 (TRADE_RETCODE_CLIENT_DISABLES_AT) before this
          check.
        - Broker disabled algo trading server-side
          → ``trade_expert = False`` → retcode 10026.
        - Account in close-only mode (e.g., past margin call)
          → ``trade_mode != ACCOUNT_TRADE_MODE_REAL/DEMO`` or
          ``trade_allowed = False``.

        Returns ``(ok, reason)``. ``ok=True`` means orders should go through;
        ``ok=False`` means the bot should not bother sending them and the
        operator needs to flip a switch in MT5.
        """
        info = mt5.account_info()
        if info is None:
            err = mt5.last_error()
            return False, f"mt5.account_info() returned None: {err}"
        if not getattr(info, "trade_allowed", True):
            return False, (
                "account_info.trade_allowed=False — Algo Trading is OFF in "
                "the MT5 terminal (toolbar button) or in Tools → Options → "
                "Expert Advisors"
            )
        if not getattr(info, "trade_expert", True):
            return False, (
                "account_info.trade_expert=False — broker has disabled "
                "automated trading on this account server-side"
            )
        return True, "ok"

    def fetch_balance_operations(self, days: int = 365) -> list[BalanceOp]:
        """
        Pull *cash* balance operations only (deposits / withdrawals /
        credits) from MT5 history for the last ``days`` days. For the
        full account ledger including closed-trade P/L use
        :meth:`fetch_account_ledger`.
        """
        try:
            # MT5 filters history on broker-local time (Helsinki), not UTC —
            # widen to_dt by +1 day to avoid dropping recent entries.
            now_utc = datetime.now(tz=timezone.utc)
            to_dt = now_utc + timedelta(days=1)
            from_dt = now_utc - timedelta(days=max(1, int(days)))
            deals = mt5.history_deals_get(from_dt, to_dt)
            if deals is None:
                return []
            balance_types = {mt5.DEAL_TYPE_BALANCE, mt5.DEAL_TYPE_CREDIT}
            ops = [
                BalanceOp.from_mt5(d)
                for d in deals
                if getattr(d, "type", None) in balance_types
            ]
            ops.sort(key=lambda op: op.time, reverse=True)
            return ops
        except Exception as exc:
            logger.warning("fetch_balance_operations failed: %s", exc)
            return []

    def fetch_account_ledger(self, days: int = 365) -> list[BalanceOp]:
        """
        Full MT5-style "Account History" ledger: deposits + withdrawals
        + credits **+ closed-trade P/L** in one chronological list.

        For closed trades we emit one BalanceOp per closing deal (entry
        type == DEAL_ENTRY_OUT, OUT_BY, or INOUT) with `type='trade'`,
        amount = realized profit (deal.profit + commission + swap),
        comment = "<symbol> <buy/sell> close", ticket = deal.position_id.

        Returns ledger entries newest first. Empty list on failure (never
        raises); errors are logged at WARNING level.
        """
        try:
            # See fetch_balance_operations re: broker-TZ shift.
            now_utc = datetime.now(tz=timezone.utc)
            to_dt = now_utc + timedelta(days=1)
            from_dt = now_utc - timedelta(days=max(1, int(days)))
            deals = mt5.history_deals_get(from_dt, to_dt)
            if deals is None:
                return []
        except Exception as exc:
            logger.warning("fetch_account_ledger failed: %s", exc)
            return []

        balance_types = {mt5.DEAL_TYPE_BALANCE, mt5.DEAL_TYPE_CREDIT}
        # Closing entries — these are the deals that realize P/L
        out_entries = {
            getattr(mt5, "DEAL_ENTRY_OUT", 1),
            getattr(mt5, "DEAL_ENTRY_OUT_BY", 4),
            getattr(mt5, "DEAL_ENTRY_INOUT", 2),
        }

        # Group deals by position so we can sum commission/swap across the
        # whole round trip (the broker charges commission on BOTH the open
        # AND close deals — close-only summing would under-count by ~50%).
        # Profit is stored on the close leg (full round-trip P/L).
        from collections import defaultdict
        by_pos: dict[int, list] = defaultdict(list)
        ledger: list[BalanceOp] = []
        for d in deals:
            d_type = getattr(d, "type", None)
            if d_type in balance_types:
                ledger.append(BalanceOp.from_mt5(d))
                continue
            pid = int(getattr(d, "position_id", 0) or 0)
            if pid == 0:
                continue
            by_pos[pid].append(d)

        for pid, pd_list in by_pos.items():
            close_legs = [
                d for d in pd_list
                if getattr(d, "entry", None) in out_entries
            ]
            if not close_legs:
                continue  # still open
            close_legs.sort(key=lambda d: d.time)
            close = close_legs[-1]
            profit = float(getattr(close, "profit", 0.0) or 0.0)
            commission = sum(
                float(getattr(d, "commission", 0.0) or 0.0) for d in pd_list
            )
            swap = sum(
                float(getattr(d, "swap", 0.0) or 0.0) for d in pd_list
            )
            net = profit + commission + swap

            symbol = str(getattr(close, "symbol", "") or "")
            d_t = getattr(close, "type", None)
            side = (
                "buy" if d_t == mt5.DEAL_TYPE_BUY
                else "sell" if d_t == mt5.DEAL_TYPE_SELL
                else "?"
            )
            ledger.append(BalanceOp(
                time=_broker_ts_to_utc(int(close.time)).replace(tzinfo=timezone.utc).isoformat(),
                type="trade",
                amount=net,
                comment=f"{symbol} {side} close",
                ticket=pid,
            ))

        ledger.sort(key=lambda op: op.time, reverse=True)
        return ledger
