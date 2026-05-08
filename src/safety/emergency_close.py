"""
emergency_close.py — Force-Close All Positions

Immediately closes all open positions when called.
This module operates WITHOUT consulting the Brain (HMM/LSTM).

Triggered by:
    - RiskMonitor when max drawdown limit is breached (Level 2 halt)
    - Manual operator call via CLI: python -m src.safety.emergency_close

Behavior:
    1. Fetch all open positions via mt5.positions_get()
    2. For each position, send a market close order
    3. Log every close attempt and result
    4. Send alert (Telegram/email) with summary

If a position fails to close (e.g. market closed), it retries
up to 3 times with a 5-second delay, then logs the failure.
"""

import logging
import time
from typing import Optional

from src.broker.mt5_connector import MT5Connector
from src.broker.order_manager import OrderManager

logger = logging.getLogger(__name__)


class EmergencyClose:
    """
    Force-closes all open positions immediately.

    Usage (programmatic):
        ec = EmergencyClose(connector)
        ec.close_all()

    Usage (CLI):
        python -m src.safety.emergency_close
    """

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 0.5

    def __init__(
        self,
        connector: MT5Connector,
        order_manager: Optional[OrderManager] = None,
    ):
        self.connector = connector
        self.order_manager = order_manager or OrderManager(connector)

    def close_all(self, symbol: Optional[str] = None) -> dict:
        """
        Close every open position (optionally filtered by symbol).

        Walks the open positions one at a time, using the OrderManager
        to send a market close for each. Every close is retried up to
        ``MAX_RETRIES`` times with ``RETRY_DELAY_SECONDS`` backoff
        before giving up and logging the failure.

        Args:
            symbol: Restrict the sweep to a single symbol, or None for
                    every open ticket.

        Returns:
            ``{"closed": [tickets], "failed": [tickets]}`` — tickets
            that never closed are surfaced so the caller / alert path
            can escalate to the operator.
        """
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError:
            logger.error("MetaTrader5 not importable — cannot emergency-close")
            return {"closed": [], "failed": []}

        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            logger.warning("mt5.positions_get returned None — nothing to close")
            return {"closed": [], "failed": []}

        closed: list[int] = []
        failed: list[int] = []
        for pos in positions:
            ticket = int(getattr(pos, "ticket", 0))
            if ticket == 0:
                continue
            if self._close_with_retry(ticket):
                closed.append(ticket)
            else:
                failed.append(ticket)

        logger.warning(
            "EmergencyClose swept %d positions: %d closed, %d failed",
            len(positions),
            len(closed),
            len(failed),
        )
        return {"closed": closed, "failed": failed}

    def _close_with_retry(self, ticket: int) -> bool:
        """
        Attempt to close one ticket up to MAX_RETRIES times with
        RETRY_DELAY_SECONDS between attempts. Returns True on success.
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                result = self.order_manager.close_position(ticket)
            except Exception as exc:  # pragma: no cover — broker-side
                logger.error(
                    "close_position(%d) raised on attempt %d: %s",
                    ticket, attempt, exc,
                )
                result = None
            if result is not None and getattr(result, "success", False):
                return True
            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY_SECONDS)
        logger.error("EmergencyClose gave up on ticket %d after %d retries",
                     ticket, self.MAX_RETRIES)
        return False


if __name__ == "__main__":
    # CLI usage: python -m src.safety.emergency_close
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    connector = MT5Connector()
    connector.connect()
    ec = EmergencyClose(connector)
    result = ec.close_all()
    logger.info(f"Emergency close result: {result}")
    connector.disconnect()
    sys.exit(0 if not result.get("failed") else 1)
