"""
position_sizer.py — Strategy-Aware Fixed-Fractional Position Sizing

Computes lot size for a trade using the 1% risk-per-trade rule, scaled
by the active strategy's allocation percentage and halved whenever the
signal's uncertainty flag is set.

Formula
-------
    risk_usd        = equity * (max_risk_pct / 100)
    price_distance  = abs(entry_price - stop_price)
    base_lot        = risk_usd / (price_distance * contract_size)

    # Wave 6 fix #16 — gates stack via min(), not product. The caller
    # passes an already-stacked size_multiplier (typically
    # ``min(signal.size_discount, circuit_breaker.current_size_multiplier)``),
    # and if the legacy ``uncertainty_mode=True`` flag is set we fold
    # 0.5 into the multiplier via min() rather than compounding it.
    effective_mult  = min(size_multiplier, 0.5) if uncertainty_mode else size_multiplier

    final_lot       = base_lot * strategy.allocation_pct * effective_mult
    final_lot       = floor_to_step(final_lot, volume_step)
    final_lot       = clip(final_lot, volume_min, volume_max)

The ``contract_size`` factor converts the price distance into an
actual USD loss per lot (for XAUUSD: 100 oz/lot; for BTCUSD: 1 BTC/lot
on most MT5 brokers). We read it from the symbol spec passed in — the
caller is responsible for fetching ``mt5.symbol_info(symbol)`` and
packaging the fields we need, so this module stays free of MT5 imports
and is fully unit-testable.
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SymbolSpec:
    """
    Subset of ``mt5.symbol_info`` fields the sizer needs.

    Decouples PositionSizer from the MT5 library so the math can be
    exercised in isolation. Populate from ``mt5.symbol_info(symbol)``
    at the broker layer and pass in.

    Notes on the broker-authoritative fields
    ----------------------------------------
    MT5 already exposes exactly how much USD (deposit currency) 1.0 lot
    makes/loses per minimum price increment. When populated, ``tick_value``
    and ``tick_size`` give the sizer a leverage/currency-neutral path:

        loss_per_lot_usd = (price_distance / tick_size) * tick_value
        lot              = risk_usd / loss_per_lot_usd

    Works for every symbol type (XAU, forex, indices, crypto). The
    ``contract_size`` + ``quote_currency`` fields are a fallback when the
    two tick-* fields are unavailable (e.g. offline tests).
    """

    symbol: str
    contract_size: float = 1.0   # trade_contract_size from mt5
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    # MT5 authoritative fields — leave 0.0 to force the fallback formula
    tick_value: float = 0.0      # profit/loss in USD per 1 tick per 1.0 lot
    tick_size: float = 0.0       # minimum price increment
    quote_currency: str = "USD"  # fallback only — used when tick_value is 0


@dataclass
class SizingResult:
    """
    Result returned by ``PositionSizer.calculate()``.

    A ``lot_size == 0.0`` result means "do not trade" — callers must
    check this before passing the result to the broker layer.
    """

    symbol: str
    lot_size: float
    risk_amount_usd: float
    entry_price: float
    stop_price: float
    reason: str = ""


class PositionSizer:
    """
    Strategy-aware fixed-fractional sizer.

    Usage
    -----
        sizer = PositionSizer(max_risk_pct=1.0)
        result = sizer.calculate(
            symbol_spec=SymbolSpec(symbol="XAUUSD", contract_size=100),
            entry_price=2000.0,
            stop_price=1985.0,
            equity=10_000.0,
            allocation_pct=0.95,
            uncertainty_mode=False,
        )
        if result.lot_size > 0:
            order_manager.place_order(..., lot_size=result.lot_size)
    """

    def __init__(self, max_risk_pct: float = 1.0):
        if max_risk_pct <= 0:
            raise ValueError("max_risk_pct must be > 0")
        self.max_risk_pct = max_risk_pct

    def calculate(
        self,
        symbol_spec: SymbolSpec,
        entry_price: float,
        stop_price: float,
        equity: float,
        allocation_pct: float,
        uncertainty_mode: bool = False,
        size_multiplier: float = 1.0,
    ) -> SizingResult:
        """
        Compute the lot size for a trade.

        Args:
            symbol_spec:        SymbolSpec with contract_size + volume bounds
            entry_price:        Planned entry price
            stop_price:         Planned initial stop price
            equity:             Current account equity in account currency
            allocation_pct:     Strategy.allocation_pct (0.0 – 1.0)
            uncertainty_mode:   Signal uncertainty flag — halves the lot
            size_multiplier:    Optional external multiplier (e.g. from
                                CircuitBreaker.current_size_multiplier())

        Returns a SizingResult. If any input is degenerate (zero
        equity, stop equal to entry, negative risk) lot_size is 0.0
        and ``reason`` explains why.
        """
        if equity <= 0.0:
            return self._reject(
                symbol_spec, entry_price, stop_price,
                f"equity<=0 ({equity})",
            )
        if allocation_pct <= 0.0:
            return self._reject(
                symbol_spec, entry_price, stop_price,
                f"allocation_pct<=0 ({allocation_pct})",
            )
        if size_multiplier <= 0.0:
            return self._reject(
                symbol_spec, entry_price, stop_price,
                f"size_multiplier<=0 ({size_multiplier})",
            )

        price_distance = abs(entry_price - stop_price)
        if price_distance <= 0.0:
            return self._reject(
                symbol_spec, entry_price, stop_price,
                "entry_price == stop_price (undefined risk)",
            )

        if symbol_spec.contract_size <= 0.0:
            return self._reject(
                symbol_spec, entry_price, stop_price,
                f"contract_size<=0 ({symbol_spec.contract_size})",
            )

        risk_usd = equity * (self.max_risk_pct / 100.0)

        # -- Compute loss-per-lot in USD --
        # Preferred path: MT5 broker-authoritative tick_value/tick_size.
        # Fallback path: explicit quote-currency correction for USD-base
        # pairs (USDJPY, USDCAD, USDCHF) whose PnL is naturally in the
        # quote currency and needs to be converted back to USD.
        if symbol_spec.tick_value > 0.0 and symbol_spec.tick_size > 0.0:
            # Tick path — correct for every symbol type including XAU,
            # JPY pairs, indices, crypto. No currency heuristics needed.
            n_ticks = price_distance / symbol_spec.tick_size
            loss_per_lot_usd = n_ticks * symbol_spec.tick_value
        else:
            # Fallback: for XXX/USD pairs (EUR/USD, GBP/USD, XAU/USD)
            # PnL is already in USD = price_distance * contract_size.
            # For USD/XXX pairs (USD/JPY, USD/CAD, USD/CHF) the raw
            # product is in the quote currency (JPY, CAD, CHF) and must
            # be converted back to USD at the entry price.
            loss_per_lot_quote = price_distance * symbol_spec.contract_size
            if symbol_spec.quote_currency.upper() != "USD" and entry_price > 0:
                loss_per_lot_usd = loss_per_lot_quote / entry_price
            else:
                loss_per_lot_usd = loss_per_lot_quote

        if loss_per_lot_usd <= 0.0:
            return self._reject(
                symbol_spec, entry_price, stop_price,
                f"loss_per_lot_usd<=0 ({loss_per_lot_usd})",
            )

        raw_lot = risk_usd / loss_per_lot_usd

        # Wave 6 fix #16: gates stack via min(), not product. If the
        # caller still passes uncertainty_mode=True, fold 0.5 into the
        # multiplier via min() so a caller that also passes an
        # already-stacked size_multiplier doesn't end up at 0.25 from
        # the old compound rule. Real upstream callers (main.py) now
        # stack signal.size_discount with the circuit-breaker multiplier
        # themselves and set uncertainty_mode=False here.
        effective_mult = (
            min(size_multiplier, 0.5) if uncertainty_mode else size_multiplier
        )
        scaled = raw_lot * allocation_pct * effective_mult

        floored = self._floor_to_step(scaled, symbol_spec.volume_step)
        clipped = max(
            symbol_spec.volume_min if floored > 0 else 0.0,
            min(floored, symbol_spec.volume_max),
        )

        # If after flooring the lot falls below the broker's volume_min,
        # we still emit 0.0 — the trade is too small to submit.
        if clipped < symbol_spec.volume_min:
            return self._reject(
                symbol_spec, entry_price, stop_price,
                f"lot={clipped:.4f} below volume_min={symbol_spec.volume_min}",
            )

        realized_risk_usd = clipped * price_distance * symbol_spec.contract_size
        return SizingResult(
            symbol=symbol_spec.symbol,
            lot_size=clipped,
            risk_amount_usd=realized_risk_usd,
            entry_price=entry_price,
            stop_price=stop_price,
            reason=(
                f"risk_usd={risk_usd:.2f} alloc={allocation_pct:.2f} "
                f"mult={size_multiplier:.2f} uncertainty={uncertainty_mode} "
                f"effective_mult={effective_mult:.2f}"
            ),
        )

    @staticmethod
    def _floor_to_step(value: float, step: float) -> float:
        if step <= 0.0:
            return value
        return math.floor(value / step) * step

    @staticmethod
    def _reject(
        symbol_spec: SymbolSpec,
        entry_price: float,
        stop_price: float,
        reason: str,
    ) -> SizingResult:
        logger.info(
            "PositionSizer rejected %s: %s", symbol_spec.symbol, reason
        )
        return SizingResult(
            symbol=symbol_spec.symbol,
            lot_size=0.0,
            risk_amount_usd=0.0,
            entry_price=entry_price,
            stop_price=stop_price,
            reason=reason,
        )
