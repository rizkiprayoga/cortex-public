"""
routes/live.py — Live trading state endpoints.

GET  /api/live/state          → top-level dashboard cards
GET  /api/live/signals/{sym}  → latest SignalResult for a symbol
GET  /api/live/positions      → all tracked open positions
GET  /api/live/breaker        → circuit breaker snapshot
GET  /api/live/stream         → SSE stream (2s interval)
"""

import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from src.api.auth import get_current_user
from src.strategy.exit_manager import ExitManager
from src.api.schemas import (
    AccountResponse,
    BreakerResponse,
    CandlesResponse,
    LiveStateResponse,
    OHLCVBarResponse,
    PositionResponse,
    RegimeResponse,
    SignalResponse,
)
from src.api.sse import stream_live_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/live", tags=["live"])


def _get_live_state(request: Request):
    """Retrieve LiveState from app.state."""
    return request.app.state.live_state


def _regime_to_response(regime) -> RegimeResponse:
    """Convert a RegimeResult dataclass to a Pydantic RegimeResponse."""
    all_probs = regime.all_probabilities
    if isinstance(all_probs, np.ndarray):
        all_probs = all_probs.tolist()

    all_vols = regime.all_expected_vols
    if isinstance(all_vols, np.ndarray):
        all_vols = all_vols.tolist()

    return RegimeResponse(
        symbol=regime.symbol,
        regime_index=regime.regime_index,
        regime_label=regime.regime_label,
        state_probability=regime.state_probability,
        position_multiplier=regime.position_multiplier,
        all_probabilities=all_probs,
        expected_volatility=regime.expected_volatility,
        all_expected_vols=all_vols,
    )


def _signal_to_response(signal) -> SignalResponse:
    """Convert a SignalResult dataclass to a Pydantic SignalResponse."""
    regime_resp = _regime_to_response(signal.regime)
    return SignalResponse(
        symbol=signal.symbol,
        should_trade=signal.should_trade,
        direction=signal.direction,
        combined_score=signal.combined_score,
        regime=regime_resp,
        lstm_prediction=signal.lstm_prediction,
        confidence=signal.confidence,
        bar_timestamp=signal.bar_timestamp,
        uncertainty_mode=signal.uncertainty_mode,
        size_discount=signal.size_discount,
        reasoning=list(signal.reasoning),
    )


@router.get("/state", response_model=LiveStateResponse)
async def get_live_state(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """Top-level dashboard state: account, breaker, signals, positions."""
    ls = _get_live_state(request)

    # Touch the dashboard lock idle timer
    ls.dashboard_lock.touch()

    # Account
    account = None
    try:
        snap = ls.account_monitor.get_info()
        account = AccountResponse(
            balance=snap.balance,
            equity=snap.equity,
            margin=snap.margin,
            free_margin=snap.free_margin,
            margin_level=snap.margin_level,
            floating_pnl=snap.floating_pnl,
            open_positions=snap.open_positions,
        )
    except Exception:
        pass

    # Breaker — read cached snapshot from last RiskMonitor cycle (30s)
    # so the UI shows real DD% instead of hardcoded zeros.
    breaker = None
    try:
        cb = ls.circuit_breaker
        last = cb.get_last_snapshot() if hasattr(cb, "get_last_snapshot") else None
        if last is not None:
            breaker = BreakerResponse(
                multiplier=last.multiplier,
                requires_flat=last.requires_flat,
                active_breakers=list(last.active_breakers),
                daily_dd_pct=float(last.daily_dd_pct),
                weekly_dd_pct=float(last.weekly_dd_pct),
                peak_dd_pct=float(last.peak_dd_pct),
                reason=last.reason,
                consecutive_losses=cb.consecutive_losses(),
                consecutive_loss_limit=cb.consecutive_loss_limit,
            )
        else:
            # RiskMonitor hasn't run a cycle yet — fall back to zeros
            breaker = BreakerResponse(
                multiplier=cb.current_size_multiplier(),
                requires_flat=cb.requires_flat(),
                active_breakers=cb.active_breakers(),
                daily_dd_pct=0.0,
                weekly_dd_pct=0.0,
                peak_dd_pct=0.0,
                reason="warming up" if not cb.active_breakers() else "active",
                consecutive_losses=cb.consecutive_losses(),
                consecutive_loss_limit=cb.consecutive_loss_limit,
            )
    except Exception:
        pass

    # Peak equity
    peak_equity = 0.0
    try:
        peak_equity = ls.risk_monitor.get_peak_equity()
    except Exception:
        pass

    # Per-symbol signals — use the per-symbol cache so the dashboard
    # shows all 4 regime cards instead of just whichever symbol the
    # trading loop touched most recently.
    signals: dict = {}
    try:
        per_sym = getattr(ls.combiner, "last_signal_by_symbol", None) or {}
        for sym, sig in per_sym.items():
            signals[sym] = _signal_to_response(sig)
        # Backwards-compat fallback: if the cache is empty (e.g. older
        # combiner without the dict), fall back to last_signal so we
        # at least show one card.
        if not signals and ls.combiner.last_signal is not None:
            signals[ls.combiner.last_signal.symbol] = _signal_to_response(
                ls.combiner.last_signal
            )
    except Exception:
        pass

    return LiveStateResponse(
        account=account,
        breaker=breaker,
        peak_equity=peak_equity,
        bot_status=ls.bot_control.status.value,
        positions_count=len(ls.tracked_positions),
        signals=signals,
    )


@router.get("/signals/{symbol}", response_model=SignalResponse)
async def get_signal(
    symbol: str,
    request: Request,
    _user: str = Depends(get_current_user),
):
    """Latest signal for a specific symbol."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    # Prefer the per-symbol cache (combiner.last_signal_by_symbol) so the
    # detail page renders correctly regardless of which symbol ran last.
    # Falls back to last_signal only if the per-symbol map doesn't exist.
    per_sym = getattr(ls.combiner, "last_signal_by_symbol", None) or {}
    sig = per_sym.get(symbol)
    if sig is None:
        # Last-resort: legacy single-slot cache, iff the last-run symbol matches.
        legacy = ls.combiner.last_signal
        if legacy is not None and legacy.symbol == symbol:
            sig = legacy
    if sig is not None:
        return _signal_to_response(sig)

    # Cold start — the combiner hasn't run this symbol yet. Return a stub
    # SignalResponse at 200 with a "warming up" reasoning line rather than
    # 404'ing the dashboard. The frontend Signals detail page then renders
    # an informative card instead of a generic error.
    stub_regime = RegimeResponse(
        symbol=symbol,
        regime_index=2,                 # Neutral
        regime_label="Neutral",
        state_probability=0.0,
        position_multiplier=0.0,
        all_probabilities=[0.0, 0.0, 1.0, 0.0, 0.0],
        expected_volatility=0.0,
        all_expected_vols=None,
    )
    return SignalResponse(
        symbol=symbol,
        should_trade=False,
        direction=None,
        combined_score=0.0,
        regime=stub_regime,
        lstm_prediction=0.0,
        confidence=0.0,
        bar_timestamp=None,
        uncertainty_mode=False,
        size_discount=1.0,
        reasoning=[
            f"No signal for {symbol} has been computed yet.",
            "Bot is warming up — next H4 bar close will produce the first signal.",
        ],
    )


@router.get("/positions", response_model=list[PositionResponse])
async def get_positions(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """All tracked open positions."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    # Fetch current bid/ask for floating P/L computation
    _ticks: dict[str, tuple[float, float]] = {}  # symbol → (bid, ask)
    try:
        import MetaTrader5 as mt5
        for pos in ls.tracked_positions.values():
            if pos.symbol not in _ticks:
                tick = mt5.symbol_info_tick(pos.symbol)
                if tick is not None:
                    _ticks[pos.symbol] = (tick.bid, tick.ask)
    except Exception as exc:
        logger.debug("MT5 tick fetch failed: %s", exc)

    positions = []
    for pos in ls.tracked_positions.values():
        # Defensive getattr — OpenPosition dataclass (exit_manager.py) does
        # not define tier_2_done / max_price / min_price; reconciled
        # positions also lack tp_price. Pydantic from_attributes raises
        # AttributeError on missing fields, so project explicitly.
        tp = getattr(pos, "tp_price", None)
        if tp == 0.0:
            tp = None  # legacy reconciled rows have tp_price=0.0 sentinel

        # Per-position floating P/L via MT5 order_calc_profit
        _tick = _ticks.get(pos.symbol)
        cur_price = None
        if _tick is not None:
            cur_price = _tick[0] if pos.direction == "buy" else _tick[1]  # bid for buy, ask for sell
        fl_pnl = None
        risk_usd = None
        if cur_price is not None:
            try:
                _ot = (mt5.ORDER_TYPE_BUY if pos.direction == "buy"
                       else mt5.ORDER_TYPE_SELL)
                _calc = mt5.order_calc_profit(
                    _ot, pos.symbol, float(pos.volume),
                    float(pos.entry_price), float(cur_price),
                )
                if _calc is not None:
                    fl_pnl = round(float(_calc), 2)
                # 1R in account currency = abs(profit if SL hit). Same MT5
                # call, swap close_price for the initial stop. Used by the
                # dashboard for $ tickmarks under the R-multiple bar.
                # Guard: legacy reconciled positions can have initial_stop=0.0
                # (TierStateStore rows pre-dating initial_stop_R tracking).
                # Calling order_calc_profit with close_price=0 returns a
                # nonsensical magnitude — skip and leave risk_usd=None so
                # the dashboard hides the tickmarks instead of showing junk.
                if float(pos.initial_stop) > 0.0:
                    _risk_calc = mt5.order_calc_profit(
                        _ot, pos.symbol, float(pos.volume),
                        float(pos.entry_price), float(pos.initial_stop),
                    )
                    if _risk_calc is not None:
                        risk_usd = round(abs(float(_risk_calc)), 2)
            except Exception as exc:
                logger.debug("order_calc_profit failed for %s: %s", pos.symbol, exc)

        # Time-exit countdown — mirrors ExitManager._h1_bars_elapsed so the
        # UI matches the actual bar counter the bot uses to decide closes.
        te_bars = getattr(pos, "time_exit_bars", 0) or None
        te_remaining_sec = None
        if te_bars and getattr(pos, "opened_at", None) is not None:
            try:
                _bars_elapsed = ExitManager._h1_bars_elapsed(
                    pos, now=datetime.now(tz=timezone.utc),
                )
                _bars_remaining = max(0, int(te_bars) - int(_bars_elapsed))
                te_remaining_sec = _bars_remaining * 3600
            except Exception as exc:
                logger.debug("time-exit countdown calc failed for %s: %s",
                             pos.symbol, exc)

        positions.append(
            PositionResponse(
                symbol=pos.symbol,
                ticket=pos.ticket,
                direction=pos.direction,
                entry_price=pos.entry_price,
                initial_stop=pos.initial_stop,
                current_stop=pos.current_stop,
                take_profit=tp,
                volume=pos.volume,
                initial_volume=pos.initial_volume,
                atr_trail_mult=pos.atr_trail_mult,
                strategy_name=getattr(pos, "strategy_name", ""),
                tier_1_done=getattr(pos, "tier_1_done", False),
                tier_2_done=getattr(pos, "tier_2_done", False),
                max_price=getattr(pos, "max_price", None),
                min_price=getattr(pos, "min_price", None),
                opened_at=getattr(pos, "opened_at", None),
                current_price=cur_price,
                floating_pnl=fl_pnl,
                risk_dollars=risk_usd,
                time_exit_bars=te_bars,
                time_exit_remaining_sec=te_remaining_sec,
            )
        )
    return positions


@router.get("/breaker", response_model=BreakerResponse)
async def get_breaker(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """Current circuit breaker state."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    cb = ls.circuit_breaker
    last = cb.get_last_snapshot() if hasattr(cb, "get_last_snapshot") else None
    if last is not None:
        return BreakerResponse(
            multiplier=last.multiplier,
            requires_flat=last.requires_flat,
            active_breakers=list(last.active_breakers),
            daily_dd_pct=float(last.daily_dd_pct),
            weekly_dd_pct=float(last.weekly_dd_pct),
            peak_dd_pct=float(last.peak_dd_pct),
            reason=last.reason,
            consecutive_losses=cb.consecutive_losses(),
            consecutive_loss_limit=cb.consecutive_loss_limit,
        )
    return BreakerResponse(
        multiplier=cb.current_size_multiplier(),
        requires_flat=cb.requires_flat(),
        active_breakers=cb.active_breakers(),
        daily_dd_pct=0.0,
        weekly_dd_pct=0.0,
        peak_dd_pct=0.0,
        reason="warming up" if not cb.active_breakers() else "active",
        consecutive_losses=cb.consecutive_losses(),
        consecutive_loss_limit=cb.consecutive_loss_limit,
    )


@router.get("/stream")
async def live_stream(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """SSE stream of live state updates (every 2 seconds)."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    return EventSourceResponse(stream_live_state(ls, interval=2.0))


# Allowed chart timeframes — lock down input so an attacker can't probe
# arbitrary strings against the OHLCV query (SQLAlchemy parameterizes
# anyway, but defense in depth).
_CANDLE_TIMEFRAMES = {"M15", "H1", "H4", "D1", "W1"}
_CANDLE_LIMIT_MAX = 1000


@router.get("/candles/{symbol}", response_model=CandlesResponse)
async def get_candles(
    symbol: str,
    request: Request,
    timeframe: str = "H1",
    limit: int = 300,
    _user: str = Depends(get_current_user),
):
    """Recent OHLCV bars for a symbol, for the frontend live chart.

    Pulls from the `ohlcv_bars` DB cache (populated by the trading loop
    + scripts/backfill_ohlcv.py). Returns bars in chronological order,
    oldest first, up to `limit` bars counting backward from now.
    """
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    # Normalize + validate inputs
    symbol_norm = symbol.upper()
    tf_norm = timeframe.upper()
    if tf_norm not in _CANDLE_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported timeframe '{timeframe}'. "
                   f"Allowed: {sorted(_CANDLE_TIMEFRAMES)}",
        )
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    if limit > _CANDLE_LIMIT_MAX:
        limit = _CANDLE_LIMIT_MAX

    # Push LIMIT into the SQL query. Prior implementation fetched
    # every bar then `.tail(limit)` in Python — for symbols with years
    # of backfilled history (EURUSD/USDCAD) this was 7-9 seconds per
    # request, 90% of dashboard wait time per the P-1 perf capture.
    # The DB does ORDER BY bar_timestamp DESC LIMIT N using the
    # existing (symbol, timeframe, bar_timestamp) index.
    df = await ls.data_store.get_ohlcv_range(
        symbol_norm, tf_norm, limit=limit,
    )
    if df is None or df.empty:
        return CandlesResponse(symbol=symbol_norm, timeframe=tf_norm, bars=[])

    bars: list[OHLCVBarResponse] = []
    for ts, row in df.iterrows():
        try:
            # `ts` is a pandas Timestamp
            time_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        except Exception:
            time_str = str(ts)
        bars.append(OHLCVBarResponse(
            time=time_str,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0) or 0.0),
        ))

    return CandlesResponse(symbol=symbol_norm, timeframe=tf_norm, bars=bars)
