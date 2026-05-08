"""
routes/config.py — Live config hot-reload endpoints.

GET   /api/config/risk  → current risk parameters
POST  /api/config/risk  → update risk parameters (hot-reload)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import get_current_user
from src.api.schemas import RiskConfigResponse, RiskConfigUpdateRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])

# Hard-halt knobs require explicit confirmation string
HARD_HALT_FIELDS = {
    "max_daily_loss_hard_pct",
    "max_weekly_loss_hard_pct",
    "max_peak_drawdown_pct",
}
CONFIRM_TOKEN = "CONFIRM_HARD_HALT_CHANGE"


def _get_live_state(request: Request):
    return request.app.state.live_state


@router.get("/risk", response_model=RiskConfigResponse)
async def get_risk_config(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """Read current risk configuration from live objects."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    cb = ls.circuit_breaker
    pm = ls.portfolio

    return RiskConfigResponse(
        max_daily_loss_soft_pct=cb.max_daily_loss_soft_pct,
        max_daily_loss_hard_pct=cb.max_daily_loss_hard_pct,
        max_weekly_loss_soft_pct=cb.max_weekly_loss_soft_pct,
        max_weekly_loss_hard_pct=cb.max_weekly_loss_hard_pct,
        max_peak_drawdown_pct=cb.max_peak_drawdown_pct,
        max_position_size_pct=pm.max_used_margin_pct_total,
        max_total_exposure_pct=pm.max_used_margin_pct_total,
        free_margin_reserve_pct=pm.free_margin_reserve_pct,
        max_concurrent_per_symbol=pm.max_concurrent_per_symbol,
        max_concurrent_total=pm.max_concurrent_total,
        max_daily_trades=pm.max_daily_trades,
    )


@router.post("/risk", response_model=RiskConfigResponse)
async def update_risk_config(
    body: RiskConfigUpdateRequest,
    request: Request,
    _user: str = Depends(get_current_user),
):
    """
    Update risk parameters on live objects and persist to settings.yaml.

    Hard-halt knobs (daily_hard, weekly_hard, peak) require
    ``confirmation: "CONFIRM_HARD_HALT_CHANGE"`` in the request body.
    """
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    updates = body.model_dump(exclude_none=True, exclude={"confirmation"})
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    # Check confirmation for hard-halt fields
    touching_hard = set(updates.keys()) & HARD_HALT_FIELDS
    if touching_hard and body.confirmation != CONFIRM_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Changing {', '.join(sorted(touching_hard))} requires "
                f'confirmation: "{CONFIRM_TOKEN}"'
            ),
        )

    cb = ls.circuit_breaker
    pm = ls.portfolio

    # Apply to live objects via their thread-safe setters
    setter_map = {
        "max_daily_loss_soft_pct": cb.set_daily_soft,
        "max_daily_loss_hard_pct": cb.set_daily_hard,
        "max_weekly_loss_soft_pct": cb.set_weekly_soft,
        "max_weekly_loss_hard_pct": cb.set_weekly_hard,
        "max_peak_drawdown_pct": cb.set_peak,
        "max_daily_trades": pm.set_max_daily_trades,
        "max_concurrent_per_symbol": pm.set_max_concurrent_per_symbol,
        "max_concurrent_total": pm.set_max_concurrent_total,
        "max_total_exposure_pct": pm.set_max_used_margin_pct_total,
        "max_position_size_pct": pm.set_max_used_margin_pct_total,
        "free_margin_reserve_pct": pm.set_free_margin_reserve_pct,
    }

    errors = []
    for field_name, value in updates.items():
        setter = setter_map.get(field_name)
        if setter is None:
            continue
        try:
            setter(value)
        except ValueError as exc:
            errors.append(f"{field_name}: {exc}")

    if errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(errors),
        )

    # Persist to settings.yaml via ConfigStore
    config_store = getattr(ls, "config_store", None)
    if config_store is not None:
        risk = config_store.read_risk_section()
        risk.update(updates)
        config_store.write_risk_section(risk)

    # Return updated state
    return RiskConfigResponse(
        max_daily_loss_soft_pct=cb.max_daily_loss_soft_pct,
        max_daily_loss_hard_pct=cb.max_daily_loss_hard_pct,
        max_weekly_loss_soft_pct=cb.max_weekly_loss_soft_pct,
        max_weekly_loss_hard_pct=cb.max_weekly_loss_hard_pct,
        max_peak_drawdown_pct=cb.max_peak_drawdown_pct,
        max_position_size_pct=pm.max_used_margin_pct_total,
        max_total_exposure_pct=pm.max_used_margin_pct_total,
        free_margin_reserve_pct=pm.free_margin_reserve_pct,
        max_concurrent_per_symbol=pm.max_concurrent_per_symbol,
        max_concurrent_total=pm.max_concurrent_total,
        max_daily_trades=pm.max_daily_trades,
    )
