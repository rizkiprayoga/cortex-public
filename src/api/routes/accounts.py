"""
accounts.py — Account management API routes.

Provides endpoints to view the current MT5 account and switch between
demo/live accounts. Data is segmented by mt5_account in the database
so demo data doesn't contaminate live dashboard views.

Security Audit fixes applied:
- H1: Credentials NEVER transmitted over API — account slots resolve
      to env vars (MT5_SLOT_<NAME>_LOGIN/PASSWORD/SERVER)
- C5: tracked_positions mutations protected by positions_lock
- C6: Account switch requires bot to be PAUSED (prevents mid-order
      MT5 shutdown race)
- C7: combiner.reset_state() scheduled via call_soon_threadsafe
- H8: current_account_id writes protected by account_lock
"""

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import get_current_user
from src.api.live_state import BotStatus
from src.api.schemas import (
    AccountInfoResponse,
    AccountRegisterRequest,
    AccountSlotInfo,
    AccountSlotsResponse,
    AccountSwitchRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/accounts", tags=["accounts"])


def _get_live_state(request: Request):
    return request.app.state.live_state


def _resolve_slot_credentials(slot: str) -> Optional[dict]:
    """
    Resolve an account slot name to credentials from environment vars.

    Example: slot="demo" → MT5_SLOT_DEMO_LOGIN, MT5_SLOT_DEMO_PASSWORD,
                           MT5_SLOT_DEMO_SERVER
    Special: slot="default" → MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    """
    slot_upper = slot.upper().strip()
    if not slot_upper or not slot_upper.replace("_", "").isalnum():
        return None

    # "default" maps to the base MT5_LOGIN/PASSWORD/SERVER env vars
    if slot_upper == "DEFAULT":
        login = os.environ.get("MT5_LOGIN", "").strip()
        password = os.environ.get("MT5_PASSWORD", "").strip()
        server = os.environ.get("MT5_SERVER", "").strip()
    else:
        login = os.environ.get(f"MT5_SLOT_{slot_upper}_LOGIN", "").strip()
        password = os.environ.get(f"MT5_SLOT_{slot_upper}_PASSWORD", "").strip()
        server = os.environ.get(f"MT5_SLOT_{slot_upper}_SERVER", "").strip()

    if not (login and password and server):
        return None
    try:
        return {"login": int(login), "password": password, "server": server}
    except ValueError:
        return None


_SLOT_RE = re.compile(r"^[A-Za-z0-9_]{1,30}$")
_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


def _scan_slots(current_account_id: Optional[int]) -> list[AccountSlotInfo]:
    """Scan os.environ for all MT5_SLOT_*_LOGIN entries + default account."""
    slots: list[AccountSlotInfo] = []

    # Include the default MT5_LOGIN account as "default" slot
    default_login = os.environ.get("MT5_LOGIN", "").strip()
    default_server = os.environ.get("MT5_SERVER", "").strip()
    if default_login and default_server:
        try:
            login_val = int(default_login)
            slots.append(AccountSlotInfo(
                slot="default",
                login=login_val,
                server=default_server,
                is_current=(current_account_id == login_val),
            ))
        except ValueError:
            pass

    # Scan for MT5_SLOT_*_LOGIN entries
    for key in os.environ:
        if not (key.startswith("MT5_SLOT_") and key.endswith("_LOGIN")):
            continue
        slot_name = key[9:-6].lower()  # MT5_SLOT_DEMO_LOGIN → "demo"
        login_str = os.environ[key].strip()
        if not login_str:
            continue
        try:
            login_val = int(login_str)
        except ValueError:
            continue
        server = os.environ.get(
            f"MT5_SLOT_{slot_name.upper()}_SERVER", ""
        ).strip()
        if not server:
            continue
        slots.append(AccountSlotInfo(
            slot=slot_name,
            login=login_val,
            server=server,
            is_current=(current_account_id == login_val),
        ))
    return slots


def _append_slot_to_env(
    slot_name: str, login: int, password: str, server: str,
) -> None:
    """Append MT5_SLOT_<NAME>_* lines to the .env file."""
    upper = slot_name.upper()
    block = (
        f"\n# --- MT5 Account Slot: {slot_name} ---\n"
        f"MT5_SLOT_{upper}_LOGIN={login}\n"
        f"MT5_SLOT_{upper}_PASSWORD={password}\n"
        f"MT5_SLOT_{upper}_SERVER={server}\n"
    )
    with open(_ENV_PATH, "a", encoding="utf-8") as f:
        f.write(block)


@router.get("/slots", response_model=AccountSlotsResponse)
async def list_account_slots(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """List all configured MT5 account slots."""
    ls = _get_live_state(request)
    current_id = ls.get_account_id()
    slots = _scan_slots(current_id)
    current_slot = next(
        (s.slot for s in slots if s.is_current), None,
    )
    return AccountSlotsResponse(slots=slots, current_slot=current_slot)


@router.post("/register", response_model=AccountInfoResponse)
async def register_account(
    request: Request,
    body: AccountRegisterRequest,
    _user: str = Depends(get_current_user),
):
    """
    Register a new MT5 account slot.

    Credentials are sent once over HTTP (localhost only) and persisted
    to .env as MT5_SLOT_<NAME>_* environment variables.
    """
    ls = _get_live_state(request)
    slot = body.slot_name.strip()

    # Validate slot name
    if not _SLOT_RE.match(slot):
        raise HTTPException(
            status_code=400,
            detail="Slot name must be 1-30 alphanumeric/underscore characters.",
        )

    # Check slot doesn't already exist
    upper = slot.upper()
    if os.environ.get(f"MT5_SLOT_{upper}_LOGIN"):
        raise HTTPException(
            status_code=409,
            detail=f"Account slot '{slot}' already exists.",
        )

    # Persist to .env and load into current process
    with ls.account_lock:
        _append_slot_to_env(slot, body.login, body.password, body.server)
    os.environ[f"MT5_SLOT_{upper}_LOGIN"] = str(body.login)
    os.environ[f"MT5_SLOT_{upper}_PASSWORD"] = body.password
    os.environ[f"MT5_SLOT_{upper}_SERVER"] = body.server

    logger.info("Registered new account slot '%s' (login %d on %s)",
                slot, body.login, body.server)

    # Auto-switch if requested
    if body.auto_switch:
        switch_body = AccountSwitchRequest(slot=slot)
        return await switch_account(request, switch_body, _user)

    return AccountInfoResponse(
        account_id=body.login,
        server=body.server,
    )


@router.get("/current", response_model=AccountInfoResponse)
async def get_current_account(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """Return info about the currently connected MT5 account."""
    ls = _get_live_state(request)
    info = mt5.account_info()
    if info is None:
        with ls.account_lock:
            return AccountInfoResponse(
                account_id=ls.current_account_id,
                is_demo=False,
            )
    return AccountInfoResponse(
        account_id=int(info.login),
        server=info.server,
        is_demo=bool(info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO),
        balance=float(info.balance),
        equity=float(info.equity),
    )


@router.post("/switch", response_model=AccountInfoResponse)
async def switch_account(
    request: Request,
    body: AccountSwitchRequest,
    _user: str = Depends(get_current_user),
):
    """
    Switch to a different MT5 account by slot name.

    Credentials are loaded from environment (MT5_SLOT_<NAME>_*).
    Requires the bot to be PAUSED or STOPPED to prevent mid-order
    MT5 shutdown races (Audit C6).
    """
    ls = _get_live_state(request)

    # C6: require bot to be paused/stopped before switching MT5 account
    bot_status = ls.bot_control.status if ls.bot_control else None
    if bot_status == BotStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Pause or stop the bot before switching accounts. "
                   "Active trading loop may have in-flight MT5 calls.",
        )

    # H1: resolve slot to env-var credentials (never accept password over HTTP)
    creds = _resolve_slot_credentials(body.slot)
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown or misconfigured account slot '{body.slot}'. "
                   f"Set MT5_SLOT_<NAME>_LOGIN/PASSWORD/SERVER in .env",
        )

    # Safety: block if positions are open
    with ls.positions_lock:
        open_count = len(ls.tracked_positions) if ls.tracked_positions else 0
    if open_count > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot switch account with {open_count} open positions. "
                   f"Close them first.",
        )

    # Find the connector
    connector = None
    if hasattr(ls, "order_manager") and ls.order_manager is not None:
        connector = getattr(ls.order_manager, "connector", None)
    if connector is None and hasattr(ls, "account_monitor") and ls.account_monitor is not None:
        connector = getattr(ls.account_monitor, "connector", None)

    if connector is None:
        raise HTTPException(
            status_code=500,
            detail="No MT5 connector available for account switching",
        )

    try:
        connector.connect_with_creds(
            creds["login"], creds["password"], creds["server"],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Clear trading state for the new account (C5: lock-protected)
    with ls.positions_lock:
        if ls.tracked_positions is not None:
            ls.tracked_positions.clear()

    # C7: schedule combiner reset on the main event loop — avoid
    # cross-thread mutation of combiner state
    if ls.combiner is not None:
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(ls.combiner.reset_state)
        except RuntimeError:
            # No running loop — direct call is safe (we're alone)
            ls.combiner.reset_state()

    # H8: update current_account_id under lock
    info = mt5.account_info()
    with ls.account_lock:
        if info is not None:
            ls.current_account_id = int(info.login)
        else:
            ls.current_account_id = creds["login"]

    logger.info("Account switched to slot '%s' (login %d on %s)",
                 body.slot, creds["login"], creds["server"])

    return AccountInfoResponse(
        account_id=ls.current_account_id,
        server=creds["server"],
        is_demo=bool(info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO) if info else False,
        balance=float(info.balance) if info else 0.0,
        equity=float(info.equity) if info else 0.0,
    )
