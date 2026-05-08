"""
routes/invariants.py — Invariant findings feed for the dashboard Health card.

GET /api/invariants/recent → last N invariant findings (violations only).

Drives the Health card on the Overview screen. Silent days = card shows
"All checks passing". Any WARN/ALERT/CRITICAL shows up here so the
operator can see problems without Telegram spam.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from src.api.auth import get_current_user
from src.safety.invariants import Severity, get_registry

router = APIRouter(prefix="/api/invariants", tags=["invariants"])


class InvariantFindingDTO(BaseModel):
    ts: str
    invariant: str
    severity: str
    passed: bool
    message: str
    symbol: Optional[str] = None
    context: dict = {}


class InvariantFeedResponse(BaseModel):
    findings: list[InvariantFindingDTO]
    count: int


@router.get("/recent", response_model=InvariantFeedResponse)
def get_recent(
    limit: int = Query(50, ge=1, le=500),
    severity: Optional[str] = Query(None, description="WARN | ALERT | CRITICAL"),
    _user: str = Depends(get_current_user),
) -> InvariantFeedResponse:
    sev = None
    if severity:
        try:
            sev = Severity(severity.upper())
        except ValueError:
            sev = None
    findings = get_registry().recent(limit=limit, severity=sev)
    return InvariantFeedResponse(
        findings=[InvariantFindingDTO(**f.__dict__) for f in findings],
        count=len(findings),
    )
