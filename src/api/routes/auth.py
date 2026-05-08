"""
routes/auth.py — Authentication endpoints.

POST /api/auth/login  → username + password → JWT
GET  /api/auth/me     → current user (requires valid token)
"""

import logging
import os
import time
import threading
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import create_access_token, get_current_user, verify_password
from src.api.schemas import LoginRequest, TokenResponse, UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# Login rate limiting (in-memory, single-process)
# ---------------------------------------------------------------------------
_rate_lock = threading.Lock()
_attempts: dict[str, list[float]] = defaultdict(list)  # IP → timestamps
_consecutive_failures: dict[str, int] = defaultdict(int)  # IP → count
_blocked_until: dict[str, float] = {}  # IP → unblock timestamp

MAX_ATTEMPTS_PER_MINUTE = 5
MAX_CONSECUTIVE_FAILURES = 10
BLOCK_DURATION_SECONDS = 900  # 15 minutes


def _check_rate_limit(client_ip: str) -> None:
    """Raise 429 if the IP has exceeded the rate limit."""
    now = time.time()
    with _rate_lock:
        # Check block
        if client_ip in _blocked_until:
            if now < _blocked_until[client_ip]:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many failed attempts. Try again later.",
                )
            else:
                del _blocked_until[client_ip]
                _consecutive_failures[client_ip] = 0

        # Sliding window: remove attempts older than 60s
        recent = [t for t in _attempts[client_ip] if now - t < 60]
        _attempts[client_ip] = recent

        if len(recent) >= MAX_ATTEMPTS_PER_MINUTE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Wait 60 seconds.",
            )


def _record_attempt(client_ip: str, success: bool) -> None:
    """Record a login attempt for rate limiting."""
    now = time.time()
    with _rate_lock:
        _attempts[client_ip].append(now)
        if success:
            _consecutive_failures[client_ip] = 0
        else:
            _consecutive_failures[client_ip] += 1
            if _consecutive_failures[client_ip] >= MAX_CONSECUTIVE_FAILURES:
                _blocked_until[client_ip] = now + BLOCK_DURATION_SECONDS
                logger.warning(
                    "IP %s blocked for %ds after %d consecutive failures",
                    client_ip,
                    BLOCK_DURATION_SECONDS,
                    MAX_CONSECUTIVE_FAILURES,
                )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request):
    """Authenticate and return a JWT access token."""
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    # Audit H5: validate username against configured DASHBOARD_USERNAME
    # (defaults to "admin"). Prevents untrusted user input in audit logs.
    expected_user = os.environ.get("DASHBOARD_USERNAME", "admin")
    if body.username != expected_user or not verify_password(body.password):
        _record_attempt(client_ip, success=False)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    _record_attempt(client_ip, success=True)
    token = create_access_token(subject=body.username)
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
async def me(username: str = Depends(get_current_user)):
    """Return the current authenticated user."""
    return UserResponse(username=username)
