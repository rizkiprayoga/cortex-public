"""
auth.py — Single-user JWT authentication for the dashboard.

Password hash stored in env var ``DASHBOARD_PW_HASH`` (bcrypt).
JWT signing secret stored in ``DASHBOARD_JWT_SECRET`` (HMAC-SHA256).
Token TTL: 12 hours.

Usage
-----
    # In route handlers:
    user = Depends(get_current_user)   # raises 401 if invalid/expired
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt as _bcrypt_lib
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

logger = logging.getLogger(__name__)

# Lazy import for jose — only needed at runtime.
_jwt_mod = None


def _get_jwt():
    global _jwt_mod
    if _jwt_mod is None:
        import jose.jwt  # type: ignore[import-untyped]
        _jwt_mod = jose.jwt
    return _jwt_mod


# JWT settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 12

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password with bcrypt. Useful for initial setup."""
    return _bcrypt_lib.hashpw(
        plain_password.encode("utf-8"), _bcrypt_lib.gensalt()
    ).decode("utf-8")


def _get_jwt_secret() -> str:
    secret = os.environ.get("DASHBOARD_JWT_SECRET", "")
    if not secret:
        raise RuntimeError(
            "DASHBOARD_JWT_SECRET environment variable is not set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return secret


def _get_pw_hash() -> str:
    pw_hash = os.environ.get("DASHBOARD_PW_HASH", "")
    if not pw_hash:
        raise RuntimeError(
            "DASHBOARD_PW_HASH environment variable is not set. "
            "Generate one with: python -c \"from src.api.auth import hash_password; "
            "print(hash_password('your-password'))\""
        )
    return pw_hash


def verify_password(plain_password: str) -> bool:
    """Check a plaintext password against the stored bcrypt hash."""
    try:
        pw_hash = _get_pw_hash()
        return _bcrypt_lib.checkpw(
            plain_password.encode("utf-8"), pw_hash.encode("utf-8")
        )
    except Exception:
        return False


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT with the given subject and expiry."""
    expire = datetime.now(tz=timezone.utc) + (
        expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    )
    payload = {"sub": subject, "exp": expire}
    return _get_jwt().encode(payload, _get_jwt_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and verify a JWT. Raises on invalid/expired."""
    jwt = _get_jwt()
    try:
        return jwt.decode(token, _get_jwt_secret(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    FastAPI dependency: extract and validate the JWT from the
    Authorization header. Returns the username (``sub`` claim).
    """
    payload = decode_access_token(token)
    username: Optional[str] = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    return username
