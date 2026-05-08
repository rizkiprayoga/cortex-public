"""
api_smoke.py — Hit every registered GET endpoint, flag non-200.

Called on bot startup and every 15 minutes by the APScheduler job in
main.py. Failures fire the ``api.route_healthy`` invariant (ALERT), so
a route throwing a 500 (like the Frankfurt-ZoneInfo bug from 2026-04-15)
surfaces in Telegram within minutes instead of when an operator notices.

Design notes
------------
- Uses httpx over the same localhost port the dashboard hits. That way
  we exercise the real auth, middleware, and routing stack rather than
  a TestClient in-memory shortcut.
- Authenticates with DASHBOARD_USERNAME / DASHBOARD_PASSWORD from env;
  if neither is set, we fall back to unauthenticated routes only.
- Skips routes whose path contains a path parameter (``{ticket}``) since
  we don't know what value is valid; those routes would need targeted
  tests. Listed in ``SKIP_PATH_PATTERNS`` for explicit opt-out too.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.safety.invariants import Severity, check

DEFAULT_BASE_URL = os.getenv("CORTEX_SMOKE_BASE_URL", "http://127.0.0.1:8787")
DEFAULT_TIMEOUT = 10.0

# SSE/streaming routes: their response body is intentionally unbounded.
# A plain ``client.get()`` hangs because new events keep resetting the
# read-idle timer, so the job stacks past the 15-min scheduler interval
# and fires spurious ``api.route_healthy`` ALERTs. We probe these via
# ``client.stream()`` instead — the context manager yields once headers
# are received, so we validate status_code without touching the body.
# Register new SSE endpoints here when adding them to the API.
SSE_PATHS: frozenset[str] = frozenset({"/api/live/stream"})
SSE_TIMEOUT = 5.0

# Paths we intentionally do not hit (destructive, need POST, or param-keyed).
SKIP_PATH_PATTERNS: tuple[str, ...] = (
    "{",          # path parameter — needs a valid value
    "/auth/",     # login/logout — exercised implicitly below
    "/system/restart",
    "/docs",      # FastAPI interactive docs
    "/openapi",
    "/redoc",
)


@dataclass
class SmokeResult:
    path: str
    status: int
    ok: bool
    error: Optional[str] = None


@dataclass
class LoginOutcome:
    token: Optional[str]
    configured: bool       # True iff creds were set in env
    login_status: int      # HTTP status from /api/auth/login, 0 on network error
    error: Optional[str] = None


async def _login(client: httpx.AsyncClient) -> LoginOutcome:
    user = os.getenv("DASHBOARD_USERNAME")
    pw = os.getenv("DASHBOARD_PASSWORD")
    if not user or not pw:
        return LoginOutcome(token=None, configured=False, login_status=0)
    try:
        r = await client.post(
            "/api/auth/login",
            json={"username": user, "password": pw},
            timeout=DEFAULT_TIMEOUT,
        )
        if r.status_code != 200:
            return LoginOutcome(
                token=None, configured=True, login_status=r.status_code,
                error=(r.text[:200] if r.text else None),
            )
        return LoginOutcome(
            token=r.json().get("access_token"),
            configured=True, login_status=200,
        )
    except Exception as exc:
        return LoginOutcome(
            token=None, configured=True, login_status=0, error=str(exc)[:200],
        )


def _enumerate_get_paths(app) -> list[str]:
    """Pull every registered GET path from a FastAPI app."""
    paths: list[str] = []
    for route in getattr(app, "routes", []):
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", None)
        if not path or "GET" not in methods:
            continue
        if any(skip in path for skip in SKIP_PATH_PATTERNS):
            continue
        paths.append(path)
    return sorted(set(paths))


async def run_smoke(
    app=None,
    base_url: str = DEFAULT_BASE_URL,
    paths: Optional[Iterable[str]] = None,
) -> list[SmokeResult]:
    """
    Hit every GET route. Returns per-path results. Fires an invariant for
    each non-200 outcome.
    """
    if paths is None and app is not None:
        paths = _enumerate_get_paths(app)
    elif paths is None:
        # No app + no explicit list → caller likely wants the known public routes.
        paths = ["/api/live/state", "/api/news/blackouts", "/api/news/events",
                 "/api/invariants/recent"]

    async with httpx.AsyncClient(base_url=base_url, timeout=DEFAULT_TIMEOUT) as client:
        outcome = await _login(client)
        # Emit a dedicated invariant for login state so an auth breakage
        # surfaces as one ALERT instead of N route-401 ALERTs. A 401/403
        # response (wrong creds, dashboard locked, rate-limited) means the
        # route matched and its auth layer is running — healthy, not a
        # breakage. Only 5xx and connection errors (status=0) flag broken
        # auth that an operator needs paged about.
        auth_ok = (
            (not outcome.configured)
            or outcome.token is not None
            or outcome.login_status in (401, 403)
        )
        check(
            "auth.smoke_login_ok",
            auth_ok,
            severity=Severity.ALERT,
            dedup_key="auth.smoke_login_ok",
            context={
                "configured": outcome.configured,
                "login_status": outcome.login_status,
                "error": outcome.error,
            },
            message=(
                f"dashboard login failed (status={outcome.login_status})"
                if not auth_ok
                else "ok"
            ),
        )
        headers = (
            {"Authorization": f"Bearer {outcome.token}"}
            if outcome.token else {}
        )

        # Run requests concurrently (bounded) so a few slow routes can't
        # stretch the whole smoke past the 15-min scheduler interval.
        sem = asyncio.Semaphore(5)

        async def _probe(path: str) -> SmokeResult:
            async with sem:
                try:
                    if path in SSE_PATHS:
                        # SSE body never ends — validate that headers arrive
                        # cleanly and abort. ``client.stream()`` yields the
                        # response as soon as headers are received; exiting
                        # the context manager closes the connection without
                        # reading body.
                        async with client.stream(
                            "GET", path, headers=headers, timeout=SSE_TIMEOUT,
                        ) as r:
                            ok = (r.status_code < 400) or r.status_code in (401, 403)
                            return SmokeResult(
                                path=path, status=r.status_code, ok=ok,
                                error=None if ok else f"SSE headers={r.status_code}",
                            )
                    r = await client.get(path, headers=headers)
                    # 401/403 means the route matched and its auth layer
                    # is working as designed (e.g. dashboard-locked flag).
                    # That is a HEALTHY route — not a broken endpoint.
                    # Only 4xx-non-auth and 5xx indicate real breakage.
                    ok = (r.status_code < 400) or r.status_code in (401, 403)
                    return SmokeResult(
                        path=path, status=r.status_code, ok=ok,
                        error=None if ok else (r.text[:200] if r.text else None),
                    )
                except Exception as exc:
                    return SmokeResult(path=path, status=0, ok=False, error=str(exc)[:200])

        results = await asyncio.gather(*(_probe(p) for p in paths))

        for res in results:
            check(
                "api.route_healthy",
                res.ok,
                severity=Severity.ALERT,
                dedup_key=f"api.route_healthy:{res.path}",
                context={"path": res.path, "status": res.status, "error": res.error},
                message=f"{res.path} returned {res.status}",
            )

    return list(results)


async def run_smoke_from_app(app) -> list[SmokeResult]:
    """Convenience wrapper for the APScheduler job in main.py."""
    return await run_smoke(app=app)


def cli() -> None:
    """CLI entry so the check can be run standalone from a shell."""
    import argparse
    ap = argparse.ArgumentParser(description="Ping every dashboard GET route.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = ap.parse_args()
    results = asyncio.run(run_smoke(base_url=args.base_url))
    failures = [r for r in results if not r.ok]
    for r in results:
        sym = "OK " if r.ok else "ERR"
        print(f"  {sym} {r.status:>4} {r.path}")
    print(f"\n{len(results)} routes, {len(failures)} failing.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
