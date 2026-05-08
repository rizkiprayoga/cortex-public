"""
app.py — FastAPI application factory.

``build_app(live_state)`` creates the FastAPI instance with all routers
registered, auth middleware, CORS, request timeout, and the dashboard
lock gate. The app is served by uvicorn as an asyncio task alongside
the trading loop in main.py.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.live_state import LiveState
from src.api.routes import accounts as account_routes
from src.api.routes import auth as auth_routes
from src.api.routes import backtest as backtest_routes
from src.api.routes import config as config_routes
from src.api.routes import history as history_routes
from src.api.routes import invariants as invariants_routes
from src.api.routes import live as live_routes
from src.api.routes import models as models_routes
from src.api.routes import news as news_routes
from src.api.routes import system as system_routes

logger = logging.getLogger(__name__)

# Paths that are always accessible regardless of lock state
LOCK_EXEMPT_PATHS = frozenset({
    "/api/system/health",
    "/api/system/lock-status",
    "/api/system/unlock",
})

# Request timeout (seconds) — protects the trading loop from slow handlers
REQUEST_TIMEOUT = 3.0

# ── API perf logger (P-1 investigation) ───────────────────────────────
# One JSONL line per HTTP request: timestamp, method, path, query, status,
# duration_ms, response bytes. Off by default; turn on with
# CORTEX_API_PERF_LOG=1 in .env before restart. Analysis via
# scripts/analyze_api_perf.py. Safe to leave on longer-term — writes are
# fire-and-forget and ~50 bytes per request — but it's scoped for
# investigation so a future cleanup may gate or remove it.
_API_PERF_LOG_PATH = Path("data/logs/api_perf.jsonl")
_API_PERF_ROTATE_BYTES = 50 * 1024 * 1024  # 50 MB, same policy as invariants
_API_PERF_SKIP_PATHS: tuple[str, ...] = (
    "/ui/assets/",
    "/static/",
    "/favicon.ico",
    "/api/live/stream",  # SSE: permanent-open; duration would be misleading
)


def _api_perf_enabled() -> bool:
    """Read the env flag at request time (not import time) so flipping
    the variable takes effect on the next HTTP request even within a
    single Python process — matters during dev, irrelevant for prod."""
    return os.environ.get("CORTEX_API_PERF_LOG", "0") == "1"


def _api_perf_write(record: dict) -> None:
    """Append one JSONL line; size-rotate at the threshold. Never raises."""
    try:
        if (
            _API_PERF_LOG_PATH.exists()
            and _API_PERF_LOG_PATH.stat().st_size >= _API_PERF_ROTATE_BYTES
        ):
            archive = _API_PERF_LOG_PATH.with_suffix(".jsonl.1")
            if archive.exists():
                archive.unlink()
            _API_PERF_LOG_PATH.rename(archive)
    except OSError:
        pass
    try:
        _API_PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _API_PERF_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
    except OSError:
        pass


def build_app(live_state: LiveState) -> FastAPI:
    """
    Construct the FastAPI application.

    Args:
        live_state: Shared state container with refs to all trading objects.

    Returns:
        Configured FastAPI instance ready for uvicorn.
    """
    # Disable OpenAPI docs in production — prevents API enumeration by
    # unauthenticated attackers (Security Audit C3).
    # Enable only when CORTEX_ENABLE_API_DOCS=1 is set.
    _docs_enabled = os.environ.get("CORTEX_ENABLE_API_DOCS", "0") == "1"
    app = FastAPI(
        title="Cortex Trading Bot Dashboard",
        version="0.12.0",
        docs_url="/api/docs" if _docs_enabled else None,
        redoc_url="/api/redoc" if _docs_enabled else None,
        openapi_url="/api/openapi.json" if _docs_enabled else None,
    )

    # Store live_state on app.state so route handlers can access it
    app.state.live_state = live_state

    # ── CORS ──────────────────────────────────────────────────────────
    # Security Audit C2: Wildcard + allow_credentials is insecure.
    # Origins must come from CORS_ALLOWED_ORIGINS env var (comma-separated).
    # Defaults to localhost origins for dev.
    _cors_origins_env = os.environ.get(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8787,http://127.0.0.1:8787",
    )
    _cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    )

    # ── GZip compression (P-1 stage 2a) ───────────────────────────────
    # Compress responses above 1KB. On the dashboard's JSON-heavy
    # payloads (candles in particular — 60KB/symbol pre-gzip) this
    # typically shrinks network bytes 5-10×. Starlette's GZipMiddleware
    # is streaming-aware — it flushes SSE progressively rather than
    # buffering, so /api/live/stream keeps working. Browsers
    # auto-decompress via the Accept-Encoding: gzip header they send
    # by default, so no frontend change is required.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # ── API perf logger ────────────────────────────────────────────────
    # First in the decorator chain → outermost wrapper around all other
    # app.middleware handlers, so duration_ms includes time spent in the
    # security headers / dashboard lock / no-cache / timeout middlewares
    # (anything that actually affects user-perceived latency).
    @app.middleware("http")
    async def api_perf_logger(request: Request, call_next):
        if not _api_perf_enabled():
            return await call_next(request)
        path = request.url.path
        if any(path.startswith(p) for p in _API_PERF_SKIP_PATHS):
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        resp_bytes = None
        try:
            cl = response.headers.get("content-length")
            if cl is not None:
                resp_bytes = int(cl)
        except (TypeError, ValueError):
            pass
        # Fire-and-forget: any failure in the perf logger must never
        # propagate out and fail the user's request. The inner writer
        # already swallows OSError, but wrap broadly so a bug in the
        # logger (JSON encoder, attribute access, whatever) can't
        # take down /api/system/health.
        try:
            _api_perf_write({
                "ts":          datetime.now(tz=timezone.utc).isoformat(),
                "method":      request.method,
                "path":        path,
                "query":       request.url.query or None,
                "status":      response.status_code,
                "duration_ms": duration_ms,
                "resp_bytes":  resp_bytes,
            })
        except Exception:
            pass
        return response

    # ── Security headers middleware (Audit H6) ────────────────────────
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            # @fontsource packages inline fonts as data: URIs in the bundled CSS
            # (Inter + JetBrains Mono). Allow those without widening default-src.
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
        return response

    # ── Dashboard lock middleware ─────────────────────────────────────
    @app.middleware("http")
    async def dashboard_lock_gate(request: Request, call_next):
        """
        When the dashboard is locked, reject all API requests except
        health and lock-status. Returns 403 with minimal info.
        """
        path = request.url.path

        # Always allow exempt paths and SPA static files
        if path in LOCK_EXEMPT_PATHS or path.startswith("/ui"):
            return await call_next(request)

        # Check lock — only applies to /api/* routes
        if path.startswith("/api") and live_state.dashboard_lock.is_locked:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Dashboard locked"},
            )

        return await call_next(request)

    # ── No-cache headers for API responses ────────────────────────────
    # Browsers can sometimes hold onto API JSON responses through their
    # disk cache. Force no-store on every /api/ response so the dashboard
    # always sees fresh data without manual page reload.
    @app.middleware("http")
    async def no_cache_api(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    # ── Long-lived cache for Vite hashed assets ──────────────────────
    # Vite produces content-hashed filenames (e.g. index-abc123.js),
    # so they're safe to cache indefinitely — a new hash = new URL.
    @app.middleware("http")
    async def cache_static_assets(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/ui/assets/"):
            response.headers["Cache-Control"] = (
                "public, max-age=31536000, immutable"
            )
        return response

    # ── Request timeout middleware ────────────────────────────────────
    @app.middleware("http")
    async def timeout_middleware(request: Request, call_next):
        """
        Cap request handling at REQUEST_TIMEOUT seconds to prevent
        a misbehaving handler from stalling the trading loop.
        SSE streams are exempt (they're long-lived by design).
        """
        if request.url.path.endswith("/stream"):
            return await call_next(request)

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": "Request timed out"},
            )

    # ── Routers ───────────────────────────────────────────────────────
    app.include_router(account_routes.router)
    app.include_router(auth_routes.router)
    app.include_router(live_routes.router)
    app.include_router(system_routes.router)
    app.include_router(history_routes.router)
    app.include_router(config_routes.router)
    app.include_router(backtest_routes.router)
    app.include_router(models_routes.router)
    app.include_router(news_routes.router)
    app.include_router(invariants_routes.router)

    # ── SPA static files ────────────────────────────────────────────────
    # Serve the Vite-built SPA from src/api/static/dist/
    _dist_dir = Path(__file__).resolve().parent / "static" / "dist"
    if _dist_dir.is_dir():
        # Mount static assets (JS, CSS, etc.) at /ui/assets/
        _assets_dir = _dist_dir / "assets"
        if _assets_dir.is_dir():
            app.mount(
                "/ui/assets",
                StaticFiles(directory=str(_assets_dir)),
                name="spa-assets",
            )

        # SPA catch-all: serve static files if they exist, otherwise
        # index.html for client-side routing by React Router.
        # Security Audit C1: path traversal protection — resolve target
        # path and verify it stays within _dist_dir.
        _dist_resolved = _dist_dir.resolve()

        @app.get("/ui/{full_path:path}")
        async def spa_catchall(full_path: str):
            # Try to serve as a static file first (favicon.svg, etc.)
            if full_path:
                static_file = (_dist_dir / full_path).resolve()
                # Verify resolved path is inside _dist_dir (prevents traversal)
                try:
                    static_file.relative_to(_dist_resolved)
                except ValueError:
                    return JSONResponse(
                        status_code=404,
                        content={"detail": "Not found"},
                    )
                if static_file.is_file():
                    return FileResponse(str(static_file))
            # Fall back to index.html for SPA routing
            index = _dist_dir / "index.html"
            if index.is_file():
                return FileResponse(str(index))
            return JSONResponse(
                status_code=404,
                content={"detail": "SPA not built. Run: cd frontend && npm run build"},
            )
    else:
        @app.get("/ui/{full_path:path}")
        async def spa_not_built(full_path: str):
            return JSONResponse(
                status_code=404,
                content={"detail": "SPA not built. Run: cd frontend && npm run build"},
            )

    # ── Root redirect ────────────────────────────────────────────────
    @app.get("/")
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/ui/")

    logger.info("FastAPI app built with %d routes", len(app.routes))
    return app
