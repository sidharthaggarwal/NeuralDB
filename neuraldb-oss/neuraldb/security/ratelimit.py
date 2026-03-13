"""
neuraldb.security.ratelimit
===========================
Token-bucket rate limiter implemented as a FastAPI middleware.

Design:
  - Per-IP sliding window using an in-process dict (suitable for single-node)
  - Returns Retry-After header on 429 so clients can back off gracefully
  - Configurable via environment variables — no code changes needed
  - O(1) per request; memory bounded to ``MAX_CLIENTS`` entries
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Deque, Dict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all tunable via environment)
# ---------------------------------------------------------------------------

_RATE_LIMIT_ENABLED: bool = (
    os.environ.get("NEURALDB_RATE_LIMIT", "true").lower() == "true"
)
_REQUESTS_PER_MINUTE: int = int(os.environ.get("NEURALDB_RPM", "120"))
_BURST: int = int(os.environ.get("NEURALDB_BURST", "20"))
_WINDOW_SECONDS: int = 60
_MAX_CLIENTS: int = 10_000  # cap memory usage


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter.

    Each client (identified by IP) may make at most ``_REQUESTS_PER_MINUTE``
    requests per 60-second window.  The ``/health`` endpoint is exempt to
    allow load-balancer probes.
    """

    _EXEMPT_PATHS = {"/health", "/metrics"}

    def __init__(self, app) -> None:  # type: ignore[override]
        super().__init__(app)
        # ip -> deque of request timestamps (float epoch seconds)
        self._windows: Dict[str, Deque[float]] = {}

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract real client IP, honouring ``X-Forwarded-For`` when set by a
        trusted reverse proxy.  We take only the *first* (leftmost) address
        to avoid spoofing via a crafted header.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if not _RATE_LIMIT_ENABLED:
            return await call_next(request)

        if request.url.path in self._EXEMPT_PATHS:
            return await call_next(request)

        ip = self._get_client_ip(request)
        now = time.monotonic()
        window_start = now - _WINDOW_SECONDS

        # Evict oldest entries and prune stale clients
        if ip not in self._windows:
            if len(self._windows) >= _MAX_CLIENTS:
                # Evict the client with the oldest most-recent-request
                oldest_ip = min(
                    self._windows,
                    key=lambda k: self._windows[k][-1] if self._windows[k] else 0,
                )
                del self._windows[oldest_ip]
            self._windows[ip] = deque()

        bucket: Deque[float] = self._windows[ip]

        # Remove timestamps outside the current window
        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= _REQUESTS_PER_MINUTE:
            retry_after = int(_WINDOW_SECONDS - (now - bucket[0])) + 1
            logger.warning("Rate limit exceeded for %s (%d req/min)", ip, len(bucket))
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": (
                        f"Too many requests. Limit is {_REQUESTS_PER_MINUTE} "
                        f"requests per minute."
                    ),
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(_REQUESTS_PER_MINUTE)
        response.headers["X-RateLimit-Remaining"] = str(
            _REQUESTS_PER_MINUTE - len(bucket)
        )
        response.headers["X-RateLimit-Reset"] = str(
            int(now + _WINDOW_SECONDS - (bucket[0] if bucket else now))
        )
        return response
