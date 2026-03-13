"""
neuraldb.middleware.logging
===========================
Structured request/response logging middleware.

Emits one log line per request with:
  - method, path, status code, duration
  - client IP (after header normalisation)
  - request ID (injected as X-Request-ID response header)

Sensitive fields (X-API-Key, Authorization) are NEVER logged.
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("neuraldb.access")

_SENSITIVE_HEADERS = frozenset({"x-api-key", "authorization", "cookie", "set-cookie"})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured access logger with request-ID correlation."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Attach request-id so downstream code can reference it
        request.state.request_id = request_id

        response: Response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )

        logger.info(
            "%s %s %s %.2fms ip=%s rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            client_ip,
            request_id,
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response
