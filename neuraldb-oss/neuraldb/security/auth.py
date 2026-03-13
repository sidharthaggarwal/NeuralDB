"""
neuraldb.security.auth
======================
Authentication and authorization for the NeuralDB REST API.

Supports:
  - Static API key authentication (header: X-API-Key)
  - Optional HMAC-signed token authentication

Security design principles:
  - Constant-time comparison to prevent timing attacks
  - No key material ever logged
  - All failures return identical HTTP 401 (no enumeration)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from typing import Optional

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_KEY: str = os.environ.get("NEURALDB_API_KEY", "")
_AUTH_ENABLED: bool = os.environ.get("NEURALDB_AUTH", "true").lower() == "true"

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or missing authentication credentials.",
    headers={"WWW-Authenticate": "ApiKey"},
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _constant_time_equals(a: str, b: str) -> bool:
    """
    Compare two strings in constant time to mitigate timing side-channels.

    Uses ``hmac.compare_digest`` on UTF-8-encoded bytes so the comparison
    time is independent of where (if at all) the strings differ.
    """
    return hmac.compare_digest(
        a.encode("utf-8", errors="replace"),
        b.encode("utf-8", errors="replace"),
    )


def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> bool:
    """
    FastAPI dependency that validates the ``X-API-Key`` request header.

    Raises ``HTTP 401`` on any authentication failure.  The error message is
    intentionally generic — we never reveal *why* authentication failed.
    """
    if not _AUTH_ENABLED:
        return True

    if not _API_KEY:
        # Server is misconfigured — fail closed, never open.
        logger.critical(
            "NEURALDB_API_KEY environment variable is not set. "
            "All requests are being rejected."
        )
        raise _UNAUTHORIZED

    if not x_api_key:
        logger.warning(
            "Rejected unauthenticated request from %s %s",
            request.client.host if request.client else "unknown",
            request.url.path,
        )
        raise _UNAUTHORIZED

    if not _constant_time_equals(x_api_key, _API_KEY):
        logger.warning(
            "Rejected request with invalid API key from %s",
            request.client.host if request.client else "unknown",
        )
        raise _UNAUTHORIZED

    return True
