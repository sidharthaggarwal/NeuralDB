"""
neuraldb.security.validation
=============================
Input validation, sanitization, and schema enforcement.

All user-supplied data passes through these validators before it touches
the storage engine.  This is the "trust boundary" — everything inside is
considered clean; everything outside is untrusted.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status

# ---------------------------------------------------------------------------
# Limits  (tunable — keep them tight for production)
# ---------------------------------------------------------------------------

MAX_DB_NAME_LEN: int = 64
MAX_LANE_NAME_LEN: int = 64
MAX_RECORD_DATA_BYTES: int = 1_048_576  # 1 MiB per record
MAX_EMBEDDING_DIM: int = 4_096
MAX_AIQL_LEN: int = 4_096
MAX_RELATION_COUNT: int = 64
MAX_DATA_KEYS: int = 128
MAX_STRING_VALUE_LEN: int = 65_536  # 64 KiB per string field
MAX_GRAPH_DEPTH: int = 10
MAX_TOP_K: int = 1_000

# Allowlists
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
# AIQL injection: block comment sequences and stacked semicolons
_AIQL_FORBIDDEN_RE = re.compile(r"(--|;{2,}|/\*|\*/|xp_|EXEC\s|EXECUTE\s)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_error(detail: str, status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def sanitize_name(name: str, field: str = "name") -> str:
    """
    Validate an identifier (database name, lane name).

    Rules:
      - 1–64 alphanumeric characters, hyphens, or underscores
      - No path traversal sequences
      - No null bytes
    """
    if not name:
        raise _http_error(f"'{field}' must not be empty.")
    if "\x00" in name:
        raise _http_error(f"'{field}' contains null byte.")
    if not _SAFE_NAME_RE.match(name):
        raise _http_error(
            f"'{field}' must be 1–64 characters: a-z, A-Z, 0-9, hyphen, or underscore."
        )
    return name


def sanitize_aiql(query: str) -> str:
    """
    Validate an AIQL query string.

    Rejects strings that contain SQL-injection–style comment sequences or
    stacked-statement patterns that are invalid in AIQL.
    """
    if not query or not query.strip():
        raise _http_error("AIQL query must not be empty.")
    if len(query) > MAX_AIQL_LEN:
        raise _http_error(
            f"AIQL query too long: {len(query)} characters (max {MAX_AIQL_LEN})."
        )
    if _AIQL_FORBIDDEN_RE.search(query):
        raise _http_error("AIQL query contains forbidden character sequence.")
    return query.strip()


def sanitize_record_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize the user-supplied data payload of a record.

    Rules:
      - Must be a non-empty dict
      - At most MAX_DATA_KEYS keys
      - Keys must be safe identifiers
      - String values are NFC-normalised and length-capped
      - No nested dicts beyond depth 3 (prevents payload amplification)
    """
    if not isinstance(data, dict):
        raise _http_error("Record 'data' must be a JSON object.")
    if not data:
        raise _http_error("Record 'data' must not be empty.")
    if len(data) > MAX_DATA_KEYS:
        raise _http_error(f"Record 'data' has too many keys (max {MAX_DATA_KEYS}).")

    return _sanitize_dict(data, depth=0)


def _sanitize_dict(obj: Dict[str, Any], depth: int) -> Dict[str, Any]:
    if depth > 3:
        raise _http_error("Record 'data' nesting exceeds maximum depth of 3.")

    clean: Dict[str, Any] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            raise _http_error("All data keys must be strings.")
        if not _SAFE_NAME_RE.match(k):
            raise _http_error(
                f"Data key '{k}' contains invalid characters. "
                "Use a-z, A-Z, 0-9, hyphen, or underscore."
            )
        clean[k] = _sanitize_value(v, depth)
    return clean


def _sanitize_value(v: Any, depth: int) -> Any:
    if isinstance(v, str):
        if len(v) > MAX_STRING_VALUE_LEN:
            raise _http_error(
                f"String value too long ({len(v)} chars, max {MAX_STRING_VALUE_LEN})."
            )
        # Normalize to NFC; strip null bytes
        return unicodedata.normalize("NFC", v).replace("\x00", "")
    if isinstance(v, dict):
        return _sanitize_dict(v, depth + 1)
    if isinstance(v, list):
        return [_sanitize_value(item, depth) for item in v]
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    # Coerce anything else to string
    return str(v)[:MAX_STRING_VALUE_LEN]


def sanitize_embedding(
    embedding: Optional[List[float]],
    expected_dim: Optional[int] = None,
) -> Optional[List[float]]:
    """Validate an embedding vector."""
    if embedding is None:
        return None
    if not isinstance(embedding, list):
        raise _http_error("'embedding' must be a JSON array of floats.")
    if len(embedding) == 0:
        raise _http_error("'embedding' must not be empty.")
    if len(embedding) > MAX_EMBEDDING_DIM:
        raise _http_error(
            f"'embedding' has {len(embedding)} dimensions "
            f"(max {MAX_EMBEDDING_DIM})."
        )
    if expected_dim and len(embedding) != expected_dim:
        raise _http_error(
            f"'embedding' has {len(embedding)} dimensions; "
            f"expected {expected_dim} (configured vector_dim)."
        )
    for i, x in enumerate(embedding):
        if not isinstance(x, (int, float)):
            raise _http_error(f"'embedding[{i}]' is not a number.")
        if x != x:  # NaN check
            raise _http_error(f"'embedding[{i}]' is NaN.")
    return [float(x) for x in embedding]


def sanitize_confidence(value: float, field: str = "confidence") -> float:
    """Clamp confidence to [0.0, 1.0] and reject NaN/Inf."""
    if value != value or value == float("inf") or value == float("-inf"):
        raise _http_error(f"'{field}' must be a finite number between 0.0 and 1.0.")
    return max(0.0, min(1.0, float(value)))


def sanitize_relations(
    relations: Optional[List[List[Any]]],
) -> Optional[List[tuple]]:
    """Validate the relations list: [[target_id, rel_type, weight], ...]"""
    if relations is None:
        return None
    if len(relations) > MAX_RELATION_COUNT:
        raise _http_error(
            f"Too many relations: {len(relations)} (max {MAX_RELATION_COUNT})."
        )
    result = []
    for i, rel in enumerate(relations):
        if not isinstance(rel, (list, tuple)) or len(rel) != 3:
            raise _http_error(
                f"relations[{i}] must be [target_id, relation_type, weight]."
            )
        target_id, rel_type, weight = rel
        if not isinstance(target_id, str) or not target_id:
            raise _http_error(f"relations[{i}].target_id must be a non-empty string.")
        if not isinstance(rel_type, str) or not rel_type:
            raise _http_error(f"relations[{i}].relation_type must be a non-empty string.")
        if not isinstance(weight, (int, float)):
            raise _http_error(f"relations[{i}].weight must be a number.")
        result.append((str(target_id), str(rel_type)[:64], float(weight)))
    return result
