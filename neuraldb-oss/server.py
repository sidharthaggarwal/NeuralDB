"""
NeuralDB REST API Server
========================
Production-grade FastAPI application exposing NeuralDB over HTTP.

Design decisions:
  - Separation of concerns: security layer → validation → engine
  - All user input passes through ``neuraldb.security.validation`` before
    reaching the storage engine.
  - Authentication uses constant-time comparison (timing-safe).
  - Rate limiting is enforced per IP via a sliding-window token bucket.
  - Structured logging on every request; sensitive headers never logged.
  - Graceful startup/shutdown with database persistence.
  - Health and readiness probes suitable for Kubernetes / ECS.
  - OpenAPI docs auto-generated; served at /docs and /redoc.

Environment variables (all have safe defaults for local dev):
  NEURALDB_API_KEY       API key for authentication (required in production)
  NEURALDB_AUTH          "true" | "false"  (default: true)
  NEURALDB_DATA_PATH     Where to persist databases  (default: ./data)
  NEURALDB_VECTOR_DIM    Embedding dimensions         (default: 128)
  NEURALDB_RATE_LIMIT    "true" | "false"             (default: true)
  NEURALDB_RPM           Requests per minute per IP   (default: 120)
  NEURALDB_BURST         Max burst size               (default: 20)
  PORT                   HTTP port                    (default: 8000)
  LOG_LEVEL              Python log level             (default: INFO)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging — configure before any other imports that might emit log records
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NeuralDB imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))

from neuraldb import NeuralDB, LaneType  # noqa: E402
from neuraldb.security.auth import verify_api_key  # noqa: E402
from neuraldb.security.ratelimit import RateLimitMiddleware  # noqa: E402
from neuraldb.security.validation import (  # noqa: E402
    sanitize_aiql,
    sanitize_confidence,
    sanitize_embedding,
    sanitize_name,
    sanitize_record_data,
    sanitize_relations,
    MAX_GRAPH_DEPTH,
    MAX_TOP_K,
)
from neuraldb.middleware.logging import RequestLoggingMiddleware  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATA_PATH: str = os.environ.get("NEURALDB_DATA_PATH", "./data")
_VECTOR_DIM: int = int(os.environ.get("NEURALDB_VECTOR_DIM", "128"))
_PORT: int = int(os.environ.get("PORT", "8000"))

# ---------------------------------------------------------------------------
# In-process database registry
# ---------------------------------------------------------------------------

_databases: Dict[str, NeuralDB] = {}


def _get_db(name: str) -> NeuralDB:
    """Retrieve an open database or raise 404."""
    sanitize_name(name, "database name")
    if name not in _databases:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Database '{name}' not found. Create it first via POST /db/{{name}}/create.",
        )
    return _databases[name]


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup  : create the data directory and reload any persisted databases.
    On shutdown : flush all open databases to disk.
    """
    # ── Startup ──────────────────────────────────────────────────────────
    os.makedirs(_DATA_PATH, exist_ok=True)
    loaded = 0
    for entry in os.scandir(_DATA_PATH):
        if entry.is_dir() and os.path.exists(
            os.path.join(entry.path, "neuraldb.pkl")
        ):
            try:
                _databases[entry.name] = NeuralDB(
                    entry.name, path=_DATA_PATH, config={"vector_dim": _VECTOR_DIM}
                )
                loaded += 1
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Failed to load database '%s': %s", entry.name, exc)

    logger.info(
        "NeuralDB server started. data_path=%s vector_dim=%d databases_loaded=%d",
        _DATA_PATH,
        _VECTOR_DIM,
        loaded,
    )

    yield  # ── Application runs ──────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────
    for name, db in _databases.items():
        try:
            db.save()
            logger.info("Persisted database '%s'", name)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to persist database '%s': %s", name, exc)
    logger.info("NeuralDB server stopped.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NeuralDB API",
    description="""
## NeuralDB — AI-Native Database Engine

NeuralDB is a database engine purpose-built for AI applications.

### Key Capabilities

| Feature | Description |
|---|---|
| **Confidence-Aware Records** | Every record carries a certainty score `0.0–1.0` |
| **Temporal Decay** | Memories fade via configurable exponential half-life |
| **Memory Lanes** | Episodic · Semantic · Working · Procedural · Associative · Sensory |
| **Vector Search** | `score = cosine_similarity × confidence^0.5` |
| **Knowledge Graph** | Typed, weighted edges with depth-limited traversal |
| **AIQL** | `RECALL` · `REMEMBER` · `FORGET` · `TRAVERSE` · `CONSOLIDATE` · `REFLECT` |

### Authentication

All endpoints (except `/health`) require the `X-API-Key` header.

```
X-API-Key: <your-api-key>
```

### Quick Start

```bash
# 1. Create a database
curl -X POST /db/mydb/create -H "X-API-Key: $KEY"

# 2. Create a memory lane
curl -X POST /db/mydb/lanes \\
  -H "X-API-Key: $KEY" \\
  -d '{"name": "knowledge", "lane_type": "semantic"}'

# 3. Store a fact with confidence
curl -X POST /db/mydb/insert \\
  -H "X-API-Key: $KEY" \\
  -d '{"lane": "knowledge", "data": {"entity": "Paris"}, "confidence": 0.99}'

# 4. Query with AIQL
curl -X POST /db/mydb/query \\
  -H "X-API-Key: $KEY" \\
  -d '{"aiql": "RECALL TOP 5 FROM knowledge WHERE confidence > 0.8"}'
```
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── Middleware (order matters: outermost = first to see request) ────────────
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "Accept"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms", "X-RateLimit-Remaining"],
)

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class CreateDBRequest(BaseModel):
    """Optional configuration overrides when creating a database."""
    config: Dict[str, Any] = Field(default_factory=dict, description="Config overrides")


class CreateLaneRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="Lane identifier")
    lane_type: str = Field(
        default="semantic",
        description="Memory type: episodic | semantic | working | procedural | associative | sensory",
    )
    decay_half_life: Optional[float] = Field(
        None,
        gt=0,
        description="Half-life in seconds for temporal decay. None = no decay.",
    )
    schema: Optional[Dict[str, str]] = Field(
        None, description="Optional field-type hints for documentation"
    )

    @field_validator("lane_type")
    @classmethod
    def _valid_lane_type(cls, v: str) -> str:
        valid = {t.value for t in LaneType}
        if v not in valid:
            raise ValueError(f"lane_type must be one of {sorted(valid)}")
        return v


class InsertRequest(BaseModel):
    lane: str = Field(..., min_length=1, max_length=64)
    data: Dict[str, Any] = Field(..., description="Record payload — any JSON object")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Certainty score 0.0 (unknown) – 1.0 (certain)",
    )
    embedding: Optional[List[float]] = Field(
        None, description="Vector embedding for semantic search"
    )
    relations: Optional[List[List[Any]]] = Field(
        None,
        description="Knowledge graph edges: [[target_id, relation_type, weight], ...]",
    )
    ttl: Optional[float] = Field(
        None, gt=0, description="Time-to-live override in seconds"
    )
    provenance: Optional[str] = Field(
        None, max_length=512, description="Data source / origin"
    )


class UpdateRequest(BaseModel):
    data: Optional[Dict[str, Any]] = Field(None, description="Fields to merge into existing data")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None


class QueryRequest(BaseModel):
    aiql: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="AIQL query string",
        examples=["RECALL TOP 5 FROM knowledge WHERE confidence > 0.8"],
    )


class SearchRequest(BaseModel):
    vector: List[float] = Field(..., min_length=1, description="Query embedding vector")
    lane: Optional[str] = Field(None, description="Restrict search to this lane (optional)")
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class TraverseRequest(BaseModel):
    node_id: str = Field(..., min_length=1, max_length=128)
    rel_type: Optional[str] = Field(None, max_length=64)
    depth: int = Field(default=2, ge=1, le=MAX_GRAPH_DEPTH)
    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # type: ignore[type-arg]
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred.",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# ---------------------------------------------------------------------------
# Routes — System
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_description="Server health and loaded databases",
)
def health() -> Dict[str, Any]:
    """
    Liveness / readiness probe.

    This endpoint is **exempt from authentication and rate-limiting** so that
    load-balancer health checks and Kubernetes probes always succeed.
    """
    return {
        "status": "healthy",
        "engine": "NeuralDB",
        "version": "1.0.0",
        "timestamp": time.time(),
        "databases_loaded": list(_databases.keys()),
        "vector_dim": _VECTOR_DIM,
    }


# ---------------------------------------------------------------------------
# Routes — Database
# ---------------------------------------------------------------------------


@app.post(
    "/db/{name}/create",
    tags=["Database"],
    status_code=status.HTTP_201_CREATED,
    summary="Create a database",
)
def create_database(
    name: str = Path(..., description="Database name"),
    req: CreateDBRequest = CreateDBRequest(),
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Create a new NeuralDB database.

    The database is persisted to disk automatically (every 10 writes) and on
    server shutdown.  The name must consist of alphanumeric characters, hyphens,
    and underscores only.
    """
    sanitize_name(name, "database name")
    if name in _databases:
        return {"message": f"Database '{name}' already exists.", "created": False}

    config = {"vector_dim": _VECTOR_DIM, **req.config}
    db = NeuralDB(name, path=_DATA_PATH, config=config)
    _databases[name] = db
    logger.info("Created database '%s'", name)
    return {"message": f"Database '{name}' created.", "created": True, "config": config}


@app.get("/db/{name}/stats", tags=["Database"], summary="Database statistics")
def get_stats(
    name: str = Path(...),
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Return full health report: record counts, confidence distributions, write/read totals."""
    return _get_db(name).stats_report()


@app.delete(
    "/db/{name}",
    tags=["Database"],
    summary="Delete a database",
    status_code=status.HTTP_200_OK,
)
def delete_database(
    name: str = Path(...),
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Close and remove a database from the server.

    This does **not** delete the on-disk files — it only unloads the database
    from memory.  To permanently delete the data, remove the directory from
    the server filesystem.
    """
    sanitize_name(name, "database name")
    if name not in _databases:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Database '{name}' not found.")
    del _databases[name]
    logger.info("Unloaded database '%s'", name)
    return {"message": f"Database '{name}' unloaded.", "deleted": True}


# ---------------------------------------------------------------------------
# Routes — Lanes
# ---------------------------------------------------------------------------


@app.post(
    "/db/{name}/lanes",
    tags=["Lanes"],
    status_code=status.HTTP_201_CREATED,
    summary="Create a memory lane",
)
def create_lane(
    name: str = Path(...),
    req: CreateLaneRequest = ...,
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Create a typed memory lane inside a database.

    **Lane types and their defaults:**

    | Type | Typical use | Default half-life |
    |---|---|---|
    | `semantic` | Facts and knowledge | None (permanent) |
    | `episodic` | Events, conversations | 7 days |
    | `working` | Active task context | 1 hour |
    | `procedural` | Workflows, rules | None |
    | `associative` | Pure relationship data | None |
    | `sensory` | Raw embeddings | 5 minutes |
    """
    db = _get_db(name)
    sanitize_name(req.name, "lane name")
    lane_type = LaneType(req.lane_type)
    lane = db.create_lane(req.name, lane_type=lane_type,
                          decay_half_life=req.decay_half_life, schema=req.schema)
    db.save()
    return {
        "lane": req.name,
        "type": lane.lane_type.value,
        "decay_half_life": lane.decay_half_life,
        "created": True,
    }


@app.get("/db/{name}/lanes", tags=["Lanes"], summary="List memory lanes")
def list_lanes(
    name: str = Path(...),
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    db = _get_db(name)
    return {
        "lanes": [
            {
                "name": lane_name,
                "type": lane.lane_type.value,
                "active_records": lane.count(active_only=True),
                "total_records": lane.count(active_only=False),
                "decay_half_life": lane.decay_half_life,
            }
            for lane_name, lane in db.lanes.items()
        ]
    }


# ---------------------------------------------------------------------------
# Routes — Records
# ---------------------------------------------------------------------------


@app.post(
    "/db/{name}/insert",
    tags=["Records"],
    status_code=status.HTTP_201_CREATED,
    summary="Insert a record",
)
def insert_record(
    name: str = Path(...),
    req: InsertRequest = ...,
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Insert a new record into a memory lane.

    The record is stored with AI-native metadata: confidence score, creation
    timestamp, version counter, and optional vector embedding / graph relations.

    Returns the ``record_id`` UUID for future retrieval or graph linking.
    """
    db = _get_db(name)
    sanitize_name(req.lane, "lane name")
    clean_data = sanitize_record_data(req.data)
    clean_embedding = sanitize_embedding(req.embedding, expected_dim=_VECTOR_DIM)
    clean_confidence = sanitize_confidence(req.confidence)
    clean_relations = sanitize_relations(req.relations)

    record_id = db.insert(
        lane_name=req.lane,
        data=clean_data,
        confidence=clean_confidence,
        embedding=clean_embedding,
        relations=clean_relations,
        ttl=req.ttl,
        provenance=req.provenance,
    )
    return {
        "record_id": record_id,
        "lane": req.lane,
        "confidence": clean_confidence,
    }


@app.get(
    "/db/{name}/get/{lane}/{record_id}",
    tags=["Records"],
    summary="Get a record by ID",
)
def get_record(
    name: str = Path(...),
    lane: str = Path(...),
    record_id: str = Path(..., min_length=36, max_length=36),
    respect_decay: bool = Query(default=True, description="Return null for expired records"),
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Fetch a single record by its UUID.

    When ``respect_decay=true`` (default), records whose effective confidence
    has dropped below the expiry threshold return ``404 Not Found`` — the same
    response as a record that was never inserted.  This is intentional: from
    the caller's perspective, an expired memory is as good as gone.
    """
    db = _get_db(name)
    sanitize_name(lane, "lane name")
    record = db.get(lane, record_id, respect_decay=respect_decay)
    if record is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Record '{record_id}' not found or has expired in lane '{lane}'.",
        )
    return record


@app.patch(
    "/db/{name}/update/{lane}/{record_id}",
    tags=["Records"],
    summary="Update a record",
)
def update_record(
    name: str = Path(...),
    lane: str = Path(...),
    record_id: str = Path(..., min_length=36, max_length=36),
    req: UpdateRequest = ...,
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Partially update an existing record.  Only supplied fields are changed.
    """
    db = _get_db(name)
    sanitize_name(lane, "lane name")
    clean_data = sanitize_record_data(req.data) if req.data else None
    clean_embedding = sanitize_embedding(req.embedding, expected_dim=_VECTOR_DIM)
    clean_confidence = sanitize_confidence(req.confidence) if req.confidence is not None else None

    success = db.update(lane, record_id,
                        data=clean_data, confidence=clean_confidence,
                        embedding=clean_embedding)
    if not success:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Record '{record_id}' not found in lane '{lane}'.",
        )
    return {"updated": True, "record_id": record_id}


# ---------------------------------------------------------------------------
# Routes — AIQL Query
# ---------------------------------------------------------------------------


@app.post(
    "/db/{name}/query",
    tags=["Query (AIQL)"],
    summary="Execute an AIQL query",
)
def execute_query(
    name: str = Path(...),
    req: QueryRequest = ...,
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Execute an **AIQL (AI Intelligence Query Language)** statement.

    AIQL is designed around cognitive memory operations rather than
    traditional SQL CRUD.

    **Supported verbs:**

    ```
    RECALL [TOP n] FROM <lane> [WHERE <conditions>] [ORDER BY confidence|age] [FUZZY]
    REMEMBER INTO <lane> SET key=val [WITH CONFIDENCE 0.9]
    FORGET FROM <lane> WHERE <conditions>
    TRAVERSE FROM <node_id> [VIA <rel_type>] [DEPTH n]
    CONSOLIDATE <source_lane> INTO <target_lane> WHERE confidence > n
    REINFORCE <lane> WHERE <conditions> BY <amount>
    DOUBT <lane> WHERE <conditions> BY <amount>
    REFLECT ON <lane|database>
    ```

    **WHERE operators:** `=`, `!=`, `>`, `>=`, `<`, `<=`, `CONTAINS`, `LIKE`

    **Special WHERE fields:** `confidence`, `age` (seconds), `version`

    **Examples:**
    ```
    RECALL TOP 10 FROM knowledge WHERE confidence > 0.8 ORDER BY confidence DESC
    RECALL FROM episodes WHERE session = sess_001 FUZZY
    FORGET FROM working WHERE age > 3600
    CONSOLIDATE episodes INTO knowledge WHERE confidence > 0.75
    REINFORCE knowledge WHERE entity = Paris BY 0.05
    REFLECT ON database
    ```
    """
    db = _get_db(name)
    clean_aiql = sanitize_aiql(req.aiql)
    result = db.query(clean_aiql)
    if "error" in result:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# Routes — Vector Search
# ---------------------------------------------------------------------------


@app.post(
    "/db/{name}/search",
    tags=["Search"],
    summary="Semantic vector similarity search",
)
def vector_search(
    name: str = Path(...),
    req: SearchRequest = ...,
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Find records semantically similar to the given query vector.

    NeuralDB uses **confidence-weighted similarity**:

    ```
    score = cosine_similarity(query, record) × effective_confidence^0.5
    ```

    This ensures that a record with 0.5 similarity but 0.99 confidence scores
    higher than a 0.8-similar record with 0.1 confidence — which is the right
    behaviour when you need *reliable* semantic matches.

    Results are ordered by ``_combined_score`` descending.
    """
    db = _get_db(name)
    clean_vector = sanitize_embedding(req.vector, expected_dim=_VECTOR_DIM)
    if req.lane:
        sanitize_name(req.lane, "lane name")
    results = db.similarity_search(
        query_vector=clean_vector,  # type: ignore[arg-type]
        lane_name=req.lane,
        top_k=req.top_k,
        min_confidence=req.min_confidence,
        min_similarity=req.min_similarity,
    )
    return {"results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Routes — Knowledge Graph
# ---------------------------------------------------------------------------


@app.post(
    "/db/{name}/traverse",
    tags=["Graph"],
    summary="Traverse the knowledge graph",
)
def traverse_graph(
    name: str = Path(...),
    req: TraverseRequest = ...,
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    BFS traversal of the knowledge graph starting at ``node_id``.

    Returns a subgraph JSON containing all reachable nodes and edges up to
    ``depth`` hops.  Optionally filtered by ``rel_type`` and ``min_weight``.
    """
    db = _get_db(name)
    return db.traverse_graph(
        start_id=req.node_id,
        rel_type=req.rel_type,
        depth=req.depth,
        min_weight=req.min_weight,
    )


# ---------------------------------------------------------------------------
# Routes — Memory Management
# ---------------------------------------------------------------------------


@app.post(
    "/db/{name}/consolidate",
    tags=["Memory"],
    summary="Consolidate memories between lanes",
)
def consolidate(
    name: str = Path(...),
    source: str = Query(..., description="Source lane"),
    target: str = Query(..., description="Target lane"),
    min_confidence: float = Query(default=0.7, ge=0.0, le=1.0),
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Move records with ``effective_confidence >= min_confidence`` from ``source``
    to ``target`` lane.  Duplicate records (>90% field match) are merged rather
    than duplicated — their confidence is bumped instead.

    This models the neuroscience concept of **memory consolidation**, where
    short-term episodic memories are transferred to long-term semantic memory
    during sleep.
    """
    db = _get_db(name)
    sanitize_name(source, "source lane"), sanitize_name(target, "target lane")
    count = db.consolidate(source, target, min_confidence=min_confidence)
    return {"consolidated": count, "from": source, "to": target}


@app.delete(
    "/db/{name}/purge",
    tags=["Memory"],
    summary="Purge expired records",
    status_code=status.HTTP_200_OK,
)
def purge_expired(
    name: str = Path(...),
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Remove all records whose effective confidence has dropped below threshold."""
    db = _get_db(name)
    count = db.purge_expired()
    db.save()
    return {"purged": count}


@app.post(
    "/db/{name}/save",
    tags=["Database"],
    summary="Persist database to disk",
)
def save_database(
    name: str = Path(...),
    _auth: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Manually trigger a persistence flush."""
    db = _get_db(name)
    db.save()
    return {"saved": True, "database": name}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=_PORT,
        reload=False,
        access_log=False,  # We use our own structured logger
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
