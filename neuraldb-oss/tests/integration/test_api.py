"""
tests/integration/test_api.py
==============================
Integration tests for the NeuralDB REST API.

Uses FastAPI's TestClient (synchronous httpx under the hood) so no real
HTTP server is required.  Each test class gets its own isolated database
via a module-scoped fixture to avoid inter-test pollution.
"""

from __future__ import annotations

import math
import random
import sys
import os

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Disable auth and rate limiting for tests
os.environ["NEURALDB_AUTH"] = "false"
os.environ["NEURALDB_RATE_LIMIT"] = "false"
os.environ["NEURALDB_VECTOR_DIM"] = "32"

from server import app  # noqa: E402

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key", "Content-Type": "application/json"}

_DB = "test_integration"
_LANE = "knowledge"


def _rand_vec(dim: int = 32) -> list:
    random.seed(42)
    v = [random.gauss(0, 1) for _ in range(dim)]
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    """Create a database and lane once for all tests in this module."""
    client.post(f"/db/{_DB}/create", headers=HEADERS)
    client.post(
        f"/db/{_DB}/lanes",
        headers=HEADERS,
        json={"name": _LANE, "lane_type": "semantic"},
    )
    yield
    client.delete(f"/db/{_DB}", headers=HEADERS)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_no_auth_required(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_contains_engine_name(self):
        r = client.get("/health")
        assert r.json()["engine"] == "NeuralDB"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class TestDatabase:
    def test_create_db_returns_201(self, tmp_path):
        r = client.post("/db/new_test_db/create", headers=HEADERS)
        assert r.status_code == 201

    def test_create_existing_db_idempotent(self):
        client.post(f"/db/{_DB}/create", headers=HEADERS)
        r = client.post(f"/db/{_DB}/create", headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["created"] is False

    def test_get_stats_returns_200(self):
        r = client.get(f"/db/{_DB}/stats", headers=HEADERS)
        assert r.status_code == 200
        assert "lanes" in r.json()

    def test_get_nonexistent_db_stats_returns_404(self):
        r = client.get("/db/does_not_exist/stats", headers=HEADERS)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------


class TestLanes:
    def test_list_lanes_returns_200(self):
        r = client.get(f"/db/{_DB}/lanes", headers=HEADERS)
        assert r.status_code == 200
        assert "lanes" in r.json()

    def test_create_lane_semantic(self):
        r = client.post(
            f"/db/{_DB}/lanes",
            headers=HEADERS,
            json={"name": "extra_facts", "lane_type": "semantic"},
        )
        assert r.status_code == 201
        assert r.json()["type"] == "semantic"

    def test_create_lane_invalid_type_returns_422(self):
        r = client.post(
            f"/db/{_DB}/lanes",
            headers=HEADERS,
            json={"name": "bad", "lane_type": "invalid_type"},
        )
        assert r.status_code == 422

    def test_create_lane_with_custom_decay(self):
        r = client.post(
            f"/db/{_DB}/lanes",
            headers=HEADERS,
            json={"name": "short_lived", "lane_type": "working", "decay_half_life": 60},
        )
        assert r.status_code == 201
        assert r.json()["decay_half_life"] == 60


# ---------------------------------------------------------------------------
# Records — Insert
# ---------------------------------------------------------------------------


class TestInsert:
    def test_insert_returns_201_with_record_id(self):
        r = client.post(
            f"/db/{_DB}/insert",
            headers=HEADERS,
            json={"lane": _LANE, "data": {"entity": "Paris"}, "confidence": 0.99},
        )
        assert r.status_code == 201
        assert "record_id" in r.json()
        assert len(r.json()["record_id"]) == 36  # UUID

    def test_insert_empty_data_returns_422(self):
        r = client.post(
            f"/db/{_DB}/insert",
            headers=HEADERS,
            json={"lane": _LANE, "data": {}, "confidence": 0.9},
        )
        assert r.status_code == 422

    def test_insert_invalid_confidence_returns_422(self):
        r = client.post(
            f"/db/{_DB}/insert",
            headers=HEADERS,
            json={"lane": _LANE, "data": {"k": "v"}, "confidence": 5.0},
        )
        assert r.status_code == 422

    def test_insert_with_embedding(self):
        r = client.post(
            f"/db/{_DB}/insert",
            headers=HEADERS,
            json={
                "lane": _LANE,
                "data": {"k": "v"},
                "confidence": 0.8,
                "embedding": _rand_vec(32),
            },
        )
        assert r.status_code == 201

    def test_insert_wrong_embedding_dim_returns_422(self):
        r = client.post(
            f"/db/{_DB}/insert",
            headers=HEADERS,
            json={
                "lane": _LANE,
                "data": {"k": "v"},
                "embedding": [0.1] * 999,  # wrong dim
            },
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Records — Get / Update
# ---------------------------------------------------------------------------


class TestGetUpdate:
    def test_get_existing_record(self):
        ins = client.post(
            f"/db/{_DB}/insert",
            headers=HEADERS,
            json={"lane": _LANE, "data": {"entity": "Rome"}, "confidence": 0.95},
        )
        rid = ins.json()["record_id"]
        r = client.get(f"/db/{_DB}/get/{_LANE}/{rid}", headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["entity"] == "Rome"

    def test_get_nonexistent_returns_404(self):
        r = client.get(
            f"/db/{_DB}/get/{_LANE}/00000000-0000-0000-0000-000000000000",
            headers=HEADERS,
        )
        assert r.status_code == 404

    def test_update_record(self):
        ins = client.post(
            f"/db/{_DB}/insert",
            headers=HEADERS,
            json={"lane": _LANE, "data": {"entity": "Berlin", "pop": 100}},
        )
        rid = ins.json()["record_id"]
        upd = client.patch(
            f"/db/{_DB}/update/{_LANE}/{rid}",
            headers=HEADERS,
            json={"data": {"pop": 3700000}},
        )
        assert upd.status_code == 200
        fetched = client.get(f"/db/{_DB}/get/{_LANE}/{rid}", headers=HEADERS)
        assert fetched.json()["pop"] == 3700000


# ---------------------------------------------------------------------------
# AIQL
# ---------------------------------------------------------------------------


class TestAIQL:
    def test_recall_returns_200(self):
        r = client.post(
            f"/db/{_DB}/query",
            headers=HEADERS,
            json={"aiql": f"RECALL TOP 5 FROM {_LANE}"},
        )
        assert r.status_code == 200
        assert "records" in r.json()

    def test_reflect_returns_stats(self):
        r = client.post(
            f"/db/{_DB}/query",
            headers=HEADERS,
            json={"aiql": "REFLECT ON database"},
        )
        assert r.status_code == 200
        assert "reflection" in r.json()

    def test_injection_attempt_rejected(self):
        r = client.post(
            f"/db/{_DB}/query",
            headers=HEADERS,
            json={"aiql": "RECALL FROM facts -- DROP TABLE"},
        )
        assert r.status_code == 422  # blocked by sanitize_aiql

    def test_empty_aiql_returns_422(self):
        r = client.post(
            f"/db/{_DB}/query",
            headers=HEADERS,
            json={"aiql": ""},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Vector Search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_search_returns_results(self):
        # Insert a record with an embedding first
        vec = _rand_vec(32)
        client.post(
            f"/db/{_DB}/insert",
            headers=HEADERS,
            json={"lane": _LANE, "data": {"tag": "searchable"}, "embedding": vec},
        )
        r = client.post(
            f"/db/{_DB}/search",
            headers=HEADERS,
            json={"vector": vec, "top_k": 5},
        )
        assert r.status_code == 200
        assert "results" in r.json()

    def test_search_nan_vector_rejected(self):
        r = client.post(
            f"/db/{_DB}/search",
            headers=HEADERS,
            json={"vector": [float("nan")] * 32, "top_k": 5},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Request ID propagation
# ---------------------------------------------------------------------------


class TestRequestID:
    def test_response_has_request_id_header(self):
        r = client.get("/health")
        assert "x-request-id" in r.headers

    def test_response_has_timing_header(self):
        r = client.get("/health")
        assert "x-response-time-ms" in r.headers
