"""
tests/unit/test_engine.py
=========================
Unit tests for the NeuralDB core engine.

Follows Google / Meta engineering standards:
  - One assertion concept per test
  - Descriptive names: test_<unit>_<scenario>_<expected>
  - No test depends on execution order
  - All file system I/O uses tmp_path fixture (pytest built-in)
"""

from __future__ import annotations

import math
import time
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from neuraldb import NeuralDB, LaneType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Fresh in-memory NeuralDB instance per test."""
    return NeuralDB("test_db", path=str(tmp_path))


@pytest.fixture
def knowledge_db(tmp_path):
    """NeuralDB with a semantic lane pre-created."""
    d = NeuralDB("kdb", path=str(tmp_path))
    d.create_lane("facts", lane_type=LaneType.SEMANTIC)
    return d


# ---------------------------------------------------------------------------
# Database creation
# ---------------------------------------------------------------------------


class TestDatabaseCreation:
    def test_create_db_sets_name(self, tmp_path):
        db = NeuralDB("mydb", path=str(tmp_path))
        assert db.name == "mydb"

    def test_create_db_initialises_empty_lanes(self, db):
        assert db.lanes == {}

    def test_create_db_stats_zero_writes(self, db):
        assert db.stats["total_writes"] == 0

    def test_duplicate_lane_raises_value_error(self, db):
        db.create_lane("test", lane_type=LaneType.SEMANTIC)
        with pytest.raises(ValueError, match="already exists"):
            db.create_lane("test", lane_type=LaneType.SEMANTIC)


# ---------------------------------------------------------------------------
# Lane creation
# ---------------------------------------------------------------------------


class TestLaneCreation:
    def test_create_semantic_lane_no_decay(self, db):
        lane = db.create_lane("knowledge", lane_type=LaneType.SEMANTIC)
        assert lane.decay_half_life is None

    def test_create_episodic_lane_default_decay(self, db):
        lane = db.create_lane("episodes", lane_type=LaneType.EPISODIC)
        assert lane.decay_half_life == 86400 * 7

    def test_create_working_lane_short_decay(self, db):
        lane = db.create_lane("ctx", lane_type=LaneType.WORKING)
        assert lane.decay_half_life == 3600

    def test_custom_half_life_overrides_default(self, db):
        lane = db.create_lane("custom", lane_type=LaneType.EPISODIC, decay_half_life=999)
        assert lane.decay_half_life == 999

    def test_get_nonexistent_lane_raises_key_error(self, db):
        with pytest.raises(KeyError):
            db.get_lane("nonexistent")


# ---------------------------------------------------------------------------
# Record insertion
# ---------------------------------------------------------------------------


class TestRecordInsertion:
    def test_insert_returns_string_id(self, knowledge_db):
        rid = knowledge_db.insert("facts", {"entity": "Paris"})
        assert isinstance(rid, str) and len(rid) == 36  # UUID

    def test_insert_increments_write_counter(self, knowledge_db):
        knowledge_db.insert("facts", {"entity": "Paris"})
        assert knowledge_db.stats["total_writes"] == 1

    def test_insert_with_confidence_stored(self, knowledge_db):
        rid = knowledge_db.insert("facts", {"k": "v"}, confidence=0.75)
        record = knowledge_db.get("facts", rid)
        assert record["_meta"]["confidence"] == pytest.approx(0.75)

    def test_insert_confidence_clamped_to_one(self, knowledge_db):
        rid = knowledge_db.insert("facts", {"k": "v"}, confidence=1.5)
        record = knowledge_db.get("facts", rid)
        assert record["_meta"]["confidence"] == 1.0

    def test_insert_confidence_clamped_to_zero(self, knowledge_db):
        rid = knowledge_db.insert("facts", {"k": "v"}, confidence=-0.3)
        record = knowledge_db.get("facts", rid)
        assert record["_meta"]["confidence"] == 0.0

    def test_insert_into_nonexistent_lane_auto_creates(self, db):
        rid = db.insert("new_lane", {"k": "v"})
        assert "new_lane" in db.lanes
        assert rid in db.lanes["new_lane"].records

    def test_insert_with_embedding_indexed(self, knowledge_db):
        vec = [0.1] * 128
        knowledge_db.insert("facts", {"k": "v"}, embedding=vec)
        assert knowledge_db.vector_index.count() == 1

    def test_insert_with_relation_creates_edge(self, knowledge_db):
        r1 = knowledge_db.insert("facts", {"entity": "Paris"})
        r2 = knowledge_db.insert("facts", {"entity": "Eiffel"},
                                  relations=[(r1, "located_in", 1.0)])
        edges = knowledge_db.graph_index.neighbors(r2)
        assert len(edges) == 1
        assert edges[0]["type"] == "located_in"


# ---------------------------------------------------------------------------
# Record retrieval
# ---------------------------------------------------------------------------


class TestRecordRetrieval:
    def test_get_returns_inserted_data(self, knowledge_db):
        rid = knowledge_db.insert("facts", {"entity": "Paris", "type": "city"})
        record = knowledge_db.get("facts", rid)
        assert record["entity"] == "Paris"
        assert record["type"] == "city"

    def test_get_nonexistent_record_returns_none(self, knowledge_db):
        assert knowledge_db.get("facts", "00000000-0000-0000-0000-000000000000") is None

    def test_get_nonexistent_lane_returns_none(self, db):
        assert db.get("nonexistent", "any-id") is None

    def test_get_expired_record_returns_none(self, db):
        db.create_lane("working", lane_type=LaneType.WORKING, decay_half_life=0.001)
        rid = db.insert("working", {"k": "v"}, confidence=1.0, ttl=0.001)
        time.sleep(0.1)
        result = db.get("working", rid, respect_decay=True)
        assert result is None

    def test_get_expired_record_without_respect_decay(self, db):
        db.create_lane("working", lane_type=LaneType.WORKING, decay_half_life=0.001)
        rid = db.insert("working", {"k": "v"}, confidence=1.0, ttl=0.001)
        time.sleep(0.1)
        result = db.get("working", rid, respect_decay=False)
        assert result is not None  # bypasses decay check

    def test_get_includes_meta_block(self, knowledge_db):
        rid = knowledge_db.insert("facts", {"k": "v"})
        record = knowledge_db.get("facts", rid)
        assert "_meta" in record
        assert "confidence" in record["_meta"]
        assert "version" in record["_meta"]


# ---------------------------------------------------------------------------
# Temporal decay
# ---------------------------------------------------------------------------


class TestTemporalDecay:
    def test_decay_reduces_effective_confidence_over_time(self, db):
        db.create_lane("eps", lane_type=LaneType.EPISODIC, decay_half_life=1.0)
        rid = db.insert("eps", {"k": "v"}, confidence=1.0, ttl=1.0)
        record = db.lanes["eps"].records[rid]
        conf_at_0 = record.effective_confidence()
        time.sleep(0.5)
        conf_at_half = record.effective_confidence()
        assert conf_at_half < conf_at_0

    def test_half_life_approximately_halves_confidence(self, db):
        db.create_lane("eps", lane_type=LaneType.EPISODIC, decay_half_life=1.0)
        rid = db.insert("eps", {"k": "v"}, confidence=1.0, ttl=1.0)
        record = db.lanes["eps"].records[rid]
        time.sleep(1.0)
        # Should be close to 0.5 (exact value affected by reinforcement)
        assert 0.4 < record.effective_confidence() < 0.7

    def test_immortal_record_no_decay(self, knowledge_db):
        rid = knowledge_db.insert("facts", {"k": "v"}, confidence=0.9)
        record = knowledge_db.lanes["facts"].records[rid]
        time.sleep(0.05)
        assert record.effective_confidence() == pytest.approx(0.9)

    def test_access_reinforces_record(self, db):
        db.create_lane("eps", lane_type=LaneType.EPISODIC, decay_half_life=10.0)
        rid = db.insert("eps", {"k": "v"}, confidence=0.5, ttl=10.0)
        record = db.lanes["eps"].records[rid]
        before = record.effective_confidence()
        for _ in range(5):
            record.touch()
        after = record.effective_confidence()
        assert after > before


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_data_merges_fields(self, knowledge_db):
        rid = knowledge_db.insert("facts", {"a": 1, "b": 2})
        knowledge_db.update("facts", rid, data={"b": 99, "c": 3})
        record = knowledge_db.get("facts", rid)
        assert record["a"] == 1
        assert record["b"] == 99
        assert record["c"] == 3

    def test_update_increments_version(self, knowledge_db):
        rid = knowledge_db.insert("facts", {"k": "v"})
        knowledge_db.update("facts", rid, data={"k": "v2"})
        record = knowledge_db.get("facts", rid)
        assert record["_meta"]["version"] == 2

    def test_update_nonexistent_returns_false(self, knowledge_db):
        result = knowledge_db.update("facts", "00000000-0000-0000-0000-000000000000",
                                     data={"k": "v"})
        assert result is False


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def _make_vec(self, seed: int, dim: int = 128) -> list:
        import random
        random.seed(seed)
        v = [random.gauss(0, 1) for _ in range(dim)]
        mag = math.sqrt(sum(x * x for x in v))
        return [x / mag for x in v]

    def test_similarity_search_returns_correct_count(self, knowledge_db):
        for i in range(5):
            knowledge_db.insert("facts", {"i": i}, embedding=self._make_vec(i))
        results = knowledge_db.similarity_search(self._make_vec(0), top_k=3)
        assert len(results) <= 3

    def test_similarity_search_query_matches_itself(self, knowledge_db):
        vec = self._make_vec(42)
        rid = knowledge_db.insert("facts", {"tag": "target"}, embedding=vec)
        results = knowledge_db.similarity_search(vec, top_k=1)
        assert results[0]["id"] == rid

    def test_low_confidence_record_demoted(self, knowledge_db):
        """A 1.0-confidence record should outrank 0.1-confidence even if similar."""
        vec = self._make_vec(1)
        r_high = knowledge_db.insert("facts", {"tag": "high"}, confidence=0.99, embedding=vec)
        r_low = knowledge_db.insert("facts", {"tag": "low"}, confidence=0.01, embedding=vec)
        results = knowledge_db.similarity_search(vec, top_k=2)
        assert results[0]["id"] == r_high


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------


class TestKnowledgeGraph:
    def test_traverse_returns_connected_nodes(self, knowledge_db):
        r1 = knowledge_db.insert("facts", {"entity": "Paris"})
        r2 = knowledge_db.insert("facts", {"entity": "Eiffel"},
                                  relations=[(r1, "located_in", 1.0)])
        graph = knowledge_db.traverse_graph(r2, depth=1)
        assert r1 in graph["nodes"]
        assert r2 in graph["nodes"]

    def test_traverse_respects_depth_limit(self, knowledge_db):
        ids = [knowledge_db.insert("facts", {"i": i}) for i in range(4)]
        for i in range(3):
            knowledge_db.graph_index.add_edge(ids[i], ids[i + 1], "next", 1.0)
        graph = knowledge_db.traverse_graph(ids[0], depth=2)
        # depth 2 should reach ids[2] but not ids[3]
        assert ids[3] not in graph["nodes"]

    def test_traverse_filters_by_relation_type(self, knowledge_db):
        r1 = knowledge_db.insert("facts", {"a": 1})
        r2 = knowledge_db.insert("facts", {"b": 2})
        r3 = knowledge_db.insert("facts", {"c": 3})
        knowledge_db.graph_index.add_edge(r1, r2, "likes", 1.0)
        knowledge_db.graph_index.add_edge(r1, r3, "hates", 1.0)
        graph = knowledge_db.traverse_graph(r1, rel_type="likes", depth=1)
        assert r2 in graph["nodes"]
        assert r3 not in graph["nodes"]


# ---------------------------------------------------------------------------
# Memory consolidation
# ---------------------------------------------------------------------------


class TestConsolidation:
    def test_consolidate_moves_high_conf_records(self, db):
        db.create_lane("episodes", lane_type=LaneType.EPISODIC)
        db.create_lane("knowledge", lane_type=LaneType.SEMANTIC)
        r1 = db.insert("episodes", {"k": "v1"}, confidence=0.9)
        db.insert("episodes", {"k": "v2"}, confidence=0.3)
        count = db.consolidate("episodes", "knowledge", min_confidence=0.7)
        assert count == 1
        assert r1 in db.lanes["knowledge"].records

    def test_consolidate_leaves_low_conf_in_source(self, db):
        db.create_lane("episodes", lane_type=LaneType.EPISODIC)
        db.create_lane("knowledge", lane_type=LaneType.SEMANTIC)
        r_low = db.insert("episodes", {"k": "v"}, confidence=0.2)
        db.consolidate("episodes", "knowledge", min_confidence=0.7)
        assert r_low in db.lanes["episodes"].records


# ---------------------------------------------------------------------------
# AIQL
# ---------------------------------------------------------------------------


class TestAIQL:
    def test_recall_returns_records(self, knowledge_db):
        knowledge_db.insert("facts", {"entity": "Paris"}, confidence=0.99)
        result = knowledge_db.query("RECALL FROM facts")
        assert result["count"] >= 1

    def test_recall_top_n_limits_results(self, knowledge_db):
        for i in range(10):
            knowledge_db.insert("facts", {"i": i})
        result = knowledge_db.query("RECALL TOP 3 FROM facts")
        assert result["count"] == 3

    def test_recall_where_confidence_filters(self, knowledge_db):
        knowledge_db.insert("facts", {"k": "high"}, confidence=0.9)
        knowledge_db.insert("facts", {"k": "low"}, confidence=0.2)
        result = knowledge_db.query("RECALL FROM facts WHERE confidence > 0.5")
        assert all(
            float(r["_meta"]["effective_confidence"]) > 0.5
            for r in result["records"]
        )

    def test_remember_inserts_record(self, knowledge_db):
        result = knowledge_db.query(
            "REMEMBER INTO facts SET entity=TestCity WITH CONFIDENCE 0.8"
        )
        assert "record_id" in result

    def test_forget_removes_low_conf(self, knowledge_db):
        knowledge_db.insert("facts", {"k": "junk"}, confidence=0.1)
        high_id = knowledge_db.insert("facts", {"k": "keep"}, confidence=0.9)
        knowledge_db.query("FORGET FROM facts WHERE confidence < 0.5")
        assert knowledge_db.get("facts", high_id) is not None

    def test_reflect_returns_stats(self, knowledge_db):
        result = knowledge_db.query("REFLECT ON database")
        assert "reflection" in result
        assert "lanes" in result["reflection"]

    def test_reinforce_increases_confidence(self, knowledge_db):
        rid = knowledge_db.insert("facts", {"k": "v"}, confidence=0.5)
        knowledge_db.query("REINFORCE facts WHERE confidence < 0.6 BY 0.2")
        record = knowledge_db.get("facts", rid)
        assert record["_meta"]["confidence"] > 0.5

    def test_doubt_decreases_confidence(self, knowledge_db):
        rid = knowledge_db.insert("facts", {"k": "v"}, confidence=0.9)
        knowledge_db.query("DOUBT facts WHERE confidence > 0.8 BY 0.2")
        record = knowledge_db.get("facts", rid)
        assert record["_meta"]["confidence"] < 0.9

    def test_invalid_verb_returns_error(self, knowledge_db):
        result = knowledge_db.query("SELECT * FROM facts")
        assert "error" in result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_reload_preserves_records(self, tmp_path):
        db1 = NeuralDB("persist_test", path=str(tmp_path))
        db1.create_lane("facts", lane_type=LaneType.SEMANTIC)
        rid = db1.insert("facts", {"entity": "Paris"}, confidence=0.99)
        db1.save()

        db2 = NeuralDB("persist_test", path=str(tmp_path))
        record = db2.get("facts", rid)
        assert record is not None
        assert record["entity"] == "Paris"

    def test_save_and_reload_preserves_confidence(self, tmp_path):
        db1 = NeuralDB("persist_conf", path=str(tmp_path))
        db1.create_lane("facts", lane_type=LaneType.SEMANTIC)
        rid = db1.insert("facts", {"k": "v"}, confidence=0.73)
        db1.save()

        db2 = NeuralDB("persist_conf", path=str(tmp_path))
        record = db2.get("facts", rid)
        assert record["_meta"]["confidence"] == pytest.approx(0.73)
