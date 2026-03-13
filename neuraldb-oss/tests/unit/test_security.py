"""
tests/unit/test_security.py
============================
Tests for the security layer: auth, rate limiting, and input validation.
"""

from __future__ import annotations

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from neuraldb.security.validation import (
    sanitize_name,
    sanitize_aiql,
    sanitize_record_data,
    sanitize_embedding,
    sanitize_confidence,
    sanitize_relations,
)
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# sanitize_name
# ---------------------------------------------------------------------------


class TestSanitizeName:
    def test_valid_alphanumeric(self):
        assert sanitize_name("mydb") == "mydb"

    def test_valid_with_underscore_hyphen(self):
        assert sanitize_name("my-db_v2") == "my-db_v2"

    def test_empty_raises(self):
        with pytest.raises(HTTPException):
            sanitize_name("")

    def test_space_raises(self):
        with pytest.raises(HTTPException):
            sanitize_name("bad name")

    def test_path_traversal_raises(self):
        with pytest.raises(HTTPException):
            sanitize_name("../../etc/passwd")

    def test_null_byte_raises(self):
        with pytest.raises(HTTPException):
            sanitize_name("bad\x00name")

    def test_too_long_raises(self):
        with pytest.raises(HTTPException):
            sanitize_name("a" * 65)

    def test_sql_injection_raises(self):
        with pytest.raises(HTTPException):
            sanitize_name("name; DROP TABLE")


# ---------------------------------------------------------------------------
# sanitize_aiql
# ---------------------------------------------------------------------------


class TestSanitizeAIQL:
    def test_valid_recall(self):
        q = "RECALL TOP 5 FROM knowledge WHERE confidence > 0.8"
        assert sanitize_aiql(q) == q

    def test_empty_raises(self):
        with pytest.raises(HTTPException):
            sanitize_aiql("")

    def test_too_long_raises(self):
        with pytest.raises(HTTPException):
            sanitize_aiql("RECALL " + "x" * 5000)

    def test_sql_comment_raises(self):
        with pytest.raises(HTTPException):
            sanitize_aiql("RECALL FROM facts -- drop table")

    def test_stacked_semicolons_raises(self):
        with pytest.raises(HTTPException):
            sanitize_aiql("RECALL FROM facts;; DROP TABLE facts")

    def test_block_comment_raises(self):
        with pytest.raises(HTTPException):
            sanitize_aiql("RECALL /* admin */ FROM facts")


# ---------------------------------------------------------------------------
# sanitize_record_data
# ---------------------------------------------------------------------------


class TestSanitizeRecordData:
    def test_valid_data_passthrough(self):
        data = {"entity": "Paris", "pop": 2161000}
        result = sanitize_record_data(data)
        assert result == data

    def test_empty_dict_raises(self):
        with pytest.raises(HTTPException):
            sanitize_record_data({})

    def test_non_dict_raises(self):
        with pytest.raises(HTTPException):
            sanitize_record_data("not a dict")  # type: ignore[arg-type]

    def test_invalid_key_raises(self):
        with pytest.raises(HTTPException):
            sanitize_record_data({"bad key!": "value"})

    def test_null_byte_in_value_stripped(self):
        result = sanitize_record_data({"k": "val\x00ue"})
        assert "\x00" not in result["k"]

    def test_string_normalized_to_nfc(self):
        # Café in NFD decomposed form vs NFC
        nfd = "Cafe\u0301"   # e + combining accent
        nfc = "Caf\xe9"      # é precomposed
        result = sanitize_record_data({"name": nfd})
        assert result["name"] == nfc

    def test_too_many_keys_raises(self):
        big = {f"k{i}": i for i in range(200)}
        with pytest.raises(HTTPException):
            sanitize_record_data(big)

    def test_nested_dict_allowed(self):
        data = {"meta": {"source": "wiki", "year": 2024}}
        result = sanitize_record_data(data)
        assert result["meta"]["source"] == "wiki"

    def test_deep_nesting_raises(self):
        deep = {"a": {"b": {"c": {"d": {"e": "too deep"}}}}}
        with pytest.raises(HTTPException):
            sanitize_record_data(deep)


# ---------------------------------------------------------------------------
# sanitize_embedding
# ---------------------------------------------------------------------------


class TestSanitizeEmbedding:
    def test_none_returns_none(self):
        assert sanitize_embedding(None) is None

    def test_valid_vector(self):
        v = [0.1] * 128
        result = sanitize_embedding(v)
        assert len(result) == 128

    def test_empty_vector_raises(self):
        with pytest.raises(HTTPException):
            sanitize_embedding([])

    def test_nan_raises(self):
        with pytest.raises(HTTPException):
            sanitize_embedding([float("nan")] * 10)

    def test_wrong_dim_raises(self):
        with pytest.raises(HTTPException):
            sanitize_embedding([0.1] * 64, expected_dim=128)

    def test_non_numeric_raises(self):
        with pytest.raises(HTTPException):
            sanitize_embedding(["not", "a", "float"])  # type: ignore[arg-type]

    def test_int_values_coerced_to_float(self):
        result = sanitize_embedding([1, 2, 3])
        assert all(isinstance(x, float) for x in result)


# ---------------------------------------------------------------------------
# sanitize_confidence
# ---------------------------------------------------------------------------


class TestSanitizeConfidence:
    def test_zero_point_five(self):
        assert sanitize_confidence(0.5) == pytest.approx(0.5)

    def test_above_one_clamped(self):
        assert sanitize_confidence(1.5) == 1.0

    def test_below_zero_clamped(self):
        assert sanitize_confidence(-0.1) == 0.0

    def test_nan_raises(self):
        with pytest.raises(HTTPException):
            sanitize_confidence(float("nan"))

    def test_inf_raises(self):
        with pytest.raises(HTTPException):
            sanitize_confidence(float("inf"))


# ---------------------------------------------------------------------------
# sanitize_relations
# ---------------------------------------------------------------------------


class TestSanitizeRelations:
    def test_none_returns_none(self):
        assert sanitize_relations(None) is None

    def test_valid_relations(self):
        rels = [["some-uuid", "located_in", 0.9]]
        result = sanitize_relations(rels)
        assert result[0] == ("some-uuid", "located_in", 0.9)

    def test_wrong_length_raises(self):
        with pytest.raises(HTTPException):
            sanitize_relations([["id", "rel"]])  # missing weight

    def test_empty_target_id_raises(self):
        with pytest.raises(HTTPException):
            sanitize_relations([["", "rel", 1.0]])

    def test_too_many_raises(self):
        rels = [["id", "rel", 1.0]] * 100
        with pytest.raises(HTTPException):
            sanitize_relations(rels)
