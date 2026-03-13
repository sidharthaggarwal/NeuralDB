"""
NeuralDB Core Engine
====================
A database engine purpose-built for AI applications.

Key innovations:
  1. CONFIDENCE-AWARE storage: every record carries a certainty score
  2. SEMANTIC memory lanes: episodic, semantic, working, procedural
  3. VECTOR-NATIVE: embeddings are first-class citizens, not add-ons
  4. TEMPORAL DECAY: data relevance degrades by configurable half-life
  5. RELATION GRAPHS: entities link with typed, weighted edges
  6. AIQL: AI-oriented query language with fuzzy + probabilistic ops
"""

import json
import math
import time
import uuid
import pickle
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .record import NeuralRecord
from .memory_lane import MemoryLane, LaneType
from ..indexes.vector_index import VectorIndex
from ..indexes.graph_index import GraphIndex
from ..query.parser import AIQLParser
from ..query.executor import QueryExecutor


class NeuralDB:
    """
    NeuralDB — The AI-Native Database Engine

    Designed around how AI systems actually think:
    - Multiple memory types (episodic, semantic, working, procedural)
    - Confidence + temporal decay on all records
    - Native vector similarity
    - Knowledge graph traversal
    - Fuzzy, probabilistic queries via AIQL
    """

    VERSION = "1.0.0"
    ENGINE = "NeuralDB"

    def __init__(self, name: str, path: str = "./neuraldb_data", config: dict = None):
        self.name = name
        self.path = os.path.join(path, name)
        self.config = config or {}
        os.makedirs(self.path, exist_ok=True)

        # Core storage: lanes (like tables, but memory-typed)
        self.lanes: Dict[str, MemoryLane] = {}

        # Indexes
        self.vector_index = VectorIndex(dimensions=self.config.get("vector_dim", 128))
        self.graph_index = GraphIndex()

        # Query engine
        self.parser = AIQLParser()
        self.executor = QueryExecutor(self)

        # Stats
        self.stats = {
            "created_at": time.time(),
            "total_writes": 0,
            "total_reads": 0,
            "total_queries": 0,
        }

        # Load existing data if present
        self._load()
        print(f"[NeuralDB v{self.VERSION}] Database '{name}' initialized ✓")

    # ─── Lane Management (like CREATE TABLE but memory-aware) ───────────────

    def create_lane(self, name: str, lane_type: LaneType = LaneType.SEMANTIC,
                    decay_half_life: float = None, schema: dict = None) -> MemoryLane:
        """
        Create a memory lane (typed collection).
        
        Lane types mirror cognitive memory systems:
          EPISODIC    — Events with timestamps (conversations, logs)
          SEMANTIC    — Facts and knowledge (entities, concepts)
          WORKING     — Short-lived context (active task state)
          PROCEDURAL  — Action patterns (workflows, steps)
          ASSOCIATIVE — Pure relationship data
        """
        if name in self.lanes:
            raise ValueError(f"Lane '{name}' already exists")

        lane = MemoryLane(name=name, lane_type=lane_type,
                         decay_half_life=decay_half_life, schema=schema)
        self.lanes[name] = lane
        print(f"  ✦ Lane '{name}' created [{lane_type.value}]")
        return lane

    def get_lane(self, name: str) -> MemoryLane:
        if name not in self.lanes:
            raise KeyError(f"Lane '{name}' does not exist")
        return self.lanes[name]

    # ─── Write Operations ────────────────────────────────────────────────────

    def insert(self, lane_name: str, data: dict,
               confidence: float = 1.0,
               embedding: List[float] = None,
               relations: List[Tuple[str, str, float]] = None,
               ttl: float = None,
               provenance: str = None) -> str:
        """
        Insert a record with AI-native metadata.
        
        Args:
            lane_name:   Target memory lane
            data:        The payload dict
            confidence:  Certainty score 0.0–1.0 (default 1.0)
            embedding:   Optional vector representation
            relations:   List of (target_id, relation_type, weight) tuples
            ttl:         Time-to-live in seconds (overrides lane decay)
        
        Returns:
            record_id: Unique identifier for this record
        """
        if lane_name not in self.lanes:
            # Auto-create lane if it doesn't exist
            self.create_lane(lane_name)

        lane = self.lanes[lane_name]
        record = NeuralRecord(
            data=data,
            confidence=confidence,
            embedding=embedding,
            ttl=ttl or lane.decay_half_life,
            provenance=provenance
        )

        lane.records[record.id] = record

        # Index embedding if provided
        if embedding:
            self.vector_index.add(record.id, embedding, lane_name)

        # Add graph relations
        if relations:
            for (target_id, rel_type, weight) in relations:
                self.graph_index.add_edge(record.id, target_id, rel_type, weight)

        self.stats["total_writes"] += 1
        self._autosave()
        return record.id

    def update(self, lane_name: str, record_id: str,
               data: dict = None, confidence: float = None,
               embedding: List[float] = None) -> bool:
        """Update a record, optionally adjusting its confidence."""
        lane = self.lanes.get(lane_name)
        if not lane or record_id not in lane.records:
            return False

        record = lane.records[record_id]
        if data:
            record.data.update(data)
        if confidence is not None:
            record.confidence = max(0.0, min(1.0, confidence))
        if embedding:
            record.embedding = embedding
            self.vector_index.add(record_id, embedding, lane_name)

        record.updated_at = time.time()
        record.version += 1
        self._autosave()
        return True

    def decay_confidence(self, lane_name: str, record_id: str, amount: float):
        """
        Explicitly decay confidence of a record (e.g. after conflicting evidence).
        Models Bayesian belief updating.
        """
        lane = self.lanes.get(lane_name)
        if lane and record_id in lane.records:
            r = lane.records[record_id]
            r.confidence = max(0.0, r.confidence - amount)

    # ─── Read Operations ─────────────────────────────────────────────────────

    def get(self, lane_name: str, record_id: str,
            respect_decay: bool = True) -> Optional[dict]:
        """
        Fetch a record by ID.
        If respect_decay=True, returns None for records whose temporal
        relevance has dropped below a threshold.
        """
        lane = self.lanes.get(lane_name)
        if not lane or record_id not in lane.records:
            return None

        record = lane.records[record_id]

        if respect_decay and record.is_expired():
            return None

        self.stats["total_reads"] += 1
        return record.to_dict(include_meta=True)

    def similarity_search(self, query_vector: List[float], lane_name: str = None,
                          top_k: int = 10, min_confidence: float = 0.0,
                          min_similarity: float = 0.0) -> List[dict]:
        """
        Find records semantically similar to a query vector.
        Unique: combines cosine similarity with confidence weighting.
        Final score = similarity * confidence^0.5  (confidence dampens, not dominates)
        """
        results = self.vector_index.search(query_vector, lane_name, top_k * 3)

        enriched = []
        for (record_id, similarity, rec_lane) in results:
            lane = self.lanes.get(rec_lane)
            if not lane or record_id not in lane.records:
                continue

            record = lane.records[record_id]
            if record.is_expired():
                continue

            eff_confidence = record.effective_confidence()
            if eff_confidence < min_confidence:
                continue

            # Confidence-weighted similarity score
            combined_score = similarity * (eff_confidence ** 0.5)

            if combined_score < min_similarity:
                continue

            enriched.append({
                **record.to_dict(include_meta=True),
                "_similarity": round(similarity, 4),
                "_confidence_score": round(eff_confidence, 4),
                "_combined_score": round(combined_score, 4),
                "_lane": rec_lane,
            })

        # Sort by combined score
        enriched.sort(key=lambda x: x["_combined_score"], reverse=True)
        self.stats["total_reads"] += 1
        return enriched[:top_k]

    def traverse_graph(self, start_id: str, relation_type: str = None,
                       depth: int = 2, min_weight: float = 0.0) -> dict:
        """
        Traverse the knowledge graph from a node.
        Returns a subgraph dict with nodes + edges.
        """
        return self.graph_index.traverse(start_id, relation_type, depth, min_weight)

    def associative_recall(self, lane_name: str, partial_data: dict,
                           min_confidence: float = 0.1,
                           fuzzy: bool = True) -> List[dict]:
        """
        Pattern-match records using partial data (like human associative memory).
        With fuzzy=True, partial string matches count.
        """
        lane = self.lanes.get(lane_name)
        if not lane:
            return []

        matches = []
        for record_id, record in lane.records.items():
            if record.is_expired():
                continue
            if record.effective_confidence() < min_confidence:
                continue

            score = record.match_score(partial_data, fuzzy=fuzzy)
            if score > 0:
                matches.append({
                    **record.to_dict(include_meta=True),
                    "_match_score": round(score, 4),
                    "_lane": lane_name,
                })

        matches.sort(key=lambda x: x["_match_score"], reverse=True)
        return matches

    # ─── AIQL Query Interface ────────────────────────────────────────────────

    def query(self, aiql_string: str) -> dict:
        """
        Execute an AIQL (AI Intelligence Query Language) query.
        
        Example queries:
          RECALL FROM knowledge WHERE confidence > 0.8
          REMEMBER INTO episodes (data, confidence) VALUES ({...}, 0.9)
          ASSOCIATE facts WITH embedding NEAR [0.1, 0.3, ...]  TOP 5
          FORGET FROM working WHERE age > 3600
          TRAVERSE graph FROM 'node_id' VIA 'relates_to' DEPTH 3
          CONSOLIDATE episodes INTO knowledge WHERE confidence > 0.7
        """
        ast = self.parser.parse(aiql_string)
        result = self.executor.execute(ast)
        self.stats["total_queries"] += 1
        return result

    # ─── Memory Management ───────────────────────────────────────────────────

    def consolidate(self, source_lane: str, target_lane: str,
                    min_confidence: float = 0.7,
                    merge_duplicates: bool = True) -> int:
        """
        Consolidate memories: move high-confidence records from one lane
        to another (mimics sleep memory consolidation).
        Returns count of records consolidated.
        """
        source = self.lanes.get(source_lane)
        target = self.lanes.get(target_lane)
        if not source or not target:
            return 0

        moved = 0
        to_delete = []
        for record_id, record in source.records.items():
            if record.effective_confidence() >= min_confidence:
                if merge_duplicates:
                    # Check for similar records in target
                    existing = self.associative_recall(target_lane, record.data, fuzzy=False)
                    if existing and existing[0]["_match_score"] > 0.9:
                        # Merge: boost confidence of existing record
                        existing_id = existing[0]["id"]
                        target.records[existing_id].confidence = min(
                            1.0, target.records[existing_id].confidence + 0.1
                        )
                        to_delete.append(record_id)
                        moved += 1
                        continue

                # Move record to target lane
                target.records[record_id] = record
                to_delete.append(record_id)
                moved += 1

        for rid in to_delete:
            del source.records[rid]

        print(f"  ✦ Consolidated {moved} records: '{source_lane}' → '{target_lane}'")
        self._autosave()
        return moved

    def purge_expired(self) -> int:
        """Remove all expired/decayed records across all lanes."""
        count = 0
        for lane in self.lanes.values():
            expired = [rid for rid, r in lane.records.items() if r.is_expired()]
            for rid in expired:
                del lane.records[rid]
                self.vector_index.remove(rid)
            count += len(expired)
        if count:
            print(f"  ✦ Purged {count} expired records")
        self._autosave()
        return count

    # ─── Analytics & Introspection ───────────────────────────────────────────

    def stats_report(self) -> dict:
        """Full database health and statistics report."""
        lane_stats = {}
        for name, lane in self.lanes.items():
            records = list(lane.records.values())
            active = [r for r in records if not r.is_expired()]
            confs = [r.effective_confidence() for r in active]
            lane_stats[name] = {
                "type": lane.lane_type.value,
                "total_records": len(records),
                "active_records": len(active),
                "expired_records": len(records) - len(active),
                "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0,
                "min_confidence": round(min(confs), 4) if confs else 0,
                "max_confidence": round(max(confs), 4) if confs else 0,
            }
        return {
            "engine": self.ENGINE,
            "version": self.VERSION,
            "database": self.name,
            "uptime_seconds": round(time.time() - self.stats["created_at"], 1),
            "total_writes": self.stats["total_writes"],
            "total_reads": self.stats["total_reads"],
            "total_queries": self.stats["total_queries"],
            "total_vectors": self.vector_index.count(),
            "total_edges": self.graph_index.edge_count(),
            "lanes": lane_stats,
        }

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _autosave(self):
        # Save every 10 writes
        if self.stats["total_writes"] % 10 == 0:
            self.save()

    def save(self):
        db_file = os.path.join(self.path, "neuraldb.pkl")
        state = {
            "name": self.name,
            "config": self.config,
            "lanes": self.lanes,
            "vector_index": self.vector_index,
            "graph_index": self.graph_index,
            "stats": self.stats,
        }
        with open(db_file, "wb") as f:
            pickle.dump(state, f)

    def _load(self):
        db_file = os.path.join(self.path, "neuraldb.pkl")
        if os.path.exists(db_file):
            with open(db_file, "rb") as f:
                state = pickle.load(f)
            self.lanes = state["lanes"]
            self.vector_index = state["vector_index"]
            self.graph_index = state["graph_index"]
            self.stats = state["stats"]
            print(f"  ✦ Loaded existing database with {sum(len(l.records) for l in self.lanes.values())} records")
