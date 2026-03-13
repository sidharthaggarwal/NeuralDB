"""
VectorIndex — Cosine similarity search with confidence weighting.
Pure numpy implementation; no external vector DB needed.
In production, this would swap to HNSW or FAISS.
"""

import math
from typing import List, Tuple, Optional, Dict


class VectorIndex:
    """
    Flat vector index with cosine similarity.
    Tracks lane membership for filtered searches.
    """

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions
        # {record_id: (vector, lane_name)}
        self._store: Dict[str, Tuple[List[float], str]] = {}

    def add(self, record_id: str, vector: List[float], lane_name: str):
        """Add or update a vector."""
        if len(vector) != self.dimensions:
            # Pad or truncate to fit dimensions
            if len(vector) < self.dimensions:
                vector = vector + [0.0] * (self.dimensions - len(vector))
            else:
                vector = vector[:self.dimensions]
        
        normalized = self._normalize(vector)
        self._store[record_id] = (normalized, lane_name)

    def remove(self, record_id: str):
        """Remove a vector."""
        self._store.pop(record_id, None)

    def search(self, query_vector: List[float], lane_name: str = None,
               top_k: int = 10) -> List[Tuple[str, float, str]]:
        """
        Find top-k similar vectors.
        Returns: list of (record_id, similarity, lane_name)
        """
        if not self._store:
            return []

        # Normalize query
        if len(query_vector) != self.dimensions:
            if len(query_vector) < self.dimensions:
                query_vector = query_vector + [0.0] * (self.dimensions - len(query_vector))
            else:
                query_vector = query_vector[:self.dimensions]

        query_norm = self._normalize(query_vector)

        results = []
        for record_id, (vec, lane) in self._store.items():
            # Filter by lane if specified
            if lane_name and lane != lane_name:
                continue

            similarity = self._cosine_similarity(query_norm, vec)
            results.append((record_id, similarity, lane))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self._store)

    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0:
            return vector
        return [x / magnitude for x in vector]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
