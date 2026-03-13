"""
NeuralRecord — AI-native data record
Every record in NeuralDB carries confidence, temporal decay, and version history.
"""

import time
import uuid
import math
from typing import Any, Dict, List, Optional


class NeuralRecord:
    """
    A record that behaves like a memory, not just a row.
    
    Key innovations over traditional DB records:
    - confidence: epistemological certainty (0=unknown, 1=certain)
    - temporal_decay: relevance fades over time unless reinforced
    - version history: all mutations tracked
    - provenance: source/origin tracking
    - embedding: vector representation for semantic search
    """

    def __init__(self, data: dict, confidence: float = 1.0,
                 embedding: List[float] = None, ttl: float = None,
                 provenance: str = None):
        
        self.id = str(uuid.uuid4())
        self.data = data
        
        # AI-native metadata
        self.confidence = max(0.0, min(1.0, confidence))
        self.embedding = embedding
        self.provenance = provenance  # source of this knowledge
        
        # Temporal
        self.created_at = time.time()
        self.updated_at = time.time()
        self.accessed_at = time.time()
        self.access_count = 0
        self.ttl = ttl  # half-life in seconds, None = immortal
        
        # Versioning
        self.version = 1
        self.history = []  # list of (timestamp, change_summary)
        
        # Reinforcement tracking (accessing boosts recall strength)
        self.reinforcement = 1.0

    def effective_confidence(self) -> float:
        """
        True confidence after applying temporal decay.
        Uses exponential decay: C_eff = C_base * e^(-t/half_life)
        Reinforcement from accesses partially counteracts decay.
        """
        if self.ttl is None:
            return self.confidence
        
        age = time.time() - self.updated_at
        # Decay constant from half-life
        decay_constant = math.log(2) / self.ttl
        temporal_factor = math.exp(-decay_constant * age)
        
        # Reinforcement bonus (each access adds a small boost)
        reinforcement_bonus = math.log1p(self.access_count) * 0.05
        
        return min(1.0, self.confidence * temporal_factor + reinforcement_bonus)

    def is_expired(self, threshold: float = 0.01) -> bool:
        """
        A record is 'expired' when its effective confidence drops below threshold.
        This is fundamentally different from TTL deletion — it's semantic expiry.
        """
        if self.ttl is None:
            return False
        return self.effective_confidence() < threshold

    def touch(self):
        """
        Record an access — reinforces memory (mimics spaced repetition).
        Each access slightly increases retention.
        """
        self.accessed_at = time.time()
        self.access_count += 1
        self.reinforcement = min(3.0, self.reinforcement + 0.1)

    def match_score(self, query: dict, fuzzy: bool = True) -> float:
        """
        Compute how well this record matches a partial query dict.
        Returns 0.0–1.0 match score.
        """
        if not query:
            return 0.0

        total_weight = 0.0
        matched_weight = 0.0

        for key, query_val in query.items():
            if key not in self.data:
                total_weight += 1.0
                continue

            record_val = self.data[key]
            total_weight += 1.0

            if record_val == query_val:
                matched_weight += 1.0
            elif fuzzy:
                # Fuzzy string matching
                if isinstance(query_val, str) and isinstance(record_val, str):
                    q = query_val.lower()
                    r = record_val.lower()
                    if q in r or r in q:
                        matched_weight += 0.7
                    elif any(word in r for word in q.split()):
                        matched_weight += 0.4
                # Numeric proximity
                elif isinstance(query_val, (int, float)) and isinstance(record_val, (int, float)):
                    if record_val != 0:
                        ratio = min(query_val, record_val) / max(query_val, record_val)
                        matched_weight += ratio

        if total_weight == 0:
            return 0.0

        return matched_weight / total_weight

    def to_dict(self, include_meta: bool = False) -> dict:
        """Serialize record to dictionary."""
        result = {
            "id": self.id,
            **self.data,
        }
        if include_meta:
            result["_meta"] = {
                "confidence": round(self.confidence, 4),
                "effective_confidence": round(self.effective_confidence(), 4),
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "version": self.version,
                "access_count": self.access_count,
                "has_embedding": self.embedding is not None,
                "provenance": self.provenance,
                "expired": self.is_expired(),
            }
        return result

    def __repr__(self):
        return (f"NeuralRecord(id={self.id[:8]}..., "
                f"conf={self.confidence:.2f}, "
                f"eff_conf={self.effective_confidence():.2f})")
