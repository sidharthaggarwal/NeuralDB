"""
MemoryLane — Typed memory collections
Mirrors human cognitive memory architecture.
"""

from enum import Enum
from typing import Dict, Optional


class LaneType(Enum):
    """
    Memory lane types inspired by cognitive science:
    
    EPISODIC    — Autobiographical events (conversations, interactions, logs)
                  High temporal sensitivity; decays unless revisited
    
    SEMANTIC    — General world knowledge (facts, entities, concepts)
                  Stable; low decay; high confidence threshold
    
    WORKING     — Active context (current task state, session data)
                  Very short TTL; ephemeral by design
    
    PROCEDURAL  — How-to knowledge (workflows, rules, action sequences)
                  Confidence reinforced by successful execution
    
    ASSOCIATIVE — Pure relational data (connections between entities)
                  Queried primarily through graph traversal
    
    SENSORY     — Raw input representations (embeddings of raw data)
                  Very high decay; only persistent via consolidation
    """
    EPISODIC    = "episodic"
    SEMANTIC    = "semantic"
    WORKING     = "working"
    PROCEDURAL  = "procedural"
    ASSOCIATIVE = "associative"
    SENSORY     = "sensory"


# Default decay half-lives (seconds) per lane type
DEFAULT_HALF_LIVES = {
    LaneType.EPISODIC:    86400 * 7,   # 1 week
    LaneType.SEMANTIC:    None,         # No decay (permanent facts)
    LaneType.WORKING:     3600,         # 1 hour
    LaneType.PROCEDURAL:  None,         # No decay
    LaneType.ASSOCIATIVE: None,         # No decay
    LaneType.SENSORY:     300,          # 5 minutes
}


class MemoryLane:
    """
    A typed, schema-optional collection of NeuralRecords.
    Think of it as a table that knows what kind of memory it holds.
    """

    def __init__(self, name: str, lane_type: LaneType = LaneType.SEMANTIC,
                 decay_half_life: float = None, schema: dict = None):
        
        self.name = name
        self.lane_type = lane_type
        self.schema = schema  # optional field type hints
        
        # Auto-assign default half-life based on type
        self.decay_half_life = (decay_half_life 
                                if decay_half_life is not None 
                                else DEFAULT_HALF_LIVES.get(lane_type))
        
        # Records: {id -> NeuralRecord}
        self.records: Dict[str, any] = {}

        # Lane-level metadata
        self.created_at = None
        self.total_inserted = 0

        import time
        self.created_at = time.time()

    def count(self, active_only: bool = True) -> int:
        if active_only:
            return sum(1 for r in self.records.values() if not r.is_expired())
        return len(self.records)

    def __repr__(self):
        return (f"MemoryLane(name='{self.name}', "
                f"type={self.lane_type.value}, "
                f"records={len(self.records)})")
