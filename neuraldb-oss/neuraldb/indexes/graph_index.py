"""
GraphIndex — Weighted, typed knowledge graph.
Supports traversal, path finding, and subgraph extraction.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple


class Edge:
    def __init__(self, source: str, target: str, rel_type: str, weight: float = 1.0):
        self.source = source
        self.target = target
        self.rel_type = rel_type
        self.weight = weight

    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "type": self.rel_type,
            "weight": self.weight
        }


class GraphIndex:
    """
    Directed weighted graph with typed edges.
    Used for entity relationships, concept networks, causal chains.
    """

    def __init__(self):
        # adjacency: {node_id: [Edge, ...]}
        self._adj: Dict[str, List[Edge]] = defaultdict(list)
        # reverse adjacency for incoming edges
        self._rev: Dict[str, List[Edge]] = defaultdict(list)
        self._edge_count = 0

    def add_edge(self, source: str, target: str, rel_type: str, weight: float = 1.0):
        """Add a directed typed edge."""
        edge = Edge(source, target, rel_type, weight)
        self._adj[source].append(edge)
        self._rev[target].append(edge)
        self._edge_count += 1

    def remove_node(self, node_id: str):
        """Remove a node and all its edges."""
        # Remove outgoing
        if node_id in self._adj:
            for edge in self._adj[node_id]:
                self._rev[edge.target] = [
                    e for e in self._rev[edge.target] if e.source != node_id
                ]
            del self._adj[node_id]

        # Remove incoming
        if node_id in self._rev:
            for edge in self._rev[node_id]:
                self._adj[edge.source] = [
                    e for e in self._adj[edge.source] if e.target != node_id
                ]
            del self._rev[node_id]

    def traverse(self, start_id: str, rel_type: str = None,
                 depth: int = 2, min_weight: float = 0.0) -> dict:
        """
        BFS traversal from start node.
        Returns subgraph: {nodes: set, edges: list}
        """
        visited: Set[str] = set()
        nodes: Set[str] = {start_id}
        edges: List[dict] = []
        queue = deque([(start_id, 0)])

        while queue:
            node_id, current_depth = queue.popleft()

            if node_id in visited or current_depth >= depth:
                continue

            visited.add(node_id)

            for edge in self._adj.get(node_id, []):
                if edge.weight < min_weight:
                    continue
                if rel_type and edge.rel_type != rel_type:
                    continue

                nodes.add(edge.target)
                edges.append(edge.to_dict())
                queue.append((edge.target, current_depth + 1))

        return {
            "start": start_id,
            "nodes": list(nodes),
            "edges": edges,
            "depth": depth,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    def neighbors(self, node_id: str, rel_type: str = None,
                  direction: str = "out") -> List[dict]:
        """Get immediate neighbors of a node."""
        if direction == "out":
            edges = self._adj.get(node_id, [])
        elif direction == "in":
            edges = self._rev.get(node_id, [])
        else:
            edges = self._adj.get(node_id, []) + self._rev.get(node_id, [])

        if rel_type:
            edges = [e for e in edges if e.rel_type == rel_type]

        return [e.to_dict() for e in edges]

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """BFS shortest path between two nodes."""
        if source == target:
            return [source]

        visited = {source}
        queue = deque([(source, [source])])

        while queue:
            node, path = queue.popleft()
            for edge in self._adj.get(node, []):
                if edge.target == target:
                    return path + [edge.target]
                if edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge.target]))

        return None  # No path found

    def edge_count(self) -> int:
        return self._edge_count

    def node_count(self) -> int:
        all_nodes = set(self._adj.keys()) | set(self._rev.keys())
        return len(all_nodes)
