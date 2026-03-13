"""
AIQL Query Executor
Executes parsed AIQL ASTs against the NeuralDB engine.
"""

import time
from typing import Any, Dict, List


class QueryExecutor:
    def __init__(self, db):
        self.db = db

    def execute(self, ast: dict) -> dict:
        verb = ast.get("verb")
        start = time.time()

        try:
            if verb == "RECALL":
                result = self._exec_recall(ast)
            elif verb == "REMEMBER":
                result = self._exec_remember(ast)
            elif verb == "FORGET":
                result = self._exec_forget(ast)
            elif verb == "ASSOCIATE":
                result = self._exec_associate(ast)
            elif verb == "TRAVERSE":
                result = self._exec_traverse(ast)
            elif verb == "CONSOLIDATE":
                result = self._exec_consolidate(ast)
            elif verb in ("REINFORCE", "DOUBT"):
                result = self._exec_adjust(ast)
            elif verb == "REFLECT":
                result = self._exec_reflect(ast)
            else:
                return {"error": f"Unknown verb: {verb}"}
        except Exception as e:
            return {"error": str(e), "verb": verb}

        elapsed = round((time.time() - start) * 1000, 2)
        result["_query_time_ms"] = elapsed
        result["_verb"] = verb
        return result

    def _exec_recall(self, ast: dict) -> dict:
        lane_name = ast["lane"]
        lane = self.db.lanes.get(lane_name)
        if not lane:
            return {"records": [], "count": 0, "lane": lane_name,
                    "message": f"Lane '{lane_name}' not found"}

        records = []
        for record_id, record in lane.records.items():
            if record.is_expired():
                continue

            eff_conf = record.effective_confidence()

            # Apply WHERE conditions
            if ast["conditions"]:
                if not self._apply_conditions(record, ast["conditions"], ast.get("fuzzy", False)):
                    continue

            records.append({
                **record.to_dict(include_meta=True),
                "_effective_confidence": round(eff_conf, 4),
                "_age_seconds": round(time.time() - record.created_at, 1),
            })

        # Sort
        order_by = ast.get("order_by", "confidence")
        desc = ast.get("order_dir", "DESC") == "DESC"

        if order_by == "confidence":
            records.sort(key=lambda r: r["_effective_confidence"], reverse=desc)
        elif order_by == "age":
            records.sort(key=lambda r: r["_age_seconds"], reverse=not desc)

        # Apply TOP limit
        top = ast.get("top")
        if top:
            records = records[:top]

        # Touch records (reinforce access)
        for r in records:
            if r["id"] in lane.records:
                lane.records[r["id"]].touch()

        self.db.stats["total_reads"] += 1
        return {"records": records, "count": len(records), "lane": lane_name}

    def _exec_remember(self, ast: dict) -> dict:
        record_id = self.db.insert(
            lane_name=ast["lane"],
            data=ast["data"],
            confidence=ast.get("confidence", 1.0),
            embedding=ast.get("embedding"),
        )
        return {"record_id": record_id, "lane": ast["lane"],
                "message": f"Memory stored in '{ast['lane']}'"}

    def _exec_forget(self, ast: dict) -> dict:
        lane = self.db.lanes.get(ast["lane"])
        if not lane:
            return {"deleted": 0, "message": f"Lane '{ast['lane']}' not found"}

        to_delete = []
        for record_id, record in lane.records.items():
            if not ast["conditions"] or self._apply_conditions(record, ast["conditions"]):
                to_delete.append(record_id)

        for rid in to_delete:
            del lane.records[rid]
            self.db.vector_index.remove(rid)

        return {"deleted": len(to_delete), "lane": ast["lane"]}

    def _exec_associate(self, ast: dict) -> dict:
        if not ast.get("vector"):
            return {"error": "ASSOCIATE requires WITH VECTOR [...]"}

        results = self.db.similarity_search(
            query_vector=ast["vector"],
            lane_name=ast.get("lane"),
            top_k=ast.get("top", 10),
            min_confidence=ast.get("min_confidence", 0.0),
            min_similarity=ast.get("min_similarity", 0.0),
        )
        return {"records": results, "count": len(results)}

    def _exec_traverse(self, ast: dict) -> dict:
        result = self.db.traverse_graph(
            start_id=ast["node_id"],
            rel_type=ast.get("rel_type"),
            depth=ast.get("depth", 2),
            min_weight=ast.get("min_weight", 0.0),
        )
        return {"graph": result}

    def _exec_consolidate(self, ast: dict) -> dict:
        count = self.db.consolidate(
            source_lane=ast["source"],
            target_lane=ast["target"],
            min_confidence=ast.get("min_confidence", 0.7),
        )
        return {"consolidated": count,
                "from": ast["source"], "to": ast["target"]}

    def _exec_adjust(self, ast: dict) -> dict:
        lane = self.db.lanes.get(ast["lane"])
        if not lane:
            return {"adjusted": 0}

        verb = ast["verb"]
        amount = ast.get("amount", 0.1)
        adjusted = 0

        for record_id, record in lane.records.items():
            if not ast["conditions"] or self._apply_conditions(record, ast["conditions"]):
                if verb == "REINFORCE":
                    record.confidence = min(1.0, record.confidence + amount)
                else:  # DOUBT
                    record.confidence = max(0.0, record.confidence - amount)
                adjusted += 1

        return {"adjusted": adjusted, "lane": ast["lane"], "verb": verb}

    def _exec_reflect(self, ast: dict) -> dict:
        target = ast.get("target", "database")
        if target == "database":
            return {"reflection": self.db.stats_report()}
        else:
            lane = self.db.lanes.get(target)
            if not lane:
                return {"error": f"Lane '{target}' not found"}
            records = list(lane.records.values())
            active = [r for r in records if not r.is_expired()]
            return {
                "reflection": {
                    "lane": target,
                    "type": lane.lane_type.value,
                    "total": len(records),
                    "active": len(active),
                    "expired": len(records) - len(active),
                    "avg_confidence": round(
                        sum(r.effective_confidence() for r in active) / max(len(active), 1), 4
                    ),
                    "decay_half_life": lane.decay_half_life,
                }
            }

    def _apply_conditions(self, record, conditions: list, fuzzy: bool = False) -> bool:
        """Check if a record satisfies all conditions."""
        for cond in conditions:
            field = cond["field"]
            op = cond["op"]
            target = cond["value"]

            # Special fields
            if field.lower() == "confidence":
                val = record.effective_confidence()
            elif field.lower() == "age":
                val = time.time() - record.created_at
            elif field.lower() == "version":
                val = record.version
            else:
                val = record.data.get(field)
                if val is None:
                    return False

            if not self._compare(val, op, target, fuzzy):
                return False
        return True

    def _compare(self, val, op: str, target, fuzzy: bool = False) -> bool:
        try:
            if op == "=":
                if fuzzy and isinstance(val, str) and isinstance(target, str):
                    return target.lower() in val.lower()
                return val == target
            elif op == "!=":
                return val != target
            elif op == ">":
                return float(val) > float(target)
            elif op == ">=":
                return float(val) >= float(target)
            elif op == "<":
                return float(val) < float(target)
            elif op == "<=":
                return float(val) <= float(target)
            elif op == "CONTAINS":
                return str(target).lower() in str(val).lower()
            elif op == "LIKE":
                return str(target).lower() in str(val).lower()
        except (TypeError, ValueError):
            return False
        return False
