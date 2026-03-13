"""
AIQL — AI Intelligence Query Language
======================================
A query language designed for how AI systems reason about data.

Core verbs reflect AI memory operations, not SQL CRUD:
  RECALL      — Retrieve memories (like SELECT but confidence-aware)
  REMEMBER    — Store new knowledge (like INSERT but with certainty)
  FORGET      — Remove/decay records (TTL or confidence-based)
  ASSOCIATE   — Semantic/vector search
  TRAVERSE    — Graph walk
  CONSOLIDATE — Move memories between lanes (sleep-like consolidation)
  REINFORCE   — Boost confidence of records
  DOUBT       — Decay confidence of records
  REFLECT     — Introspect on the database itself

Grammar:
  RECALL [TOP n] FROM <lane> [WHERE <conditions>] [ORDER BY confidence|age|score]
  REMEMBER INTO <lane> SET <key>=<val>, ... [WITH CONFIDENCE <0-1>]
  FORGET FROM <lane> WHERE <conditions>
  ASSOCIATE FROM <lane> WITH VECTOR [<floats>] TOP <n>
  TRAVERSE FROM <node_id> VIA <rel_type> DEPTH <n>
  CONSOLIDATE <lane> INTO <lane> WHERE confidence > <n>
  REINFORCE <lane> WHERE <conditions> BY <amount>
  DOUBT <lane> WHERE <conditions> BY <amount>
  REFLECT ON <lane|database>
"""

import re
import json
from typing import Any, Dict, List, Optional


class AIQLToken:
    KEYWORD = "KEYWORD"
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"
    OPERATOR = "OPERATOR"
    VECTOR = "VECTOR"
    DICT = "DICT"
    EOF = "EOF"


class ParseError(Exception):
    pass


class AIQLParser:
    """Parse AIQL strings into AST (Abstract Syntax Tree) dicts."""

    KEYWORDS = {
        "RECALL", "REMEMBER", "FORGET", "ASSOCIATE", "TRAVERSE",
        "CONSOLIDATE", "REINFORCE", "DOUBT", "REFLECT",
        "FROM", "INTO", "SET", "WHERE", "WITH", "TOP", "VIA", "DEPTH",
        "ORDER", "BY", "CONFIDENCE", "AGE", "ASC", "DESC",
        "VECTOR", "NEAR", "GRAPH", "ON", "AND", "OR", "NOT",
        "VALUES", "AS", "LIMIT", "FUZZY", "MIN", "MAX",
    }

    def parse(self, query: str) -> dict:
        """Parse an AIQL query string into an AST dict."""
        query = query.strip()
        if not query:
            raise ParseError("Empty query")

        upper = query.upper().strip()

        if upper.startswith("RECALL"):
            return self._parse_recall(query)
        elif upper.startswith("REMEMBER"):
            return self._parse_remember(query)
        elif upper.startswith("FORGET"):
            return self._parse_forget(query)
        elif upper.startswith("ASSOCIATE"):
            return self._parse_associate(query)
        elif upper.startswith("TRAVERSE"):
            return self._parse_traverse(query)
        elif upper.startswith("CONSOLIDATE"):
            return self._parse_consolidate(query)
        elif upper.startswith("REINFORCE"):
            return self._parse_reinforce(query)
        elif upper.startswith("DOUBT"):
            return self._parse_doubt(query)
        elif upper.startswith("REFLECT"):
            return self._parse_reflect(query)
        else:
            raise ParseError(f"Unknown AIQL verb. Query must start with: "
                           f"RECALL, REMEMBER, FORGET, ASSOCIATE, TRAVERSE, "
                           f"CONSOLIDATE, REINFORCE, DOUBT, or REFLECT")

    def _parse_recall(self, query: str) -> dict:
        """
        RECALL [TOP n] FROM <lane> [WHERE <conditions>]
               [ORDER BY confidence|age|score] [FUZZY]
        """
        ast = {"verb": "RECALL", "top": None, "lane": None,
               "conditions": [], "order_by": "confidence",
               "order_dir": "DESC", "fuzzy": False, "min_confidence": 0.0}

        upper = query.upper()

        # TOP n
        top_m = re.search(r'\bTOP\s+(\d+)\b', upper)
        if top_m:
            ast["top"] = int(top_m.group(1))

        # FROM <lane>
        from_m = re.search(r'\bFROM\s+(\w+)\b', upper)
        if not from_m:
            raise ParseError("RECALL requires FROM <lane>")
        ast["lane"] = query.split()[query.upper().split().index("FROM") + 1]

        # FUZZY flag
        if "FUZZY" in upper:
            ast["fuzzy"] = True

        # WHERE conditions
        where_m = re.search(r'\bWHERE\s+(.+?)(?:\s+ORDER\s+BY|\s+FUZZY|$)',
                            query, re.IGNORECASE)
        if where_m:
            ast["conditions"] = self._parse_conditions(where_m.group(1))

        # ORDER BY
        order_m = re.search(r'\bORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?\b',
                            query, re.IGNORECASE)
        if order_m:
            ast["order_by"] = order_m.group(1).lower()
            if order_m.group(2):
                ast["order_dir"] = order_m.group(2).upper()

        return ast

    def _parse_remember(self, query: str) -> dict:
        """
        REMEMBER INTO <lane> SET key=val, key2=val2 
                  [WITH CONFIDENCE <float>] [PROVENANCE <string>]
        """
        ast = {"verb": "REMEMBER", "lane": None, "data": {},
               "confidence": 1.0, "provenance": None, "ttl": None}

        # INTO <lane>
        into_m = re.search(r'\bINTO\s+(\w+)\b', query, re.IGNORECASE)
        if not into_m:
            raise ParseError("REMEMBER requires INTO <lane>")
        ast["lane"] = into_m.group(1)

        # WITH CONFIDENCE
        conf_m = re.search(r'\bWITH\s+CONFIDENCE\s+([0-9.]+)\b', query, re.IGNORECASE)
        if conf_m:
            ast["confidence"] = float(conf_m.group(1))

        # SET key=val pairs or JSON
        set_m = re.search(r'\bSET\s+(.+?)(?:\s+WITH\s+CONFIDENCE|\s+PROVENANCE|$)',
                          query, re.IGNORECASE | re.DOTALL)
        if set_m:
            set_str = set_m.group(1).strip()
            # Try JSON first
            if set_str.startswith("{"):
                try:
                    ast["data"] = json.loads(set_str)
                except json.JSONDecodeError:
                    ast["data"] = self._parse_kv_pairs(set_str)
            else:
                ast["data"] = self._parse_kv_pairs(set_str)

        return ast

    def _parse_forget(self, query: str) -> dict:
        """FORGET FROM <lane> WHERE <conditions>"""
        ast = {"verb": "FORGET", "lane": None, "conditions": []}

        from_m = re.search(r'\bFROM\s+(\w+)\b', query, re.IGNORECASE)
        if from_m:
            ast["lane"] = from_m.group(1)

        where_m = re.search(r'\bWHERE\s+(.+)$', query, re.IGNORECASE)
        if where_m:
            ast["conditions"] = self._parse_conditions(where_m.group(1))

        return ast

    def _parse_associate(self, query: str) -> dict:
        """ASSOCIATE FROM <lane> WITH VECTOR [floats] TOP n"""
        ast = {"verb": "ASSOCIATE", "lane": None, "vector": None,
               "top": 10, "min_confidence": 0.0, "min_similarity": 0.0}

        from_m = re.search(r'\bFROM\s+(\w+)\b', query, re.IGNORECASE)
        if from_m:
            ast["lane"] = from_m.group(1)

        vec_m = re.search(r'\bVECTOR\s*(\[.+?\])', query, re.IGNORECASE)
        if vec_m:
            ast["vector"] = json.loads(vec_m.group(1))

        top_m = re.search(r'\bTOP\s+(\d+)\b', query, re.IGNORECASE)
        if top_m:
            ast["top"] = int(top_m.group(1))

        conf_m = re.search(r'\bMIN\s+CONFIDENCE\s+([0-9.]+)\b', query, re.IGNORECASE)
        if conf_m:
            ast["min_confidence"] = float(conf_m.group(1))

        return ast

    def _parse_traverse(self, query: str) -> dict:
        """TRAVERSE FROM <node_id> [VIA <rel_type>] [DEPTH n]"""
        ast = {"verb": "TRAVERSE", "node_id": None, "rel_type": None,
               "depth": 2, "min_weight": 0.0}

        from_m = re.search(r'\bFROM\s+[\'"]?([a-zA-Z0-9_-]+)[\'"]?\b',
                           query, re.IGNORECASE)
        if from_m:
            ast["node_id"] = from_m.group(1)

        via_m = re.search(r'\bVIA\s+[\'"]?(\w+)[\'"]?\b', query, re.IGNORECASE)
        if via_m:
            ast["rel_type"] = via_m.group(1)

        depth_m = re.search(r'\bDEPTH\s+(\d+)\b', query, re.IGNORECASE)
        if depth_m:
            ast["depth"] = int(depth_m.group(1))

        return ast

    def _parse_consolidate(self, query: str) -> dict:
        """CONSOLIDATE <source_lane> INTO <target_lane> WHERE confidence > n"""
        ast = {"verb": "CONSOLIDATE", "source": None, "target": None,
               "min_confidence": 0.7}

        parts = query.split()
        upper_parts = query.upper().split()

        if len(parts) > 1:
            ast["source"] = parts[1]

        into_idx = next((i for i, p in enumerate(upper_parts) if p == "INTO"), None)
        if into_idx and into_idx + 1 < len(parts):
            ast["target"] = parts[into_idx + 1]

        conf_m = re.search(r'\bCONFIDENCE\s*[>>=]\s*([0-9.]+)', query, re.IGNORECASE)
        if conf_m:
            ast["min_confidence"] = float(conf_m.group(1))

        return ast

    def _parse_reinforce(self, query: str) -> dict:
        """REINFORCE <lane> WHERE <conditions> BY <amount>"""
        return self._parse_adjust(query, "REINFORCE")

    def _parse_doubt(self, query: str) -> dict:
        """DOUBT <lane> WHERE <conditions> BY <amount>"""
        return self._parse_adjust(query, "DOUBT")

    def _parse_adjust(self, query: str, verb: str) -> dict:
        ast = {"verb": verb, "lane": None, "conditions": [], "amount": 0.1}
        parts = query.split()
        if len(parts) > 1:
            ast["lane"] = parts[1]

        where_m = re.search(r'\bWHERE\s+(.+?)(?:\s+BY\s+|$)', query, re.IGNORECASE)
        if where_m:
            ast["conditions"] = self._parse_conditions(where_m.group(1))

        by_m = re.search(r'\bBY\s+([0-9.]+)\b', query, re.IGNORECASE)
        if by_m:
            ast["amount"] = float(by_m.group(1))

        return ast

    def _parse_reflect(self, query: str) -> dict:
        """REFLECT ON <lane|DATABASE>"""
        ast = {"verb": "REFLECT", "target": "database"}
        on_m = re.search(r'\bON\s+(\w+)\b', query, re.IGNORECASE)
        if on_m:
            ast["target"] = on_m.group(1).lower()
        return ast

    def _parse_conditions(self, conditions_str: str) -> list:
        """Parse WHERE conditions into list of condition dicts."""
        conditions = []
        # Split by AND/OR
        parts = re.split(r'\s+AND\s+|\s+OR\s+', conditions_str, flags=re.IGNORECASE)

        for part in parts:
            part = part.strip()
            # Match: field operator value
            m = re.match(r'(\w+)\s*(>=|<=|!=|>|<|=|CONTAINS|LIKE)\s*(.+)',
                        part, re.IGNORECASE)
            if m:
                field = m.group(1)
                op = m.group(2).upper()
                val_str = m.group(3).strip().strip('"\'')

                # Try to cast value
                try:
                    val = float(val_str) if '.' in val_str else int(val_str)
                except ValueError:
                    val = val_str

                conditions.append({"field": field, "op": op, "value": val})

        return conditions

    def _parse_kv_pairs(self, kv_str: str) -> dict:
        """Parse 'key=val, key2=val2' into dict."""
        result = {}
        pairs = re.split(r',\s*', kv_str)
        for pair in pairs:
            if '=' in pair:
                key, _, val = pair.partition('=')
                key = key.strip()
                val = val.strip().strip('"\'')
                try:
                    val = float(val) if '.' in val else int(val)
                except ValueError:
                    pass
                result[key] = val
        return result
