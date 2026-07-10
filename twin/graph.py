from __future__ import annotations

import json
import sqlite3
import time
from typing import Iterable, Optional

import networkx as nx

from .models import TwinEdge, TwinNode

class TwinStore:
    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._g = nx.DiGraph()
        self._init_schema()
        self._load_graph()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                data    TEXT NOT NULL,
                ts      REAL
            );
            CREATE TABLE IF NOT EXISTS edges (
                src   TEXT NOT NULL,
                dst   TEXT NOT NULL,
                kind  TEXT NOT NULL DEFAULT 'influence',
                weight REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (src, dst, kind)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                node_id  TEXT,
                agent_id TEXT,
                label    TEXT,
                context  TEXT NOT NULL,
                ts       REAL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
            CREATE INDEX IF NOT EXISTS idx_ckpt_agent ON checkpoints(agent_id);
            """
        )
        self._conn.commit()

    def _load_graph(self) -> None:
        cur = self._conn.cursor()
        for row in cur.execute("SELECT node_id, data FROM nodes"):
            node = TwinNode.model_validate_json(row["data"])
            self._g.add_node(node.node_id, node=node)
        for row in cur.execute("SELECT src, dst, kind, weight FROM edges"):
            self._g.add_edge(row["src"], row["dst"], kind=row["kind"], weight=row["weight"])

    def upsert_node(self, node: TwinNode) -> None:
        if node.timestamp is None:
            node.timestamp = time.time()
        self._conn.execute(
            "INSERT INTO nodes(node_id, data, ts) VALUES(?,?,?) "
            "ON CONFLICT(node_id) DO UPDATE SET data=excluded.data, ts=excluded.ts",
            (node.node_id, node.model_dump_json(), node.timestamp),
        )
        self._conn.commit()
        self._g.add_node(node.node_id, node=node)

    def add_edge(self, edge: TwinEdge) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO edges(src, dst, kind, weight) VALUES(?,?,?,?)",
            (edge.src, edge.dst, edge.kind, edge.weight),
        )
        self._conn.commit()
        self._g.add_edge(edge.src, edge.dst, kind=edge.kind, weight=edge.weight)

    def get_node(self, node_id: str) -> Optional[TwinNode]:
        data = self._g.nodes.get(node_id, {}).get("node")
        return data

    def has_node(self, node_id: str) -> bool:
        return node_id in self._g

    def all_nodes(self) -> list[TwinNode]:
        return [d["node"] for _, d in self._g.nodes(data=True) if "node" in d]

    def all_edges(self) -> list[TwinEdge]:
        return [
            TwinEdge(src=u, dst=v, kind=d.get("kind", "influence"), weight=d.get("weight", 1.0))
            for u, v, d in self._g.edges(data=True)
        ]

    def blast_radius(self, node_id: str) -> list[str]:
        if node_id not in self._g:
            return []
        return list(nx.descendants(self._g, node_id))

    def upstream(self, node_id: str) -> list[str]:
        if node_id not in self._g:
            return []
        return list(nx.ancestors(self._g, node_id))

    def propagation_path(self, root: str, target: str) -> list[str]:
        if root not in self._g or target not in self._g:
            return []
        try:
            return nx.shortest_path(self._g, root, target)
        except nx.NetworkXNoPath:
            return []

    def roots(self) -> list[str]:
        return [n for n in self._g.nodes if self._g.in_degree(n) == 0]

    def topo_order(self) -> list[str]:
        try:
            return list(nx.topological_sort(self._g))
        except nx.NetworkXUnfeasible:
            return list(self._g.nodes)

    def successors(self, node_id: str) -> list[str]:
        return list(self._g.successors(node_id)) if node_id in self._g else []

    def predecessors(self, node_id: str) -> list[str]:
        return list(self._g.predecessors(node_id)) if node_id in self._g else []

    def save_checkpoint(self, checkpoint_id: str, node_id: str, agent_id: str,
                        context: dict, label: str = "") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints"
            "(checkpoint_id, node_id, agent_id, label, context, ts) VALUES(?,?,?,?,?,?)",
            (checkpoint_id, node_id, agent_id, label, json.dumps(context), time.time()),
        )
        self._conn.commit()

    def get_checkpoint(self, checkpoint_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT context FROM checkpoints WHERE checkpoint_id=?", (checkpoint_id,)
        ).fetchone()
        return json.loads(row["context"]) if row else None

    def latest_good_checkpoint(self, agent_id: str, before_ts: float) -> Optional[str]:
        row = self._conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE agent_id=? AND ts<=? "
            "ORDER BY ts DESC LIMIT 1",
            (agent_id, before_ts),
        ).fetchone()
        return row["checkpoint_id"] if row else None

    def set_meta(self, key: str, value: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)",
            (key, json.dumps(value)),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row["value"]) if row else None

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def reset(self) -> None:
        self._conn.executescript(
            "DELETE FROM nodes; DELETE FROM edges; DELETE FROM checkpoints; "
            "DELETE FROM meta;"
        )
        self._conn.commit()
        self._g = nx.DiGraph()
