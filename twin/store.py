from __future__ import annotations

import json
import time
from collections import deque
from typing import Iterable, Optional, Sequence

from sqlalchemy import case, delete, func, select, text, update
from sqlalchemy.engine import Engine as SAEngine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .db import (
    CheckpointRow, CounterRow, EdgeRow, FeedbackLabelRow, MetaRow, NodeRow,
)
from .models import Privilege, TwinEdge, TwinNode

VALID_LABELS = ("true_positive", "false_positive")

_DESCENDANTS_SQL = text(
    """
    WITH RECURSIVE reachable(id) AS (
        SELECT dst FROM edges WHERE src = :root
        UNION
        SELECT e.dst FROM edges e JOIN reachable r ON e.src = r.id
    )
    SELECT id FROM reachable
    """
)

_ANCESTORS_SQL = text(
    """
    WITH RECURSIVE reachable(id) AS (
        SELECT src FROM edges WHERE dst = :root
        UNION
        SELECT e.src FROM edges e JOIN reachable r ON e.dst = r.id
    )
    SELECT id FROM reachable
    """
)

_PRIV_RANK = {"high": 2, "medium": 1, "low": 0}


def _row_to_node(row: NodeRow) -> TwinNode:
    return TwinNode.model_validate(row.data)


class TwinStore:
    def __init__(self, engine: SAEngine,
                 session_factory: sessionmaker[Session]) -> None:
        self.engine = engine
        self._sf = session_factory

    def _session(self) -> Session:
        return self._sf()

    # --- nodes ---

    def upsert_node(self, node: TwinNode, session: Optional[Session] = None) -> None:
        if node.timestamp is None:
            node.timestamp = time.time()
        row = NodeRow(
            node_id=node.node_id,
            trace_id=node.trace_id,
            agent_id=node.agent_id,
            privilege=node.privilege.value,
            drift_status=node.drift.status.value,
            drift_score=node.drift.score,
            risk_type=node.drift.risk_type,
            blocked=node.blocked,
            quarantined=node.quarantined,
            ts=node.timestamp,
            data=json.loads(node.model_dump_json()),
        )
        if session is not None:
            session.merge(row)
            return
        with self._session() as s:
            s.merge(row)
            s.commit()

    def get_node(self, node_id: str) -> Optional[TwinNode]:
        with self._session() as s:
            row = s.get(NodeRow, node_id)
            return _row_to_node(row) if row else None

    def get_nodes(self, node_ids: Sequence[str]) -> dict[str, TwinNode]:
        if not node_ids:
            return {}
        out: dict[str, TwinNode] = {}
        with self._session() as s:
            for row in s.scalars(
                    select(NodeRow).where(NodeRow.node_id.in_(list(node_ids)))):
                out[row.node_id] = _row_to_node(row)
        return out

    def has_node(self, node_id: str) -> bool:
        with self._session() as s:
            return s.scalar(
                select(func.count()).select_from(NodeRow)
                .where(NodeRow.node_id == node_id)) > 0

    def existing_ids(self, node_ids: Sequence[str]) -> set[str]:
        if not node_ids:
            return set()
        with self._session() as s:
            rows = s.scalars(
                select(NodeRow.node_id)
                .where(NodeRow.node_id.in_(list(node_ids)))).all()
        return set(rows)

    def node_count(self) -> int:
        with self._session() as s:
            return int(s.scalar(select(func.count()).select_from(NodeRow)) or 0)

    def list_nodes(self, trace_id: Optional[str] = None,
                   status: Optional[str] = None,
                   agent_id: Optional[str] = None,
                   blocked: Optional[bool] = None,
                   since: Optional[float] = None,
                   limit: int = 100, offset: int = 0) -> tuple[list[TwinNode], int]:
        stmt = select(NodeRow)
        if trace_id:
            stmt = stmt.where(NodeRow.trace_id == trace_id)
        if status:
            stmt = stmt.where(NodeRow.drift_status == status)
        if agent_id:
            stmt = stmt.where(NodeRow.agent_id == agent_id)
        if blocked is not None:
            stmt = stmt.where(NodeRow.blocked == blocked)
        if since is not None:
            stmt = stmt.where(NodeRow.ts >= since)
        with self._session() as s:
            total = int(s.scalar(
                select(func.count()).select_from(stmt.subquery())) or 0)
            rows = s.scalars(
                stmt.order_by(NodeRow.ts.asc(), NodeRow.node_id.asc())
                .limit(limit).offset(offset)).all()
        return [_row_to_node(r) for r in rows], total

    def worst_node(self) -> Optional[TwinNode]:
        with self._session() as s:
            rows = s.scalars(
                select(NodeRow).where(NodeRow.quarantined.is_(False))
                .order_by(NodeRow.drift_score.desc(), NodeRow.ts.desc())
                .limit(25)).all()
        if not rows:
            return None
        best = max(rows, key=lambda r: (
            r.drift_score, _PRIV_RANK.get(r.privilege, 0), r.ts or 0.0))
        return _row_to_node(best)

    def trace_anchor_task(self, trace_id: str) -> Optional[str]:
        if not trace_id:
            return None
        with self._session() as s:
            row = s.execute(
                select(NodeRow.data).where(
                    NodeRow.trace_id == trace_id)
                .order_by(NodeRow.ts.asc()).limit(5)).all()
        for (data,) in row:
            task = (data or {}).get("task", "")
            if task:
                return task
        return None

    def list_traces(self, limit: int = 50, offset: int = 0,
                    flagged_only: bool = False) -> tuple[list[dict], int]:
        flagged_expr = func.sum(
            case((NodeRow.drift_status == "flagged", 1), else_=0))
        blocked_expr = func.sum(case((NodeRow.blocked.is_(True), 1), else_=0))
        base = (
            select(
                NodeRow.trace_id.label("trace_id"),
                func.count().label("span_count"),
                func.max(NodeRow.drift_score).label("max_score"),
                func.min(NodeRow.ts).label("first_ts"),
                func.max(NodeRow.ts).label("last_ts"),
                flagged_expr.label("flagged"),
                blocked_expr.label("blocked"),
            )
            .where(NodeRow.trace_id != "")
            .group_by(NodeRow.trace_id)
        )
        if flagged_only:
            base = base.having(flagged_expr > 0)
        with self._session() as s:
            total = int(s.scalar(
                select(func.count()).select_from(base.subquery())) or 0)
            rows = s.execute(
                base.order_by(func.max(NodeRow.ts).desc())
                .limit(limit).offset(offset)).all()
        traces = [
            {"trace_id": r.trace_id, "span_count": int(r.span_count or 0),
             "max_score": float(r.max_score or 0.0),
             "flagged_count": int(r.flagged or 0),
             "blocked_count": int(r.blocked or 0),
             "first_ts": r.first_ts, "last_ts": r.last_ts}
            for r in rows
        ]
        return traces, total

    # --- edges ---

    def add_edge(self, edge: TwinEdge, session: Optional[Session] = None) -> None:
        row = EdgeRow(src=edge.src, dst=edge.dst, kind=edge.kind,
                      weight=edge.weight)
        if session is not None:
            session.merge(row)
            return
        with self._session() as s:
            s.merge(row)
            s.commit()

    def set_edge_weight(self, src: str, dst: str, weight: float) -> bool:
        with self._session() as s:
            res = s.execute(
                update(EdgeRow).where(EdgeRow.src == src, EdgeRow.dst == dst)
                .values(weight=weight))
            s.commit()
            return res.rowcount > 0

    def edges_for_nodes(self, node_ids: Iterable[str]) -> list[TwinEdge]:
        ids = list(node_ids)
        if not ids:
            return []
        with self._session() as s:
            rows = s.scalars(
                select(EdgeRow).where(EdgeRow.src.in_(ids),
                                      EdgeRow.dst.in_(ids))).all()
        return [TwinEdge(src=r.src, dst=r.dst, kind=r.kind, weight=r.weight)
                for r in rows]

    def successors(self, node_id: str) -> list[str]:
        with self._session() as s:
            return list(s.scalars(
                select(EdgeRow.dst).where(EdgeRow.src == node_id)).all())

    def predecessors(self, node_id: str) -> list[str]:
        with self._session() as s:
            return list(s.scalars(
                select(EdgeRow.src).where(EdgeRow.dst == node_id)).all())

    def blast_radius(self, node_id: str) -> list[str]:
        with self._session() as s:
            rows = s.execute(_DESCENDANTS_SQL, {"root": node_id}).scalars().all()
        return [r for r in rows if r != node_id]

    def upstream(self, node_id: str) -> list[str]:
        with self._session() as s:
            rows = s.execute(_ANCESTORS_SQL, {"root": node_id}).scalars().all()
        return [r for r in rows if r != node_id]

    def propagation_path(self, root: str, target: str) -> list[str]:
        if root == target:
            return [root]
        scope = set(self.blast_radius(root)) | {root, target}
        adj: dict[str, list[str]] = {}
        for e in self.edges_for_nodes(scope):
            adj.setdefault(e.src, []).append(e.dst)
        prev: dict[str, str] = {}
        q: deque[str] = deque([root])
        seen = {root}
        while q:
            cur = q.popleft()
            if cur == target:
                break
            for nxt in adj.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    prev[nxt] = cur
                    q.append(nxt)
        if target not in seen:
            return []
        path = [target]
        while path[-1] != root:
            path.append(prev[path[-1]])
        return list(reversed(path))

    def topo_order(self, node_ids: Iterable[str]) -> list[str]:
        ids = set(node_ids)
        edges = self.edges_for_nodes(ids)
        indeg = {n: 0 for n in ids}
        adj: dict[str, list[str]] = {n: [] for n in ids}
        for e in edges:
            adj[e.src].append(e.dst)
            indeg[e.dst] += 1
        q = deque(sorted(n for n, d in indeg.items() if d == 0))
        out: list[str] = []
        while q:
            cur = q.popleft()
            out.append(cur)
            for nxt in adj[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)
        if len(out) != len(ids):
            out.extend(sorted(ids - set(out)))
        return out

    # --- checkpoints ---

    def save_checkpoint(self, checkpoint_id: str, node_id: str, agent_id: str,
                        context: dict, label: str = "",
                        session: Optional[Session] = None) -> None:
        row = CheckpointRow(checkpoint_id=checkpoint_id, node_id=node_id,
                            agent_id=agent_id, label=label, context=context,
                            ts=time.time())
        if session is not None:
            session.merge(row)
            return
        with self._session() as s:
            s.merge(row)
            s.commit()

    def get_checkpoint(self, checkpoint_id: str) -> Optional[dict]:
        with self._session() as s:
            row = s.get(CheckpointRow, checkpoint_id)
            return dict(row.context) if row else None

    def latest_good_checkpoint(self, agent_id: str,
                               before_ts: float) -> Optional[str]:
        with self._session() as s:
            row = s.scalars(
                select(CheckpointRow.checkpoint_id)
                .where(CheckpointRow.agent_id == agent_id,
                       CheckpointRow.ts <= before_ts)
                .order_by(CheckpointRow.ts.desc()).limit(1)).first()
            return row

    # --- meta & counters ---

    def set_meta(self, key: str, value) -> None:
        with self._session() as s:
            s.merge(MetaRow(key=key, value=value))
            s.commit()

    def get_meta(self, key: str):
        with self._session() as s:
            row = s.get(MetaRow, key)
            return row.value if row else None

    def incr_counters(self, deltas: dict[str, float]) -> None:
        if not deltas:
            return
        with self._session() as s:
            for key, delta in deltas.items():
                res = s.execute(
                    update(CounterRow).where(CounterRow.key == key)
                    .values(value=CounterRow.value + delta))
                if res.rowcount == 0:
                    try:
                        with s.begin_nested():
                            s.add(CounterRow(key=key, value=delta))
                    except IntegrityError:
                        s.execute(
                            update(CounterRow).where(CounterRow.key == key)
                            .values(value=CounterRow.value + delta))
            s.commit()

    def get_counters(self, keys: Sequence[str]) -> dict[str, float]:
        out = {k: 0.0 for k in keys}
        if not keys:
            return out
        with self._session() as s:
            for row in s.scalars(
                    select(CounterRow).where(CounterRow.key.in_(list(keys)))):
                out[row.key] = float(row.value)
        return out

    # --- feedback labels (threshold calibration) ---

    def save_label(self, node_id: str, label: str, score: float,
                   workflow: str, drift_status: str, labeled_by: str,
                   note: str = "") -> None:
        with self._session() as s:
            s.merge(FeedbackLabelRow(
                node_id=node_id, label=label, score=score, workflow=workflow,
                drift_status=drift_status, labeled_by=labeled_by, note=note,
                ts=time.time()))
            s.commit()

    def labeled_points(self, workflow: Optional[str] = None
                       ) -> list[tuple[float, str, str]]:
        """Return ``(score, label, workflow)`` for every human-labelled node."""
        stmt = select(FeedbackLabelRow.score, FeedbackLabelRow.label,
                      FeedbackLabelRow.workflow)
        if workflow is not None:
            stmt = stmt.where(FeedbackLabelRow.workflow == workflow)
        with self._session() as s:
            return [(float(sc), lb, wf) for sc, lb, wf in s.execute(stmt).all()]

    def label_count(self) -> int:
        with self._session() as s:
            return int(s.scalar(
                select(func.count()).select_from(FeedbackLabelRow)) or 0)

    # --- retention / lifecycle ---

    def prune_older_than(self, cutoff_ts: float) -> dict[str, int]:
        with self._session() as s:
            old_ids = set(s.scalars(
                select(NodeRow.node_id).where(NodeRow.ts < cutoff_ts)).all())
            if not old_ids:
                return {"nodes": 0, "edges": 0, "checkpoints": 0}
            ids = list(old_ids)
            edges = 0
            nodes = 0
            ckpts = 0
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                edges += s.execute(
                    delete(EdgeRow).where(
                        (EdgeRow.src.in_(chunk)) | (EdgeRow.dst.in_(chunk)))
                ).rowcount
                ckpts += s.execute(
                    delete(CheckpointRow).where(CheckpointRow.node_id.in_(chunk))
                ).rowcount
                nodes += s.execute(
                    delete(NodeRow).where(NodeRow.node_id.in_(chunk))
                ).rowcount
            s.commit()
        return {"nodes": nodes, "edges": edges, "checkpoints": ckpts}

    def reset(self) -> None:
        with self._session() as s:
            s.execute(delete(EdgeRow))
            s.execute(delete(CheckpointRow))
            s.execute(delete(NodeRow))
            s.execute(delete(FeedbackLabelRow))
            s.execute(delete(MetaRow))
            s.execute(delete(CounterRow))
            s.commit()
