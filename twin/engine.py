from __future__ import annotations

import logging
import time
from typing import Optional

from sqlalchemy import text

from . import attribution, metrics, replay
from .audit import AuditLog
from .config import Settings, get_settings
from .db import build_engine, build_session_factory, init_schema
from .detection import DetectionPolicy
from .embeddings import build_embedder
from .llm import build_judges
from .models import DriftStatus, Span, Tier, TwinNode
from .remediation import RemediationEngine
from .router import CostRouter
from .security import ApiKeyManager, RateLimiter
from .store import TwinStore

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.db_engine = build_engine(self.settings)
        init_schema(self.db_engine)
        self.session_factory = build_session_factory(self.db_engine)

        self.store = TwinStore(self.db_engine, self.session_factory)
        self.audit = AuditLog(self.session_factory)
        self.policy = DetectionPolicy.from_settings(self.settings)
        self.embedder = build_embedder(self.settings)
        self.judges = build_judges(self.settings, self.policy)
        self.router = CostRouter(
            self.store, self.embedder, self.judges, self.policy,
            low_sample_rate=self.settings.low_privilege_sample_rate,
            cache_size=self.settings.detection_cache_size,
        )
        self.remediation = RemediationEngine(
            self.store, self.audit, self.session_factory)
        self.keys = ApiKeyManager(self.session_factory)
        self.rate_limiter = RateLimiter(self.settings.rate_limit_per_minute)
        self.started_at = time.time()

    def close(self) -> None:
        try:
            self.db_engine.dispose()
        except Exception:
            pass

    # --- ingestion ---

    def ingest(self, span: Span, idempotent: bool = True) -> TwinNode:
        if idempotent and self.store.has_node(span.span_id):
            return self.store.get_node(span.span_id)
        node = self.router.ingest(span)
        d = node.drift

        metrics.SPANS_INGESTED.labels(status=d.status.value).inc()
        metrics.TIER_DECISIONS.labels(tier=d.tier_reached.value).inc()

        if node.blocked:
            metrics.BLOCKED_ACTIONS.inc()
            self.audit.record("inline-rail", "action.blocked", node.node_id,
                              detail=node.blocked_reason)
        if d.tier_reached in (Tier.SMALL_JUDGE, Tier.DEEP_ESCALATION):
            self.audit.record("system", "escalation", node.node_id,
                              detail=f"escalated to {d.tier_reached.value}; "
                                     f"score={d.score:.2f} risk={d.risk_type}")
        if d.status in (DriftStatus.FLAGGED, DriftStatus.WATCH):
            self.audit.record("system", "detection", node.node_id,
                              detail=f"{d.status.value} score={d.score:.2f} "
                                     f"risk={d.risk_type}: {d.rationale}")
        if d.status == DriftStatus.FLAGGED:
            metrics.DRIFT_FLAGGED.labels(
                risk_type=d.risk_type or "unclassified").inc()
            self._auto_propose(node)
        return node

    def _auto_propose(self, node: TwinNode) -> None:
        try:
            root_id = attribution.find_root_cause(
                self.store, node.node_id, self.policy)
            root = self.store.get_node(root_id)
            if root is None:
                return
            actions = attribution.propose_remediation(self.store, root)
            registered = self.remediation.register(actions)
            for _ in registered:
                metrics.REMEDIATION_EVENTS.labels(event="proposed").inc()
        except Exception:
            log.exception("auto-propose remediation failed for %s",
                          node.node_id)

    # --- analysis ---

    def narrative(self, target_id: Optional[str] = None):
        return attribution.build_narrative(self.store, target_id, self.policy)

    def incidents(self, limit: int = 50, offset: int = 0):
        return attribution.list_incidents(self.store, self.policy,
                                          limit=limit, offset=offset)

    def whatif(self, root_id: str):
        return replay.build_preview(self.store, root_id, self.embedder,
                                    self.policy)

    def graph_state(self, trace_id: str) -> dict:
        nodes, total = self.store.list_nodes(trace_id=trace_id, limit=2000)
        node_ids = [n.node_id for n in nodes]
        edges = self.store.edges_for_nodes(node_ids)
        return {
            "nodes": [n.model_dump(mode="json") for n in nodes],
            "edges": [e.model_dump(mode="json") for e in edges],
            "trace_id": trace_id,
            "node_count": total,
        }

    def guard_report(self, limit: int = 100, offset: int = 0) -> dict:
        blocked, total = self.store.list_nodes(blocked=True, limit=limit,
                                               offset=offset)
        return {
            "blocked_count": total,
            "blocked": [
                {"node_id": n.node_id, "agent_id": n.agent_id,
                 "trace_id": n.trace_id,
                 "tool_calls": [c.name for c in n.tool_calls],
                 "reason": n.blocked_reason}
                for n in blocked
            ],
        }

    def cost(self) -> dict:
        return self.router.ledger()

    def compliance(self) -> dict:
        return self.audit.compliance_report()

    # --- ops ---

    def db_ok(self) -> bool:
        try:
            with self.session_factory() as s:
                s.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def info(self) -> dict:
        return {
            "version": "1.0.0",
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "database": {
                "dialect": self.db_engine.dialect.name,
                "ok": self.db_ok(),
            },
            "nodes_in_twin": self.store.node_count(),
            "audit_entries": self.audit.count(),
            "auth_enabled": self.settings.auth_enabled,
            "embeddings": self.embedder.info(),
            "judge": self.judges.info(),
            "policy": {
                "flag_threshold": self.policy.flag_threshold,
                "watch_threshold": self.policy.watch_threshold,
                "hard_flag_severity": self.policy.hard_flag_severity,
                "low_privilege_sample_rate":
                    self.settings.low_privilege_sample_rate,
                "dangerous_tools": sorted(self.policy.dangerous_tools),
            },
        }

    def run_retention(self, days: int, actor: str = "system") -> dict:
        cutoff = time.time() - days * 86400.0
        result = self.store.prune_older_than(cutoff)
        self.audit.record(actor, "retention.pruned", "twin",
                          detail=f"removed {result['nodes']} node(s), "
                                 f"{result['edges']} edge(s), "
                                 f"{result['checkpoints']} checkpoint(s) "
                                 f"older than {days}d")
        return result
