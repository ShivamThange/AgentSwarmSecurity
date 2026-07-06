"""Engine — wires the twin, router, detection, remediation and audit into one
facade the API and CLI drive. This is the detect -> attribute -> remediate ->
redeploy loop of Section 12.4, running within the cost envelope.

The twin (graph + checkpoints + audit chain + cost snapshot) is DURABLE: point the
engine at a file path and it survives a process restart, which is the whole moat —
cross-run, cross-time causal state (Section 8.1). `load_or_seed` reconstructs the
derived incident view from the persisted graph without re-ingesting anything.
"""
from __future__ import annotations

from . import attribution, guard, llm, scenario, whatif
from .audit import AuditLog
from .graph import TwinStore
from .models import DriftStatus, Span, Tier
from .remediation import RemediationEngine
from .router import CostRouter

_LEDGER_KEY = "cost_ledger"


class Engine:
    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self.store = TwinStore(db_path)
        self.audit = AuditLog(conn=self.store._conn)
        # Hybrid judge: online LLM if configured via env, else offline StubJudge.
        self.llm_config = llm.LLMConfig.from_env()
        self.router = CostRouter(self.store, judge=llm.build_judge(self.llm_config))
        self.remediation = RemediationEngine(self.store, self.audit)
        self.incident_narrative = None  # cached CausalNarrative for the seeded run
        self.boot_mode = "empty"        # "seeded" | "loaded" | "empty"

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Release the underlying database connection (Windows needs this before
        the file can be removed)."""
        self.store.close()

    def reset(self) -> None:
        self.store.reset()
        self.audit.reset()
        self.router.ledger.__init__()
        self.router._cache.clear()
        self.router._chain_anchor.clear()
        self.remediation._actions.clear()

    def ingest(self, span: Span, idempotent: bool = True):
        # Idempotent on span id: replaying the same span never double-counts the
        # cost ledger or duplicates a node (endpoint-hardening requirement).
        if idempotent and self.store.has_node(span.span_id):
            return self.store.get_node(span.span_id)
        node = self.router.ingest(span)
        d = node.drift
        if node.blocked:
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
        return node

    def seed(self, background: int = 200) -> None:
        """Reset and replay the beachhead scenario through the full pipeline,
        against a realistic volume of clean org-wide background traffic."""
        self.reset()
        # background load first, so the incident lands into an already-busy twin
        for span in scenario.background_spans(background):
            self.ingest(span)
        for span in scenario.build_spans():
            self.ingest(span)
        # attribute the incident once and register its proposed remediation so the
        # supervisor can approve/reject concrete actions.
        self.incident_narrative = attribution.build_narrative(self.store)
        if self.incident_narrative:
            self.remediation.register(self.incident_narrative.recommended_remediation)
        # persist the cost snapshot so a reloaded twin shows the same figures
        self.store.set_meta(_LEDGER_KEY, self.router.ledger.as_dict())
        self.boot_mode = "seeded"

    def load_or_seed(self, background: int = 200) -> str:
        """If the persisted twin already has data, reconstruct the derived view
        from it (proving cross-restart persistence). Otherwise seed fresh."""
        if self.store.all_nodes():
            self._rebuild_from_store()
            self.boot_mode = "loaded"
        else:
            self.seed(background=background)
        return self.boot_mode

    def _rebuild_from_store(self) -> None:
        """Rebuild in-memory derived state (narrative, proposed remediation, cost
        ledger) from the persisted graph WITHOUT re-ingesting any span."""
        snap = self.store.get_meta(_LEDGER_KEY)
        if snap:
            self.router.ledger.load_dict(snap)
        self.incident_narrative = attribution.build_narrative(self.store)
        if self.incident_narrative and not self.remediation.all_actions():
            self.remediation.register(self.incident_narrative.recommended_remediation)

    # ------------------------------------------------------------------ #
    def narrative(self, target_id: str | None = None):
        return attribution.build_narrative(self.store, target_id)

    def whatif(self, root_id: str):
        return whatif.build_preview(self.store, root_id)

    def graph_state(self, trace: str | None = None) -> dict:
        """Graph view. `trace` focuses on one incident (default: the whole twin).
        Background traffic lives in the twin but is filtered out of the focused
        incident view."""
        nodes = self.store.all_nodes()
        if trace:
            nodes = [n for n in nodes if n.trace_id == trace]
        node_ids = {n.node_id for n in nodes}
        edges = [e for e in self.store.all_edges()
                 if e.src in node_ids and e.dst in node_ids]
        return {
            "nodes": [n.model_dump(mode="json") for n in nodes],
            "edges": [e.model_dump(mode="json") for e in edges],
            "total_nodes_in_twin": len(self.store.all_nodes()),
            "focused_trace": trace,
        }

    def cost(self) -> dict:
        return self.router.ledger.as_dict()

    def compliance(self) -> dict:
        return self.audit.compliance_report()

    def guard_report(self) -> dict:
        """Inline-rail decisions across the twin: what was stopped pre-execution."""
        blocked = [n for n in self.store.all_nodes() if n.blocked]
        return {
            "blocked_count": len(blocked),
            "blocked": [
                {"node_id": n.node_id, "agent_id": n.agent_id,
                 "tool_calls": [c.name for c in n.tool_calls],
                 "reason": n.blocked_reason}
                for n in blocked
            ],
        }

    def llm_info(self) -> dict:
        """Which judge backs the escalation tier (online LLM vs offline stub)."""
        return self.llm_config.info()

    def persistence_info(self) -> dict:
        """Evidence that the twin is durable, not just backed by code that could
        be. Includes a live blast-radius read off the (possibly reloaded) graph."""
        root = self.incident_narrative.root_cause_node if self.incident_narrative else None
        return {
            "db_path": self.db_path,
            "durable": self.db_path != ":memory:",
            "boot_mode": self.boot_mode,   # "seeded" on first boot, "loaded" after
            "nodes_in_twin": len(self.store.all_nodes()),
            "audit_entries": len(self.audit.entries()),
            "audit_chain_valid": self.audit.verify_chain(),
            "sample_query": {
                "root_cause": root,
                "blast_radius": sorted(self.store.blast_radius(root)) if root else [],
            },
        }
