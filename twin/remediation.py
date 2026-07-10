from __future__ import annotations

import time

from .audit import AuditLog
from .graph import TwinStore
from .models import (
    ActionStatus, DriftStatus, RemediationAction, RemediationKind,
)

AUTO_APPROVE_KINDS: set[RemediationKind] = set()

class RemediationEngine:
    def __init__(self, store: TwinStore, audit: AuditLog) -> None:
        self.store = store
        self.audit = audit
        self._actions: dict[str, RemediationAction] = {}

    def register(self, actions: list[RemediationAction]) -> None:
        for a in actions:
            a.timestamp = a.timestamp or time.time()
            self._actions[a.action_id] = a
            self.audit.record("system", "remediation.proposed", a.node_id,
                              detail=f"{a.kind.value}: {a.rationale}", reversible=a.reversible)

    def get(self, action_id: str) -> RemediationAction | None:
        return self._actions.get(action_id)

    def pending(self) -> list[RemediationAction]:
        return [a for a in self._actions.values() if a.status == ActionStatus.PROPOSED]

    def all_actions(self) -> list[RemediationAction]:
        return list(self._actions.values())

    def approve(self, action_id: str, approver: str = "supervisor") -> RemediationAction:
        a = self._actions[action_id]
        if a.status not in (ActionStatus.PROPOSED,):
            return a
        a.status = ActionStatus.APPROVED
        a.approved_by = approver
        self.audit.record(approver, "remediation.approved", a.node_id,
                          detail=f"{a.kind.value} authorised", reversible=a.reversible)
        return self._apply(a, approver)

    def reject(self, action_id: str, approver: str = "supervisor") -> RemediationAction:
        a = self._actions[action_id]
        a.status = ActionStatus.REJECTED
        a.approved_by = approver
        self.audit.record(approver, "remediation.rejected", a.node_id,
                          detail=f"{a.kind.value} declined by operator")
        return a

    def maybe_auto_apply(self, action_id: str) -> RemediationAction | None:
        a = self._actions[action_id]
        if a.kind in AUTO_APPROVE_KINDS:
            return self.approve(action_id, approver="auto-policy")
        return None

    def _apply(self, a: RemediationAction, actor: str) -> RemediationAction:
        node = self.store.get_node(a.node_id)
        if node is None:
            a.status = ActionStatus.REJECTED
            return a

        if a.kind == RemediationKind.ROLLBACK:
            a.before_state = _snapshot(node)
            ckpt = a.params.get("checkpoint_id")
            ctx = self.store.get_checkpoint(ckpt) if ckpt else None
            if ctx:
                node.declared_intent = ctx.get("declared_intent", node.declared_intent)
                node.output = ctx.get("output", node.output)
                node.task = ctx.get("task_spec", node.task)
            node.drift.status = DriftStatus.CONTAINED
            node.drift.score = min(node.drift.score, 0.2)
            a.after_state = _snapshot(node)

        elif a.kind == RemediationKind.QUARANTINE:
            a.before_state = {"quarantined": node.quarantined, "status": node.drift.status.value}
            node.quarantined = True
            node.drift.status = DriftStatus.CONTAINED
            a.after_state = {"quarantined": True, "status": node.drift.status.value}

        elif a.kind == RemediationKind.MESSAGE_FILTER:
            dst = a.params.get("edge_to")
            a.before_state = {"edge_to": dst, "filtered": False}

            for e in self.store.all_edges():
                if e.src == a.node_id and e.dst == dst:
                    e.weight = 0.0
                    self.store.add_edge(e)

            child = self.store.get_node(dst)
            if child and child.drift.status in (DriftStatus.FLAGGED, DriftStatus.WATCH):
                child.drift.status = DriftStatus.CONTAINED
                child.drift.score = min(child.drift.score, 0.25)
                self.store.upsert_node(child)
            a.after_state = {"edge_to": dst, "filtered": True}

        elif a.kind == RemediationKind.CONTEXT_EDIT:
            a.before_state = _snapshot(node)
            patch = a.params.get("patch", {})
            if "declared_intent" in patch:
                node.declared_intent = patch["declared_intent"]
            if "task" in patch:
                node.task = patch["task"]
            node.drift.status = DriftStatus.CONTAINED
            a.after_state = _snapshot(node)

        elif a.kind == RemediationKind.REDEPLOY:
            a.before_state = {"quarantined": node.quarantined}
            node.quarantined = False
            node.drift.status = DriftStatus.OK
            a.after_state = {"quarantined": False}

        self.store.upsert_node(node)
        a.status = ActionStatus.APPLIED
        event = "quarantine" if a.kind == RemediationKind.QUARANTINE else "remediation.applied"
        self.audit.record(actor, event, a.node_id,
                          detail=f"{a.kind.value} applied; reversible={a.reversible}",
                          reversible=a.reversible)
        return a

    def revert(self, action_id: str, actor: str = "supervisor") -> RemediationAction:
        a = self._actions[action_id]
        if a.status != ActionStatus.APPLIED:
            return a
        node = self.store.get_node(a.node_id)
        if node is not None and a.before_state:
            bs = a.before_state
            if a.kind == RemediationKind.QUARANTINE:
                node.quarantined = bs.get("quarantined", False)
                node.drift.status = DriftStatus(bs.get("status", "flagged"))
            elif a.kind == RemediationKind.MESSAGE_FILTER:
                dst = bs.get("edge_to")
                for e in self.store.all_edges():
                    if e.src == a.node_id and e.dst == dst:
                        e.weight = 1.0
                        self.store.add_edge(e)
            else:
                node.declared_intent = bs.get("declared_intent", node.declared_intent)
                node.output = bs.get("output", node.output)
                node.task = bs.get("task", node.task)
                node.quarantined = bs.get("quarantined", node.quarantined)
                node.drift.status = DriftStatus(bs.get("status", node.drift.status.value))
            self.store.upsert_node(node)
        a.status = ActionStatus.REVERTED
        self.audit.record(actor, "remediation.reverted", a.node_id,
                          detail=f"{a.kind.value} reverted to prior state")
        return a

def _snapshot(node) -> dict:
    return {
        "declared_intent": node.declared_intent,
        "output": node.output,
        "task": node.task,
        "quarantined": node.quarantined,
        "status": node.drift.status.value,
    }
