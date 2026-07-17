from __future__ import annotations

import json
import time
import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from .audit import AuditLog
from .db import RemediationRow
from .models import (
    ActionStatus, DriftStatus, RemediationAction, RemediationKind,
)
from .store import TwinStore


def _row_to_action(row: RemediationRow) -> RemediationAction:
    return RemediationAction.model_validate(row.data)


def _action_row(a: RemediationAction) -> RemediationRow:
    return RemediationRow(
        action_id=a.action_id, node_id=a.node_id, kind=a.kind.value,
        status=a.status.value, ts=a.timestamp or time.time(),
        data=json.loads(a.model_dump_json()),
    )


class RemediationEngine:
    def __init__(self, store: TwinStore, audit: AuditLog,
                 session_factory: sessionmaker[Session]) -> None:
        self.store = store
        self.audit = audit
        self._sf = session_factory

    def _dedupe_key(self, a: RemediationAction) -> tuple:
        return (a.node_id, a.kind.value, a.params.get("edge_to"))

    def register(self, actions: list[RemediationAction]) -> list[RemediationAction]:
        if not actions:
            return []
        registered: list[RemediationAction] = []
        with self._sf() as s:
            node_ids = {a.node_id for a in actions}
            existing_rows = s.scalars(
                select(RemediationRow).where(
                    RemediationRow.node_id.in_(list(node_ids)),
                    RemediationRow.status.in_(
                        [ActionStatus.PROPOSED.value,
                         ActionStatus.APPROVED.value,
                         ActionStatus.APPLIED.value]))).all()
            existing_keys = {
                self._dedupe_key(_row_to_action(r)) for r in existing_rows}
            for a in actions:
                if self._dedupe_key(a) in existing_keys:
                    continue
                a.timestamp = a.timestamp or time.time()
                s.add(_action_row(a))
                self.audit.record(
                    a.proposed_by, "remediation.proposed", a.node_id,
                    detail=f"{a.kind.value}: {a.rationale}",
                    reversible=a.reversible, session=s)
                existing_keys.add(self._dedupe_key(a))
                registered.append(a)
            s.commit()
        return registered

    def get(self, action_id: str) -> Optional[RemediationAction]:
        with self._sf() as s:
            row = s.get(RemediationRow, action_id)
            return _row_to_action(row) if row else None

    def list_actions(self, status: Optional[str] = None,
                     node_id: Optional[str] = None,
                     limit: int = 100,
                     offset: int = 0) -> tuple[list[RemediationAction], int]:
        stmt = select(RemediationRow)
        if status:
            stmt = stmt.where(RemediationRow.status == status)
        if node_id:
            stmt = stmt.where(RemediationRow.node_id == node_id)
        with self._sf() as s:
            total = int(s.scalar(
                select(func.count()).select_from(stmt.subquery())) or 0)
            rows = s.scalars(
                stmt.order_by(RemediationRow.ts.desc())
                .limit(limit).offset(offset)).all()
        return [_row_to_action(r) for r in rows], total

    def _transition(self, action_id: str, expected: ActionStatus
                    ) -> tuple[Optional[Session], Optional[RemediationAction], str]:
        s = self._sf()
        row = s.execute(
            select(RemediationRow)
            .where(RemediationRow.action_id == action_id)
            .with_for_update()).scalar_one_or_none()
        if row is None:
            s.close()
            return None, None, "not_found"
        action = _row_to_action(row)
        if action.status != expected:
            s.close()
            return None, action, "wrong_state"
        return s, action, "ok"

    def _save(self, s: Session, a: RemediationAction) -> None:
        row = s.get(RemediationRow, a.action_id)
        row.status = a.status.value
        row.data = json.loads(a.model_dump_json())

    def approve(self, action_id: str, approver: str) -> RemediationAction:
        s, a, state = self._transition(action_id, ActionStatus.PROPOSED)
        if s is None:
            if state == "not_found":
                raise KeyError(action_id)
            raise ValueError(
                f"cannot approve an action in state '{a.status.value}'")
        try:
            a.status = ActionStatus.APPROVED
            a.approved_by = approver
            self.audit.record(approver, "remediation.approved", a.node_id,
                              detail=f"{a.kind.value} authorised",
                              reversible=a.reversible, session=s)
            a = self._apply(s, a, approver)
            self._save(s, a)
            s.commit()
            return a
        finally:
            s.close()

    def reject(self, action_id: str, approver: str) -> RemediationAction:
        s, a, state = self._transition(action_id, ActionStatus.PROPOSED)
        if s is None:
            if state == "not_found":
                raise KeyError(action_id)
            raise ValueError(
                f"cannot reject an action in state '{a.status.value}'")
        try:
            a.status = ActionStatus.REJECTED
            a.approved_by = approver
            self.audit.record(approver, "remediation.rejected", a.node_id,
                              detail=f"{a.kind.value} declined by operator",
                              session=s)
            self._save(s, a)
            s.commit()
            return a
        finally:
            s.close()

    def _apply(self, s: Session, a: RemediationAction,
               actor: str) -> RemediationAction:
        node = self.store.get_node(a.node_id)
        if node is None:
            a.status = ActionStatus.REJECTED
            return a

        if a.kind == RemediationKind.ROLLBACK:
            a.before_state = _snapshot(node)
            ckpt = a.params.get("checkpoint_id")
            ctx = self.store.get_checkpoint(ckpt) if ckpt else None
            if ctx:
                node.declared_intent = ctx.get("declared_intent",
                                               node.declared_intent)
                node.output = ctx.get("output", node.output)
                node.task = ctx.get("task_spec", node.task)
            node.drift.status = DriftStatus.CONTAINED
            node.drift.score = min(node.drift.score, 0.2)
            a.after_state = _snapshot(node)

        elif a.kind == RemediationKind.QUARANTINE:
            a.before_state = {"quarantined": node.quarantined,
                              "status": node.drift.status.value}
            node.quarantined = True
            node.drift.status = DriftStatus.CONTAINED
            a.after_state = {"quarantined": True,
                             "status": node.drift.status.value}

        elif a.kind == RemediationKind.MESSAGE_FILTER:
            dst = a.params.get("edge_to")
            a.before_state = {"edge_to": dst, "filtered": False}
            if dst:
                self.store.set_edge_weight(a.node_id, dst, 0.0)
                child = self.store.get_node(dst)
                if child and child.drift.status in (DriftStatus.FLAGGED,
                                                    DriftStatus.WATCH):
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
        event = ("quarantine" if a.kind == RemediationKind.QUARANTINE
                 else "remediation.applied")
        self.audit.record(actor, event, a.node_id,
                          detail=f"{a.kind.value} applied; "
                                 f"reversible={a.reversible}",
                          reversible=a.reversible, session=s)
        return a

    def revert(self, action_id: str, actor: str) -> RemediationAction:
        s, a, state = self._transition(action_id, ActionStatus.APPLIED)
        if s is None:
            if state == "not_found":
                raise KeyError(action_id)
            raise ValueError(
                f"cannot revert an action in state '{a.status.value}' "
                f"(only APPLIED actions are reversible)")
        try:
            node = self.store.get_node(a.node_id)
            if node is not None and a.before_state:
                bs = a.before_state
                if a.kind == RemediationKind.QUARANTINE:
                    node.quarantined = bs.get("quarantined", False)
                    node.drift.status = DriftStatus(bs.get("status", "flagged"))
                elif a.kind == RemediationKind.MESSAGE_FILTER:
                    dst = bs.get("edge_to")
                    if dst:
                        self.store.set_edge_weight(a.node_id, dst, 1.0)
                else:
                    node.declared_intent = bs.get("declared_intent",
                                                  node.declared_intent)
                    node.output = bs.get("output", node.output)
                    node.task = bs.get("task", node.task)
                    node.quarantined = bs.get("quarantined", node.quarantined)
                    node.drift.status = DriftStatus(
                        bs.get("status", node.drift.status.value))
                self.store.upsert_node(node)
            a.status = ActionStatus.REVERTED
            self.audit.record(actor, "remediation.reverted", a.node_id,
                              detail=f"{a.kind.value} reverted to prior state",
                              session=s)
            self._save(s, a)
            s.commit()
            return a
        finally:
            s.close()


def new_action(node_id: str, kind: RemediationKind, params: dict,
               rationale: str, proposed_by: str = "system") -> RemediationAction:
    return RemediationAction(
        action_id=f"act::{uuid.uuid4().hex[:12]}",
        node_id=node_id, kind=kind, params=params,
        rationale=rationale, reversible=True, proposed_by=proposed_by,
    )


def _snapshot(node) -> dict:
    return {
        "declared_intent": node.declared_intent,
        "output": node.output,
        "task": node.task,
        "quarantined": node.quarantined,
        "status": node.drift.status.value,
    }
