from __future__ import annotations

import pytest
from sqlalchemy import select, update

from twin.db import AuditRow
from twin.engine import Engine
from twin.models import ActionStatus, DriftStatus, RemediationKind


def _rollback_action(engine: Engine):
    actions, _ = engine.remediation.list_actions()
    return next(a for a in actions if a.kind == RemediationKind.ROLLBACK)


def test_remediation_lifecycle_is_reversible_and_audited(seeded: Engine):
    rollback = _rollback_action(seeded)
    applied = seeded.remediation.approve(rollback.action_id, "alice")
    assert applied.status == ActionStatus.APPLIED
    assert applied.approved_by == "alice"
    assert seeded.store.get_node("A2").drift.status == DriftStatus.CONTAINED

    reverted = seeded.remediation.revert(rollback.action_id, "alice")
    assert reverted.status == ActionStatus.REVERTED

    entries, _ = seeded.audit.entries(limit=500)
    actions_logged = {e.action for e in entries}
    assert {"remediation.proposed", "remediation.approved",
            "remediation.applied", "remediation.reverted"} <= actions_logged
    actors = {e.actor for e in entries if e.action == "remediation.approved"}
    assert "alice" in actors


def test_invalid_transitions_are_rejected(seeded: Engine):
    rollback = _rollback_action(seeded)
    seeded.remediation.approve(rollback.action_id, "alice")
    with pytest.raises(ValueError):
        seeded.remediation.approve(rollback.action_id, "alice")
    with pytest.raises(ValueError):
        seeded.remediation.reject(rollback.action_id, "alice")
    with pytest.raises(KeyError):
        seeded.remediation.approve("act::missing", "alice")


def test_message_filter_zeroes_edge_and_revert_restores(seeded: Engine):
    actions, _ = seeded.remediation.list_actions()
    mf = next(a for a in actions
              if a.kind == RemediationKind.MESSAGE_FILTER
              and a.params.get("edge_to") == "A3")
    seeded.remediation.approve(mf.action_id, "bob")
    edges = {(e.src, e.dst): e.weight
             for e in seeded.store.edges_for_nodes(["A2", "A3"])}
    assert edges[("A2", "A3")] == 0.0
    seeded.remediation.revert(mf.action_id, "bob")
    edges = {(e.src, e.dst): e.weight
             for e in seeded.store.edges_for_nodes(["A2", "A3"])}
    assert edges[("A2", "A3")] == 1.0


def test_audit_chain_is_tamper_evident(seeded: Engine):
    assert seeded.audit.verify_chain()["valid"] is True

    with seeded.session_factory() as s:
        row = s.scalars(select(AuditRow).order_by(AuditRow.seq).limit(1)).one()
        s.execute(update(AuditRow).where(AuditRow.seq == row.seq)
                  .values(detail="tampered"))
        s.commit()

    verdict = seeded.audit.verify_chain()
    assert verdict["valid"] is False
    assert verdict["broken_at_seq"] is not None


def test_compliance_report_maps_frameworks(seeded: Engine):
    rep = seeded.compliance()
    assert rep["chain_valid"] is True
    assert any("EU AI Act" in k for k in rep["coverage_by_clause"])
    assert any("NIST AI RMF" in k for k in rep["coverage_by_clause"])


def test_retention_prunes_nodes_but_never_audit(seeded: Engine):
    audit_before = seeded.audit.count()
    nodes_before = seeded.store.node_count()
    result = seeded.run_retention(days=1)
    # fixture spans have year-2023 timestamps, background spans are "now"
    assert result["nodes"] > 0
    assert seeded.store.node_count() < nodes_before
    assert seeded.audit.count() >= audit_before
