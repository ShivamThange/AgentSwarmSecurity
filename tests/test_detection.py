from __future__ import annotations

from twin.detection import StubJudge
from twin.engine import Engine
from twin.models import DriftStatus, Privilege

from . import fixtures
from .conftest import make_settings


def test_injection_is_detected(seeded: Engine):
    a2 = seeded.store.get_node("A2")
    assert a2.drift.status == DriftStatus.FLAGGED
    assert a2.drift.risk_type == "prompt_injection"
    assert a2.drift.injection_introduced is True


def test_dangerous_endpoint_flagged_on_behaviour_not_words(seeded: Engine):
    a7 = seeded.store.get_node("A7")
    assert a7.drift.stated_vs_revealed >= 0.6
    assert a7.drift.risk_type == "tool_misuse"
    assert a7.drift.judge is not None
    assert a7.drift.judge.serves_goal is False


def test_clean_control_branch_stays_clean(seeded: Engine):
    assert seeded.store.get_node("A1").drift.status == DriftStatus.OK
    assert seeded.store.get_node("A6").drift.status == DriftStatus.OK


def test_blast_radius_and_root_cause(seeded: Engine):
    assert set(seeded.store.blast_radius("A2")) == {"A3", "A5", "A7"}
    nar = seeded.narrative()
    assert nar is not None
    assert nar.root_cause_node == "A2"
    assert nar.propagation_path == ["A2", "A3", "A5", "A7"]


def test_high_privilege_always_covered(seeded: Engine):
    a7 = seeded.store.get_node("A7")
    assert a7.drift.tier_reached.value != "sampled"
    assert a7.privilege == Privilege.HIGH


def test_cost_ledger_is_measured_not_modelled(seeded: Engine):
    c = seeded.cost()
    assert c["measured"] is True
    assert c["cache_hits"] > 0
    assert c["spans_seen"] == 67
    assert c["baseline_tokens"] > 0
    assert c["added_tokens"] == 0
    assert c["overhead_pct"] == 0.0


def test_inline_rail_blocks_transfer_pre_execution(seeded: Engine):
    a7 = seeded.store.get_node("A7")
    assert a7.blocked is True
    assert all("BLOCKED" in eff for eff in a7.effects), a7.effects
    assert seeded.guard_report()["blocked_count"] == 1
    assert a7.drift.status == DriftStatus.FLAGGED


def test_detection_is_structural_not_keyword(tmp_path):
    for vid, spans in fixtures.all_variants():
        e = Engine(make_settings(tmp_path,
                                 database_url=f"sqlite:///{tmp_path}/{vid}.db"))
        try:
            for s in spans:
                e.ingest(s)
            retriever = e.store.get_node(f"{vid}-A2")
            text = (retriever.declared_intent + " " + retriever.output + " " +
                    " ".join(retriever.effects)).lower()
            assert not any(m in text for m in StubJudge.INJECTION_MARKERS), vid
            assert retriever.drift.status == DriftStatus.FLAGGED, \
                (vid, retriever.drift.score)
            assert retriever.drift.risk_type == "prompt_injection", vid
            assert retriever.drift.injection_introduced is True, vid
            assert retriever.drift.svr_primary == "injection_introduced", vid
        finally:
            e.close()


def test_whatif_saves_inherited_but_not_intrinsic(seeded: Engine):
    wi = seeded.whatif("A2")
    assert set(wi.contained_nodes) == {"A3", "A5"}
    assert set(wi.remediated_blast_radius) == {"A7"}
    assert wi.projected_drift_after["A3"] < 0.35 <= wi.projected_drift_before["A3"]
    assert wi.projected_drift_after["A7"] >= 0.6


def test_flag_auto_proposes_remediation(seeded: Engine):
    actions, total = seeded.remediation.list_actions()
    assert total >= 2
    kinds = {(a.node_id, a.kind.value) for a in actions}
    assert ("A2", "rollback") in kinds
    assert ("A2", "quarantine") in kinds
    assert ("A2", "message_filter") in kinds


def test_incidents_listing(seeded: Engine):
    incidents, total = seeded.incidents()
    assert total == 1
    inc = incidents[0]
    assert inc["trace_id"] == fixtures.INCIDENT_TRACE
    assert inc["root_cause_node"] == "A2"
    assert inc["flagged_count"] >= 2


def test_taxonomy_enrichment(seeded: Engine):
    a2 = seeded.store.get_node("A2")
    assert a2.drift.risk_id == "RT1.1"
    assert a2.drift.owasp_ref == "LLM01"
