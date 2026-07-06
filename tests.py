"""End-to-end tests for the drift-containment pipeline.

Run standalone:   python tests.py
Or with pytest:   pytest tests.py
Covers the load-bearing claims of the brief: detection catches the injection,
attribution finds the true root cause, blast radius is correct, remediation is
reversible and audited, the audit chain is tamper-evident, what-if is advisory,
and the cost envelope stays low-single-digit under realistic load.
"""
import os
import subprocess
import sys
import tempfile

from twin import Engine
from twin.detection import StubJudge
from twin.models import ActionStatus, DriftStatus, RemediationKind, Privilege


def fresh() -> Engine:
    e = Engine()
    e.seed()
    return e


def test_injection_is_detected():
    e = fresh()
    a2 = e.store.get_node("A2")
    assert a2.drift.status == DriftStatus.FLAGGED
    assert a2.drift.risk_type == "prompt_injection"


def test_dangerous_endpoint_flagged_on_behaviour_not_words():
    e = fresh()
    a7 = e.store.get_node("A7")
    # declared intent is benign; behaviour (transfer_funds) is not -> high gap
    assert a7.drift.stated_vs_revealed >= 0.6
    assert a7.drift.risk_type == "tool_misuse"
    assert a7.drift.judge is not None and a7.drift.judge.serves_goal is False


def test_clean_control_branch_stays_clean():
    e = fresh()
    assert e.store.get_node("A1").drift.status == DriftStatus.OK   # orchestrator
    assert e.store.get_node("A6").drift.status == DriftStatus.OK   # low-priv notifier


def test_blast_radius_and_root_cause():
    e = fresh()
    assert set(e.store.blast_radius("A2")) == {"A3", "A5", "A7"}
    nar = e.incident_narrative
    assert nar.root_cause_node == "A2"
    assert nar.propagation_path == ["A2", "A3", "A5", "A7"]


def test_high_privilege_always_covered():
    e = fresh()
    # A7 is HIGH privilege -> must never be sampled out
    assert e.store.get_node("A7").drift.tier_reached.value != "sampled"
    assert e.store.get_node("A7").privilege == Privilege.HIGH


def test_cost_envelope_is_low_single_digit():
    e = fresh()
    c = e.cost()
    assert c["overhead_pct"] < 5.0, c
    assert c["cache_hits"] > 0            # dedup actually fired
    # the incident chain (A2->A3->A5->A7) is the only deep-escalated traffic
    assert c["tier_counts"]["deep_escalation"] <= 5, c


def test_remediation_is_reversible_and_audited():
    e = fresh()
    rollback = next(a for a in e.remediation.all_actions()
                    if a.kind == RemediationKind.ROLLBACK)
    e.remediation.approve(rollback.action_id)
    assert e.store.get_node("A2").drift.status == DriftStatus.CONTAINED
    assert e.remediation.get(rollback.action_id).status == ActionStatus.APPLIED
    e.remediation.revert(rollback.action_id)
    assert e.remediation.get(rollback.action_id).status == ActionStatus.REVERTED


def test_audit_chain_is_tamper_evident():
    e = fresh()
    assert e.audit.verify_chain() is True
    # tamper with an entry and prove the chain breaks
    e.audit._entries[2].detail = "tampered"
    assert e.audit.verify_chain() is False


def test_whatif_saves_inherited_but_not_intrinsic():
    e = fresh()
    wi = e.whatif("A2")
    assert set(wi.contained_nodes) == {"A3", "A5"}     # inherited -> recoverable
    assert set(wi.remediated_blast_radius) == {"A7"}   # own fault -> still needs action


def test_compliance_report_maps_frameworks():
    e = fresh()
    rep = e.compliance()
    assert rep["chain_valid"] is True
    assert any("EU AI Act" in k for k in rep["coverage_by_clause"])
    assert any("NIST AI RMF" in k for k in rep["coverage_by_clause"])


def test_detection_is_structural_not_keyword():
    """Held-out injections with NO planted marker phrases must still flag, via the
    structural stated-vs-revealed signal — proving detection is not a keyword mirror."""
    from twin import scenario_variants
    for vid, spans in scenario_variants.all_variants():
        e = Engine()
        for s in spans:
            e.ingest(s)
        retriever = e.store.get_node(f"{vid}-A2")
        text = (retriever.declared_intent + " " + retriever.output + " " +
                " ".join(retriever.effects)).lower()
        # no injection-marker keyword is present in this variant...
        assert not any(m in text for m in StubJudge.INJECTION_MARKERS), vid
        # ...yet it is caught structurally as an introduced prompt injection
        assert retriever.drift.status == DriftStatus.FLAGGED, (vid, retriever.drift.score)
        assert retriever.drift.risk_type == "prompt_injection", vid
        assert retriever.drift.injection_introduced is True, vid
        assert retriever.drift.svr_primary == "injection_introduced", vid


def test_persistence_survives_process_restart():
    """The twin is durable, not just backed by code that could be: write in one
    process, read the SAME graph back in a SEPARATE process, no re-ingest."""
    db = os.path.join(tempfile.gettempdir(), "twin_persist_test.db")
    for p in (db, db + "-wal", db + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    try:
        w = subprocess.run([sys.executable, "persistence_check.py", "write", db],
                           capture_output=True, text=True)
        assert w.returncode == 0, w.stderr
        r = subprocess.run([sys.executable, "persistence_check.py", "read", db],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "PERSISTENCE OK" in r.stdout, r.stdout
    finally:
        for p in (db, db + "-wal", db + "-shm"):
            if os.path.exists(p):
                os.remove(p)


def test_inline_rail_blocks_transfer_pre_execution():
    """The dangerous transfer is STOPPED synchronously by the inline rail, not
    merely flagged after the fact."""
    e = fresh()
    a7 = e.store.get_node("A7")
    assert a7.blocked is True
    assert all("BLOCKED" in eff for eff in a7.effects), a7.effects
    assert e.guard_report()["blocked_count"] == 1
    # the monitor still records the attempt for attribution (defence in depth)
    assert a7.drift.status == DriftStatus.FLAGGED


def test_whatif_is_a_real_replay_not_a_partition():
    """The what-if recomputes downstream drift by re-running detection with the
    root's context corrected — inherited nodes' projected score genuinely drops."""
    e = fresh()
    wi = e.whatif("A2")
    # A3 inherited the contamination: its projected drift falls sharply after the fix
    assert wi.projected_drift_after["A3"] < 0.35 <= wi.projected_drift_before["A3"]
    assert "A3" in wi.contained_nodes and "A5" in wi.contained_nodes
    # A7 owns an undeclared dangerous tool: it stays flagged despite the fix
    assert "A7" in wi.remediated_blast_radius
    assert wi.projected_drift_after["A7"] >= 0.6


def test_api_hardening():
    """Endpoint contract: 404 / 422 / 409, idempotent ingest, error envelope,
    verified audit. Skipped if the FastAPI TestClient stack is unavailable."""
    try:
        from fastapi.testclient import TestClient
    except Exception:  # pragma: no cover - optional dependency
        print("    (skipped: TestClient unavailable)")
        return
    db = os.path.join(tempfile.gettempdir(), "twin_api_test.db")
    for p in (db, db + "-wal", db + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    os.environ["TWIN_DB"] = db
    import app as appmod
    try:
        c = TestClient(appmod.app)
        assert c.get("/api/health").json()["durable"] is True
        assert c.get("/api/node/NOPE").status_code == 404
        # malformed span -> 422 with the standard envelope
        r = c.post("/api/spans", json=[{"span_id": "x"}])
        assert r.status_code == 422 and r.json()["error"]["type"] == "validation_error"
        # idempotent ingest: a replayed span is skipped, not double-counted
        span = {"span_id": "ZZ1", "trace_id": "t", "agent_id": "a",
                "task_spec": "read", "declared_intent": "read", "output": "read done"}
        assert c.post("/api/spans", json=[span]).json()["ingested"] == ["ZZ1"]
        assert c.post("/api/spans", json=[span]).json()["skipped_duplicates"] == ["ZZ1"]
        # invalid remediation transitions -> 409
        acts = c.get("/api/remediation").json()["actions"]
        rb = next(a for a in acts if a["kind"] == "rollback")["action_id"]
        assert c.post(f"/api/remediation/{rb}/approve").status_code == 200
        assert c.post(f"/api/remediation/{rb}/approve").status_code == 409
        assert c.post(f"/api/remediation/{rb}/revert").status_code == 200
        assert c.post(f"/api/remediation/{rb}/revert").status_code == 409
        assert c.get("/api/audit").json()["chain_valid"] is True
    finally:
        appmod.engine.close()
        os.environ.pop("TWIN_DB", None)
        for p in (db, db + "-wal", db + "-shm"):
            if os.path.exists(p):
                os.remove(p)


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    passed = 0
    for t in ALL:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as ex:
            print(f"  FAIL  {t.__name__}: {ex}")
        except Exception as ex:  # noqa
            print(f"  ERROR {t.__name__}: {ex!r}")
    print(f"\n{passed}/{len(ALL)} tests passed")
