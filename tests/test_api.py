from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import create_app

from . import fixtures
from .conftest import make_settings

ADMIN_KEY = "test-admin-key-000111222333"


@pytest.fixture
def client(tmp_path):
    settings = make_settings(tmp_path, auth_enabled=True,
                             bootstrap_admin_key=ADMIN_KEY,
                             max_batch_size=100)
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _mkkey(client, role: str) -> str:
    r = client.post("/api/admin/keys",
                    headers={"X-API-Key": ADMIN_KEY},
                    json={"name": f"test-{role}", "role": role})
    assert r.status_code == 200, r.text
    return r.json()["api_key"]


def _spans_payload():
    return [s.model_dump(mode="json") for s in fixtures.incident_spans()]


def test_health_is_public(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["database_ok"] is True


def test_requests_without_key_are_rejected(client):
    assert client.get("/api/traces").status_code == 401
    assert client.post("/api/spans", json=[]).status_code == 401


def test_invalid_key_is_rejected(client):
    r = client.get("/api/traces", headers={"X-API-Key": "twin_bogus_nope"})
    assert r.status_code == 401


def test_rbac_enforced(client):
    viewer = _mkkey(client, "viewer")
    ingest = _mkkey(client, "ingest")

    r = client.post("/api/spans", headers={"X-API-Key": viewer},
                    json=_spans_payload())
    assert r.status_code == 403

    r = client.get("/api/traces", headers={"X-API-Key": ingest})
    assert r.status_code == 403

    r = client.get("/api/admin/keys", headers={"X-API-Key": viewer})
    assert r.status_code == 403

    r = client.post("/api/spans", headers={"X-API-Key": ingest},
                    json=_spans_payload())
    assert r.status_code == 200


def test_bearer_token_also_works(client):
    r = client.get("/api/traces",
                   headers={"Authorization": f"Bearer {ADMIN_KEY}"})
    assert r.status_code == 200


def test_ingest_idempotency_and_validation(client):
    ingest = _mkkey(client, "ingest")
    payload = _spans_payload()

    r = client.post("/api/spans", headers={"X-API-Key": ingest}, json=payload)
    assert r.status_code == 200
    assert len(r.json()["ingested"]) == len(payload)

    r = client.post("/api/spans", headers={"X-API-Key": ingest}, json=payload)
    assert r.json()["ingested"] == []
    assert len(r.json()["skipped_duplicates"]) == len(payload)

    r = client.post("/api/spans", headers={"X-API-Key": ingest},
                    json=[{"span_id": "x"}])
    assert r.status_code == 422
    assert r.json()["error"]["type"] == "validation_error"


def test_batch_cap(client):
    ingest = _mkkey(client, "ingest")
    span = fixtures.incident_spans()[0].model_dump(mode="json")
    batch = []
    for i in range(101):
        s = dict(span)
        s["span_id"] = f"cap-{i}"
        batch.append(s)
    r = client.post("/api/spans", headers={"X-API-Key": ingest}, json=batch)
    assert r.status_code == 413


def test_full_incident_flow_over_api(client):
    operator = _mkkey(client, "operator")
    h = {"X-API-Key": operator}

    r = client.post("/api/spans", headers=h, json=_spans_payload())
    assert r.status_code == 200

    incidents = client.get("/api/incidents", headers=h).json()
    assert incidents["total"] == 1
    assert incidents["incidents"][0]["root_cause_node"] == "A2"

    nar = client.get("/api/narrative", headers=h).json()
    assert nar["root_cause_node"] == "A2"

    graph = client.get("/api/graph",
                       params={"trace": fixtures.INCIDENT_TRACE},
                       headers=h).json()
    assert len(graph["nodes"]) == 7

    assert client.get("/api/node/NOPE", headers=h).status_code == 404

    wi = client.get("/api/whatif/A2", headers=h).json()
    assert set(wi["contained_nodes"]) == {"A3", "A5"}

    acts = client.get("/api/remediation", headers=h).json()["actions"]
    rb = next(a for a in acts if a["kind"] == "rollback")["action_id"]

    assert client.post(f"/api/remediation/{rb}/approve",
                       headers=h).status_code == 200
    assert client.post(f"/api/remediation/{rb}/approve",
                       headers=h).status_code == 409
    assert client.post(f"/api/remediation/{rb}/revert",
                       headers=h).status_code == 200
    assert client.post(f"/api/remediation/{rb}/revert",
                       headers=h).status_code == 409

    audit = client.get("/api/audit", headers=h,
                       params={"limit": 500}).json()
    approvers = {e["actor"] for e in audit["entries"]
                 if e["action"] == "remediation.approved"}
    assert "test-operator" in approvers

    assert client.get("/api/audit/verify", headers=h).json()["valid"] is True

    guard = client.get("/api/guard", headers=h).json()
    assert guard["blocked_count"] == 1

    cost = client.get("/api/cost", headers=h).json()
    assert cost["measured"] is True


def test_admin_key_lifecycle(client):
    viewer_key = _mkkey(client, "viewer")
    keys = client.get("/api/admin/keys",
                      headers={"X-API-Key": ADMIN_KEY}).json()["keys"]
    target = next(k for k in keys if k["name"] == "test-viewer")

    r = client.post(f"/api/admin/keys/{target['key_id']}/disable",
                    headers={"X-API-Key": ADMIN_KEY})
    assert r.status_code == 200
    r = client.get("/api/traces", headers={"X-API-Key": viewer_key})
    assert r.status_code == 401


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"twin_spans_ingested_total" in r.content
