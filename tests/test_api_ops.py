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
    r = client.post("/api/admin/keys", headers={"X-API-Key": ADMIN_KEY},
                    json={"name": f"test-{role}", "role": role})
    assert r.status_code == 200, r.text
    return r.json()["api_key"]


def _seed(client, headers):
    payload = [s.model_dump(mode="json") for s in fixtures.incident_spans()]
    assert client.post("/api/spans", headers=headers,
                       json=payload).status_code == 200


def test_escalation_and_compliance_map_endpoints(client):
    operator = _mkkey(client, "operator")
    h = {"X-API-Key": operator}
    _seed(client, h)

    esc = client.get("/api/escalation", headers=h)
    assert esc.status_code == 200
    body = esc.json()
    assert "ratio" in body and "anomaly" in body

    cmap = client.get("/api/compliance/map", headers=h).json()
    assert cmap["clause_count"] > 0
    assert "detection" in cmap["actions"]


def test_label_and_calibration_endpoints(client):
    operator = _mkkey(client, "operator")
    h = {"X-API-Key": operator}
    _seed(client, h)

    # A2 is the flagged injection node.
    r = client.post("/api/nodes/A2/label", headers=h,
                    json={"label": "true_positive", "note": "confirmed"})
    assert r.status_code == 200
    assert r.json()["label"] == "true_positive"

    # bad label -> 422; unknown node -> 404
    assert client.post("/api/nodes/A2/label", headers=h,
                       json={"label": "bogus"}).status_code == 422
    assert client.post("/api/nodes/NOPE/label", headers=h,
                       json={"label": "true_positive"}).status_code == 404

    cal = client.get("/api/calibration", headers=h).json()
    assert cal["total_labels"] == 1
    assert "overall" in cal


def test_label_requires_operate_permission(client):
    viewer = _mkkey(client, "viewer")
    operator = _mkkey(client, "operator")
    _seed(client, {"X-API-Key": operator})

    # viewer can read calibration but cannot label
    assert client.get("/api/calibration",
                      headers={"X-API-Key": viewer}).status_code == 200
    r = client.post("/api/nodes/A2/label", headers={"X-API-Key": viewer},
                    json={"label": "true_positive"})
    assert r.status_code == 403
