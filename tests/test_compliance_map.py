from __future__ import annotations

import json

from twin.audit import DEFAULT_COMPLIANCE_MAP, load_compliance_map
from twin.engine import Engine

from .conftest import make_settings


def test_load_none_returns_defaults():
    m = load_compliance_map(None)
    assert m == {k: list(v) for k, v in DEFAULT_COMPLIANCE_MAP.items()}
    # a copy, not the module-level object
    assert m is not DEFAULT_COMPLIANCE_MAP


def test_load_merges_and_overrides(tmp_path):
    path = tmp_path / "cmap.json"
    path.write_text(json.dumps({
        "detection": ["ACME-CTRL-1 — custom detection control"],
        "custom.action": ["ACME-CTRL-9 — bespoke clause"],
    }))
    m = load_compliance_map(str(path))
    assert m["detection"] == ["ACME-CTRL-1 — custom detection control"]
    assert m["custom.action"] == ["ACME-CTRL-9 — bespoke clause"]
    # untouched default keys survive
    assert "remediation.applied" in m


def test_malformed_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"detection": "not-a-list"}')
    m = load_compliance_map(str(path))
    assert m == {k: list(v) for k, v in DEFAULT_COMPLIANCE_MAP.items()}


def test_missing_file_falls_back(tmp_path):
    m = load_compliance_map(str(tmp_path / "does-not-exist.json"))
    assert "detection" in m


def test_engine_uses_configured_map(tmp_path):
    path = tmp_path / "cmap.json"
    path.write_text(json.dumps({
        "feedback": ["ACME-HUMAN-REVIEW — org policy 4.1"]}))
    settings = make_settings(tmp_path, compliance_map_path=str(path))
    e = Engine(settings)
    try:
        active = e.audit.active_map()
        assert "ACME-HUMAN-REVIEW — org policy 4.1" in active["clauses"]
        assert active["actions"]["feedback"] == [
            "ACME-HUMAN-REVIEW — org policy 4.1"]
    finally:
        e.close()
