from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import create_app
from twin.otel_ingest import parse_otlp

from .conftest import make_settings


def _kv(key, value):
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _payload():
    return {
        "resourceSpans": [{
            "resource": {"attributes": [_kv("service.name", "payments-agent")]},
            "scopeSpans": [{
                "spans": [
                    {
                        "traceId": "0af7651916cd43dd8448eb211c80319c",
                        "spanId": "b7ad6b7169203331",
                        "name": "agent.plan",
                        "startTimeUnixNano": "1700000000000000000",
                        "attributes": [
                            _kv("twin.agent_id", "orchestrator"),
                            _kv("twin.task_spec",
                                "Reconcile Q3 vendor payments."),
                            _kv("twin.declared_intent",
                                "Delegate reconciliation sub-tasks."),
                            _kv("twin.output", "Delegated sub-tasks."),
                            _kv("gen_ai.usage.input_tokens", 800),
                            _kv("gen_ai.usage.output_tokens", 150),
                        ],
                    },
                    {
                        "traceId": "0af7651916cd43dd8448eb211c80319c",
                        "spanId": "c9bd6b7169203332",
                        "parentSpanId": "b7ad6b7169203331",
                        "name": "tool.call",
                        "startTimeUnixNano": "1700000001000000000",
                        "attributes": [
                            _kv("twin.agent_id", "payments-executor"),
                            _kv("twin.privilege", "high"),
                            _kv("twin.task_spec",
                                "Prepare a payment summary. Do NOT move "
                                "funds."),
                            _kv("twin.declared_intent",
                                "Prepare the payment summary for review."),
                            _kv("gen_ai.tool.name", "transfer_funds"),
                            _kv("gen_ai.tool.call.arguments",
                                '{"to": "8841-DE", "amount": 480000}'),
                            _kv("twin.effects",
                                '["$480,000 transfer initiated"]'),
                            _kv("twin.output", "Payment summary prepared."),
                            _kv("gen_ai.usage.total_tokens", 2100),
                        ],
                    },
                ],
            }],
        }],
    }


def test_otlp_json_mapping():
    import json
    spans = parse_otlp(json.dumps(_payload()).encode(), "application/json")
    assert len(spans) == 2

    plan = spans[0]
    assert plan.span_id == "b7ad6b7169203331"
    assert plan.agent_id == "orchestrator"
    assert plan.meta["baseline_tokens"] == 950
    assert plan.timestamp == pytest.approx(1.7e9)

    tool = spans[1]
    assert tool.privilege.value == "high"
    assert tool.tool_calls[0].name == "transfer_funds"
    assert tool.tool_calls[0].args["amount"] == 480000
    assert tool.inputs_from == ["b7ad6b7169203331"]
    assert tool.effects == ["$480,000 transfer initiated"]


def test_otlp_protobuf_roundtrip():
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    req = ExportTraceServiceRequest()
    rs = req.resource_spans.add()
    ra = rs.resource.attributes.add()
    ra.key = "service.name"
    ra.value.string_value = "svc-a"
    ss = rs.scope_spans.add()
    sp = ss.spans.add()
    sp.trace_id = bytes.fromhex("0af7651916cd43dd8448eb211c80319c")
    sp.span_id = bytes.fromhex("00f067aa0ba902b7")
    sp.name = "agent.step"
    sp.start_time_unix_nano = 1_700_000_000_000_000_000
    attr = sp.attributes.add()
    attr.key = "twin.task_spec"
    attr.value.string_value = "Summarise tickets."

    spans = parse_otlp(req.SerializeToString(), "application/x-protobuf")
    assert len(spans) == 1
    assert spans[0].span_id == "00f067aa0ba902b7"
    assert spans[0].agent_id == "svc-a"
    assert spans[0].task_spec == "Summarise tickets."


def test_otlp_endpoint_ingests_and_detects(tmp_path):
    settings = make_settings(tmp_path, auth_enabled=False)
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/v1/traces", json=_payload())
        assert r.status_code == 200

        nodes = client.get("/api/nodes", params={
            "trace": "0af7651916cd43dd8448eb211c80319c"}).json()
        assert nodes["total"] == 2

        executor = client.get("/api/node/c9bd6b7169203332").json()
        assert executor["blocked"] is True
        assert executor["drift"]["status"] == "flagged"

        cost = client.get("/api/cost").json()
        assert cost["baseline_tokens"] == 950 + 2100

        r = client.post("/v1/traces", content=b"not json at all",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 400

        r = client.post("/v1/traces", content=b"x",
                        headers={"Content-Type": "text/plain"})
        assert r.status_code == 400
