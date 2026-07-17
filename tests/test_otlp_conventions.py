from __future__ import annotations

import json

from twin.otel_ingest import parse_otlp


def _kv(key, value):
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _span(attributes, span_id="aa11", name="llm", trace="t1", parent=None):
    sp = {"traceId": trace, "spanId": span_id, "name": name,
          "attributes": attributes}
    if parent:
        sp["parentSpanId"] = parent
    return sp


def _parse(spans):
    payload = {"resourceSpans": [{
        "resource": {"attributes": [_kv("service.name", "svc")]},
        "scopeSpans": [{"spans": spans}]}]}
    return parse_otlp(json.dumps(payload).encode(), "application/json")


def test_openinference_llm_span_maps_output_and_tool_calls():
    spans = _parse([_span([
        _kv("openinference.span.kind", "LLM"),
        _kv("graph.node.id", "retriever-node"),
        _kv("input.value", "Retrieve Q3 vendor invoices."),
        _kv("llm.output_messages.0.message.role", "assistant"),
        _kv("llm.output_messages.0.message.content",
            "Retrieved 42 invoices."),
        _kv("llm.output_messages.0.message.tool_calls.0.tool_call.function.name",
            "query_datastore"),
        _kv("llm.output_messages.0.message.tool_calls.0.tool_call.function."
            "arguments", '{"period": "Q3"}'),
        _kv("llm.token_count.prompt", 800),
        _kv("llm.token_count.completion", 120),
    ])])
    assert len(spans) == 1
    s = spans[0]
    assert s.agent_id == "retriever-node"
    assert s.task_spec == "Retrieve Q3 vendor invoices."
    assert s.output == "Retrieved 42 invoices."
    assert s.tool_calls[0].name == "query_datastore"
    assert s.tool_calls[0].args["period"] == "Q3"
    assert s.meta["baseline_tokens"] == 920
    assert s.meta["instrumentation"] == "openinference"


def test_langfuse_span_maps_input_output_and_vendor():
    spans = _parse([_span([
        _kv("langfuse.trace.name", "reconciliation-agent"),
        _kv("langfuse.observation.type", "generation"),
        _kv("langfuse.observation.input", "Summarise the ticket."),
        _kv("langfuse.observation.output", "Ticket summarised."),
    ])])
    s = spans[0]
    assert s.agent_id == "reconciliation-agent"
    assert s.agent_role == "generation"
    assert s.task_spec == "Summarise the ticket."
    assert s.output == "Ticket summarised."
    assert s.meta["instrumentation"] == "langfuse"


def test_langgraph_parent_becomes_input_edge():
    spans = _parse([
        _span([_kv("graph.node.id", "planner")], span_id="p1", name="plan"),
        _span([_kv("graph.node.id", "worker"),
               _kv("graph.node.parent_id", "p1")],
              span_id="w1", name="work"),
    ])
    worker = next(s for s in spans if s.span_id == "w1")
    assert worker.inputs_from == ["p1"]


def test_input_messages_used_when_no_explicit_input():
    spans = _parse([_span([
        _kv("llm.input_messages.0.message.role", "system"),
        _kv("llm.input_messages.0.message.content", "You are helpful."),
        _kv("llm.input_messages.1.message.role", "user"),
        _kv("llm.input_messages.1.message.content", "Book me a flight."),
    ])])
    assert spans[0].task_spec == "Book me a flight."


def test_twin_native_attributes_take_precedence():
    spans = _parse([_span([
        _kv("twin.agent_id", "native-agent"),
        _kv("graph.node.id", "oi-node"),
        _kv("twin.output", "native output"),
        _kv("output.value", "oi output"),
    ])])
    s = spans[0]
    assert s.agent_id == "native-agent"
    assert s.output == "native output"
