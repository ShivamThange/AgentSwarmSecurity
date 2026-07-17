from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from .models import Privilege, Span, ToolCall

log = logging.getLogger(__name__)

PROTOBUF_CONTENT_TYPE = "application/x-protobuf"
JSON_CONTENT_TYPE = "application/json"

# Flattened OpenInference message attribute, e.g.
#   llm.output_messages.0.message.content
_MSG_PREFIX_RE = re.compile(r"^llm\.(input|output)_messages\.(\d+)\.message\.")


def _first(attrs: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v = attrs.get(k)
        if v is not None and v != "":
            return v
    return None


class OTLPParseError(ValueError):
    pass


def _any_value_json(v: dict) -> Any:
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "arrayValue" in v:
        return [_any_value_json(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {kv["key"]: _any_value_json(kv.get("value", {}))
                for kv in v["kvlistValue"].get("values", [])}
    if "bytesValue" in v:
        return v["bytesValue"]
    return None


def _attrs_json(attr_list: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kv in attr_list or []:
        out[kv.get("key", "")] = _any_value_json(kv.get("value", {}))
    return out


def _any_value_pb(v) -> Any:
    which = v.WhichOneof("value")
    if which is None:
        return None
    val = getattr(v, which)
    if which == "array_value":
        return [_any_value_pb(x) for x in val.values]
    if which == "kvlist_value":
        return {kv.key: _any_value_pb(kv.value) for kv in val.values}
    return val


def _attrs_pb(attr_list) -> dict[str, Any]:
    return {kv.key: _any_value_pb(kv.value) for kv in attr_list}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                return [value]
        return [value] if value else []
    return [value]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _indexed_messages(
    attrs: dict[str, Any], kind: str
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[int, dict[str, Any]]]]:
    """Reassemble OpenInference flattened chat messages of one ``kind``.

    Returns ``(messages, tool_calls)`` where ``messages`` maps message index to
    ``{"role", "content"}`` and ``tool_calls`` maps message index to a dict of
    tool-call index -> ``{"name", "arguments"}``.
    """
    prefix = f"llm.{kind}_messages."
    messages: dict[int, dict[str, Any]] = {}
    tool_calls: dict[int, dict[int, dict[str, Any]]] = {}
    for key, val in attrs.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].split(".")
        if len(parts) < 3 or parts[1] != "message":
            continue
        try:
            midx = int(parts[0])
        except ValueError:
            continue
        field = parts[2]
        if field in ("content", "role"):
            messages.setdefault(midx, {})[field] = _as_str(val)
        elif (field == "tool_calls" and len(parts) >= 7
              and parts[4] == "tool_call" and parts[5] == "function"):
            try:
                tcidx = int(parts[3])
            except ValueError:
                continue
            tc = tool_calls.setdefault(midx, {}).setdefault(tcidx, {})
            if parts[6] == "name":
                tc["name"] = _as_str(val)
            elif parts[6] == "arguments":
                tc["arguments"] = val
    return messages, tool_calls


def _message_content(attrs: dict[str, Any], kind: str,
                     role: Optional[str] = None) -> str:
    messages, _ = _indexed_messages(attrs, kind)
    if not messages:
        return ""
    candidates = sorted(messages.items())
    if role is not None:
        matching = [(i, m) for i, m in candidates
                    if str(m.get("role", "")).lower() == role]
        if matching:
            candidates = matching
    for _idx, msg in reversed(candidates):
        content = msg.get("content")
        if content:
            return content
    return ""


def _openinference_tool_calls(attrs: dict[str, Any]) -> list[ToolCall]:
    _, tool_calls = _indexed_messages(attrs, "output")
    calls: list[ToolCall] = []
    for _midx, tcs in sorted(tool_calls.items()):
        for _tcidx, tc in sorted(tcs.items()):
            name = tc.get("name")
            if not name:
                continue
            args: dict[str, Any] = {}
            raw_args = tc.get("arguments")
            if raw_args:
                try:
                    parsed = (json.loads(raw_args) if isinstance(raw_args, str)
                              else raw_args)
                    if isinstance(parsed, dict):
                        args = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            calls.append(ToolCall(name=str(name), args=args))
    return calls


def _tool_calls_from(attrs: dict[str, Any]) -> list[ToolCall]:
    raw = attrs.get("twin.tool_calls")
    calls: list[ToolCall] = []
    if raw:
        try:
            items = json.loads(raw) if isinstance(raw, str) else raw
            for item in items:
                if isinstance(item, str):
                    calls.append(ToolCall(name=item))
                elif isinstance(item, dict) and item.get("name"):
                    calls.append(ToolCall(name=item["name"],
                                          args=item.get("args") or {}))
        except (json.JSONDecodeError, TypeError):
            log.warning("unparseable twin.tool_calls attribute")
    if calls:
        return calls

    tool_name = attrs.get("gen_ai.tool.name") or attrs.get("tool.name")
    if tool_name:
        args: dict[str, Any] = {}
        raw_args = (attrs.get("gen_ai.tool.call.arguments")
                    or attrs.get("tool.parameters"))
        if raw_args:
            try:
                parsed = (json.loads(raw_args) if isinstance(raw_args, str)
                          else raw_args)
                if isinstance(parsed, dict):
                    args = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        calls.append(ToolCall(name=str(tool_name), args=args))
        return calls

    # OpenInference / Langfuse LLM spans carry tool calls inside the flattened
    # output-message attributes.
    return _openinference_tool_calls(attrs)


def _instrumentation_vendor(attrs: dict[str, Any]) -> Optional[str]:
    for key in attrs:
        if key.startswith("langfuse."):
            return "langfuse"
    for key in attrs:
        if key.startswith("openinference.") or key.startswith("llm.") \
                or key.startswith("input.") or key.startswith("output."):
            return "openinference"
    for key in attrs:
        if key.startswith("gen_ai."):
            return "otel-genai"
    return None


def _privilege_from(attrs: dict[str, Any]) -> Privilege:
    raw = str(attrs.get("twin.privilege", "") or "").lower()
    try:
        return Privilege(raw)
    except ValueError:
        return Privilege.LOW


def _baseline_tokens(attrs: dict[str, Any]) -> int:
    for total_key in ("gen_ai.usage.total_tokens", "llm.token_count.total"):
        v = attrs.get(total_key)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    total = 0
    for key in ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens",
                "gen_ai.usage.prompt_tokens", "gen_ai.usage.completion_tokens",
                "llm.token_count.prompt", "llm.token_count.completion"):
        v = attrs.get(key)
        if v is not None:
            try:
                total += int(v)
            except (TypeError, ValueError):
                pass
    return total


def _map_span(trace_id: str, span_id: str, parent_span_id: Optional[str],
              name: str, start_ns: int, attrs: dict[str, Any],
              link_span_ids: list[str],
              resource_attrs: dict[str, Any]) -> Span:
    agent_id = _as_str(
        _first(attrs, "twin.agent_id", "gen_ai.agent.name", "gen_ai.agent.id",
               "graph.node.id", "langfuse.trace.name")
        or resource_attrs.get("service.name")
        or "unknown-agent")
    task_spec = _as_str(
        _first(attrs, "twin.task_spec", "gen_ai.agent.task", "input.value",
               "langfuse.observation.input")
        or _message_content(attrs, "input", role="user")
        or name)
    output = _as_str(
        _first(attrs, "twin.output", "gen_ai.completion", "output.value",
               "langfuse.observation.output")
        or _message_content(attrs, "output"))

    inputs_from = [str(x) for x in _as_list(attrs.get("twin.inputs_from"))]
    if not inputs_from:
        inputs_from = list(link_span_ids)
        parent = parent_span_id or _as_str(attrs.get("graph.node.parent_id"))
        if parent:
            inputs_from.append(parent)
    inputs_from = list(dict.fromkeys(x for x in inputs_from if x))

    logprob = attrs.get("twin.logprob_confidence")
    try:
        logprob = float(logprob) if logprob is not None else None
    except (TypeError, ValueError):
        logprob = None

    schema = [str(x) for x in
              _as_list(attrs.get("twin.expected_output_schema"))] or None

    meta: dict[str, Any] = {"source": "otlp", "otel_span_name": name}
    vendor = _instrumentation_vendor(attrs)
    if vendor:
        meta["instrumentation"] = vendor
    baseline = _baseline_tokens(attrs)
    if baseline > 0:
        meta["baseline_tokens"] = baseline
    model = attrs.get("gen_ai.request.model") or attrs.get("llm.model_name")
    if model:
        meta["model"] = _as_str(model)

    return Span(
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        agent_id=agent_id,
        agent_role=_as_str(
            _first(attrs, "twin.agent_role", "gen_ai.agent.description",
                   "openinference.span.kind", "langfuse.observation.type")),
        privilege=_privilege_from(attrs),
        task_spec=task_spec,
        workflow=_as_str(attrs.get("twin.workflow", "")),
        declared_intent=_as_str(attrs.get("twin.declared_intent")),
        tool_calls=_tool_calls_from(attrs),
        effects=[_as_str(x) for x in _as_list(attrs.get("twin.effects"))],
        output=output,
        logprob_confidence=logprob,
        expected_output_schema=schema,
        inputs_from=inputs_from,
        timestamp=(start_ns / 1e9) if start_ns else None,
        meta=meta,
    )


def _parse_json_payload(body: bytes) -> list[Span]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OTLPParseError(f"invalid OTLP JSON: {exc}") from exc
    spans: list[Span] = []
    for rs in payload.get("resourceSpans", []):
        resource_attrs = _attrs_json(
            (rs.get("resource") or {}).get("attributes", []))
        for ss in rs.get("scopeSpans", []) or rs.get("instrumentationLibrarySpans", []):
            for sp in ss.get("spans", []):
                trace_id = str(sp.get("traceId", ""))
                span_id = str(sp.get("spanId", ""))
                if not span_id:
                    continue
                parent = str(sp.get("parentSpanId", "") or "") or None
                links = [str(ln.get("spanId", ""))
                         for ln in sp.get("links", []) if ln.get("spanId")]
                try:
                    start_ns = int(sp.get("startTimeUnixNano", 0) or 0)
                except (TypeError, ValueError):
                    start_ns = 0
                spans.append(_map_span(
                    trace_id, span_id, parent, str(sp.get("name", "")),
                    start_ns, _attrs_json(sp.get("attributes", [])),
                    links, resource_attrs))
    return spans


def _parse_protobuf_payload(body: bytes) -> list[Span]:
    try:
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )
    except ImportError as exc:
        raise OTLPParseError(
            "protobuf OTLP support requires the 'opentelemetry-proto' "
            "package; send OTLP JSON or install requirements") from exc
    req = ExportTraceServiceRequest()
    try:
        req.ParseFromString(body)
    except Exception as exc:
        raise OTLPParseError(f"invalid OTLP protobuf: {exc}") from exc
    spans: list[Span] = []
    for rs in req.resource_spans:
        resource_attrs = _attrs_pb(rs.resource.attributes)
        for ss in rs.scope_spans:
            for sp in ss.spans:
                span_id = sp.span_id.hex()
                if not span_id:
                    continue
                parent = sp.parent_span_id.hex() or None
                links = [ln.span_id.hex() for ln in sp.links if ln.span_id]
                spans.append(_map_span(
                    sp.trace_id.hex(), span_id, parent, sp.name,
                    sp.start_time_unix_nano, _attrs_pb(sp.attributes),
                    links, resource_attrs))
    return spans


def parse_otlp(body: bytes, content_type: str) -> list[Span]:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct == PROTOBUF_CONTENT_TYPE:
        return _parse_protobuf_payload(body)
    if ct in (JSON_CONTENT_TYPE, ""):
        return _parse_json_payload(body)
    raise OTLPParseError(
        f"unsupported content type '{content_type}'; use "
        f"{PROTOBUF_CONTENT_TYPE} or {JSON_CONTENT_TYPE}")


def empty_export_response(content_type: str) -> tuple[bytes, str]:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct == PROTOBUF_CONTENT_TYPE:
        try:
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceResponse,
            )
            return (ExportTraceServiceResponse().SerializeToString(),
                    PROTOBUF_CONTENT_TYPE)
        except ImportError:
            pass
    return b"{}", JSON_CONTENT_TYPE
