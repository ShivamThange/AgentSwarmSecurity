from __future__ import annotations

"""Convert AutoGen / LangGraph agent handoffs into twin Spans.

These converters take the plain message/state shapes those frameworks already
produce (dicts for AutoGen, message objects or dicts for LangGraph) rather than
importing the frameworks, so a single adapter works across framework versions
and neither package is a runtime dependency of the twin. Emit the resulting
spans to a running twin with :class:`TwinTap` or by POSTing them to
``/api/spans`` yourself.
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Iterable, Optional

from ..models import Privilege, Span, ToolCall

log = logging.getLogger(__name__)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _content(msg: Any) -> str:
    content = _get(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # LangChain content blocks: [{"type": "text", "text": ...}, ...]
        parts = []
        for block in content:
            text = _get(block, "text") if not isinstance(block, str) else block
            if text:
                parts.append(str(text))
        return " ".join(parts)
    return "" if content is None else str(content)


def _tool_calls(msg: Any) -> list[ToolCall]:
    raw = _get(msg, "tool_calls") or []
    calls: list[ToolCall] = []
    for tc in raw:
        # LangChain style: {"name": ..., "args": {...}}
        name = _get(tc, "name")
        args = _get(tc, "args")
        if not name:
            # OpenAI / AutoGen style: {"function": {"name", "arguments"}}
            fn = _get(tc, "function") or {}
            name = _get(fn, "name")
            args = _get(fn, "arguments")
        if not name:
            continue
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                args = parsed if isinstance(parsed, dict) else {"_raw": args}
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": args}
        calls.append(ToolCall(name=str(name), args=dict(args or {})))
    return calls


def _privilege(value: Any) -> Privilege:
    try:
        return Privilege(str(value).lower())
    except ValueError:
        return Privilege.LOW


def from_autogen_messages(
    messages: Iterable[Any],
    trace_id: str,
    *,
    task_spec: str = "",
    privileges: Optional[dict[str, str]] = None,
    id_prefix: str = "",
) -> list[Span]:
    """Convert an AutoGen/AG2 message list into a linked chain of spans.

    Each named turn becomes one span; ``inputs_from`` links it to the previous
    turn so cross-agent propagation is captured. ``privileges`` optionally maps
    an agent name to ``"high" | "medium" | "low"``.
    """
    privileges = privileges or {}
    spans: list[Span] = []
    prev_id: Optional[str] = None
    idx = 0
    for msg in messages:
        content = _content(msg)
        tool_calls = _tool_calls(msg)
        if not content and not tool_calls:
            continue
        agent_id = str(_get(msg, "name") or _get(msg, "role") or "agent")
        span_id = f"{id_prefix}{trace_id}-{idx}"
        spans.append(Span(
            span_id=span_id,
            trace_id=trace_id,
            agent_id=agent_id,
            agent_role=str(_get(msg, "role") or ""),
            privilege=_privilege(privileges.get(agent_id, "low")),
            task_spec=task_spec,
            declared_intent=content,
            output=content,
            tool_calls=tool_calls,
            inputs_from=[prev_id] if prev_id else [],
        ))
        prev_id = span_id
        idx += 1
    return spans


def from_langgraph_updates(
    updates: Iterable[Any],
    trace_id: str,
    *,
    task_spec: str = "",
    privileges: Optional[dict[str, str]] = None,
    id_prefix: str = "",
) -> list[Span]:
    """Convert a LangGraph ``stream(..., stream_mode="updates")`` sequence.

    Each streamed update is ``{node_name: state_delta}``; every node execution
    becomes a span linked to the node that ran before it. Output and tool calls
    are read from the last message in the node's ``messages`` delta when present.
    """
    privileges = privileges or {}
    spans: list[Span] = []
    prev_id: Optional[str] = None
    idx = 0
    for update in updates:
        if not isinstance(update, dict):
            continue
        for node_name, state in update.items():
            output = ""
            tool_calls: list[ToolCall] = []
            declared = ""
            if isinstance(state, dict):
                msgs = state.get("messages")
                if isinstance(msgs, list) and msgs:
                    output = _content(msgs[-1])
                    tool_calls = _tool_calls(msgs[-1])
                    if len(msgs) > 1:
                        declared = _content(msgs[-2])
                if not output:
                    for key in ("output", "result", "response"):
                        if state.get(key):
                            output = str(state[key])
                            break
            else:
                output = _content(state)
            span_id = f"{id_prefix}{trace_id}-{idx}"
            spans.append(Span(
                span_id=span_id,
                trace_id=trace_id,
                agent_id=str(node_name),
                agent_role="langgraph_node",
                privilege=_privilege(privileges.get(str(node_name), "low")),
                task_spec=task_spec,
                declared_intent=declared or output,
                output=output,
                tool_calls=tool_calls,
                inputs_from=[prev_id] if prev_id else [],
            ))
            prev_id = span_id
            idx += 1
    return spans


class TwinTap:
    """Minimal, dependency-free client that emits spans to a running twin.

    Uses the stdlib so it can be dropped into an agent process without pulling
    extra packages. Emission never raises into the caller's agent loop: on
    failure it logs and returns the error so instrumentation can't take down the
    workload it observes.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def emit(self, spans: list[Span]) -> dict:
        if not spans:
            return {"ingested": [], "skipped_duplicates": [], "failed": []}
        body = json.dumps([json.loads(s.model_dump_json())
                           for s in spans]).encode()
        req = urllib.request.Request(
            self.base_url + "/api/spans", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("X-API-Key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            log.warning("twin tap emit failed: HTTP %s %s", exc.code, detail)
            return {"error": f"http_{exc.code}", "detail": detail}
        except (urllib.error.URLError, OSError) as exc:
            log.warning("twin tap emit failed: %s", exc)
            return {"error": "unreachable", "detail": str(exc)}
