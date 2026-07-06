"""Inline deterministic hard-stop rail (Section 7).

Distinct from the async twin monitor. The monitor is out-of-loop and observational
(it *notices*); this rail is synchronous and sits in the action path (it *stops*).
Before a dangerous tool executes, a purely deterministic check either allows or
denies it — no inference, no model, agent-inaccessible. In production this is a
NeMo-Guardrails / policy rail; here it is a compact built-in.

The two coexist by design: the rail blocks the money synchronously, while the twin
records the *attempt* so attribution can still trace it back to the root cause.
"""
from __future__ import annotations

from dataclasses import dataclass

from .detection import DANGEROUS_TOOLS
from .models import Span

# Task phrasings that explicitly forbid an action, e.g. "Do NOT move funds".
_PROHIBITIONS = ("do not", "don't", "must not", "never", "without moving")


@dataclass
class GuardDecision:
    tool: str
    allowed: bool
    rule: str
    reason: str


def check_span(span: Span) -> list[GuardDecision]:
    """Evaluate every tool call in a span against the sensitive-action gate,
    synchronously, before any effect is committed."""
    decisions: list[GuardDecision] = []
    declared = span.declared_intent.lower()
    task = span.task_spec.lower()
    for c in span.tool_calls:
        if c.name not in DANGEROUS_TOOLS:
            continue
        justified = (c.name.replace("_", " ") in declared or c.name in declared
                     or c.name.replace("_", " ") in task)
        # An explicit prohibition in the task is a hard deny regardless.
        prohibited = any(p in task for p in _PROHIBITIONS)
        allowed = justified and not prohibited
        if allowed:
            reason = f"'{c.name}' is declared and permitted by the task spec"
        elif prohibited:
            reason = (f"'{c.name}' blocked: the task spec explicitly forbids this "
                      f"class of action")
        else:
            reason = (f"'{c.name}' blocked: no declared intent authorises this "
                      f"sensitive action")
        decisions.append(GuardDecision(
            tool=c.name, allowed=allowed, rule="sensitive_action_gate", reason=reason))
    return decisions


def blocked_decisions(span: Span) -> list[GuardDecision]:
    return [d for d in check_span(span) if not d.allowed]
