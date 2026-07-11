from __future__ import annotations

from dataclasses import dataclass

from .detection import DEFAULT_POLICY, DetectionPolicy
from .models import Span


@dataclass
class GuardDecision:
    tool: str
    allowed: bool
    rule: str
    reason: str


def check_span(span: Span,
               policy: DetectionPolicy = DEFAULT_POLICY) -> list[GuardDecision]:
    decisions: list[GuardDecision] = []
    declared = span.declared_intent.lower()
    task = span.task_spec.lower()
    for c in span.tool_calls:
        if c.name not in policy.dangerous_tools:
            continue
        justified = (c.name.replace("_", " ") in declared or c.name in declared
                     or c.name.replace("_", " ") in task)
        prohibited = any(p in task for p in policy.prohibition_markers)
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
            tool=c.name, allowed=allowed, rule="sensitive_action_gate",
            reason=reason))
    return decisions


def blocked_decisions(span: Span,
                      policy: DetectionPolicy = DEFAULT_POLICY
                      ) -> list[GuardDecision]:
    return [d for d in check_span(span, policy) if not d.allowed]
