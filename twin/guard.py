from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .detection import DEFAULT_POLICY, DetectionPolicy
from .models import Span

log = logging.getLogger(__name__)


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


class GuardBackend(Protocol):
    def check_span(self, span: Span) -> list[GuardDecision]: ...
    def blocked_decisions(self, span: Span) -> list[GuardDecision]: ...
    def info(self) -> dict: ...


class NativeGuard:
    """Deterministic, zero-inference inline rail — the production default.

    This is the authoritative pre-execution gate: it runs on every commit with
    no model call and no network I/O, so it cannot be slowed or bypassed by an
    unavailable dependency. Other backends layer on top of it but never remove
    its denials.
    """

    backend = "native"

    def __init__(self, policy: DetectionPolicy = DEFAULT_POLICY) -> None:
        self.policy = policy

    def check_span(self, span: Span) -> list[GuardDecision]:
        return check_span(span, self.policy)

    def blocked_decisions(self, span: Span) -> list[GuardDecision]:
        return blocked_decisions(span, self.policy)

    def info(self) -> dict:
        return {"backend": self.backend}


def build_guard(settings: Settings, policy: DetectionPolicy) -> GuardBackend:
    native = NativeGuard(policy)
    if settings.guard_backend == "nemo":
        from .integrations.nemo_guard import NeMoGuard
        return NeMoGuard(settings, policy, native)
    return native
