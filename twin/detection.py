from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .config import DEFAULT_DANGEROUS_TOOLS, Settings
from .embeddings import Embedder, distance
from .models import (
    DeterministicFlag, DriftSignals, DriftStatus, JudgeVerdict, Span, Tier,
)

MUTATING_MARKERS = (
    "added", "created", "deleted", "removed", "transfer", "granted",
    "modified", "onboard", "registered", "approved payee", "sent to", "wrote",
)

RISK_PROMPT_INJECTION = "prompt_injection"
RISK_GOAL_MISGEN = "goal_misgeneralization"
RISK_HALLUCINATION = "hallucination"
RISK_TOOL_MISUSE = "tool_misuse"
RISK_CONTEXT_ROT = "context_rot"
RISK_CONFIDENCE_COLLAPSE = "confidence_collapse"

SVR_UNDECLARED_DANGEROUS = 0.90
SVR_INJECTION_INTRODUCED = 0.82
SVR_INJECTION_INHERITED = 0.62
SVR_FABRICATED_EFFECT = 0.80


@dataclass(frozen=True)
class DetectionPolicy:
    dangerous_tools: frozenset[str] = frozenset(DEFAULT_DANGEROUS_TOOLS)
    flag_threshold: float = 0.60
    watch_threshold: float = 0.35
    escalate_threshold: float = 0.45
    hard_flag_severity: float = 0.85
    prohibition_markers: tuple[str, ...] = (
        "do not", "don't", "must not", "never", "without moving",
    )

    @classmethod
    def from_settings(cls, settings: Settings) -> "DetectionPolicy":
        return cls(
            dangerous_tools=frozenset(settings.dangerous_tools),
            flag_threshold=settings.flag_threshold,
            watch_threshold=settings.watch_threshold,
            escalate_threshold=settings.escalate_threshold,
            hard_flag_severity=settings.hard_flag_severity,
            prohibition_markers=tuple(settings.prohibition_markers),
        )


DEFAULT_POLICY = DetectionPolicy()


def aggregate(svr: float, traj: float, flags, judge_verdict,
              structural: bool = True) -> float:
    det = max((f.severity for f in flags), default=0.0)
    struct_svr = svr if structural else 0.0
    base = 0.50 * struct_svr + 0.35 * det + 0.15 * traj
    if judge_verdict is not None and not judge_verdict.serves_goal:
        base = max(base, 0.65 + 0.30 * (judge_verdict.confidence - 0.5))
    return max(0.0, min(1.0, base))


def _behaviour_text(span: Span) -> str:
    parts = [span.output]
    for c in span.tool_calls:
        parts.append(c.name + " " + " ".join(f"{k}={v}" for k, v in c.args.items()))
    parts += span.effects
    return " ".join(p for p in parts if p)


_ACCOUNT_RE = re.compile(r"\b\d{2,}[-/][A-Za-z0-9]{1,6}\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def foreign_entities(span: Span) -> set[str]:
    home = (span.task_spec + " " + span.declared_intent).lower()
    behaviour = _behaviour_text(span)
    ents: set[str] = set()
    for m in _ACCOUNT_RE.findall(behaviour):
        if m.lower() not in home:
            ents.add(m)
    for m in _PROPER_RE.findall(behaviour):
        if m.lower() not in home:
            ents.add(m)
    return ents


def injection_is_introduced(span: Span,
                            upstream_texts: Optional[list[str]]) -> bool:
    ents = foreign_entities(span)
    if not ents:
        return False
    up = " ".join(upstream_texts or []).lower()
    return any(e.lower() not in up for e in ents)


def _is_mutating_tool(name: str, policy: DetectionPolicy) -> bool:
    return name in policy.dangerous_tools or any(
        v in name for v in ("write", "create", "add", "delete", "update",
                            "transfer", "grant", "send", "deploy", "remove"))


def run_deterministic(span: Span,
                      policy: DetectionPolicy) -> list[DeterministicFlag]:
    flags: list[DeterministicFlag] = []

    if span.expected_output_schema:
        low = span.output.lower()
        missing = [k for k in span.expected_output_schema if k.lower() not in low]
        if missing:
            flags.append(DeterministicFlag(
                rule="schema_violation",
                detail=f"output missing required fields: {', '.join(missing)}",
                severity=0.6,
            ))

    if span.logprob_confidence is not None and span.logprob_confidence < 0.45:
        flags.append(DeterministicFlag(
            rule="confidence_collapse",
            detail=f"logprob confidence {span.logprob_confidence:.2f} < 0.45",
            severity=0.5,
        ))

    declared = span.declared_intent.lower()
    task = span.task_spec.lower()
    for call in span.tool_calls:
        if call.name in policy.dangerous_tools:
            justified = call.name.replace("_", " ") in declared or \
                call.name.replace("_", " ") in task or call.name in declared
            sev = 0.9 if not justified else 0.4
            flags.append(DeterministicFlag(
                rule="sensitive_tool_call",
                detail=(f"'{call.name}' invoked" +
                        ("" if justified else " with no matching declared intent")),
                severity=sev,
            ))

    has_mutating_tool = any(_is_mutating_tool(c.name, policy)
                            for c in span.tool_calls)
    for eff in span.effects:
        e = eff.lower()
        if any(m in e for m in MUTATING_MARKERS) and not has_mutating_tool:
            flags.append(DeterministicFlag(
                rule="fabricated_effect",
                detail=f"mutating effect with no mutating tool call: {eff}",
                severity=0.8,
            ))
            break

    for eff in span.effects:
        e = eff.lower()
        if any(t in e for t in ("$", "prod", "deleted", "sent to", "granted")) \
                and e.split(":")[0] not in declared:
            flags.append(DeterministicFlag(
                rule="unexplained_effect",
                detail=f"effect not present in declared intent: {eff}",
                severity=0.55,
            ))
    return flags


def stated_vs_revealed_gap(
    span: Span,
    emb: Embedder,
    policy: DetectionPolicy,
    upstream_texts: Optional[list[str]] = None,
) -> tuple[float, str, str]:
    behaviour = _behaviour_text(span)
    if not span.declared_intent.strip() or not behaviour.strip():
        return 0.0, "insufficient text for stated-vs-revealed comparison", ""

    semantic = distance(emb.encode(span.declared_intent), emb.encode(behaviour))
    gap = semantic
    primary = "semantic"
    reasons: list[str] = []
    declared = span.declared_intent.lower()

    ents = foreign_entities(span)
    if ents:
        introduced = injection_is_introduced(span, upstream_texts)
        shown = ", ".join(sorted(ents)[:2])
        if introduced:
            if SVR_INJECTION_INTRODUCED > gap:
                gap, primary = SVR_INJECTION_INTRODUCED, "injection_introduced"
            reasons.append(f"introduces entity foreign to its task ({shown})")
        else:
            if SVR_INJECTION_INHERITED > gap:
                gap, primary = SVR_INJECTION_INHERITED, "injection_inherited"
            reasons.append(f"carries foreign entity from upstream ({shown})")

    has_mutating_tool = any(_is_mutating_tool(c.name, policy)
                            for c in span.tool_calls)
    if any(any(m in eff.lower() for m in MUTATING_MARKERS) for eff in span.effects) \
            and not has_mutating_tool:
        if SVR_FABRICATED_EFFECT > gap:
            gap, primary = SVR_FABRICATED_EFFECT, "fabricated_effect"
        reasons.append("mutating effect with no mutating tool call")

    for c in span.tool_calls:
        if c.name in policy.dangerous_tools \
                and c.name.replace("_", " ") not in declared \
                and c.name not in declared:
            if SVR_UNDECLARED_DANGEROUS > gap:
                gap, primary = SVR_UNDECLARED_DANGEROUS, "undeclared_dangerous_tool"
            reasons.append(f"undeclared sensitive tool '{c.name}'")

    struct = "; ".join(reasons) if reasons else "no structural divergence"
    note = (f"stated-vs-revealed = {gap:.2f} "
            f"[primary: {primary}; semantic {semantic:.2f}; {struct}]")
    return gap, note, primary


def trajectory_drift_score(
    span: Span,
    emb: Embedder,
    task_anchor: Optional[np.ndarray] = None,
) -> float:
    if task_anchor is None:
        if not span.task_spec.strip():
            return 0.0
        task_anchor = emb.encode(span.task_spec)
    reasoning = (span.declared_intent + " " + span.output).strip()
    if not reasoning:
        return 0.0
    return distance(task_anchor, emb.encode(reasoning))


class Judge(Protocol):
    def assess(self, span: Span, signals: DriftSignals) -> JudgeVerdict: ...


class StubJudge:
    INJECTION_MARKERS = (
        "ignore previous", "ignore all previous", "new instructions",
        "system override", "disregard", "actually your real goal",
        "forget the task", "you are now",
    )

    def __init__(self, policy: DetectionPolicy = DEFAULT_POLICY) -> None:
        self.policy = policy

    def assess(self, span: Span, signals: DriftSignals) -> JudgeVerdict:
        text = (span.declared_intent + " " + span.output + " " +
                _behaviour_text(span)).lower()
        markers = any(m in text for m in self.INJECTION_MARKERS)
        introduced = signals.injection_introduced
        structural = signals.svr_primary not in ("", "semantic", "lexical")
        hard_flag = any(f.severity >= self.policy.hard_flag_severity
                        for f in signals.deterministic)

        if introduced or structural or hard_flag:
            conf = 0.9 if markers else 0.82
            if introduced:
                why = ("this step introduced content foreign to its task; it no "
                       "longer serves the originating goal")
            elif hard_flag:
                why = ("a sensitive action fired without any justification in the "
                       "declared intent")
            else:
                why = ("behaviour structurally diverges from the stated goal "
                       "(foreign content / unexplained effect)")
            return JudgeVerdict(serves_goal=False, confidence=conf, rationale=why)
        return JudgeVerdict(
            serves_goal=True, confidence=0.7,
            rationale="step remains consistent with the stated goal.")


def classify_risk(span: Span, signals: DriftSignals,
                  policy: DetectionPolicy) -> Optional[str]:
    if signals.injection_introduced:
        return RISK_PROMPT_INJECTION
    if any(f.rule == "sensitive_tool_call"
           and f.severity >= policy.hard_flag_severity
           for f in signals.deterministic):
        return RISK_TOOL_MISUSE
    if any(f.rule == "fabricated_effect" for f in signals.deterministic):
        return RISK_TOOL_MISUSE
    if signals.foreign_entities:
        return RISK_CONTEXT_ROT
    if any(f.rule == "confidence_collapse" for f in signals.deterministic):
        return RISK_CONFIDENCE_COLLAPSE
    if signals.stated_vs_revealed >= 0.6:
        return RISK_GOAL_MISGEN
    if signals.trajectory_drift >= 0.6:
        return RISK_CONTEXT_ROT
    if any(f.rule == "schema_violation" for f in signals.deterministic):
        return RISK_HALLUCINATION
    return None


def assess_span(
    span: Span,
    emb: Embedder,
    judge: Optional[Judge] = None,
    policy: DetectionPolicy = DEFAULT_POLICY,
    upstream_texts: Optional[list[str]] = None,
    anchor_vec: Optional[np.ndarray] = None,
) -> DriftSignals:
    if judge is None:
        judge = StubJudge(policy)
    signals = DriftSignals()
    signals.deterministic = run_deterministic(span, policy)
    svr, svr_note, primary = stated_vs_revealed_gap(
        span, emb, policy, upstream_texts)
    traj = trajectory_drift_score(span, emb, anchor_vec)
    signals.stated_vs_revealed = round(svr, 3)
    signals.trajectory_drift = round(traj, 3)
    signals.svr_primary = primary
    signals.foreign_entities = sorted(foreign_entities(span))
    signals.injection_introduced = injection_is_introduced(span, upstream_texts)

    det_max = max((f.severity for f in signals.deterministic), default=0.0)
    hard_flag = det_max >= policy.hard_flag_severity
    structural = primary not in ("", "semantic", "lexical")

    if structural or hard_flag or det_max >= 0.5:
        signals.judge = judge.assess(span, signals)
        signals.tier_reached = Tier.SMALL_JUDGE
    else:
        signals.tier_reached = Tier.ZERO_INFERENCE

    final = aggregate(svr, traj, signals.deterministic, signals.judge, structural)
    signals.score = round(final, 3)
    if final >= policy.flag_threshold:
        signals.status = DriftStatus.FLAGGED
    elif final >= policy.watch_threshold:
        signals.status = DriftStatus.WATCH
    else:
        signals.status = DriftStatus.OK

    signals.risk_type = (classify_risk(span, signals, policy)
                         if signals.status != DriftStatus.OK else None)

    from . import taxonomy
    rt = taxonomy.lookup(signals.risk_type)
    if rt is not None:
        signals.risk_id = rt.risk_id
        signals.risk_name = rt.name
        signals.owasp_ref = rt.owasp_ref
        signals.risk_tier = rt.tier
    signals.rationale = "; ".join(
        [svr_note, f"trajectory drift = {traj:.2f}"]
        + [f"{f.rule}({f.severity:.2f})" for f in signals.deterministic]
    )
    return signals
