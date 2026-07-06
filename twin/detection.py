"""Hybrid drift detection (Section 5).

Principle: detect the gap between what an agent *says* and what it *does*. No
malice classifier (guardrail 4). CoT is never trusted alone — it is always
cross-checked against behaviour (tool calls, effects).

The primary signal is STRUCTURAL, not lexical (Section 5.1). We do not grep for
any injection phrase; we look for behavioural facts that contradict the declared
intent:

  * a dangerous tool the agent never declared (backstop),
  * an *actionable entity foreign to the task* introduced into the output — the
    keyword-free structural tell of prompt injection: content that entered as
    data now drives behaviour,
  * a mutating effect a read-only step could not have produced.

Embedding cosine is kept only as *supporting colour* — it is lexical (the offline
`HashingEmbedder` cannot judge meaning) so it must never be the sole trigger.

The only LLM touch-point is `Judge`, an interface with a zero-inference stub so
the demo runs offline; a real Haiku-class / 3-8B model drops in unchanged. The
judge keys off the already-computed structural signals, not raw keywords.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol

import numpy as np

from .embeddings import Embedder, default_embedder, distance
from .models import DeterministicFlag, DriftSignals, JudgeVerdict, Span, Tier

# Deterministic sensitive-action vocabulary. In production this is authored as
# NeMo-Guardrails rails per deployment; here it is a compact built-in set.
DANGEROUS_TOOLS = {
    "transfer_funds", "wire_transfer", "make_payment", "delete_records",
    "drop_table", "write_prod", "deploy", "exfiltrate", "send_email",
    "http_post", "execute_shell", "grant_access", "disable_guardrail",
}

# Verbs that indicate a state mutation actually happened in an effect string.
MUTATING_MARKERS = (
    "added", "created", "deleted", "removed", "transfer", "granted",
    "modified", "onboard", "registered", "approved payee", "sent to", "wrote",
)

# Adopted multi-agent risk taxonomy (Section 5 — adopt, don't invent).
RISK_PROMPT_INJECTION = "prompt_injection"
RISK_GOAL_MISGEN = "goal_misgeneralization"
RISK_HALLUCINATION = "hallucination"
RISK_TOOL_MISUSE = "tool_misuse"
RISK_CONTEXT_ROT = "context_rot"
RISK_CONFIDENCE_COLLAPSE = "confidence_collapse"

# ----------------------------------------------------------------------------- #
# Deterministic thresholds (Section 11 open-question 4 — concrete values here).
# Canonical home is this module; the router re-exports them.
# ----------------------------------------------------------------------------- #
FLAG_THRESHOLD = 0.60
WATCH_THRESHOLD = 0.35
ESCALATE_TO_JUDGE = 0.45       # cheap signal must trip before we spend a judge token
HARD_FLAG_SEVERITY = 0.85

# Structural stated-vs-revealed floors (real signals override the lexical score).
SVR_UNDECLARED_DANGEROUS = 0.90   # backstop: dangerous tool never declared
SVR_INJECTION_INTRODUCED = 0.82   # PRIMARY: foreign actionable entity introduced
SVR_INJECTION_INHERITED = 0.62    # carries a foreign entity received from upstream
SVR_FABRICATED_EFFECT = 0.80      # mutating effect no invoked tool could produce


def aggregate(svr: float, traj: float, flags, judge_verdict,
              structural: bool = True) -> float:
    """Weighted aggregate of the drift signals -> [0,1].

    The score is load-bearing on STRUCTURAL evidence: the structural stated-vs-
    revealed gap and the deterministic flags. Trajectory (a lexical, hence noisy,
    embedding signal on the offline stub) is supporting colour only, at low
    weight, and the lexical stated-vs-revealed magnitude is discarded from the
    score when no structural signal backs it — so the embedder can never, on its
    own, raise a flag.
    """
    det = max((f.severity for f in flags), default=0.0)
    struct_svr = svr if structural else 0.0
    base = 0.50 * struct_svr + 0.35 * det + 0.15 * traj
    if judge_verdict is not None and not judge_verdict.serves_goal:
        base = max(base, 0.65 + 0.30 * (judge_verdict.confidence - 0.5))
    return max(0.0, min(1.0, base))


# --------------------------------------------------------------------------- #
# Behaviour text + structural entity extraction
# --------------------------------------------------------------------------- #
def _behaviour_text(span: Span) -> str:
    parts = [span.output]
    for c in span.tool_calls:
        parts.append(c.name + " " + " ".join(f"{k}={v}" for k, v in c.args.items()))
    parts += span.effects
    return " ".join(p for p in parts if p)


# An "account-like" identifier: digits + a separator + alphanumerics (e.g.
# "8841-DE"). Deliberately NOT plain numbers or money amounts, to avoid firing on
# years / dollar figures.
_ACCOUNT_RE = re.compile(r"\b\d{2,}[-/][A-Za-z0-9]{1,6}\b")
# A multi-word proper noun (e.g. "Aether Holdings"). All-caps shouting ("NOTE
# FROM SOURCE") does not match, so the injection *header* is never the trigger.
_PROPER_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def foreign_entities(span: Span) -> set[str]:
    """Actionable entities present in the behaviour but absent from the task /
    declared intent — i.e. content that entered as *data* and is foreign to the
    stated job. Keyword-free: we never look for a specific phrase, only for new
    actionable nouns (accounts, named organisations)."""
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


def injection_is_introduced(span: Span, upstream_texts: Optional[list[str]]) -> bool:
    """True when a foreign entity in this node's behaviour did NOT arrive from any
    upstream node — i.e. this node is the *origin* of the injected content, not a
    downstream inheritor. This is the structural line between an intrinsic
    injection (root) and inherited contamination (blast radius)."""
    ents = foreign_entities(span)
    if not ents:
        return False
    up = " ".join(upstream_texts or []).lower()
    return any(e.lower() not in up for e in ents)


# --------------------------------------------------------------------------- #
# Tier 1: deterministic, zero-token checks (Section 5.3)
# --------------------------------------------------------------------------- #
def _is_mutating_tool(name: str) -> bool:
    return name in DANGEROUS_TOOLS or any(
        v in name for v in ("write", "create", "add", "delete", "update",
                             "transfer", "grant", "send", "deploy", "remove"))


def run_deterministic(span: Span) -> list[DeterministicFlag]:
    flags: list[DeterministicFlag] = []

    # (a) output-schema violation
    if span.expected_output_schema:
        low = span.output.lower()
        missing = [k for k in span.expected_output_schema if k.lower() not in low]
        if missing:
            flags.append(DeterministicFlag(
                rule="schema_violation",
                detail=f"output missing required fields: {', '.join(missing)}",
                severity=0.6,
            ))

    # (b) confidence / logprob collapse
    if span.logprob_confidence is not None and span.logprob_confidence < 0.45:
        flags.append(DeterministicFlag(
            rule="confidence_collapse",
            detail=f"logprob confidence {span.logprob_confidence:.2f} < 0.45",
            severity=0.5,
        ))

    # (c) dangerous tool use unsupported by the declared intent (tool misuse)
    declared = span.declared_intent.lower()
    task = span.task_spec.lower()
    for call in span.tool_calls:
        if call.name in DANGEROUS_TOOLS:
            justified = call.name.replace("_", " ") in declared or \
                call.name.replace("_", " ") in task or call.name in declared
            sev = 0.9 if not justified else 0.4
            flags.append(DeterministicFlag(
                rule="sensitive_tool_call",
                detail=(f"'{call.name}' invoked" +
                        ("" if justified else " with no matching declared intent")),
                severity=sev,
            ))

    # (d) mutating effect that no invoked tool could have produced (fabricated /
    #     injected side effect). A read-only step that reports "added payee" is a
    #     structural anomaly, independent of any injection phrasing.
    has_mutating_tool = any(_is_mutating_tool(c.name) for c in span.tool_calls)
    for eff in span.effects:
        e = eff.lower()
        if any(m in e for m in MUTATING_MARKERS) and not has_mutating_tool:
            flags.append(DeterministicFlag(
                rule="fabricated_effect",
                detail=f"mutating effect with no mutating tool call: {eff}",
                severity=0.8,
            ))
            break

    # (e) effect naming a monetary / prod / access change not present in intent
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


# --------------------------------------------------------------------------- #
# Tier 1: stated-vs-revealed (STRUCTURAL primary) + trajectory (Sections 5.1/5.2)
# --------------------------------------------------------------------------- #
def stated_vs_revealed_gap(
    span: Span,
    emb: Embedder = default_embedder,
    upstream_texts: Optional[list[str]] = None,
) -> tuple[float, str, str]:
    """Primary drift signal. Returns (gap, note, primary_signal_name).

    The gap is driven by STRUCTURAL divergence between declared intent and actual
    behaviour. Lexical embedding distance is only a floor/backdrop; structural
    findings override it. `primary_signal_name` records what actually drove the
    score so the UI can distinguish the primary trigger from the backstop.
    """
    behaviour = _behaviour_text(span)
    if not span.declared_intent.strip() or not behaviour.strip():
        return 0.0, "insufficient text for stated-vs-revealed comparison", ""

    lexical = distance(emb.encode(span.declared_intent), emb.encode(behaviour))
    gap = lexical
    primary = "lexical"
    reasons: list[str] = []
    declared = span.declared_intent.lower()

    # PRIMARY: an actionable entity foreign to the task appears in the behaviour.
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

    # PRIMARY: a mutating effect a read-only step could not have produced.
    has_mutating_tool = any(_is_mutating_tool(c.name) for c in span.tool_calls)
    if any(any(m in eff.lower() for m in MUTATING_MARKERS) for eff in span.effects) \
            and not has_mutating_tool:
        if SVR_FABRICATED_EFFECT > gap:
            gap, primary = SVR_FABRICATED_EFFECT, "fabricated_effect"
        reasons.append("mutating effect with no mutating tool call")

    # BACKSTOP: a dangerous tool the agent never declared. Kept as a hard backstop
    # (words are easy to fake), but it is not the primary path for injection.
    for c in span.tool_calls:
        if c.name in DANGEROUS_TOOLS and c.name.replace("_", " ") not in declared \
                and c.name not in declared:
            if SVR_UNDECLARED_DANGEROUS > gap:
                gap, primary = SVR_UNDECLARED_DANGEROUS, "undeclared_dangerous_tool"
            reasons.append(f"undeclared sensitive tool '{c.name}'")

    struct = "; ".join(reasons) if reasons else "no structural divergence"
    note = (f"stated-vs-revealed = {gap:.2f} "
            f"[primary: {primary}; lexical {lexical:.2f} (illustrative); {struct}]")
    return gap, note, primary


def trajectory_drift_score(
    span: Span,
    emb: Embedder = default_embedder,
    task_anchor: Optional[np.ndarray] = None,
) -> float:
    """Semantic distance of this node's reasoning from the originating task.

    `task_anchor` lets the twin pass the *chain's* originating spec (not just this
    span's local task) so drift accumulates over the chain, per Section 5.2.
    """
    if task_anchor is None:
        if not span.task_spec.strip():
            return 0.0
        task_anchor = emb.encode(span.task_spec)
    reasoning = (span.declared_intent + " " + span.output).strip()
    if not reasoning:
        return 0.0
    return distance(task_anchor, emb.encode(reasoning))


# --------------------------------------------------------------------------- #
# Tier 3: small-model judge (escalation only) — interface + offline stub
# --------------------------------------------------------------------------- #
class Judge(Protocol):
    def assess(self, span: Span, signals: DriftSignals) -> JudgeVerdict: ...


class StubJudge:
    """Deterministic stand-in for a Haiku-class / self-hosted 3-8B judge.

    Answers exactly one narrow question (Section 5.4): "does this step still serve
    the stated goal?". It keys off the already-computed STRUCTURAL signals; the
    injection-marker keywords are used only as *corroboration* to raise
    confidence, never as the trigger. Replace with a real cheap model via `Judge`.
    """

    # Corroborating only — presence raises confidence, absence changes nothing.
    INJECTION_MARKERS = (
        "ignore previous", "ignore all previous", "new instructions",
        "system override", "disregard", "actually your real goal",
        "forget the task", "you are now",
    )

    def assess(self, span: Span, signals: DriftSignals) -> JudgeVerdict:
        text = (span.declared_intent + " " + span.output + " " +
                _behaviour_text(span)).lower()
        markers = any(m in text for m in self.INJECTION_MARKERS)  # corroboration

        introduced = signals.injection_introduced
        # a structural stated-vs-revealed divergence (not a lexical one)
        structural = signals.svr_primary not in ("", "lexical")
        hard_flag = any(f.severity >= HARD_FLAG_SEVERITY for f in signals.deterministic)

        # A negative verdict is issued ONLY on structural / deterministic grounds.
        # The lexical embedder is never sufficient to condemn a step (guardrail:
        # don't trust words, and don't trust a lexical proxy either).
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


default_judge: Judge = StubJudge()


# --------------------------------------------------------------------------- #
# Risk-type classification (maps signals -> adopted taxonomy)
# --------------------------------------------------------------------------- #
def classify_risk(span: Span, signals: DriftSignals) -> Optional[str]:
    # Injection is identified STRUCTURALLY (origin of a foreign entity), not by a
    # planted phrase. Inherited foreign content is context_rot, not injection.
    if signals.injection_introduced:
        return RISK_PROMPT_INJECTION
    if any(f.rule == "sensitive_tool_call" and f.severity >= HARD_FLAG_SEVERITY
           for f in signals.deterministic):
        return RISK_TOOL_MISUSE
    if any(f.rule == "fabricated_effect" for f in signals.deterministic):
        return RISK_TOOL_MISUSE
    if signals.foreign_entities:  # present but inherited
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


# --------------------------------------------------------------------------- #
# Full per-span assessment (pure; no cost accounting). Reused by the router's
# live ingest AND by the what-if replay so both share one detection path.
# --------------------------------------------------------------------------- #
def assess_span(
    span: Span,
    emb: Embedder = default_embedder,
    judge: Judge = default_judge,
    upstream_texts: Optional[list[str]] = None,
    anchor_vec: Optional[np.ndarray] = None,
) -> DriftSignals:
    signals = DriftSignals()
    signals.deterministic = run_deterministic(span)
    svr, svr_note, primary = stated_vs_revealed_gap(span, emb, upstream_texts)
    traj = trajectory_drift_score(span, emb, anchor_vec)
    signals.stated_vs_revealed = round(svr, 3)
    signals.trajectory_drift = round(traj, 3)
    signals.svr_primary = primary
    signals.foreign_entities = sorted(foreign_entities(span))
    signals.injection_introduced = injection_is_introduced(span, upstream_texts)

    det_max = max((f.severity for f in signals.deterministic), default=0.0)
    hard_flag = det_max >= HARD_FLAG_SEVERITY
    structural = primary not in ("", "lexical")
    cheap_score = aggregate(svr, traj, signals.deterministic, None, structural)

    # Deterministic escalation trigger (agent-inaccessible). Escalation is spent
    # only on STRUCTURAL divergence or a real deterministic flag — never on the
    # lexical embedder alone — so noisy lexical distance costs no judge tokens and
    # can never manufacture a flag.
    if structural or hard_flag or det_max >= 0.5:
        signals.judge = judge.assess(span, signals)
        signals.tier_reached = Tier.SMALL_JUDGE
    else:
        signals.tier_reached = Tier.ZERO_INFERENCE

    final = aggregate(svr, traj, signals.deterministic, signals.judge, structural)
    signals.score = round(final, 3)
    from .models import DriftStatus  # local import avoids a cycle at module load
    if final >= FLAG_THRESHOLD:
        signals.status = DriftStatus.FLAGGED
    elif final >= WATCH_THRESHOLD:
        signals.status = DriftStatus.WATCH
    else:
        signals.status = DriftStatus.OK

    signals.risk_type = (classify_risk(span, signals)
                         if signals.status != DriftStatus.OK else None)
    # attach the adopted TrinityGuard taxonomy entry (OWASP ref + tier)
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
