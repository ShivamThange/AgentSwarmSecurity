"""Core data models for the Drift Containment platform.

These describe the *semantics* the brief asks for (Section 8), not a wire format
prescribed anywhere. A `Span` is the already-emitted telemetry an agent produces
(the brief's non-negotiable constraint: consume reasoning agents already emit, no
new paid token stream). A `TwinNode` is that span promoted into the persistent
digital twin. Everything else hangs off those two.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Privilege(str, Enum):
    """Risk weighting for tiered coverage (Section 6.2).

    HIGH = moves money / writes prod / touches PII -> full coverage.
    LOW  = read-only / low-stakes -> thin sampling.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DriftStatus(str, Enum):
    OK = "ok"            # cleared by a cheap layer
    WATCH = "watch"      # mild drift, keep sampling
    FLAGGED = "flagged"  # crossed escalation threshold
    CONTAINED = "contained"  # remediated / quarantined


class Tier(str, Enum):
    """The cost tiers of Section 6, cheapest first."""

    ZERO_INFERENCE = "zero_inference"
    SAMPLED = "sampled"
    SMALL_JUDGE = "small_judge"
    DEEP_ESCALATION = "deep_escalation"


class RemediationKind(str, Enum):
    CONTEXT_EDIT = "context_edit"       # modify context/prompt state (never internals)
    ROLLBACK = "rollback"               # restore a known-good checkpoint
    QUARANTINE = "quarantine"           # isolate a compromised node
    MESSAGE_FILTER = "message_filter"   # drop/replace a harmful inter-agent message
    REDEPLOY = "redeploy"               # return remediated agent to service


class ActionStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    REVERTED = "reverted"


# --------------------------------------------------------------------------- #
# Telemetry contract (what agents already emit)
# --------------------------------------------------------------------------- #
class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    """OpenTelemetry-shaped unit of already-emitted agent telemetry.

    This is the ingestion contract. Note the deliberate split between what the
    agent *says* (`declared_intent`) and what it *does* (`tool_calls`, `effects`)
    — that gap is the primary drift signal (Section 5.1).
    """

    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None

    agent_id: str
    agent_role: str = ""
    privilege: Privilege = Privilege.LOW

    # The originating task specification this agent is meant to serve.
    task_spec: str = ""

    # "What it says": declared intent, from CoT / a tiny structured self-report
    # at a decision gate (Section 5). NOT a new paid reasoning pass.
    declared_intent: str = ""

    # "What it does": behavioural ground truth.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    output: str = ""

    # Zero-token deterministic signals (Section 5.3).
    logprob_confidence: Optional[float] = None  # 0..1; collapse => flag
    expected_output_schema: Optional[list[str]] = None  # required keys/tokens

    # Influence edges: run/span ids whose output entered THIS context.
    inputs_from: list[str] = Field(default_factory=list)

    timestamp: Optional[float] = None
    meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Detection output
# --------------------------------------------------------------------------- #
class DeterministicFlag(BaseModel):
    rule: str
    detail: str
    severity: float = 0.5  # 0..1


class JudgeVerdict(BaseModel):
    serves_goal: bool
    rationale: str
    confidence: float  # 0..1


class DriftSignals(BaseModel):
    """Hybrid drift metrics (Section 5), aggregated."""

    stated_vs_revealed: float = 0.0   # 0..1, primary behavioural gap
    trajectory_drift: float = 0.0     # 0..1, semantic distance from task spec
    deterministic: list[DeterministicFlag] = Field(default_factory=list)
    judge: Optional[JudgeVerdict] = None

    # Structural evidence behind stated-vs-revealed (real signals, not lexical).
    # `foreign_entities`: actionable nouns (accounts / named orgs) present in the
    # behaviour but absent from the task — content that entered as data. Keyword-
    # free, so it survives re-phrased injections. `injection_introduced` is True
    # when this node is the *origin* of such an entity (not merely inheriting it).
    foreign_entities: list[str] = Field(default_factory=list)
    injection_introduced: bool = False
    # Which stated-vs-revealed sub-signal actually fired (primary trigger vs the
    # dangerous-tool backstop) — surfaced so the UI can show what drove the flag.
    svr_primary: str = ""

    score: float = 0.0                # aggregate 0..1
    status: DriftStatus = DriftStatus.OK
    tier_reached: Tier = Tier.ZERO_INFERENCE
    # Risk taxonomy label (adopted, not invented — Section 5 / TrinityGuard).
    risk_type: Optional[str] = None
    # Adopted TrinityGuard OWASP-grounded taxonomy (twin/taxonomy.py).
    risk_id: Optional[str] = None       # e.g. "RT1.1"
    risk_name: Optional[str] = None     # e.g. "Prompt Injection"
    owasp_ref: Optional[str] = None     # e.g. "LLM01"
    risk_tier: Optional[int] = None     # 1 atomic | 2 comms | 3 system
    rationale: str = ""


# --------------------------------------------------------------------------- #
# The digital twin (Section 8.1)
# --------------------------------------------------------------------------- #
class TwinNode(BaseModel):
    node_id: str
    agent_id: str
    agent_role: str = ""
    privilege: Privilege = Privilege.LOW
    task: str = ""

    declared_intent: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    output: str = ""

    drift: DriftSignals = Field(default_factory=DriftSignals)
    checkpoint_id: Optional[str] = None
    quarantined: bool = False
    # Set by the inline deterministic rail (Section 7) when a dangerous action was
    # stopped *before* it could execute — the "we stopped it", not "we noticed it".
    blocked: bool = False
    blocked_reason: str = ""
    timestamp: Optional[float] = None
    trace_id: str = ""


class TwinEdge(BaseModel):
    src: str
    dst: str
    kind: str = "influence"  # influence | handoff
    weight: float = 1.0


# --------------------------------------------------------------------------- #
# Remediation & audit (Sections 8.3 / 8.4 / 14.2)
# --------------------------------------------------------------------------- #
class RemediationAction(BaseModel):
    action_id: str
    node_id: str
    kind: RemediationKind
    params: dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus = ActionStatus.PROPOSED
    reversible: bool = True
    proposed_by: str = "system"
    approved_by: Optional[str] = None
    before_state: dict[str, Any] = Field(default_factory=dict)
    after_state: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    timestamp: Optional[float] = None


class AuditEntry(BaseModel):
    entry_id: str
    actor: str
    action: str
    target: str
    detail: str = ""
    reversible: bool = True
    compliance_tags: list[str] = Field(default_factory=list)
    prev_hash: str = ""
    hash: str = ""
    timestamp: Optional[float] = None


# --------------------------------------------------------------------------- #
# Narrative (Section 10) & what-if (Section 14.1)
# --------------------------------------------------------------------------- #
class CausalNarrative(BaseModel):
    incident_id: str
    root_cause_node: Optional[str] = None
    root_cause_summary: str = ""
    propagation_path: list[str] = Field(default_factory=list)  # ordered node ids
    blast_radius: list[str] = Field(default_factory=list)      # all affected nodes
    recommended_remediation: list[RemediationAction] = Field(default_factory=list)
    narrative: str = ""


class WhatIfPreview(BaseModel):
    node_id: str
    do_nothing_blast_radius: list[str] = Field(default_factory=list)
    remediated_blast_radius: list[str] = Field(default_factory=list)
    contained_nodes: list[str] = Field(default_factory=list)  # nodes saved
    projected_drift_before: dict[str, float] = Field(default_factory=dict)
    projected_drift_after: dict[str, float] = Field(default_factory=dict)
    summary: str = ""
