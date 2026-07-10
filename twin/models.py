from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

class Privilege(str, Enum):

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class DriftStatus(str, Enum):
    OK = "ok"
    WATCH = "watch"
    FLAGGED = "flagged"
    CONTAINED = "contained"

class Tier(str, Enum):

    ZERO_INFERENCE = "zero_inference"
    SAMPLED = "sampled"
    SMALL_JUDGE = "small_judge"
    DEEP_ESCALATION = "deep_escalation"

class RemediationKind(str, Enum):
    CONTEXT_EDIT = "context_edit"
    ROLLBACK = "rollback"
    QUARANTINE = "quarantine"
    MESSAGE_FILTER = "message_filter"
    REDEPLOY = "redeploy"

class ActionStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    REVERTED = "reverted"

class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)

class Span(BaseModel):

    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None

    agent_id: str
    agent_role: str = ""
    privilege: Privilege = Privilege.LOW

    task_spec: str = ""

    declared_intent: str = ""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    output: str = ""

    logprob_confidence: Optional[float] = None
    expected_output_schema: Optional[list[str]] = None

    inputs_from: list[str] = Field(default_factory=list)

    timestamp: Optional[float] = None
    meta: dict[str, Any] = Field(default_factory=dict)

class DeterministicFlag(BaseModel):
    rule: str
    detail: str
    severity: float = 0.5

class JudgeVerdict(BaseModel):
    serves_goal: bool
    rationale: str
    confidence: float

class DriftSignals(BaseModel):

    stated_vs_revealed: float = 0.0
    trajectory_drift: float = 0.0
    deterministic: list[DeterministicFlag] = Field(default_factory=list)
    judge: Optional[JudgeVerdict] = None

    foreign_entities: list[str] = Field(default_factory=list)
    injection_introduced: bool = False

    svr_primary: str = ""

    score: float = 0.0
    status: DriftStatus = DriftStatus.OK
    tier_reached: Tier = Tier.ZERO_INFERENCE

    risk_type: Optional[str] = None

    risk_id: Optional[str] = None
    risk_name: Optional[str] = None
    owasp_ref: Optional[str] = None
    risk_tier: Optional[int] = None
    rationale: str = ""

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

    blocked: bool = False
    blocked_reason: str = ""
    timestamp: Optional[float] = None
    trace_id: str = ""

class TwinEdge(BaseModel):
    src: str
    dst: str
    kind: str = "influence"
    weight: float = 1.0

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

class CausalNarrative(BaseModel):
    incident_id: str
    root_cause_node: Optional[str] = None
    root_cause_summary: str = ""
    propagation_path: list[str] = Field(default_factory=list)
    blast_radius: list[str] = Field(default_factory=list)
    recommended_remediation: list[RemediationAction] = Field(default_factory=list)
    narrative: str = ""

class WhatIfPreview(BaseModel):
    node_id: str
    do_nothing_blast_radius: list[str] = Field(default_factory=list)
    remediated_blast_radius: list[str] = Field(default_factory=list)
    contained_nodes: list[str] = Field(default_factory=list)
    projected_drift_before: dict[str, float] = Field(default_factory=dict)
    projected_drift_after: dict[str, float] = Field(default_factory=dict)
    summary: str = ""
