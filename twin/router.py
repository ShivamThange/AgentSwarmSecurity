"""Tiered cost router (Section 8.2 / 6) — the architecture *is* the cost control.

Every observed span falls through the cheapest layer first and only escalates when
a cheaper layer flags:

    zero-inference  ->  sampled  ->  small-judge  ->  deep-escalation

Key properties from the brief:
  * The escalation trigger is DETERMINISTIC and computed by the platform from
    signals the monitored agent cannot see or feed back into its own context
    (Section 7 guardrail) — an attacker must not be able to suppress escalation.
  * Coverage is RISK-WEIGHTED: high-privilege agents (money/prod/PII) get full
    coverage; low-stakes agents are thinly sampled (Section 6.2).
  * Results are CACHED/DEDUPED so repeated reasoning is never analysed twice.
  * The router keeps running cost accounting to demonstrate the low-single-digit-%
    overhead envelope (Section 6, target).

Detection itself lives in `detection.assess_span` (pure, no cost side effects) so
the live path here and the what-if replay share one code path. This module only
adds tier accounting and the inline guard around that shared assessment.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from . import detection, guard
from .detection import (
    ESCALATE_TO_JUDGE, FLAG_THRESHOLD, HARD_FLAG_SEVERITY, WATCH_THRESHOLD,
)
from .embeddings import Embedder, default_embedder
from .graph import TwinStore
from .models import (
    DriftSignals, DriftStatus, Privilege, Span, Tier, TwinEdge, TwinNode,
)

# Nominal per-span cost in "token-equivalents". Zero-inference tiers cost CPU only
# (modelled as ~0 tokens); LLM tiers cost real tokens. Used only for the overhead
# demonstration (labelled "modelled" in the UI), not for control flow.
COST_BASELINE_DEFAULT = 1500   # what the monitored agent itself spends per step
COST_ZERO_INFERENCE = 0
COST_SMALL_JUDGE = 120         # a narrow Haiku-class classification
COST_DEEP_ESCALATION = 1400    # full-context deep analysis, only on flagged cases


@dataclass
class CostLedger:
    baseline_tokens: int = 0
    added_tokens: int = 0
    spans_seen: int = 0
    spans_sampled: int = 0
    cache_hits: int = 0
    tier_counts: dict = field(default_factory=lambda: {t.value: 0 for t in Tier})
    # Every token figure here is modelled from fixed constants, not measured.
    modelled: bool = True

    @property
    def overhead_pct(self) -> float:
        if self.baseline_tokens == 0:
            return 0.0
        return 100.0 * self.added_tokens / self.baseline_tokens

    def as_dict(self) -> dict:
        return {
            "baseline_tokens": self.baseline_tokens,
            "added_tokens": self.added_tokens,
            "overhead_pct": round(self.overhead_pct, 3),
            "spans_seen": self.spans_seen,
            "spans_sampled": self.spans_sampled,
            "cache_hits": self.cache_hits,
            "tier_counts": dict(self.tier_counts),
            "modelled": self.modelled,
            "note": "token figures are modelled from fixed per-tier constants, "
                    "not measured against a live model",
        }

    def load_dict(self, d: dict) -> None:
        self.baseline_tokens = d.get("baseline_tokens", 0)
        self.added_tokens = d.get("added_tokens", 0)
        self.spans_seen = d.get("spans_seen", 0)
        self.spans_sampled = d.get("spans_sampled", 0)
        self.cache_hits = d.get("cache_hits", 0)
        self.tier_counts = dict(d.get("tier_counts", {t.value: 0 for t in Tier}))


class CostRouter:
    def __init__(self, store: TwinStore, emb: Embedder = default_embedder,
                 judge: detection.Judge = detection.default_judge,
                 low_sample_rate: int = 3) -> None:
        self.store = store
        self.emb = emb
        self.judge = judge
        self.ledger = CostLedger()
        self.low_sample_rate = low_sample_rate  # sample 1-in-N low-privilege spans
        self._cache: dict[str, DriftSignals] = {}
        # per-trace originating task anchor, so drift accumulates over the chain
        self._chain_anchor: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    def _sampled(self, span: Span) -> bool:
        """Risk-weighted sampling. HIGH always covered; LOW thinned."""
        if span.privilege in (Privilege.HIGH, Privilege.MEDIUM):
            return True
        h = int(hashlib.blake2b(span.span_id.encode(), digest_size=4).hexdigest(), 16)
        return (h % self.low_sample_rate) == 0

    def _cache_key(self, span: Span) -> str:
        payload = span.declared_intent + "||" + detection._behaviour_text(span) \
            + "||" + span.task_spec
        return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()

    def _upstream_texts(self, span: Span) -> list[str]:
        """Outputs of already-ingested nodes that fed this context — the real
        upstream the structural injection test needs."""
        texts: list[str] = []
        for src in span.inputs_from:
            n = self.store.get_node(src)
            if n is not None:
                texts.append(n.output + " " + " ".join(n.effects))
        return texts

    # ------------------------------------------------------------------ #
    def ingest(self, span: Span) -> TwinNode:
        """Route one span through the tiers and promote it into the twin."""
        self.ledger.spans_seen += 1
        baseline = int(span.meta.get("baseline_tokens", COST_BASELINE_DEFAULT))
        self.ledger.baseline_tokens += baseline

        # chain anchor: first task_spec seen for this trace is the originating goal
        if span.trace_id and span.trace_id not in self._chain_anchor and span.task_spec:
            self._chain_anchor[span.trace_id] = span.task_spec
        anchor_text = self._chain_anchor.get(span.trace_id, span.task_spec)

        # ---- sampling gate (Tier 2 policy) ----
        if not self._sampled(span):
            signals = DriftSignals(
                status=DriftStatus.OK, tier_reached=Tier.SAMPLED,
                rationale="not sampled (low-privilege thinning)")
            self.ledger.tier_counts[Tier.SAMPLED.value] += 1
            return self._commit(span, signals)

        self.ledger.spans_sampled += 1

        # ---- cache / dedup ----
        key = self._cache_key(span)
        if key in self._cache:
            self.ledger.cache_hits += 1
            cached = self._cache[key].model_copy(deep=True)
            cached.rationale = "cache hit: identical reasoning already analysed"
            return self._commit(span, cached)

        # ---- Tier 1: zero-inference assessment (shared detection path) ----
        self.ledger.tier_counts[Tier.ZERO_INFERENCE.value] += 1
        self.ledger.added_tokens += COST_ZERO_INFERENCE
        anchor_vec = self.emb.encode(anchor_text) if anchor_text else None
        upstream = self._upstream_texts(span)
        signals = detection.assess_span(
            span, self.emb, self.judge, upstream_texts=upstream, anchor_vec=anchor_vec)

        # ---- tier accounting from what the assessment actually did ----
        if signals.judge is not None:
            self.ledger.tier_counts[Tier.SMALL_JUDGE.value] += 1
            self.ledger.added_tokens += COST_SMALL_JUDGE
        if signals.status == DriftStatus.FLAGGED:
            # ambiguous-but-serious cases warrant the deep engine (Section 6.4);
            # here that tier is realised by the attribution + remediation proposal.
            self.ledger.tier_counts[Tier.DEEP_ESCALATION.value] += 1
            self.ledger.added_tokens += COST_DEEP_ESCALATION
            signals.tier_reached = Tier.DEEP_ESCALATION

        self._cache[key] = signals.model_copy(deep=True)
        return self._commit(span, signals)

    # ------------------------------------------------------------------ #
    def _commit(self, span: Span, signals: DriftSignals) -> TwinNode:
        """Promote span -> twin node, run the inline guard, wire influence edges,
        snapshot context."""
        node = TwinNode(
            node_id=span.span_id,
            agent_id=span.agent_id,
            agent_role=span.agent_role,
            privilege=span.privilege,
            task=span.task_spec,
            declared_intent=span.declared_intent,
            tool_calls=span.tool_calls,
            effects=list(span.effects),
            output=span.output,
            drift=signals,
            timestamp=span.timestamp or time.time(),
            trace_id=span.trace_id,
        )

        # ---- inline deterministic hard-stop rail (synchronous, pre-effect) ----
        denials = guard.blocked_decisions(span)
        if denials:
            node.blocked = True
            node.blocked_reason = "; ".join(d.reason for d in denials)
            # the dangerous effect never lands: reflect prevention in the record
            node.effects = [f"[BLOCKED by inline rail] {e}" for e in node.effects] \
                or [f"[BLOCKED by inline rail] {d.tool} denied" for d in denials]

        # checkpoint the agent's context at this run (enables rollback)
        ckpt_id = f"ckpt::{span.span_id}"
        self.store.save_checkpoint(
            ckpt_id, span.span_id, span.agent_id,
            context={
                "task_spec": span.task_spec,
                "declared_intent": span.declared_intent,
                "output": span.output,
                "tool_calls": [c.model_dump() for c in span.tool_calls],
            },
            label=f"auto-snapshot @ {node.timestamp}",
        )
        node.checkpoint_id = ckpt_id
        self.store.upsert_node(node)

        # influence edges: each source that fed this context -> this node
        for src in span.inputs_from:
            if self.store.get_node(src) is not None:
                self.store.add_edge(TwinEdge(src=src, dst=span.span_id, kind="influence"))
        return node
