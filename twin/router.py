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

COST_BASELINE_DEFAULT = 1500
COST_ZERO_INFERENCE = 0
COST_SMALL_JUDGE = 120
COST_DEEP_ESCALATION = 1400

@dataclass
class CostLedger:
    baseline_tokens: int = 0
    added_tokens: int = 0
    spans_seen: int = 0
    spans_sampled: int = 0
    cache_hits: int = 0
    tier_counts: dict = field(default_factory=lambda: {t.value: 0 for t in Tier})

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
        self.low_sample_rate = low_sample_rate
        self._cache: dict[str, DriftSignals] = {}

        self._chain_anchor: dict[str, str] = {}

    def _sampled(self, span: Span) -> bool:
        if span.privilege in (Privilege.HIGH, Privilege.MEDIUM):
            return True
        h = int(hashlib.blake2b(span.span_id.encode(), digest_size=4).hexdigest(), 16)
        return (h % self.low_sample_rate) == 0

    def _cache_key(self, span: Span) -> str:
        payload = span.declared_intent + "||" + detection._behaviour_text(span) \
            + "||" + span.task_spec
        return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()

    def _upstream_texts(self, span: Span) -> list[str]:
        texts: list[str] = []
        for src in span.inputs_from:
            n = self.store.get_node(src)
            if n is not None:
                texts.append(n.output + " " + " ".join(n.effects))
        return texts

    def ingest(self, span: Span) -> TwinNode:
        self.ledger.spans_seen += 1
        baseline = int(span.meta.get("baseline_tokens", COST_BASELINE_DEFAULT))
        self.ledger.baseline_tokens += baseline

        if span.trace_id and span.trace_id not in self._chain_anchor and span.task_spec:
            self._chain_anchor[span.trace_id] = span.task_spec
        anchor_text = self._chain_anchor.get(span.trace_id, span.task_spec)

        if not self._sampled(span):
            signals = DriftSignals(
                status=DriftStatus.OK, tier_reached=Tier.SAMPLED,
                rationale="not sampled (low-privilege thinning)")
            self.ledger.tier_counts[Tier.SAMPLED.value] += 1
            return self._commit(span, signals)

        self.ledger.spans_sampled += 1

        key = self._cache_key(span)
        if key in self._cache:
            self.ledger.cache_hits += 1
            cached = self._cache[key].model_copy(deep=True)
            cached.rationale = "cache hit: identical reasoning already analysed"
            return self._commit(span, cached)

        self.ledger.tier_counts[Tier.ZERO_INFERENCE.value] += 1
        self.ledger.added_tokens += COST_ZERO_INFERENCE
        anchor_vec = self.emb.encode(anchor_text) if anchor_text else None
        upstream = self._upstream_texts(span)
        signals = detection.assess_span(
            span, self.emb, self.judge, upstream_texts=upstream, anchor_vec=anchor_vec)

        if signals.judge is not None:
            self.ledger.tier_counts[Tier.SMALL_JUDGE.value] += 1
            self.ledger.added_tokens += COST_SMALL_JUDGE
        if signals.status == DriftStatus.FLAGGED:

            self.ledger.tier_counts[Tier.DEEP_ESCALATION.value] += 1
            self.ledger.added_tokens += COST_DEEP_ESCALATION
            signals.tier_reached = Tier.DEEP_ESCALATION

        self._cache[key] = signals.model_copy(deep=True)
        return self._commit(span, signals)

    def _commit(self, span: Span, signals: DriftSignals) -> TwinNode:
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

        denials = guard.blocked_decisions(span)
        if denials:
            node.blocked = True
            node.blocked_reason = "; ".join(d.reason for d in denials)

            node.effects = [f"[BLOCKED by inline rail] {e}" for e in node.effects] \
                or [f"[BLOCKED by inline rail] {d.tool} denied" for d in denials]

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

        for src in span.inputs_from:
            if self.store.get_node(src) is not None:
                self.store.add_edge(TwinEdge(src=src, dst=span.span_id, kind="influence"))
        return node
