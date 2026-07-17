from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict

from . import detection
from .detection import DetectionPolicy, PolicyResolver
from .embeddings import Embedder
from .guard import GuardBackend, NativeGuard
from .llm import JudgePair
from .models import (
    DriftSignals, DriftStatus, Privilege, Span, Tier, TwinEdge, TwinNode,
)
from .store import TwinStore

C_SPANS_SEEN = "spans_seen"
C_SPANS_SAMPLED = "spans_sampled"
C_CACHE_HITS = "cache_hits"
C_BASELINE_TOKENS = "baseline_tokens"
C_JUDGE_TOKENS = "judge_tokens"
C_TIER_PREFIX = "tier_"

LEDGER_KEYS = [
    C_SPANS_SEEN, C_SPANS_SAMPLED, C_CACHE_HITS,
    C_BASELINE_TOKENS, C_JUDGE_TOKENS,
] + [C_TIER_PREFIX + t.value for t in Tier]


class _SignalsCache:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, DriftSignals] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> DriftSignals | None:
        with self._lock:
            sig = self._data.get(key)
            if sig is not None:
                self._data.move_to_end(key)
                return sig.model_copy(deep=True)
            return None

    def put(self, key: str, sig: DriftSignals) -> None:
        with self._lock:
            self._data[key] = sig.model_copy(deep=True)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)


class CostRouter:
    def __init__(self, store: TwinStore, emb: Embedder, judges: JudgePair,
                 policy: DetectionPolicy,
                 low_sample_rate: int = 3,
                 cache_size: int = 4096,
                 guard: GuardBackend | None = None,
                 resolver: PolicyResolver | None = None) -> None:
        self.store = store
        self.emb = emb
        self.judges = judges
        self.policy = policy
        self.resolver = resolver or PolicyResolver(policy)
        self.guard = guard or NativeGuard(policy)
        self.low_sample_rate = max(1, low_sample_rate)
        self._cache = _SignalsCache(cache_size)

    def _sampled(self, span: Span) -> bool:
        if span.privilege in (Privilege.HIGH, Privilege.MEDIUM):
            return True
        h = int(hashlib.blake2b(span.span_id.encode(),
                                digest_size=4).hexdigest(), 16)
        return (h % self.low_sample_rate) == 0

    def _cache_key(self, span: Span) -> str:
        payload = span.declared_intent + "||" + \
            detection._behaviour_text(span) + "||" + span.task_spec + \
            "||" + (span.workflow or "")
        return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()

    def _upstream_texts(self, span: Span) -> list[str]:
        nodes = self.store.get_nodes(span.inputs_from)
        return [n.output + " " + " ".join(n.effects) for n in nodes.values()]

    def ingest(self, span: Span) -> TwinNode:
        policy = self.resolver.resolve(span.workflow)
        deltas: dict[str, float] = {C_SPANS_SEEN: 1}
        baseline = int(span.meta.get("baseline_tokens", 0) or 0)
        if baseline > 0:
            deltas[C_BASELINE_TOKENS] = baseline

        if not self._sampled(span):
            signals = DriftSignals(
                status=DriftStatus.OK, tier_reached=Tier.SAMPLED,
                rationale="not sampled (low-privilege thinning)")
            deltas[C_TIER_PREFIX + Tier.SAMPLED.value] = 1
            self.store.incr_counters(deltas)
            return self._commit(span, signals)

        deltas[C_SPANS_SAMPLED] = 1

        key = self._cache_key(span)
        cached = self._cache.get(key)
        if cached is not None:
            deltas[C_CACHE_HITS] = 1
            cached.rationale = "cache hit: identical reasoning already analysed"
            self.store.incr_counters(deltas)
            return self._commit(span, cached)

        anchor_text = self.store.trace_anchor_task(span.trace_id) \
            or span.task_spec
        anchor_vec = self.emb.encode(anchor_text) if anchor_text else None
        upstream = self._upstream_texts(span)

        signals = detection.assess_span(
            span, self.emb, self.judges.small, policy,
            upstream_texts=upstream, anchor_vec=anchor_vec)

        if signals.judge is not None:
            deltas[C_TIER_PREFIX + Tier.SMALL_JUDGE.value] = 1
        else:
            deltas[C_TIER_PREFIX + Tier.ZERO_INFERENCE.value] = 1

        if signals.status == DriftStatus.FLAGGED and self.judges.enabled \
                and self.judges.deep is not None:
            deep_verdict = self.judges.deep.assess(span, signals)
            signals.judge = deep_verdict
            structural = signals.svr_primary not in ("", "semantic", "lexical")
            final = detection.aggregate(
                signals.stated_vs_revealed, signals.trajectory_drift,
                signals.deterministic, deep_verdict, structural)
            signals.score = round(final, 3)
            if final >= policy.flag_threshold:
                signals.status = DriftStatus.FLAGGED
            elif final >= policy.watch_threshold:
                signals.status = DriftStatus.WATCH
            else:
                signals.status = DriftStatus.OK
            signals.tier_reached = Tier.DEEP_ESCALATION
            deltas[C_TIER_PREFIX + Tier.DEEP_ESCALATION.value] = 1
        elif signals.status == DriftStatus.FLAGGED:
            signals.tier_reached = Tier.DEEP_ESCALATION
            deltas[C_TIER_PREFIX + Tier.DEEP_ESCALATION.value] = 1

        measured = self.judges.pop_measured_tokens()
        if measured:
            deltas[C_JUDGE_TOKENS] = measured

        self._cache.put(key, signals)
        self.store.incr_counters(deltas)
        return self._commit(span, signals)

    def _commit(self, span: Span, signals: DriftSignals) -> TwinNode:
        node = TwinNode(
            node_id=span.span_id,
            agent_id=span.agent_id,
            agent_role=span.agent_role,
            privilege=span.privilege,
            task=span.task_spec,
            workflow=span.workflow,
            declared_intent=span.declared_intent,
            tool_calls=span.tool_calls,
            effects=list(span.effects),
            output=span.output,
            drift=signals,
            timestamp=span.timestamp or time.time(),
            trace_id=span.trace_id,
        )

        denials = self.guard.blocked_decisions(span)
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

        if span.inputs_from:
            existing = self.store.existing_ids(span.inputs_from)
            for src in span.inputs_from:
                if src in existing:
                    self.store.add_edge(
                        TwinEdge(src=src, dst=span.span_id, kind="influence"))
        return node

    def ledger(self) -> dict:
        counters = self.store.get_counters(LEDGER_KEYS)
        baseline = counters[C_BASELINE_TOKENS]
        added = counters[C_JUDGE_TOKENS]
        overhead = 100.0 * added / baseline if baseline > 0 else None
        return {
            "baseline_tokens": int(baseline),
            "added_tokens": int(added),
            "overhead_pct": round(overhead, 3) if overhead is not None else None,
            "spans_seen": int(counters[C_SPANS_SEEN]),
            "spans_sampled": int(counters[C_SPANS_SAMPLED]),
            "cache_hits": int(counters[C_CACHE_HITS]),
            "tier_counts": {
                t.value: int(counters[C_TIER_PREFIX + t.value]) for t in Tier
            },
            "measured": True,
            "note": ("token figures are measured: baseline from telemetry "
                     "usage attributes, added from judge API usage; "
                     "overhead_pct is null until baseline usage is reported"),
        }
