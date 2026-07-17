from __future__ import annotations

import re

from . import detection
from .detection import DEFAULT_POLICY, DetectionPolicy
from .embeddings import Embedder
from .models import DriftStatus, Span, WhatIfPreview
from .store import TwinStore


def _node_to_span(node, output: str, effects: list[str]) -> Span:
    return Span(
        span_id=node.node_id, trace_id=node.trace_id,
        agent_id=node.agent_id, agent_role=node.agent_role,
        privilege=node.privilege,
        task_spec=node.task, declared_intent=node.declared_intent,
        tool_calls=node.tool_calls, effects=list(effects), output=output,
    )


def _redact(text: str, entities: set[str]) -> str:
    out = text
    for e in sorted(entities, key=len, reverse=True):
        out = re.sub(re.escape(e), "[redacted-by-remediation]", out)
    return out


def replay(store: TwinStore, root_id: str, emb: Embedder,
           policy: DetectionPolicy = DEFAULT_POLICY) -> dict:
    root = store.get_node(root_id)
    if root is None:
        return {}

    radius = set(store.blast_radius(root_id))
    scope_ids = radius | {root_id}
    order = store.topo_order(scope_ids)
    scope = [root_id] + [n for n in order if n in radius and n != root_id]

    nodes = store.get_nodes(scope)
    preds: dict[str, list[str]] = {nid: store.predecessors(nid)
                                   for nid in scope}
    outside = {p for ps in preds.values() for p in ps} - set(scope)
    nodes.update(store.get_nodes(list(outside)))

    def original_upstream(nid: str) -> str:
        parts = []
        for p in preds.get(nid, []):
            n = nodes.get(p)
            if n is not None:
                parts.append(n.output + " " + " ".join(n.effects))
        return " ".join(parts)

    corrected_out: dict[str, str] = {}
    before: dict[str, float] = {}
    after: dict[str, float] = {}
    after_status: dict[str, DriftStatus] = {}

    for nid in scope:
        node = nodes.get(nid)
        if node is None:
            continue
        before[nid] = node.drift.score

        corr_up = []
        for p in preds.get(nid, []):
            if p in corrected_out:
                corr_up.append(corrected_out[p])
            else:
                pn = nodes.get(p)
                corr_up.append(
                    (pn.output + " " + " ".join(pn.effects)) if pn else "")
        corr_up_text = " ".join(corr_up)

        orig_up = original_upstream(nid)
        span_now = _node_to_span(node, node.output, node.effects)
        ents = detection.foreign_entities(span_now)
        intrinsic_ents = {e for e in ents if e.lower() not in orig_up.lower()}
        inherited_ents = ents - intrinsic_ents

        undeclared_dangerous = any(
            c.name in policy.dangerous_tools
            and c.name.replace("_", " ") not in node.declared_intent.lower()
            and c.name not in node.declared_intent.lower()
            for c in node.tool_calls)

        is_root = nid == root_id
        own_fault = undeclared_dangerous or (intrinsic_ents and not is_root)

        if is_root:
            clean_out = _redact(node.output, intrinsic_ents)
            clean_eff = [_redact(e, intrinsic_ents) for e in node.effects]
            corrected_out[nid] = clean_out + " " + " ".join(clean_eff)
        elif own_fault:
            clean_out = _redact(node.output, inherited_ents)
            clean_eff = [_redact(e, inherited_ents) for e in node.effects]
            corrected_out[nid] = node.output + " " + " ".join(node.effects)
        else:
            clean_out = _redact(node.output, ents)
            clean_eff = [_redact(e, ents) for e in node.effects]
            corrected_out[nid] = clean_out + " " + " ".join(clean_eff)

        span_re = _node_to_span(node, clean_out, clean_eff)
        signals = detection.assess_span(
            span_re, emb, policy=policy,
            upstream_texts=[corr_up_text] if corr_up_text.strip() else [])
        after[nid] = signals.score
        after_status[nid] = signals.status

    downstream = [n for n in scope if n != root_id]
    drifted_now = [n for n in downstream
                   if nodes.get(n) is not None
                   and nodes[n].drift.status in
                   (DriftStatus.FLAGGED, DriftStatus.WATCH)]

    saved = [n for n in drifted_now
             if after_status.get(n) != DriftStatus.FLAGGED]
    still_bad = [n for n in drifted_now if n not in saved]

    return {
        "order": scope,
        "before": before,
        "after": after,
        "after_status": {k: v.value for k, v in after_status.items()},
        "do_nothing": drifted_now,
        "saved": saved,
        "still_bad": still_bad,
        "root_after": after.get(root_id, 0.0),
    }


def build_preview(store: TwinStore, root_id: str, emb: Embedder,
                  policy: DetectionPolicy = DEFAULT_POLICY
                  ) -> WhatIfPreview | None:
    root = store.get_node(root_id)
    if root is None:
        return None
    r = replay(store, root_id, emb, policy)
    if not r:
        return None

    saved, still_bad, do_nothing = r["saved"], r["still_bad"], r["do_nothing"]
    summary = (
        f"Replay over the twin: doing nothing leaves {len(do_nothing)} "
        f"downstream node(s) drifted. Substituting corrected context at "
        f"{root.agent_id} and re-running detection recovers {len(saved)} of "
        f"them (inherited contamination); {len(still_bad)} carry their own "
        f"intrinsic fault and stay flagged. No agents were re-executed."
    )
    return WhatIfPreview(
        node_id=root_id,
        do_nothing_blast_radius=do_nothing,
        remediated_blast_radius=still_bad,
        contained_nodes=saved,
        projected_drift_before={k: v for k, v in r["before"].items()
                                if k != root_id},
        projected_drift_after={k: v for k, v in r["after"].items()
                               if k != root_id},
        summary=summary,
    )
