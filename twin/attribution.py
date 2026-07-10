from __future__ import annotations

import uuid

from .graph import TwinStore
from .models import (
    CausalNarrative, DriftStatus, RemediationAction, RemediationKind, TwinNode,
)
from .router import WATCH_THRESHOLD

def _drifted(node: TwinNode) -> bool:
    return node.drift.score >= WATCH_THRESHOLD or node.drift.status in (
        DriftStatus.FLAGGED, DriftStatus.WATCH
    )

def find_root_cause(store: TwinStore, target_id: str) -> str:
    target = store.get_node(target_id)
    if target is None:
        return target_id
    candidates = set(store.upstream(target_id)) | {target_id}
    drifted = {n for n in candidates
               if (nd := store.get_node(n)) is not None and _drifted(nd)}
    if not drifted:
        return target_id

    for n in drifted:
        anc = set(store.upstream(n))
        if not (anc & drifted):
            return n

    return min(drifted, key=lambda n: store.get_node(n).timestamp or 0)

_PRIV_RANK = {"high": 2, "medium": 1, "low": 0}

def worst_node(store: TwinStore) -> TwinNode | None:
    nodes = [n for n in store.all_nodes() if not n.quarantined]
    if not nodes:
        return None

    return max(nodes, key=lambda n: (
        n.drift.score, _PRIV_RANK.get(n.privilege.value, 0), n.timestamp or 0.0))

def propose_remediation(store: TwinStore, root: TwinNode) -> list[RemediationAction]:
    actions: list[RemediationAction] = []
    downstream = store.blast_radius(root.node_id)

    good = store.latest_good_checkpoint(root.agent_id, (root.timestamp or 0) - 1e-6)
    actions.append(RemediationAction(
        action_id=f"act::{uuid.uuid4().hex[:8]}",
        node_id=root.node_id,
        kind=RemediationKind.ROLLBACK,
        params={"checkpoint_id": good},
        rationale=(f"Restore {root.agent_id} to the last context snapshot taken before "
                   f"drift crossed threshold."),
        reversible=True,
    ))

    for dst in store.successors(root.node_id):
        actions.append(RemediationAction(
            action_id=f"act::{uuid.uuid4().hex[:8]}",
            node_id=root.node_id,
            kind=RemediationKind.MESSAGE_FILTER,
            params={"edge_to": dst},
            rationale=f"Drop/replace the contaminated hand-off from {root.node_id} to {dst} "
                      f"before it propagates further.",
            reversible=True,
        ))

    if root.privilege.value in ("high", "medium") or len(downstream) >= 2:
        actions.append(RemediationAction(
            action_id=f"act::{uuid.uuid4().hex[:8]}",
            node_id=root.node_id,
            kind=RemediationKind.QUARANTINE,
            params={"blast_radius": downstream},
            rationale=f"Isolate {root.agent_id} from the graph; it influenced "
                      f"{len(downstream)} downstream node(s).",
            reversible=True,
        ))
    return actions

def build_narrative(store: TwinStore, target_id: str | None = None) -> CausalNarrative | None:
    if target_id is None:
        w = worst_node(store)
        if w is None:
            return None
        target_id = w.node_id
    target = store.get_node(target_id)
    if target is None:
        return None

    root_id = find_root_cause(store, target_id)
    root = store.get_node(root_id)
    path = store.propagation_path(root_id, target_id) or [root_id]
    blast = store.blast_radius(root_id)
    remediation = propose_remediation(store, root) if root else []

    def label(nid: str) -> str:
        n = store.get_node(nid)
        return f"{n.agent_id}" if n else nid

    risk = (root.drift.risk_type or "drift") if root else "drift"
    root_summary = (
        f"{risk.replace('_', ' ').title()} originated at {label(root_id)} "
        f"({root.agent_role or root.agent_id}). "
        f"Stated-vs-revealed gap {root.drift.stated_vs_revealed:.2f}, "
        f"trajectory drift {root.drift.trajectory_drift:.2f}."
    ) if root else "Unattributed drift."

    path_str = " -> ".join(label(n) for n in path)
    blast_str = ", ".join(sorted(label(n) for n in blast)) or "none"

    narrative = (
        f"Root cause: {root_summary}\n"
        f"Propagation path: {path_str}.\n"
        f"Blast radius: {len(blast)} downstream node(s) — {blast_str}.\n"
        f"Recommended remediation: "
        + "; ".join(f"{a.kind.value} on {label(a.node_id)}" for a in remediation)
        + "."
    )

    return CausalNarrative(
        incident_id=f"inc::{uuid.uuid4().hex[:8]}",
        root_cause_node=root_id,
        root_cause_summary=root_summary,
        propagation_path=path,
        blast_radius=blast,
        recommended_remediation=remediation,
        narrative=narrative,
    )
