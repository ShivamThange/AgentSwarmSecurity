from __future__ import annotations

import hashlib

from .detection import DEFAULT_POLICY, DetectionPolicy
from .models import (
    CausalNarrative, DriftStatus, RemediationAction, RemediationKind, TwinNode,
)
from .remediation import new_action
from .store import TwinStore


def _drifted(node: TwinNode, policy: DetectionPolicy) -> bool:
    return node.drift.score >= policy.watch_threshold or node.drift.status in (
        DriftStatus.FLAGGED, DriftStatus.WATCH
    )


def find_root_cause(store: TwinStore, target_id: str,
                    policy: DetectionPolicy = DEFAULT_POLICY) -> str:
    target = store.get_node(target_id)
    if target is None:
        return target_id
    candidate_ids = set(store.upstream(target_id)) | {target_id}
    nodes = store.get_nodes(list(candidate_ids))
    drifted = {nid for nid, n in nodes.items() if _drifted(n, policy)}
    if not drifted:
        return target_id

    for nid in drifted:
        anc = set(store.upstream(nid))
        if not (anc & drifted):
            return nid

    return min(drifted, key=lambda nid: nodes[nid].timestamp or 0)


def propose_remediation(store: TwinStore,
                        root: TwinNode) -> list[RemediationAction]:
    actions: list[RemediationAction] = []
    downstream = store.blast_radius(root.node_id)

    good = store.latest_good_checkpoint(
        root.agent_id, (root.timestamp or 0) - 1e-6)
    actions.append(new_action(
        root.node_id, RemediationKind.ROLLBACK,
        params={"checkpoint_id": good},
        rationale=(f"Restore {root.agent_id} to the last context snapshot "
                   f"taken before drift crossed threshold."),
    ))

    for dst in store.successors(root.node_id):
        actions.append(new_action(
            root.node_id, RemediationKind.MESSAGE_FILTER,
            params={"edge_to": dst},
            rationale=(f"Drop/replace the contaminated hand-off from "
                       f"{root.node_id} to {dst} before it propagates further."),
        ))

    if root.privilege.value in ("high", "medium") or len(downstream) >= 2:
        actions.append(new_action(
            root.node_id, RemediationKind.QUARANTINE,
            params={"blast_radius": downstream},
            rationale=(f"Isolate {root.agent_id} from the graph; it influenced "
                       f"{len(downstream)} downstream node(s)."),
        ))
    return actions


def build_narrative(store: TwinStore, target_id: str | None = None,
                    policy: DetectionPolicy = DEFAULT_POLICY
                    ) -> CausalNarrative | None:
    if target_id is None:
        w = store.worst_node()
        if w is None or not _drifted(w, policy):
            return None
        target_id = w.node_id
    target = store.get_node(target_id)
    if target is None:
        return None

    root_id = find_root_cause(store, target_id, policy)
    root = store.get_node(root_id)
    path = store.propagation_path(root_id, target_id) or [root_id]
    blast = store.blast_radius(root_id)
    remediation = propose_remediation(store, root) if root else []

    labels = store.get_nodes(list(set(path) | set(blast) | {root_id}))

    def label(nid: str) -> str:
        n = labels.get(nid)
        return n.agent_id if n else nid

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

    incident_id = "inc::" + hashlib.blake2b(
        f"{root_id}|{target_id}".encode(), digest_size=6).hexdigest()

    return CausalNarrative(
        incident_id=incident_id,
        root_cause_node=root_id,
        root_cause_summary=root_summary,
        propagation_path=path,
        blast_radius=blast,
        recommended_remediation=remediation,
        narrative=narrative,
    )


def list_incidents(store: TwinStore, policy: DetectionPolicy = DEFAULT_POLICY,
                   limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    traces, total = store.list_traces(limit=limit, offset=offset,
                                      flagged_only=True)
    incidents: list[dict] = []
    for t in traces:
        flagged, _ = store.list_nodes(trace_id=t["trace_id"], status="flagged",
                                      limit=1, offset=0)
        root_id = None
        risk = None
        if flagged:
            root_id = find_root_cause(store, flagged[0].node_id, policy)
            root = store.get_node(root_id)
            risk = root.drift.risk_type if root else None
        incidents.append({
            **t,
            "root_cause_node": root_id,
            "risk_type": risk,
        })
    return incidents, total
