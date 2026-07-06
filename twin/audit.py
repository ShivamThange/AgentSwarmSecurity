"""Supervisor audit trail — "who watches the watcher" (Section 8.4 / 14.2).

The remediation layer rewrites agents' context, so it is itself a powerful actor.
Every action it takes is attributable, reversible, and logged in a tamper-evident
hash chain. A configurable compliance-mapping layer tags each entry against the
frameworks the buyer is accountable to (EU AI Act, NIST AI RMF, ISO/IEC 42001) —
data 8.4 already requires, re-projected for GRC (14.2). The mapping is a thin,
swappable annotation, never baked into the core model.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from typing import Optional

from .models import AuditEntry

# Configurable compliance mapping (14.2 guardrail: swappable, not hardcoded).
# event key -> list of "Framework Clause — obligation".
DEFAULT_COMPLIANCE_MAP: dict[str, list[str]] = {
    "detection": [
        "EU AI Act Art. 15 — accuracy & robustness monitoring",
        "NIST AI RMF MEASURE-2.6 — ongoing monitoring of AI risks",
        "ISO/IEC 42001 A.6.2.6 — performance monitoring",
    ],
    "escalation": [
        "EU AI Act Art. 14 — human oversight",
        "NIST AI RMF MANAGE-2.3 — response to identified risks",
    ],
    "remediation.proposed": [
        "EU AI Act Art. 9 — risk-management measures",
        "NIST AI RMF MANAGE-1 — risk treatment planning",
    ],
    "remediation.approved": [
        "EU AI Act Art. 14 — human-in-the-loop authorisation",
        "ISO/IEC 42001 A.9.2 — operational control & authorisation",
    ],
    "remediation.applied": [
        "EU AI Act Art. 12 — automatic record-keeping (logging)",
        "EU AI Act Art. 9 — risk-management measures",
        "NIST AI RMF MANAGE-4.1 — documented risk response",
    ],
    "remediation.reverted": [
        "EU AI Act Art. 12 — record-keeping of corrective actions",
    ],
    "quarantine": [
        "EU AI Act Art. 9 — mitigation of residual risk",
        "ISO/IEC 42001 A.8.4 — incident containment",
    ],
}


class AuditLog:
    def __init__(self, conn: Optional[sqlite3.Connection] = None,
                 compliance_map: Optional[dict[str, list[str]]] = None) -> None:
        self._entries: list[AuditEntry] = []
        self._last_hash = "genesis"
        self._conn = conn
        self.compliance_map = compliance_map or DEFAULT_COMPLIANCE_MAP
        if conn is not None:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS audit ("
                "seq INTEGER PRIMARY KEY AUTOINCREMENT, entry_id TEXT, data TEXT)"
            )
            conn.commit()
            self._load()

    def _load(self) -> None:
        assert self._conn is not None
        for row in self._conn.execute("SELECT data FROM audit ORDER BY seq"):
            e = AuditEntry.model_validate_json(row[0])
            self._entries.append(e)
            self._last_hash = e.hash

    def _compliance_tags(self, action: str) -> list[str]:
        # match on exact key, then on the "prefix.*" family, then the bare prefix
        if action in self.compliance_map:
            return list(self.compliance_map[action])
        prefix = action.split(".")[0]
        return list(self.compliance_map.get(prefix, []))

    def record(self, actor: str, action: str, target: str, detail: str = "",
               reversible: bool = True,
               extra_tags: Optional[list[str]] = None) -> AuditEntry:
        ts = time.time()
        tags = self._compliance_tags(action) + (extra_tags or [])
        body = json.dumps({
            "actor": actor, "action": action, "target": target,
            "detail": detail, "ts": ts, "prev": self._last_hash,
        }, sort_keys=True)
        digest = hashlib.sha256((self._last_hash + body).encode()).hexdigest()
        entry = AuditEntry(
            entry_id=f"aud::{uuid.uuid4().hex[:8]}",
            actor=actor, action=action, target=target, detail=detail,
            reversible=reversible, compliance_tags=tags,
            prev_hash=self._last_hash, hash=digest, timestamp=ts,
        )
        self._entries.append(entry)
        self._last_hash = digest
        if self._conn is not None:
            self._conn.execute(
                "INSERT INTO audit(entry_id, data) VALUES(?,?)",
                (entry.entry_id, entry.model_dump_json()))
            self._conn.commit()
        return entry

    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def reset(self) -> None:
        """Clear the log (in-memory and persisted) and restart the hash chain."""
        self._entries = []
        self._last_hash = "genesis"
        if self._conn is not None:
            self._conn.execute("DELETE FROM audit")
            self._conn.commit()

    def verify_chain(self) -> bool:
        """Recompute the hash chain to prove the log has not been tampered with."""
        prev = "genesis"
        for e in self._entries:
            body = json.dumps({
                "actor": e.actor, "action": e.action, "target": e.target,
                "detail": e.detail, "ts": e.timestamp, "prev": prev,
            }, sort_keys=True)
            if hashlib.sha256((prev + body).encode()).hexdigest() != e.hash:
                return False
            prev = e.hash
        return True

    def compliance_report(self) -> dict:
        """Aggregate log entries by framework clause — audit-ready on demand."""
        by_clause: dict[str, int] = {}
        for e in self._entries:
            for tag in e.compliance_tags:
                by_clause[tag] = by_clause.get(tag, 0) + 1
        return {
            "total_events": len(self._entries),
            "chain_valid": self.verify_chain(),
            "coverage_by_clause": dict(sorted(by_clause.items())),
        }
