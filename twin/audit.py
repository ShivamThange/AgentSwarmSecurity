from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from .db import AuditHeadRow, AuditRow
from .models import AuditEntry

log = logging.getLogger(__name__)

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
    "remediation.rejected": [
        "EU AI Act Art. 14 — human-in-the-loop authorisation",
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
    "auth": [
        "ISO/IEC 42001 A.9.2 — operational control & authorisation",
    ],
    "retention": [
        "EU AI Act Art. 12 — record-keeping policy",
    ],
    "feedback": [
        "EU AI Act Art. 14 — human oversight",
        "NIST AI RMF MEASURE-4 — feedback from human review",
        "ISO/IEC 42001 A.6.2.8 — continual improvement from monitoring",
    ],
}


def load_compliance_map(path: Optional[str]) -> dict[str, list[str]]:
    """Merge a JSON compliance map from ``path`` over the built-in defaults.

    The file is ``{action_or_prefix: [clause, ...]}``. Its entries override
    matching default keys and add new ones. A missing path returns the defaults
    unchanged; a malformed file logs and falls back to the defaults so a bad
    config can never wipe the audit trail's compliance annotations.
    """
    merged: dict[str, list[str]] = {k: list(v)
                                    for k, v in DEFAULT_COMPLIANCE_MAP.items()}
    if not path:
        return merged
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("compliance map must be a JSON object")
        for action, clauses in data.items():
            if not isinstance(clauses, list) or not all(
                    isinstance(c, str) for c in clauses):
                raise ValueError(
                    f"compliance map entry '{action}' must be a list of "
                    f"strings")
            merged[str(action)] = list(clauses)
        log.info("loaded compliance map overrides from %s (%d entries)",
                 path, len(data))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.warning("failed to load compliance map from %s (%s: %s); using "
                    "built-in defaults", path, type(exc).__name__, exc)
    return merged


def _body(actor: str, action: str, target: str, detail: str, ts: float,
          prev: str) -> str:
    return json.dumps({
        "actor": actor, "action": action, "target": target,
        "detail": detail, "ts": ts, "prev": prev,
    }, sort_keys=True)


def _row_to_entry(row: AuditRow) -> AuditEntry:
    return AuditEntry(
        entry_id=row.entry_id, actor=row.actor, action=row.action,
        target=row.target, detail=row.detail, reversible=row.reversible,
        compliance_tags=list(row.compliance_tags or []),
        prev_hash=row.prev_hash, hash=row.hash, timestamp=row.ts,
    )


class AuditLog:
    def __init__(self, session_factory: sessionmaker[Session],
                 compliance_map: Optional[dict[str, list[str]]] = None) -> None:
        self._sf = session_factory
        self.compliance_map = compliance_map or DEFAULT_COMPLIANCE_MAP

    def _compliance_tags(self, action: str) -> list[str]:
        if action in self.compliance_map:
            return list(self.compliance_map[action])
        prefix = action.split(".")[0]
        return list(self.compliance_map.get(prefix, []))

    def record(self, actor: str, action: str, target: str, detail: str = "",
               reversible: bool = True,
               extra_tags: Optional[list[str]] = None,
               session: Optional[Session] = None) -> AuditEntry:
        if session is not None:
            return self._record_in(session, actor, action, target, detail,
                                   reversible, extra_tags)
        with self._sf() as s:
            entry = self._record_in(s, actor, action, target, detail,
                                    reversible, extra_tags)
            s.commit()
            return entry

    def _record_in(self, s: Session, actor: str, action: str, target: str,
                   detail: str, reversible: bool,
                   extra_tags: Optional[list[str]]) -> AuditEntry:
        head = s.execute(
            select(AuditHeadRow).where(AuditHeadRow.id == 1)
            .with_for_update()).scalar_one()
        ts = time.time()
        prev = head.last_hash
        digest = hashlib.sha256(
            (prev + _body(actor, action, target, detail, ts, prev)).encode()
        ).hexdigest()
        tags = self._compliance_tags(action) + (extra_tags or [])
        row = AuditRow(
            entry_id=f"aud::{uuid.uuid4().hex[:12]}",
            actor=actor, action=action, target=target, detail=detail,
            reversible=reversible, compliance_tags=tags,
            prev_hash=prev, hash=digest, ts=ts,
        )
        s.add(row)
        head.last_hash = digest
        head.last_seq = head.last_seq + 1
        return _row_to_entry(row)

    def entries(self, limit: int = 100, offset: int = 0,
                action: Optional[str] = None,
                target: Optional[str] = None) -> tuple[list[AuditEntry], int]:
        stmt = select(AuditRow)
        if action:
            stmt = stmt.where(AuditRow.action == action)
        if target:
            stmt = stmt.where(AuditRow.target == target)
        with self._sf() as s:
            total = int(s.scalar(
                select(func.count()).select_from(stmt.subquery())) or 0)
            rows = s.scalars(
                stmt.order_by(AuditRow.seq.asc())
                .limit(limit).offset(offset)).all()
        return [_row_to_entry(r) for r in rows], total

    def count(self) -> int:
        with self._sf() as s:
            return int(s.scalar(select(func.count()).select_from(AuditRow)) or 0)

    def verify_chain(self) -> dict:
        prev = "genesis"
        checked = 0
        with self._sf() as s:
            result = s.execute(
                select(AuditRow).order_by(AuditRow.seq.asc())
                .execution_options(yield_per=500))
            for row in result.scalars():
                expected = hashlib.sha256(
                    (prev + _body(row.actor, row.action, row.target,
                                  row.detail, row.ts, prev)).encode()
                ).hexdigest()
                if expected != row.hash:
                    return {"valid": False, "checked": checked,
                            "broken_at_seq": row.seq,
                            "broken_entry_id": row.entry_id}
                prev = row.hash
                checked += 1
            head = s.get(AuditHeadRow, 1)
        head_ok = head is not None and (checked == 0 or head.last_hash == prev)
        return {"valid": head_ok, "checked": checked, "broken_at_seq": None,
                "broken_entry_id": None}

    def active_map(self) -> dict:
        clauses = sorted({c for tags in self.compliance_map.values()
                          for c in tags})
        return {
            "actions": {k: list(v) for k, v in
                        sorted(self.compliance_map.items())},
            "clauses": clauses,
            "clause_count": len(clauses),
        }

    def compliance_report(self) -> dict:
        by_clause: dict[str, int] = {}
        with self._sf() as s:
            result = s.execute(
                select(AuditRow.compliance_tags)
                .execution_options(yield_per=500))
            total = 0
            for (tags,) in result:
                total += 1
                for tag in tags or []:
                    by_clause[tag] = by_clause.get(tag, 0) + 1
        verify = self.verify_chain()
        return {
            "total_events": total,
            "chain_valid": verify["valid"],
            "coverage_by_clause": dict(sorted(by_clause.items())),
        }
