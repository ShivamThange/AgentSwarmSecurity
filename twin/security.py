from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .db import ApiKeyRow

ROLE_INGEST = "ingest"
ROLE_VIEWER = "viewer"
ROLE_OPERATOR = "operator"
ROLE_ADMIN = "admin"
ROLES = (ROLE_INGEST, ROLE_VIEWER, ROLE_OPERATOR, ROLE_ADMIN)

_READ_ROLES = {ROLE_VIEWER, ROLE_OPERATOR, ROLE_ADMIN}
_INGEST_ROLES = {ROLE_INGEST, ROLE_OPERATOR, ROLE_ADMIN}
_OPERATE_ROLES = {ROLE_OPERATOR, ROLE_ADMIN}
_ADMIN_ROLES = {ROLE_ADMIN}

PERMISSIONS: dict[str, set[str]] = {
    "read": _READ_ROLES,
    "ingest": _INGEST_ROLES,
    "operate": _OPERATE_ROLES,
    "admin": _ADMIN_ROLES,
}

KEY_PREFIX = "twin"


@dataclass(frozen=True)
class Principal:
    key_id: str
    name: str
    role: str

    def can(self, permission: str) -> bool:
        return self.role in PERMISSIONS.get(permission, set())


DEV_PRINCIPAL = Principal(key_id="dev", name="dev-mode", role=ROLE_ADMIN)


def _hash_secret(full_key: str) -> str:
    return hashlib.sha256(full_key.encode("utf-8")).hexdigest()


def generate_key(role: str) -> tuple[str, str, str]:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    key_id = secrets.token_hex(6)
    secret = secrets.token_urlsafe(32)
    full_key = f"{KEY_PREFIX}_{key_id}_{secret}"
    return key_id, full_key, _hash_secret(full_key)


def parse_key_id(full_key: str) -> Optional[str]:
    parts = full_key.split("_", 2)
    if len(parts) != 3 or parts[0] != KEY_PREFIX or not parts[1]:
        return None
    return parts[1]


class ApiKeyManager:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def create(self, name: str, role: str,
               literal_key: Optional[str] = None) -> tuple[ApiKeyRow, str]:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if literal_key is not None:
            key_id = parse_key_id(literal_key)
            if key_id is None:
                key_id = hashlib.sha256(
                    literal_key.encode()).hexdigest()[:12]
            full_key = literal_key
        else:
            key_id, full_key, _ = generate_key(role)
        row = ApiKeyRow(
            key_id=key_id, name=name, key_hash=_hash_secret(full_key),
            role=role, disabled=False, created_ts=time.time(),
        )
        with self._sf() as s:
            s.merge(row)
            s.commit()
        return row, full_key

    def list_keys(self) -> list[dict]:
        with self._sf() as s:
            rows = s.scalars(select(ApiKeyRow)).all()
        return [
            {"key_id": r.key_id, "name": r.name, "role": r.role,
             "disabled": r.disabled, "created_ts": r.created_ts,
             "last_used_ts": r.last_used_ts}
            for r in rows
        ]

    def set_disabled(self, key_id: str, disabled: bool) -> bool:
        with self._sf() as s:
            row = s.get(ApiKeyRow, key_id)
            if row is None:
                return False
            row.disabled = disabled
            s.commit()
            return True

    def count(self) -> int:
        with self._sf() as s:
            from sqlalchemy import func
            return int(s.scalar(
                select(func.count()).select_from(ApiKeyRow)) or 0)

    def authenticate(self, full_key: str) -> Optional[Principal]:
        key_id = parse_key_id(full_key)
        candidate_ids = [key_id] if key_id else []
        candidate_ids.append(
            hashlib.sha256(full_key.encode()).hexdigest()[:12])
        supplied_hash = _hash_secret(full_key)
        with self._sf() as s:
            for cid in candidate_ids:
                row = s.get(ApiKeyRow, cid)
                if row is None or row.disabled:
                    continue
                if hmac.compare_digest(row.key_hash, supplied_hash):
                    row.last_used_ts = time.time()
                    s.commit()
                    return Principal(key_id=row.key_id, name=row.name,
                                     role=row.role)
        return None

    def bootstrap(self, literal_key: Optional[str]) -> Optional[str]:
        if not literal_key:
            return None
        if self.count() > 0:
            return None
        row, _ = self.create(name="bootstrap-admin", role=ROLE_ADMIN,
                             literal_key=literal_key)
        return row.key_id


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key_id: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key_id, deque())
            cutoff = now - 60.0
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True
