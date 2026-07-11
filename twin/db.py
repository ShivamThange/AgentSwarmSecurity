from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import (
    JSON, Boolean, Float, Index, Integer, String, Text, create_engine, event,
)
from sqlalchemy.engine import Engine as SAEngine
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, Session, mapped_column, sessionmaker,
)

from .config import Settings


class Base(DeclarativeBase):
    pass


class NodeRow(Base):
    __tablename__ = "nodes"

    node_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(256), default="", index=True)
    agent_id: Mapped[str] = mapped_column(String(256), default="", index=True)
    privilege: Mapped[str] = mapped_column(String(16), default="low")
    drift_status: Mapped[str] = mapped_column(String(16), default="ok", index=True)
    drift_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    blocked: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    quarantined: Mapped[bool] = mapped_column(Boolean, default=False)
    ts: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    data: Mapped[dict] = mapped_column(JSON)


Index("idx_nodes_trace_ts", NodeRow.trace_id, NodeRow.ts)


class EdgeRow(Base):
    __tablename__ = "edges"

    src: Mapped[str] = mapped_column(String(256), primary_key=True)
    dst: Mapped[str] = mapped_column(String(256), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), primary_key=True,
                                      default="influence")
    weight: Mapped[float] = mapped_column(Float, default=1.0)


Index("idx_edges_src", EdgeRow.src)
Index("idx_edges_dst", EdgeRow.dst)


class CheckpointRow(Base):
    __tablename__ = "checkpoints"

    checkpoint_id: Mapped[str] = mapped_column(String(300), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(256), default="")
    agent_id: Mapped[str] = mapped_column(String(256), default="", index=True)
    label: Mapped[str] = mapped_column(Text, default="")
    context: Mapped[dict] = mapped_column(JSON)
    ts: Mapped[float] = mapped_column(Float, default=0.0, index=True)


class RemediationRow(Base):
    __tablename__ = "remediation_actions"

    action_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(256), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="proposed", index=True)
    ts: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    data: Mapped[dict] = mapped_column(JSON)


class AuditRow(Base):
    __tablename__ = "audit_log"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entry_id: Mapped[str] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(256))
    action: Mapped[str] = mapped_column(String(128), index=True)
    target: Mapped[str] = mapped_column(String(256), index=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    reversible: Mapped[bool] = mapped_column(Boolean, default=True)
    compliance_tags: Mapped[list] = mapped_column(JSON, default=list)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64))
    ts: Mapped[float] = mapped_column(Float, default=0.0, index=True)


class AuditHeadRow(Base):
    __tablename__ = "audit_head"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    last_hash: Mapped[str] = mapped_column(String(64), default="genesis")
    last_seq: Mapped[int] = mapped_column(Integer, default=0)


class ApiKeyRow(Base):
    __tablename__ = "api_keys"

    key_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    key_hash: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_ts: Mapped[float] = mapped_column(Float, default=0.0)
    last_used_ts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class MetaRow(Base):
    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)


class CounterRow(Base):
    __tablename__ = "counters"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[float] = mapped_column(Float, default=0.0)


def build_engine(settings: Settings) -> SAEngine:
    url = settings.database_url
    kwargs: dict[str, Any] = {"echo": settings.db_echo, "future": True}
    if settings.is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        kwargs["pool_size"] = settings.db_pool_size
        kwargs["max_overflow"] = settings.db_max_overflow
        kwargs["pool_pre_ping"] = True
    engine = create_engine(url, **kwargs)

    if settings.is_sqlite:
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def init_schema(engine: SAEngine) -> None:
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        if session.get(AuditHeadRow, 1) is None:
            session.add(AuditHeadRow(id=1, last_hash="genesis", last_seq=0))
            try:
                session.commit()
            except Exception:
                session.rollback()


def build_session_factory(engine: SAEngine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
