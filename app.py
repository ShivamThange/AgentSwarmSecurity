from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from twin import metrics, otel_ingest
from twin.config import Settings, get_settings
from twin.engine import Engine
from twin.logging_config import configure_logging
from twin.models import Span
from twin.security import ROLES, Principal

log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = Engine(settings)
        app.state.engine = engine
        bootstrapped = engine.keys.bootstrap(settings.bootstrap_admin_key)
        if bootstrapped:
            log.info("bootstrap admin API key registered (key_id=%s)",
                     bootstrapped)
        if settings.auth_enabled and engine.keys.count() == 0:
            log.warning(
                "auth is enabled but no API keys exist — every request will "
                "be rejected; set TWIN_BOOTSTRAP_ADMIN_KEY or create a key "
                "with scripts/manage_keys.py")
        metrics.READY.set(1)
        try:
            yield
        finally:
            metrics.READY.set(0)
            engine.close()

    app = FastAPI(
        title="Agent Drift Containment & Remediation Twin",
        version="1.0.0",
        lifespan=lifespan,
    )

    if settings.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["Authorization", "X-API-Key", "Content-Type"])

    def engine_of(request: Request) -> Engine:
        return request.app.state.engine

    def _extract_key(request: Request) -> Optional[str]:
        key = request.headers.get("x-api-key")
        if key:
            return key.strip()
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    def require(permission: str):
        def dependency(request: Request) -> Principal:
            engine = engine_of(request)
            if not settings.auth_enabled:
                from twin.security import DEV_PRINCIPAL
                return DEV_PRINCIPAL
            raw = _extract_key(request)
            if not raw:
                raise HTTPException(
                    401, "missing API key (X-API-Key or Bearer token)")
            principal = engine.keys.authenticate(raw)
            if principal is None:
                raise HTTPException(401, "invalid or disabled API key")
            if not engine.rate_limiter.allow(principal.key_id):
                raise HTTPException(429, "rate limit exceeded for this key")
            if not principal.can(permission):
                raise HTTPException(
                    403, f"role '{principal.role}' lacks the "
                         f"'{permission}' permission")
            return principal
        return Depends(dependency)

    @app.middleware("http")
    async def _observability(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and int(length) > settings.max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={"error": {"type": "payload_too_large",
                                   "message": "request body exceeds limit"}})
        timer = metrics.Timer()
        response = await call_next(request)
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        timer.observe(request.method, route_path, response.status_code)
        return response

    def _err(status: int, message: str, kind: str = "error",
             **extra) -> JSONResponse:
        body = {"error": {"type": kind, "message": message}}
        if extra:
            body["error"].update(extra)
        return JSONResponse(status_code=status,
                            content=jsonable_encoder(body))

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(_request, exc: StarletteHTTPException):
        return _err(exc.status_code, str(exc.detail), kind="http_error")

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(_request, exc: RequestValidationError):
        return _err(422, "request failed validation", kind="validation_error",
                    detail=exc.errors())

    @app.exception_handler(Exception)
    async def _unhandled(request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method,
                      request.url.path)
        return _err(500, "internal server error", kind="internal_error")

    # --- ops ---

    @app.get("/api/health")
    def api_health(request: Request):
        engine: Engine = engine_of(request)
        db_ok = engine.db_ok()
        status = 200 if db_ok else 503
        return JSONResponse(status_code=status, content={
            "status": "ok" if db_ok else "degraded",
            "database_ok": db_ok,
            "version": app.version,
        })

    @app.get("/api/info")
    def api_info(request: Request, _p: Principal = require("read")):
        return engine_of(request).info()

    if settings.metrics_enabled:
        @app.get("/metrics")
        def api_metrics():
            payload, content_type = metrics.render()
            return Response(content=payload, media_type=content_type)

    # --- ingestion ---

    @app.post("/api/spans")
    def api_ingest(request: Request, spans: list[Span] = Body(...),
                   p: Principal = require("ingest")):
        if len(spans) > settings.max_batch_size:
            raise HTTPException(
                413, f"batch exceeds max_batch_size={settings.max_batch_size}")
        engine = engine_of(request)
        ingested, skipped, failed = [], [], []
        for s in spans:
            try:
                if engine.store.has_node(s.span_id):
                    skipped.append(s.span_id)
                    continue
                node = engine.ingest(s)
                ingested.append(node.node_id)
            except Exception:
                log.exception("failed to ingest span %s", s.span_id)
                metrics.INGEST_ERRORS.labels(reason="exception").inc()
                failed.append(s.span_id)
        return {"ingested": ingested, "skipped_duplicates": skipped,
                "failed": failed}

    @app.post("/v1/traces")
    async def api_otlp(request: Request, p: Principal = require("ingest")):
        body = await request.body()
        content_type = request.headers.get("content-type", "")
        try:
            spans = otel_ingest.parse_otlp(body, content_type)
        except otel_ingest.OTLPParseError as exc:
            metrics.INGEST_ERRORS.labels(reason="otlp_parse").inc()
            raise HTTPException(400, str(exc))
        if len(spans) > settings.max_batch_size:
            raise HTTPException(
                413, f"batch exceeds max_batch_size={settings.max_batch_size}")
        engine = engine_of(request)
        for s in spans:
            try:
                if not engine.store.has_node(s.span_id):
                    engine.ingest(s)
            except Exception:
                log.exception("failed to ingest OTLP span %s", s.span_id)
                metrics.INGEST_ERRORS.labels(reason="exception").inc()
        payload, out_type = otel_ingest.empty_export_response(content_type)
        return Response(content=payload, media_type=out_type)

    # --- twin queries ---

    @app.get("/api/traces")
    def api_traces(request: Request,
                   limit: int = Query(50, ge=1, le=500),
                   offset: int = Query(0, ge=0),
                   flagged_only: bool = False,
                   _p: Principal = require("read")):
        traces, total = engine_of(request).store.list_traces(
            limit=limit, offset=offset, flagged_only=flagged_only)
        return {"traces": traces, "total": total,
                "limit": limit, "offset": offset}

    @app.get("/api/graph")
    def api_graph(request: Request, trace: str = Query(...),
                  _p: Principal = require("read")):
        return engine_of(request).graph_state(trace)

    @app.get("/api/nodes")
    def api_nodes(request: Request,
                  trace: Optional[str] = None,
                  status: Optional[str] = None,
                  agent: Optional[str] = None,
                  blocked: Optional[bool] = None,
                  since: Optional[float] = None,
                  limit: int = Query(100, ge=1, le=1000),
                  offset: int = Query(0, ge=0),
                  _p: Principal = require("read")):
        nodes, total = engine_of(request).store.list_nodes(
            trace_id=trace, status=status, agent_id=agent, blocked=blocked,
            since=since, limit=limit, offset=offset)
        return {"nodes": [n.model_dump(mode="json") for n in nodes],
                "total": total, "limit": limit, "offset": offset}

    @app.get("/api/node/{node_id}")
    def api_node(request: Request, node_id: str,
                 _p: Principal = require("read")):
        node = engine_of(request).store.get_node(node_id)
        if node is None:
            raise HTTPException(404, f"node {node_id} not found")
        return node.model_dump(mode="json")

    @app.get("/api/incidents")
    def api_incidents(request: Request,
                      limit: int = Query(50, ge=1, le=200),
                      offset: int = Query(0, ge=0),
                      _p: Principal = require("read")):
        incidents, total = engine_of(request).incidents(limit=limit,
                                                        offset=offset)
        return {"incidents": incidents, "total": total,
                "limit": limit, "offset": offset}

    @app.get("/api/narrative")
    def api_narrative(request: Request, target: Optional[str] = None,
                      _p: Principal = require("read")):
        engine = engine_of(request)
        if target is not None and engine.store.get_node(target) is None:
            raise HTTPException(404, f"node {target} not found")
        nar = engine.narrative(target)
        if nar is None:
            raise HTTPException(404, "no drifted node to attribute")
        return nar.model_dump(mode="json")

    @app.get("/api/whatif/{root_id}")
    def api_whatif(request: Request, root_id: str,
                   _p: Principal = require("read")):
        wi = engine_of(request).whatif(root_id)
        if wi is None:
            raise HTTPException(404, f"node {root_id} not found")
        return wi.model_dump(mode="json")

    @app.post("/api/nodes/{node_id}/label")
    def api_label(request: Request, node_id: str, payload: dict = Body(...),
                  p: Principal = require("operate")):
        label = str(payload.get("label", "")).strip()
        note = str(payload.get("note", "") or "")
        try:
            return engine_of(request).label_node(node_id, label, p.name,
                                                 note=note)
        except KeyError:
            raise HTTPException(404, f"node {node_id} not found")
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/api/calibration")
    def api_calibration(request: Request,
                        target_precision: Optional[float] = Query(
                            None, ge=0.0, le=1.0),
                        _p: Principal = require("read")):
        return engine_of(request).calibration_report(
            target_precision=target_precision)

    @app.get("/api/guard")
    def api_guard(request: Request,
                  limit: int = Query(100, ge=1, le=1000),
                  offset: int = Query(0, ge=0),
                  _p: Principal = require("read")):
        return engine_of(request).guard_report(limit=limit, offset=offset)

    @app.get("/api/cost")
    def api_cost(request: Request, _p: Principal = require("read")):
        return engine_of(request).cost()

    @app.get("/api/escalation")
    def api_escalation(request: Request, _p: Principal = require("read")):
        return engine_of(request).escalation_report()

    # --- remediation ---

    @app.get("/api/remediation")
    def api_remediation(request: Request,
                        status: Optional[str] = None,
                        node: Optional[str] = None,
                        limit: int = Query(100, ge=1, le=1000),
                        offset: int = Query(0, ge=0),
                        _p: Principal = require("read")):
        actions, total = engine_of(request).remediation.list_actions(
            status=status, node_id=node, limit=limit, offset=offset)
        return {"actions": [a.model_dump(mode="json") for a in actions],
                "total": total, "limit": limit, "offset": offset}

    @app.post("/api/remediation/propose")
    def api_propose(request: Request,
                    payload: dict = Body(...),
                    p: Principal = require("operate")):
        engine = engine_of(request)
        node_id = str(payload.get("node_id", ""))
        node = engine.store.get_node(node_id)
        if node is None:
            raise HTTPException(404, f"node {node_id} not found")
        from twin import attribution
        actions = attribution.propose_remediation(engine.store, node)
        for a in actions:
            a.proposed_by = p.name
        registered = engine.remediation.register(actions)
        return {"proposed": [a.model_dump(mode="json") for a in registered]}

    def _lifecycle(request: Request, action_id: str, p: Principal,
                   op: str):
        engine = engine_of(request)
        try:
            fn = getattr(engine.remediation, op)
            a = fn(action_id, p.name)
        except KeyError:
            raise HTTPException(404, "action not found")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        metrics.REMEDIATION_EVENTS.labels(event=op).inc()
        return a.model_dump(mode="json")

    @app.post("/api/remediation/{action_id}/approve")
    def api_approve(request: Request, action_id: str,
                    p: Principal = require("operate")):
        return _lifecycle(request, action_id, p, "approve")

    @app.post("/api/remediation/{action_id}/reject")
    def api_reject(request: Request, action_id: str,
                   p: Principal = require("operate")):
        return _lifecycle(request, action_id, p, "reject")

    @app.post("/api/remediation/{action_id}/revert")
    def api_revert(request: Request, action_id: str,
                   p: Principal = require("operate")):
        return _lifecycle(request, action_id, p, "revert")

    # --- audit & compliance ---

    @app.get("/api/audit")
    def api_audit(request: Request,
                  action: Optional[str] = None,
                  target: Optional[str] = None,
                  limit: int = Query(100, ge=1, le=1000),
                  offset: int = Query(0, ge=0),
                  _p: Principal = require("read")):
        entries, total = engine_of(request).audit.entries(
            limit=limit, offset=offset, action=action, target=target)
        return {"entries": [e.model_dump(mode="json") for e in entries],
                "total": total, "limit": limit, "offset": offset}

    @app.get("/api/audit/verify")
    def api_audit_verify(request: Request, _p: Principal = require("read")):
        return engine_of(request).audit.verify_chain()

    @app.get("/api/compliance")
    def api_compliance(request: Request, _p: Principal = require("read")):
        return engine_of(request).compliance()

    @app.get("/api/compliance/map")
    def api_compliance_map(request: Request, _p: Principal = require("read")):
        return engine_of(request).audit.active_map()

    # --- admin ---

    @app.get("/api/admin/keys")
    def api_keys_list(request: Request, _p: Principal = require("admin")):
        return {"keys": engine_of(request).keys.list_keys()}

    @app.post("/api/admin/keys")
    def api_keys_create(request: Request, payload: dict = Body(...),
                        p: Principal = require("admin")):
        name = str(payload.get("name", "")).strip()
        role = str(payload.get("role", "")).strip()
        if not name:
            raise HTTPException(422, "name is required")
        if role not in ROLES:
            raise HTTPException(422, f"role must be one of {list(ROLES)}")
        engine = engine_of(request)
        row, full_key = engine.keys.create(name=name, role=role)
        engine.audit.record(p.name, "auth.key_created", row.key_id,
                            detail=f"role={role} name={name}")
        return {"key_id": row.key_id, "name": name, "role": role,
                "api_key": full_key,
                "note": "store this key now; it is not retrievable later"}

    @app.post("/api/admin/keys/{key_id}/disable")
    def api_keys_disable(request: Request, key_id: str,
                         p: Principal = require("admin")):
        engine = engine_of(request)
        if not engine.keys.set_disabled(key_id, True):
            raise HTTPException(404, "key not found")
        engine.audit.record(p.name, "auth.key_disabled", key_id)
        return {"key_id": key_id, "disabled": True}

    @app.post("/api/admin/retention")
    def api_retention(request: Request, payload: dict = Body(...),
                      p: Principal = require("admin")):
        try:
            days = int(payload.get("days"))
        except (TypeError, ValueError):
            raise HTTPException(422, "days must be an integer")
        if days < 1:
            raise HTTPException(422, "days must be >= 1")
        return engine_of(request).run_retention(days, actor=p.name)

    # --- UI ---

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB, "index.html"))

    if os.path.isdir(WEB):
        app.mount("/web", StaticFiles(directory=WEB), name="web")

    return app


app = create_app()
