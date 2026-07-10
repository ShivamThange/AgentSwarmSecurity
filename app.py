from __future__ import annotations

import os

from fastapi import Body, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from twin import Engine
from twin.models import ActionStatus, Span
from twin.scenario import INCIDENT_TRACE

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
DB_PATH = os.environ.get("TWIN_DB", os.path.join(HERE, "twin.db"))

app = FastAPI(title="Agent Drift Containment & Remediation Twin", version="0.2.0")
engine = Engine(db_path=DB_PATH)
engine.load_or_seed()

def _err(status: int, message: str, kind: str = "error", **extra) -> JSONResponse:
    body = {"error": {"type": kind, "message": message}}
    if extra:
        body["error"].update(extra)
    return JSONResponse(status_code=status, content=jsonable_encoder(body))

@app.exception_handler(StarletteHTTPException)
def _http_exc(_request, exc: StarletteHTTPException):
    return _err(exc.status_code, str(exc.detail), kind="http_error")

@app.exception_handler(RequestValidationError)
def _validation_exc(_request, exc: RequestValidationError):
    return _err(422, "request failed validation", kind="validation_error",
                detail=exc.errors())

@app.exception_handler(Exception)
def _unhandled(_request, exc: Exception):
    return _err(500, str(exc), kind="internal_error")

@app.get("/api/health")
def api_health():
    return {"status": "ok", **engine.persistence_info()}

@app.get("/api/persistence")
def api_persistence():
    return engine.persistence_info()

@app.post("/api/spans")
def api_ingest(spans: list[Span] = Body(...)):
    ingested, skipped = [], []
    for s in spans:
        if engine.store.has_node(s.span_id):
            skipped.append(s.span_id)
            continue
        node = engine.ingest(s)
        ingested.append(node.node_id)
    engine.store.set_meta("cost_ledger", engine.router.ledger.as_dict())
    return {"ingested": ingested, "skipped_duplicates": skipped,
            "cost": engine.cost()}

@app.post("/api/seed")
def api_seed(background: int = 200):
    if background < 0:
        raise HTTPException(422, "background must be >= 0")
    engine.seed(background=background)
    return {
        "ok": True,
        "incident": engine.incident_narrative.model_dump(mode="json")
        if engine.incident_narrative else None,
        "cost": engine.cost(),
    }

@app.get("/api/graph")
def api_graph(trace: str | None = INCIDENT_TRACE):
    return engine.graph_state(trace)

@app.get("/api/node/{node_id}")
def api_node(node_id: str):
    node = engine.store.get_node(node_id)
    if node is None:
        raise HTTPException(404, f"node {node_id} not found")
    return node.model_dump(mode="json")

@app.get("/api/narrative")
def api_narrative(target: str | None = None):
    if target is not None and engine.store.get_node(target) is None:
        raise HTTPException(404, f"node {target} not found")
    nar = engine.narrative(target) if target else engine.incident_narrative
    if nar is None:
        raise HTTPException(404, "no incident to attribute")
    return nar.model_dump(mode="json")

@app.get("/api/whatif/{root_id}")
def api_whatif(root_id: str):
    if engine.store.get_node(root_id) is None:
        raise HTTPException(404, f"node {root_id} not found")
    wi = engine.whatif(root_id)
    if wi is None:
        raise HTTPException(404, f"node {root_id} not found")
    return wi.model_dump(mode="json")

@app.get("/api/guard")
def api_guard():
    return engine.guard_report()

@app.get("/api/remediation")
def api_remediation():
    return {"actions": [a.model_dump(mode="json")
                        for a in engine.remediation.all_actions()]}

@app.post("/api/remediation/{action_id}/approve")
def api_approve(action_id: str, approver: str = "supervisor"):
    a = engine.remediation.get(action_id)
    if a is None:
        raise HTTPException(404, "action not found")
    if a.status != ActionStatus.PROPOSED:
        raise HTTPException(409, f"cannot approve an action in state '{a.status.value}'")
    return engine.remediation.approve(action_id, approver).model_dump(mode="json")

@app.post("/api/remediation/{action_id}/reject")
def api_reject(action_id: str, approver: str = "supervisor"):
    a = engine.remediation.get(action_id)
    if a is None:
        raise HTTPException(404, "action not found")
    if a.status != ActionStatus.PROPOSED:
        raise HTTPException(409, f"cannot reject an action in state '{a.status.value}'")
    return engine.remediation.reject(action_id, approver).model_dump(mode="json")

@app.post("/api/remediation/{action_id}/revert")
def api_revert(action_id: str, approver: str = "supervisor"):
    a = engine.remediation.get(action_id)
    if a is None:
        raise HTTPException(404, "action not found")
    if a.status != ActionStatus.APPLIED:
        raise HTTPException(409, f"cannot revert an action in state '{a.status.value}' "
                                 f"(only APPLIED actions are reversible)")
    return engine.remediation.revert(action_id, approver).model_dump(mode="json")

@app.get("/api/cost")
def api_cost():
    return engine.cost()

@app.get("/api/audit")
def api_audit():
    return {
        "entries": [e.model_dump(mode="json") for e in engine.audit.entries()],
        "chain_valid": engine.audit.verify_chain(),
    }

@app.get("/api/compliance")
def api_compliance():
    return engine.compliance()

@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))

if os.path.isdir(WEB):
    app.mount("/web", StaticFiles(directory=WEB), name="web")
