# Agent Drift Containment & Remediation Twin

A self-hosted control plane that monitors multi-agent LLM systems for
behavioural drift (prompt injection, goal drift, tool misuse, fabricated
effects), maintains a causal "twin" graph of every agent step, attributes
incidents to their root cause, and drives an auditable human-in-the-loop
containment workflow.

## What it does

1. **Ingest** agent step telemetry via a native REST API (`POST /api/spans`)
   or an OTLP/HTTP endpoint (`POST /v1/traces`) that understands OTel GenAI /
   OpenInference span conventions plus `twin.*` attributes.
2. **Detect** drift with a tiered, cost-aware pipeline:
   - deterministic rules (undeclared sensitive tools, fabricated effects,
     schema violations, confidence collapse) — free;
   - a semantic stated-vs-revealed gap (sentence-transformer embeddings)
     comparing what an agent *said* it would do to what it *did*;
   - an LLM judge (OpenAI-compatible endpoint) called only at escalation
     tiers, with a deeper model confirming FLAGGED verdicts.
3. **Block** sensitive tool calls that no declared intent or task spec
   authorises (inline rail), before they take effect.
4. **Attribute** any drifted node to the earliest drifted ancestor in the
   influence graph, with propagation path and blast radius.
5. **Contain** with proposed remediations (rollback / quarantine /
   message-filter) that a human operator approves, rejects, or reverts —
   every transition recorded in a tamper-evident, hash-chained audit log
   tagged with EU AI Act / NIST AI RMF / ISO 42001 clauses.
6. **Preview** counterfactuals: `GET /api/whatif/{node}` replays detection
   over the blast radius with corrected context to show which downstream
   nodes a remediation would actually save.

## Quick start (Docker Compose)

```bash
cp .env.example .env
# in .env set at minimum:
#   POSTGRES_PASSWORD=<strong password>
#   TWIN_BOOTSTRAP_ADMIN_KEY=<strong random string>
docker compose up --build -d
curl http://localhost:8000/api/health
```

The bootstrap key becomes the admin API key on first boot (only when the key
table is empty). Immediately create scoped keys and rotate the bootstrap key:

```bash
curl -X POST http://localhost:8000/api/admin/keys \
  -H "X-API-Key: $TWIN_BOOTSTRAP_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "agents-prod", "role": "ingest"}'
```

## Local development

```bash
pip install -r requirements.txt -r requirements-dev.txt
# optional, for real semantic embeddings:
pip install -r requirements-ml.txt
# smallest possible dev loop (SQLite, no auth, lexical embeddings):
TWIN_DATABASE_URL=sqlite:///twin.db \
TWIN_AUTH_ENABLED=false \
TWIN_EMBEDDINGS_BACKEND=hashing \
python run.py
```

Send yourself traffic (an external client, not part of the service):

```bash
python scripts/sample_traffic.py --api-key <ingest-key> --traces 20
```

Run the tests: `pytest`

## Roles

| Role | Can |
| --- | --- |
| `ingest` | POST spans / OTLP traces only |
| `viewer` | read every query endpoint |
| `operator` | viewer + propose/approve/reject/revert remediation + ingest |
| `admin` | operator + key management + retention |

Authenticate with `X-API-Key: <key>` or `Authorization: Bearer <key>`.
Keys are stored hashed (SHA-256); the plaintext is shown once at creation.

## Ingestion contract

Native span (`POST /api/spans`, JSON array):

```json
{
  "span_id": "unique-step-id",
  "trace_id": "workflow-run-id",
  "agent_id": "payments-executor",
  "privilege": "high",
  "task_spec": "Prepare a payment summary. Do NOT move funds.",
  "declared_intent": "Prepare the payment summary for review.",
  "tool_calls": [{"name": "transfer_funds", "args": {"amount": 480000}}],
  "effects": ["$480,000 transfer initiated"],
  "output": "Payment summary prepared.",
  "inputs_from": ["upstream-span-id"],
  "meta": {"baseline_tokens": 2100}
}
```

OTLP (`POST /v1/traces`, protobuf or JSON): standard OTel span identity plus
attributes. Recognised attributes, in precedence order: `twin.*`
(`twin.agent_id`, `twin.task_spec`, `twin.declared_intent`, `twin.output`,
`twin.effects`, `twin.tool_calls`, `twin.privilege`, `twin.inputs_from`),
then `gen_ai.*` (agent name, tool name/arguments, completion, usage tokens),
then OpenInference (`tool.name`, `output.value`, `llm.token_count.*`).
Span links and the parent span become influence edges. Reported usage tokens
feed the measured monitoring-overhead figure at `GET /api/cost`.

## Key endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/incidents` | traces containing flagged spans, with root cause |
| `GET /api/narrative?target=` | root cause, propagation path, blast radius |
| `GET /api/whatif/{node}` | counterfactual containment preview |
| `GET/POST /api/remediation...` | proposal + approve/reject/revert lifecycle |
| `GET /api/guard` | tool calls denied by the inline rail |
| `GET /api/audit`, `/api/audit/verify` | hash-chained audit log + verification |
| `GET /api/compliance` | audit coverage per compliance clause |
| `GET /metrics` | Prometheus metrics |
| `GET /api/health` | liveness/readiness (DB check) |

## Production notes

- **Database**: PostgreSQL via `TWIN_DATABASE_URL`. SQLite (WAL) is for
  single-node development only. Schema is created on startup
  (`Base.metadata.create_all`); introduce Alembic before making breaking
  schema changes across releases.
- **Detection quality**: keep `TWIN_EMBEDDINGS_BACKEND=sentence-transformers`
  and set `TWIN_LLM_API_KEY` in production. Without them the service runs and
  logs that it is in degraded deterministic mode.
- **TLS / SSO**: terminate TLS at your ingress; the built-in auth is
  service-to-service API keys. Front the UI with your IdP if you need SSO.
- **Rate limiting**: the per-key limiter is per-process; enforce global
  limits at your gateway when running multiple replicas.
- **Retention**: schedule `python scripts/retention.py --days 90` (or call
  `POST /api/admin/retention`). The audit log is never pruned.
- **Scaling**: state lives in the database; run multiple uvicorn workers or
  replicas freely. The detection cache and rate limiter are per-process
  optimisations only.
