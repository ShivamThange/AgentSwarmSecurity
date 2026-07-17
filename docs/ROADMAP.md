# Roadmap — Agent Drift Twin

Lean successor to the deleted `builder_context_brief`. Holds decisions + backlog only.
Not auto-loaded into the agent context by design — read it deliberately, don't paste it wholesale.

## Product in one line

A persistent **digital twin of an org's agent graph** that detects cross-agent drift/error
propagation, attributes any failure back to its root cause across the whole chain, and lets a
human supervisor remediate surgically (edit context, roll back, redeploy) — all audited and
reversible, at low-single-digit-% compute overhead, on-prem.

## Non-negotiable constraints (resolve every fork against these)

1. **No new paid token stream.** Consume reasoning agents *already emit* (CoT, tool calls, outputs). Never add a side-channel that costs tokens.
2. **Mostly zero-inference detection.** Default path is pure compute (embeddings, diffing, validators, logprob parsing). LLM is consulted only *after* a cheap signal flags something.
3. **On-prem / no data exfiltration.** Self-hosted is the default. No shipping agent internals to third-party SaaS.
4. **Detect behavioral disagreement, not malice.** Compare what an agent *says* vs. what it *does*. Never trust CoT alone; never put the monitor in the agents' training loop.
5. **Remediate on context, not internals.** No activation steering as baseline. Every remediation human-approved + logged + reversible.

## The defensible core (this is what we build)

The moat is **durable cross-agent, cross-time causal state**. Five capabilities no reused tool provides (C1–C5):

- C1 durable cross-run lineage graph · C2 backward root-cause across agents · C3 blast radius · C4 human-approved reversible context-remediation · C5 what-if replay over held state.

## Already built in `bare-code` — DO NOT rebuild

| Capability | Where |
|---|---|
| Durable store (SQLAlchemy, Postgres/SQLite) | `twin/db.py`, `twin/store.py` |
| Blast radius / upstream / propagation path | `twin/store.py`, `twin/attribution.py` |
| Backward root-cause + incident narrative | `twin/attribution.py` |
| What-if counterfactual replay (§14.1) | `twin/replay.py`, `Engine.whatif()` |
| Checkpoints + rollback + remediation loop | `twin/remediation.py`, `twin/store.py` |
| Tamper-evident audit trail + compliance report | `twin/audit.py` |
| Tiered cost router (zero-inference→sampled→small-judge→deep) | `twin/router.py` |
| Real CPU embeddings (sentence-transformers + hashing fallback) | `twin/embeddings.py` |
| Real LLM judges (OpenAI-compatible small+deep, det. fallback) | `twin/llm.py` |
| OTLP/OpenTelemetry ingest (protobuf + JSON) | `twin/otel_ingest.py` |
| Inline deterministic hard-stop rail | `twin/guard.py` |
| MAS risk taxonomy | `twin/taxonomy.py` |
| API keys, rate limiting, metrics, Docker, tests | `twin/security.py`, `twin/metrics.py`, `Dockerfile`, `tests/` |

## The gap — "reuse existing repos" (§9) — NOW WIRED

Each reuse target drops in behind an existing seam, additive and config-gated.
Every adapter degrades to the native component when its package is absent
(verify with `python scripts/validate_boundary.py`).

| Reuse target | Seam | Adapter | Enable with |
|---|---|---|---|
| **LlamaFirewall** (AlignmentCheck + PromptGuard) | `detection.Judge` / `llm.build_judges` | `integrations/llamafirewall_judge.py` | `TWIN_JUDGE_BACKEND=llamafirewall` |
| **NeMo Guardrails** (Colang) | `guard.GuardBackend` | `integrations/nemo_guard.py` | `TWIN_GUARD_BACKEND=nemo` + `TWIN_NEMO_CONFIG_PATH` |
| **Arize-Phoenix (OpenInference) / Langfuse** | `otel_ingest` conventions | broadened attribute mapping in `otel_ingest.py` | point their OTLP exporter at `/v1/traces` |
| **AG2/AutoGen / LangGraph** | new ingest adapter | `integrations/handoff.py` (+ `TwinTap`) | import the converter in the agent process |
| **TrinityGuard** | taxonomy + remediation | native `taxonomy.py` (kept; TrinityGuard not on PyPI) | — |

## Backlog (value order) — status

**T1 — close the reuse gap:** DONE
1. ✅ `LlamaFirewallJudge` behind the `Judge` seam (`integrations/llamafirewall_judge.py`).
2. ✅ NeMo Guardrails as pluggable `guard` backend, native rail as authoritative fallback (`integrations/nemo_guard.py`).
3. ✅ Phoenix (OpenInference) + Langfuse attribute conventions in `otel_ingest.py`.
4. ✅ AutoGen/LangGraph handoff tap + `TwinTap` client (`integrations/handoff.py`).

**T2 — finish specified roadmap items:** DONE
5. ✅ Executable boundary validation (`scripts/validate_boundary.py`) — asserts C1–C5 twin-native, no reused tool provides them.
6. ✅ Per-workflow threshold profiles (`detection.PolicyResolver`) + human-review label loop (`calibration.py`, `POST /api/nodes/{id}/label`, `GET /api/calibration`).
7. ✅ Escalation-rate anomaly monitor (`escalation.py`, `GET /api/escalation`, Prometheus metrics).

**T3 — near-free differentiators:**
8. ✅ Config-loadable, swappable compliance map (`audit.load_compliance_map`, `TWIN_COMPLIANCE_MAP_PATH`, `GET /api/compliance/map`).
9. ✅ What-if side-by-side already in the dashboard (`web/index.html` `previewWhatif`, do-nothing vs remediated blast radius). Optional next: surface escalation + calibration panels in the UI.

All items are additive, config-gated, and covered by tests; the service runs unchanged with zero integration packages installed.

## Open questions (close these as you build)

Substrate choice (Phoenix vs Langfuse) · concrete drift thresholds per workflow type ·
deterministic agent-inaccessible escalation trigger · which remediations may auto-run vs. always-human · exact audit schema for compliance mapping.

## Where the demo lives now

Beachhead injection scenario → `tests/fixtures.py` (vendor-payment) + `scripts/sample_traffic.py`
(live traffic generator against the HTTP API). No hardcoded `seed()` — ingest real/sample traffic.
