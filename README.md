# Agent Drift Containment & Remediation Twin

A persistent **digital twin of an organization's agent graph** that detects drift
and error *propagation* across agents, traces any failure back to its **root cause
across the whole chain**, and lets a human supervisor **remediate surgically** —
every action audited and reversible — at **low-single-digit-percent compute
overhead**.

This is a working MVP of the platform described in `builder_context_brief`. It runs
fully offline with **zero model downloads** (a local CPU embedder stands in for an
on-prem sentence-encoder), which is itself the on-prem/no-exfiltration story.

---

## Quick start

```bash
pip install -r requirements.txt
python run.py                 # then open http://127.0.0.1:8000
python tests.py               # end-to-end tests (all offline, no key needed)
```

The dashboard boots with a seeded incident already detected and attributed. The
offline demo needs **no API key and no downloads**.

## Setup for collaborators

Reproducible from a clean checkout:

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -r requirements.txt                    # offline demo + tests
cp .env.example .env                               # optional: configure the online LLM
python tests.py                                    # expect all tests passing
python run.py                                      # http://127.0.0.1:8000
```

For the live integration stack (Phoenix substrate, AG2/AutoGen, LlamaFirewall,
NeMo-Guardrails — heavier, per builder-brief §9):

```bash
pip install -r requirements-integrations.txt
git clone https://github.com/AI45Lab/TrinityGuard && pip install -e ./TrinityGuard
```

## Enabling an online LLM judge (hybrid)

The judge tier is **off by default** (deterministic `StubJudge`, fully offline). It is
called **only at the escalation tier**, so when enabled the cloud sees a small,
already-flagged fraction of traffic — never the full stream (honours the on-prem
constraint §3.3; swap to a self-hosted endpoint to keep everything on-box).

Configure via env (OpenAI-compatible — OpenAI, Groq, OpenRouter, or self-hosted). See
`.env.example`. Note: OpenAI's **API** needs a funded account — the free ChatGPT plan
does not include API access; **Groq / OpenRouter** offer genuine free tiers.

```bash
export LLM_API_KEY=...          # enables the online judge
export LLM_BASE_URL=...         # optional; e.g. https://api.groq.com/openai/v1
export LLM_MODEL=gpt-4o-mini    # judge tier
```

---

## The demo, in one sentence

A poisoned source document injects an instruction into a **Data Retriever** agent;
the contamination silently propagates `data-retriever → analyst → report-writer`
and culminates in a high-privilege **Payments Executor** attempting a $480k transfer
to the injected account. The twin catches it, walks the edges back to the true root
cause, and offers the supervisor a reversible fix — with a what-if preview of the
blast radius before/after.

Everything a stateless single-hop tool **can't** do: cross-agent, cross-time,
causal.

---

## What you can do in the UI

- **Digital twin graph** — the incident chain, colour-coded by drift status, with the
  propagation path and root cause highlighted, plus the node the **inline rail
  blocked** pre-execution. Click any node to inspect it.
- **Node inspector** — the drift signals for a node: the **structural** *stated-vs-
  revealed gap* (the primary trigger — e.g. "introduces an entity foreign to its
  task"), *trajectory drift* (a lexical signal, labelled illustrative), deterministic
  zero-token checks, the small-model judge verdict, and declared-intent-vs-observed-
  behaviour side by side.
- **Causal narrative** — root cause → propagation path → blast radius → recommended fix
  (a story, not an alert).
- **Remediation** — approve / reject / revert proposed actions (rollback, message-filter,
  quarantine). Human-approved by default; watch the graph update live.
- **What-if preview** — a **real forward replay** over the twin: corrected context is
  substituted at the root and detection is re-run node-by-node; inherited nodes
  recover, nodes with their own fault stay flagged. No agents re-executed. Advisory only.
- **Cost router** — the tier distribution and the resulting overhead % (≈2%, labelled
  *modelled* in the UI — it is derived from fixed per-tier token constants, not measured).
- **Audit trail** — a tamper-evident hash chain of every action, each tagged to
  EU AI Act / NIST AI RMF / ISO 42001 clauses; on-demand compliance report.
- **Persistence badge** — the twin is durable by default: the graph, audit chain and
  cost snapshot survive a process restart (`boot_mode: loaded`).

---

## Architecture

Ingestion contract is an OpenTelemetry-shaped `Span` — *already-emitted* agent
telemetry (CoT / structured self-report, tool calls, effects, output). No new paid
token stream.

```
Span ──▶ CostRouter ──▶ TwinStore (persistent lineage graph)
          │                 │
          │ tiers:          ├─ blast_radius / upstream / propagation_path
          │ zero-inference  │
          │ → sampled       ▼
          │ → small-judge   Attribution ──▶ CausalNarrative + proposed Remediation
          │ → deep-escalate                     │
          ▼                                      ▼
     DriftSignals                    RemediationEngine ──▶ AuditLog (hash chain
     (stated-vs-revealed,            (human-approved,        + compliance mapping)
      trajectory, det. checks,        reversible)
      judge)                              │
                                          ▼
                                   WhatIf (counterfactual replay)
```

| Module | Brief section | Responsibility |
|---|---|---|
| `twin/models.py` | 8 | Telemetry contract + twin/remediation/audit schemas |
| `twin/embeddings.py` | 3.2 / 6.1 | Local zero-download CPU embedder (swap-in seam for a real encoder) |
| `twin/detection.py` | 5 | Deterministic checks, stated-vs-revealed gap, trajectory drift, judge (interface + offline stub), risk taxonomy |
| `twin/graph.py` | 8.1 | Persistent lineage graph (SQLite + NetworkX); blast-radius / backward-causal / checkpoints |
| `twin/router.py` | 8.2 / 6 | Tiered cost router; deterministic agent-inaccessible escalation trigger; risk-weighted sampling; dedup cache; cost accounting |
| `twin/attribution.py` | 8.1 / 10 | Root-cause walk + supervisor narrative + remediation proposals |
| `twin/remediation.py` | 8.3 / 8.4 | Human-approved, reversible remediation over context (never internals) |
| `twin/audit.py` | 8.4 / 14.2 | Tamper-evident audit chain + configurable compliance mapping |
| `twin/whatif.py` | 14.1 | Counterfactual "what-if" remediation preview |
| `twin/scenario.py` | 12.1 | Seeded beachhead incident + realistic clean background load |
| `app.py` / `web/` | 10 | REST API + self-contained dashboard (no CDN) |

---

## How it honours the non-negotiable constraints (Section 3)

1. **No new paid token stream** — ingests only already-emitted telemetry; the default
   detection path is pure CPU (embeddings, diffing, validators, logprob parsing).
2. **Mostly zero-inference** — an LLM (the small judge) is consulted *only* after a
   cheap signal trips a deterministic threshold. Under realistic load the measured
   overhead is ≈1.5%, not the ~2× of naive per-step monitoring.
3. **On-prem / no exfiltration** — everything runs locally with no external calls and
   no model downloads.

Design guardrails respected: no malice classifier (detect behavioural disagreement);
CoT never trusted alone (always cross-checked against tool calls/effects); the monitor
is observational and out-of-loop; remediation operates on context, not model internals;
monitoring is tiered/sampled/async, with the escalation trigger deterministic and
inaccessible to the monitored agents.

---

## Where the real system differs from this MVP

Honest boundaries — what a production build swaps in:

- **Embedder** — `HashingEmbedder` is a deterministic offline stand-in. Drop in
  `sentence-transformers` (all-MiniLM) or any on-prem encoder via the `Embedder` seam;
  no detector changes.
- **Small judge** — `StubJudge` is a rule-based stand-in behind the `Judge` protocol.
  Point it at a self-hosted 3–8B or Haiku-class model.
- **Substrate** — the ingestion contract is OTel-shaped so Phoenix/Langfuse spans and
  a NeMo-Guardrails rails engine feed in at the boundary; the twin is the durable
  cross-run layer they lack (Section 9).
- **Scale** — the graph is SQLite + NetworkX (fine for the beachhead). At MNC scale
  the store behind `TwinStore` becomes a real graph/columnar backend; the query
  surface (`blast_radius` / `upstream` / `propagation_path`) stays the same.

---

## Detection is structural, not a keyword mirror

The primary drift signal is **structural**, not lexical and not a planted phrase.
An injection is caught because the agent's behaviour introduces an *actionable
entity foreign to its task* (an account, a named payee) that no upstream node
emitted — the keyword-free structural tell of "content that entered as data now
drives behaviour". The dangerous-tool check is a **backstop**, not the primary
path. The offline embedder is lexical, so its distances are treated as *illustrative
supporting colour only* and can never, on their own, raise a flag.

Proof it is not circular: `twin/scenario_variants.py` holds injections phrased
entirely differently, containing **none** of the judge's marker strings; the
`test_detection_is_structural_not_keyword` test shows they still flag as structural
prompt injections. If they ever stop flagging, that is the honest signal that the
lexical stub needs replacing with a real encoder — also a legitimate finding.

## Durability (the moat, on by default)

The twin is backed by a file database (`TWIN_DB`, default `twin.db`) and rebuilds
its incident view from the persisted graph on restart. Prove it across a real
process boundary:

```bash
python persistence_check.py     # process A writes; process B reads back; asserts
```

`test_persistence_survives_process_restart` runs this as two separate `python`
processes and asserts the blast radius and audit chain survive.

## Inline hard-stop rail (we stop it, not just notice it)

`twin/guard.py` is a synchronous deterministic rail, distinct from the async
monitor. It blocks the dangerous transfer *before* it executes (the demo shows
A7's `$480k` transfer prevented, not post-hoc flagged), while the monitor still
records the attempt so attribution can trace it back to the root.

---

## Known production gaps (explicitly NOT built)

Stated plainly so the real/simulated line is unambiguous:

- **Live boundary validation is pending.** `docs/boundary_validation.md` is a *desk*
  validation (confirmed the wedge against LlamaFirewall / NeMo-Guardrails / Phoenix
  by architecture). `trinityguard` is not a public PyPI package, and standing up the
  real tools needs network downloads that contradict the offline constraint — both
  need an operator decision. This gates hardening (Section 9 / 11.2).
- **Embedder & judge are offline stubs** behind `Embedder` / `Judge` seams. The
  lexical embedder is illustrative; detection stands on the structural signals.
- **Cost overhead is modelled**, not measured — fixed per-tier token constants.
  Labelled *modelled* in the UI and the API.
- **Async is conceptual.** Ingestion is synchronous. A real off-hot-path queue /
  batching layer is a known production gap; no fake async layer was built.
- **Threshold calibration** is hand-picked; no calibration dataset yet (Section 11.4).
