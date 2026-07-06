# Builder Context Brief — Multi-Agent Drift Containment & Remediation Platform

**Audience:** the engineering agent/team that will architect and build this system.
**Purpose:** give you every decision and the *reasoning behind it* so you can reach
architectural choices accurately, reuse what exists, and not re-litigate settled questions.
**Status:** pre-build. Architecture is decided at the level below; implementation detail is
yours. No code is prescribed here on purpose — choose implementations that satisfy the
stated rationale.

---

## 0. How to use this document

Every section states a **decision** and the **why**. When you face an implementation fork,
resolve it against the *why*, not just the *what*. Section 4 (Design Guardrails) lists
things you must **not** build and the reason each was ruled out — treat these as hard
constraints, not suggestions. Section 11 lists the open questions you are expected to close.

---

## 1. Product mandate

Build a persistent **digital twin of an organization's agent graph** that:
1. Detects **drift** and **error propagation** across agents in near-real-time.
2. Traces any failure back to its **root cause across the whole chain** (causal attribution).
3. Lets a **human supervisor remediate surgically** — edit an agent's context, roll back to
   a checkpoint, redeploy — with every action audited and reversible.
4. Runs at **low-single-digit-percent compute overhead** on the org's existing agent spend,
   **on-premise**.

The defensible core is the **durable, cross-agent, cross-time causal state**. Everything
else is assembled from existing open-source components.

---

## 2. Problem framing (the mental model to hold)

In production multi-agent systems, failure is not "one bad output" — it is **propagation**:
a contaminated output from one agent silently corrupts every downstream agent that consumes
it. The entire existing tooling ecosystem is **stateless and single-hop**: it evaluates one
message in isolation and keeps no memory of how a fault spread. The unsolved, valuable gap
is **durable causal state across agents and time**. Build for that gap; do not rebuild the
parts the ecosystem already covers.

---

## 3. Non-negotiable design constraints

These three constraints drive most downstream decisions. Violating any of them breaks the
product's viability.

1. **No new paid token stream.** The system must consume reasoning the agents *already
   emit* — chain-of-thought traces, tool calls, outputs, inter-agent messages. It must not
   generate a new side-channel of reasoning that costs tokens. Rationale: the buyer is an
   MNC already burning money on tokens; adding generation cost kills adoption.

2. **Detection must be mostly zero-inference.** The default detection path is pure compute
   (embeddings, diffing, validators, logprob parsing). An LLM is consulted only when a cheap
   signal has already flagged something. Rationale: an LLM that reads every reasoning step
   is a second inference pass — it roughly doubles compute. That is the single biggest cost
   trap and must be avoided architecturally, not tuned away later.

3. **On-prem / no data exfiltration.** Corps will not ship agent internals to a third-party
   SaaS. Self-hosted deployment is the default; any hosted option is secondary. Rationale:
   data sovereignty is a hard buying requirement for the target segment, and the field is
   already moving toward on-prem autonomous safety loops.

---

## 4. Design guardrails (what NOT to build, and why)

These are settled negative decisions. Do not reintroduce them.

- **Do not build a "malice" or "deceptive intent" classifier as the core detector.**
  Reliable detection of adversarial/deceptive reasoning is an unsolved research problem, and
  the overwhelming majority of real drift is *benign* (context rot, hallucination, goal
  misgeneralization). A malice-first framing produces mostly false positives. **Detect
  behavioral disagreement instead** (Section 5).

- **Do not rely on chain-of-thought as a sole or trusted signal.** CoT is what a model
  *says*, not what it computed; it can be post-hoc rationalization and a misaligned agent
  can emit clean CoT while acting otherwise. CoT is excellent for the common (benign) case
  and weak for adversarial cases. **Always cross-check CoT against behavior** (tool calls,
  effects).

- **Do not place the monitor inside the agents' training/optimization loop.** The moment
  agents are trained against a CoT-reading monitor, they learn to hide reasoning from it,
  destroying the signal. CoT monitorability is a *fragile asset* — keep the monitor strictly
  observational and out-of-loop.

- **Do not make internal-state steering ("editing the agent's thinking") the default
  remediation.** Steering model internals (activation steering) is unreliable and requires
  white-box model access — exactly the costly/closed infra we avoid. **Remediation operates
  on context, not internals** (Section 8.3). White-box activation *probing* is permitted
  only as an optional premium tier for orgs running self-hosted open models; it is never the
  baseline.

- **Do not monitor uniformly in real-time on the agent's critical path.** Uniform real-time
  monitoring is the thing that doubles cost and adds latency. **Monitoring is tiered,
  sampled, and async** (Section 6). The only inline checks are deterministic hard-stops on
  genuinely dangerous actions.

- **Do not build the observability substrate, trace format, or guardrail-primitive engine
  from scratch.** These are solved (Section 9). Reinventing them wastes the whole budget.

---

## 5. Detection design — hybrid drift metrics

**Principle: detect the gap between what an agent *says* and what it *does*.** This catches
hallucination, goal misgeneralization, and the early shape of deception without needing a
malice classifier, and it degrades gracefully.

Signals, in priority order:

1. **Stated-vs-revealed gap (primary).** Compare the agent's declared intent (from CoT /
   structured self-report) against its actual tool-call arguments and resulting effects.
   Divergence here is the single best cheap signal. Behavioral, harder to fake than words.

2. **Embedding trajectory (the drift metric).** Track semantic distance between a node's
   current reasoning and the originating task specification, over the chain and over time.
   Rising distance = drift. This is the quantitative backbone of the twin.

3. **Deterministic checks (zero-token).** Output-schema violations, tool-misuse rules,
   call-sequence anomalies, confidence/logprob collapse. Pure code; run on everything.

4. **Small-model judge (escalation only).** A cheap model answering one narrow question —
   "does this step still serve the stated goal?" — fired only when a cheaper signal trips.

**Structured self-report at gates.** Require agents to emit a tiny structured statement at
decision gates (restate goal, list assumptions, state confidence). A few structured tokens
are far cheaper to parse and diff than free-form prose and give the twin clean fields.
Note this is capture of already-emitted reasoning shaped into structure, not a new paid
reasoning pass.

**Detection is a taxonomy, not a binary.** Adopt an existing multi-agent risk taxonomy
(see TrinityGuard, Section 9) rather than inventing drift categories. Map each risk type to
which signal(s) above detect it.

---

## 6. Cost & tiering architecture

The architecture *is* the cost control. Layers, cheapest first; traffic falls through only
when a cheaper layer flags:

1. **Zero-inference default layer.** Local/CPU embeddings + structured diffing + validators
   + logprob parsing. Designed to resolve the large majority of traffic at the price of CPU.

2. **Intelligent sampling.** Analyze at **gates** (handoffs, tool calls, decision points),
   not every token — handoffs are where propagation happens. **Risk-weight** coverage: full
   coverage on high-privilege agents (those that move money, write to prod, touch PII), thin
   sampling on low-stakes agents.

3. **Small-model judge.** Haiku-class or self-hosted 3–8B, narrow classification only.
   Roughly an order of magnitude cheaper per token than a frontier model.

4. **Deep-analysis engine (the hybrid resolver / escalation tier).** When a case is
   ambiguous, escalate the flagged agent's *full context + lineage slice* to a costlier
   engine that analyzes deeply and proposes remediation. See Section 7 for why this is safe
   to adopt and where the novelty is.

**Async and off the hot path.** Detection runs asynchronously; the agent never waits on it.
Batch embeddings and judge calls. Accept seconds-to-minutes detection latency for
containment — you are quarantining blast radius, not blocking every token. **Exception:**
hard-stops on dangerous actions are inline and **deterministic (rule-based, never an LLM
call)**.

**Cache and dedupe.** Multi-agent workflows repeat highly similar reasoning; hash/embed-
cache seen patterns so the same thing is never analyzed twice.

**Target envelope:** added cost dominated by CPU + cheap embeddings, realistically low-
single-digit-percent overhead on existing agent spend — not the ~2x that naive monitoring
implies.

---

## 7. The hybrid resolver (escalation) — adopt, with guardrails

**Decision: adopt cheap-flag → escalate-to-costly-engine.** The mechanism is proven and
standard: cascade approaches query a cheap model first, score reliability, and escalate to a
stronger model only on low confidence (FrugalGPT and successors report matching a top
model's accuracy at large cost reductions; see also AutoMix, Cascade Routing, xRouter).

**Where our novelty is:** every existing cascade routes a *user query* to get a better
*answer*. We route a *flagged agent's full context + lineage* to a deep engine that then
*remediates*. We inherit a battle-tested pattern and apply it to an unsolved problem
(drift remediation). Do not try to invent the cascade from scratch — reuse the pattern,
innovate on the application.

**Two guardrails (mandatory):**
- The escalation trigger must be **deterministic and inaccessible to the monitored agents.**
  This is a safety tool; an attacker's goal is to blind it. If agents can influence when
  escalation fires, they can suppress it.
- **Threshold calibration is a first-class problem.** A bad confidence threshold either
  leaks errors (under-escalation) or blows up cost (over-escalation). Build a calibration
  dataset and a human-review feedback loop from day one. Also note cascades can be
  adversarially manipulated into cascade failure — monitor escalation rates for anomalies.

---

## 8. The build surface (this is the actual product)

Everything below is what *we* build. Describe/implement these; reuse everything in
Section 9 underneath them. Data models are described in prose deliberately — choose your own
representation that satisfies the described semantics.

### 8.1 Persistent lineage graph / digital twin — CORE IP

A durable graph that outlives individual runs. **Node = an agent run** (agent id, task,
context snapshot ref, reasoning/telemetry refs, drift scores, timestamp). **Edge = an
influence relation** ("output of run X entered the input/context of run Y"), i.e. the
propagation/provenance link. The graph must support:
- **Cross-time persistence** (survives across runs and sessions), unlike stateless tools
  that discard after each run.
- **Blast-radius queries** — given a compromised node, enumerate every downstream node it
  influenced.
- **Causal attribution across tiers** — given a system-wide failure, walk edges backward to
  the originating single-agent fault (e.g. a cascading collapse ← the initial injection on
  one agent). Treat this as provenance/causal-inference over the graph. This capability is
  explicitly the thing the closest prior art (TrinityGuard) named as needed future work.

This graph is also the **explanation engine** (Section 10) — the same edges that attribute
cause produce the supervisor's narrative.

### 8.2 Tiered cost router

The orchestrator that implements Section 6: routes each observed unit through zero-inference
→ sampled → small-judge → deep-escalation, enforcing the cost envelope. Owns sampling
policy, risk-weighting, caching, batching, and the deterministic escalation trigger. Nobody
has packaged this cost discipline for a safety product — it is part of the moat.

### 8.3 Remediation workflow (closed loop, human-approved)

Wraps the underlying allow/replace/deny primitives (from TrinityGuard, Section 9) into a
supervisor-facing loop:
- **Context edit** — modify the agent's context/prompt state (never internals).
- **Checkpoint / rollback** — restore an agent to a known-good prior context.
- **Isolate / quarantine** — remove a compromised agent from the graph so it cannot affect
  others (a named future-work item in the prior art).
- **Message filtering** — drop or replace a harmful inter-agent message before it propagates.
- **Redeploy** — return the remediated agent to service.

**Every remediation is human-approved by default, logged, and reversible.** Automated
action is permitted only for predefined, low-risk intervention strategies, and always
recorded.

### 8.4 Supervisor + audit trail ("who watches the watcher")

The remediation layer is itself a powerful actor that rewrites agents' context — an
unguarded version silently corrupts what it touches. Therefore: every supervisor action is
attributable, logged, reversible, and auditable. The audit trail is a product feature, not
an afterthought — it is what makes automated intervention trustworthy to a regulated buyer.

---

## 9. Dependency map — reuse vs. integration boundary

Reuse these; build the Section 8 layer on top. For each, the integration boundary is stated
so you know where their responsibility ends and ours begins.

| Layer | Component | Take this | Boundary — ours begins where… |
|---|---|---|---|
| Telemetry + storage + eval API | **Phoenix** *or* **Langfuse** (pick one; self-host) | OpenTelemetry span schema as the telemetry contract; scoring/eval API as where drift scores are written back | …we need durable *cross-run* graph state; they store per-run traces, not a persistent twin |
| Deterministic gates | **NeMo Guardrails** | rails engine as the enforcement primitive; we author drift-specific rails | …drift must accumulate *over a chain/session*; their rails are stateless single-message |
| Drift/alignment scanner | **LlamaFirewall — AlignmentCheck scanner** | ready-made stated-vs-revealed / goal-divergence detector for a single agent trajectory | …we need multi-hop A→B→C propagation reasoning, which it does not model |
| MAS risk taxonomy + interception + remediation primitives | **TrinityGuard** | its multi-agent risk taxonomy; its message-interception layer as our lineage tap; its allow/replace/deny as our remediation primitive | …we need always-on *production* monitoring (it targets eval runs), *closed-loop* remediation (it is diagnostic-only), and the *persistent twin* (it has none) |
| Orchestration integration | **Microsoft Agent Framework / LangGraph / AutoGen / CrewAI** | tap handoff/workflow events as gate-monitoring points | …we add safety semantics; orchestrators emit none |

**Adjacent / do not depend on:** AgentOps (fallback telemetry substrate only, overlaps the
above); FareedKhan-dev/agentic-guardrails (pattern reference / tutorial, not production
infra); initializ/guardrails (only if the stack is Go and an in-process lightweight guardrail
is wanted); ruvnet/agentic-security (different problem — code/DevSecOps scanning — out of
scope).

**Critical pre-build validation:** confirm that **TrinityGuard cannot already be configured**
to do most of what Section 8 describes. It is the closest prior art and the highest
reinvention risk. Validate the boundary above empirically before committing to build.

---

## 10. Output / UX principle

Because we hold causal state, the supervisor-facing output is a **narrative, not an alert.**
Instead of "Agent 7 flagged," the twin surfaces: root cause (e.g. an injection on Agent 2 at
a timestamp) → propagation path (handoffs to Agents 5 and 7) → blast radius (which nodes) →
recommended remediation (e.g. reset Agent 2 to checkpoint, replay). Every stateless tool in
Section 9 can only report an isolated flag; the causal narrative is the UX moat. Build the
attribution engine (8.1) and the presentation layer to produce this narrative directly.

---

## 11. Open questions you must resolve

1. **Substrate choice:** Phoenix vs. Langfuse — decide on self-host maturity, OTel fidelity,
   and eval-API fit for writing drift scores back.
2. **TrinityGuard boundary:** empirically determine what it already does vs. what we add
   (Section 9 validation). This gates the whole build plan.
3. **Twin data model:** concrete representation and store for the lineage graph that supports
   fast blast-radius and backward-causal queries at MNC scale.
4. **Drift thresholds:** define concrete embedding-distance thresholds and stated-vs-revealed
   scoring per workflow type; build the calibration dataset.
5. **Escalation policy:** the deterministic, agent-inaccessible trigger, plus rate monitoring
   for adversarial manipulation.
6. **Remediation safety model:** which (if any) interventions may be automated vs.
   always-human-approved, and the exact audit schema.

---

## 12. Recommended sequencing

1. Pick one **narrow beachhead** where drift is measurable — code agents or multi-step
   research. Do not attempt general coverage first.
2. Stand up the **substrate** (Phoenix/Langfuse) + **TrinityGuard** + **LlamaFirewall
   AlignmentCheck** as the reused base.
3. Build only the **twin graph (8.1)** + **cost router (8.2)** + **remediation loop (8.3/8.4)**
   on top.
4. Prove **containment** on the beachhead (detect → attribute → remediate → redeploy within
   the cost envelope), then generalize.

---

## 13. Decision summary (the reasoning in one place)

- Detect **behavioral disagreement**, not malice — because malice detection is unsolved and
  most drift is benign.
- Harvest **already-emitted CoT + tool-call telemetry** — because a new reasoning stream
  costs tokens the buyer won't pay.
- Cross-check CoT against behavior and keep the monitor **out of the training loop** —
  because CoT is unfaithful and monitorability is fragile.
- Make detection **tiered/sampled/async, mostly zero-inference** — because uniform LLM
  monitoring doubles cost and adds latency.
- **Reuse** substrate, gates, scanners, and MAS taxonomy; **build** only the twin, the cost
  router, and the remediation loop — because those three are the unsolved, defensible core.
- Remediate on **context, not internals**, always **human-approved and audited** — because
  internal steering needs closed infra and an unguarded editor is itself a threat.
- Adopt the **escalation cascade** pattern but apply it to **remediation**, with a
  **deterministic, agent-inaccessible trigger** — because the mechanism is proven and our
  novelty is the application, and because a safety tool must not be blindable by the agents
  it watches.

---

## 14. Market differentiator — predictive simulation & certifiable trust layer

Everything above makes the product *correct*. This section makes it **hard to copy and easy
to buy** — two additions that ride entirely on top of the twin graph (8.1) and the audit
trail (8.4), so they cost almost nothing incremental and violate none of the Section 3
constraints (no new paid token stream, no new inference on the hot path). A competitor
without a persistent causal graph cannot replicate either one — that is the point.

### 14.1 Counterfactual "what-if" remediation preview

Today, remediation (8.3) is reactive: detect → propose a fix → operator approves → fix
applies. Add a **preview step** between proposal and approval: replay the affected downstream
slice of the lineage graph with the proposed corrected context substituted in, and show the
operator a side-by-side projection — "blast radius if we do nothing" vs. "blast radius after
this remediation" — before they commit.

- **Why this is close to free:** the graph, the embeddings, and the propagation edges already
  exist (8.1); this is a replay/simulation over already-computed state, not a new inference
  pass or a new data source.
- **Why it's differentiating:** every competitor in Section 9 is diagnostic or single-shot —
  none can *simulate forward* because none holds durable cross-agent state. This converts the
  twin from a forensic tool into a decision-support tool, which is a materially different
  (and higher-value) product category to a buyer.
- **Guardrail:** the preview is advisory only. It does not change the human-approval default
  in 8.3 — it makes the human's decision better-informed, it does not replace the human.

### 14.2 Compliance-mapped audit trail

The audit trail (8.4) already logs every detection signal, decision, and remediation action
because trustworthy automation requires it. Expose that same log through a **compliance
mapping layer** that tags each entry against clauses of relevant frameworks the target MNC
buyer is already accountable to (e.g. EU AI Act Article 9/15 risk-management and logging
duties, NIST AI RMF, ISO/IEC 42001) and can generate an audit-ready report on demand.

- **Why this is close to free:** no new data is captured — it is a mapping/labeling layer
  over data 8.4 already requires you to keep.
- **Why it's differentiating:** it moves the purchase decision out of a single budget
  (security/eng) and into a second one (compliance/GRC), which widens the buying committee
  and shortens procurement cycles for regulated enterprises — the same buyer segment implied
  by the on-prem constraint in Section 3.
- **Guardrail:** treat the specific framework list as configurable, not hardcoded — regulatory
  mappings change and vary by geography and industry; the mapping layer should be a thin,
  swappable annotation on top of the audit schema (Section 11, open question 6), not baked
  into the core data model.

Both additions are **optional, additive layers**. Do not let either one grow into a
prerequisite for the core loop (detect → attribute → remediate) — that loop must work and be
sellable on its own, per Section 12's sequencing.
