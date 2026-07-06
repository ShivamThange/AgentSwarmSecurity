# Boundary Validation — does an existing tool already do the 8.x layers?

**Purpose (Section 9 / 11.2 — the gate on the whole build plan).** Before hardening
any of the novel layers, confirm that the "reuse, don't rebuild" boundary is real:
that no existing guardrail/observability tool already provides the durable
cross-run causal twin, backward root-cause attribution, blast radius, human-approved
reversible context-remediation, and what-if replay. If one does, we reuse it. If
none do, the wedge is confirmed and we build exactly those layers and nothing else.

**Status: DESK VALIDATION (partial).** A live stand-up of every tool was **not**
completed in this environment. See *Method & honesty* and the open blocker at the
end — one item needs your intervention.

---

## Method & honesty

What I actually ran (evidence, reproducible):

```
python -m pip index versions trinityguard      -> ERROR: No matching distribution found
python -m pip index versions llamafirewall     -> 1.0.3 (and older)
python -m pip index versions nemoguardrails     -> 0.23.0 (and older)
python -m pip index versions arize-phoenix      -> 17.19.0 (and older)
```

- **TrinityGuard is not a public PyPI package** — it cannot be `pip install`-ed here.
  It is either proprietary/internal, a research artifact, or a naming mismatch
  (e.g. LlamaGuard / a vendor product). **This blocks a literal "stand up
  TrinityGuard".** It needs your input (see blocker).
- LlamaFirewall, NeMo-Guardrails and Arize-Phoenix are real and installable, but a
  full live stand-up pulls large model/runtime downloads (torch, transformers,
  model weights, a Phoenix server). That directly contradicts the product's own
  offline / no-download constraint (Section 3.3), so I did **not** install them
  unilaterally — it is a decision for you (see blocker).

The capability analysis below is therefore a **desk validation** grounded in each
tool's documented architecture, not a live configuration attempt. Every claim is
falsifiable by the live procedure in the appendix; treat any "cannot" as a
hypothesis to confirm on a live stand-up, not a settled fact.

---

## The 8.x capabilities under test

| # | Capability (brief) | One-line test |
|---|---|---|
| C1 | Durable cross-run/cross-session lineage graph | Restart the process — is the multi-agent influence graph still queryable? |
| C2 | Backward causal root-cause across agents | Given a flagged agent, does it name the *originating* agent up-chain? |
| C3 | Blast radius (forward propagation set) | Does it enumerate every downstream consumer a bad output reached? |
| C4 | Human-approved, reversible remediation on **context** | Can an operator approve a fix that edits context (not weights) and revert it? |
| C5 | Counterfactual what-if replay over held state | "If I fix the root, who recovers?" simulated from stored state? |
| D  | Per-hop drift/injection detection | Does it flag a single step where behaviour ≠ declared intent? |
| R  | Inline deterministic hard-stop rail | Does it synchronously block a dangerous action pre-execution? |

C1–C5 are the wedge. D and R are the commodity layers we expect to **reuse**.

---

## Tool-by-tool

### LlamaFirewall (Meta) — a per-hop agent scanner
- **What it is:** a real-time guardrail framework. Scanners: **PromptGuard**
  (jailbreak/injection classifier), **AlignmentCheck** (audits an agent's
  chain-of-thought for goal-hijacking / misalignment on the current trajectory),
  **CodeShield** (insecure-code static analysis), plus regex/custom scanners.
- **Covers:** **D** strongly. AlignmentCheck is the closest external analogue to our
  stated-vs-revealed signal — but it is *single-agent, single-trajectory,
  in-loop*. PromptGuard is a good drop-in for injection detection at ingest.
- **Does not cover (hypotheses):** C1 (no persistence across runs — it scans an
  interaction and returns a decision; it holds no cross-run graph), C2/C3 (no
  multi-agent lineage, so no backward attribution or blast radius), C4 (it
  allows/blocks; it has no human-approval + reversible context-remediation loop),
  C5 (no held state to replay).
- **Verdict:** **reuse as a detector** behind our `Judge`/detector seam. It is a
  scanner, not the twin. Wedge intact.

### NeMo-Guardrails (NVIDIA) — programmable rails
- **What it is:** input/output/dialog/execution rails authored in Colang;
  deterministic policy enforcement around an LLM app.
- **Covers:** **R** directly — this is the production form of our inline
  deterministic hard-stop (`twin/guard.py`). Also contributes to **D** via rails.
- **Does not cover (hypotheses):** C1–C3, C5 (rails are per-invocation policy; there
  is no durable cross-agent causal graph, no attribution, no blast radius, no
  replay). C4 partially — rails can deny/replace, but not the audited, reversible,
  human-in-the-loop remediation-over-context loop with rollback checkpoints.
- **Verdict:** **reuse as the rail engine** for R (and some D). Not the twin.

### Arize-Phoenix — LLM observability / tracing
- **What it is:** OpenTelemetry-based tracing and evals for LLM/agent apps; persists
  spans/traces; retrieval and agent-trace views; eval runners.
- **Covers:** the **substrate**. It persists OTel spans — which is exactly why our
  ingestion contract is OTel-shaped (Section 3.1): Phoenix is a natural upstream
  data source we ingest from.
- **Does not cover (hypotheses):** C2/C3 as first-class causal queries (it shows
  traces and lets you eval spans; it does not maintain an *influence* lineage graph
  with `blast_radius`/`upstream`/root-cause as native operations), C4 (observability,
  not remediation — it does not act on the system), C5 (no counterfactual replay).
- **Verdict:** **reuse as the telemetry source / UI-adjacent layer.** Not the twin.

### TrinityGuard — not resolvable here
- **What I found:** not on PyPI (probe above). Cannot be stood up in this
  environment. Whatever it maps to, the capability questions C1–C5 remain the test;
  if it is another per-hop guard/observability tool (the common category), the
  analysis above applies and the wedge holds. **Needs your input to resolve.**

---

## Conclusion (desk)

Across the three tools that *are* installable, the coverage splits cleanly:

- **Reused, not rebuilt:** D (LlamaFirewall PromptGuard/AlignmentCheck), R (NeMo
  rails), telemetry substrate (Phoenix OTel spans).
- **Not provided by any of them — the wedge:** C1 durable cross-run causal twin,
  C2 backward root-cause across agents, C3 blast radius, C4 human-approved
  reversible context-remediation, C5 what-if replay over held state.

These C1–C5 are precisely the layers this MVP builds (`twin/graph.py`,
`twin/attribution.py`, `twin/remediation.py`, `twin/replay.py`) and nothing more —
the detectors and rails sit behind swap-in seams so the commodity work is delegated
to the tools above. **On the desk evidence, the wedge is confirmed.** It is not yet
*live*-confirmed; that is the remaining gate.

---

## Appendix — how to convert this to a LIVE validation

Run in a network-enabled, non-air-gapped dev box (NOT the product runtime, which
stays offline). ~Confirms or refutes each "cannot" above.

```bash
# 1. Detector reuse (LlamaFirewall) — confirm it has no cross-run graph API
pip install llamafirewall
#   feed it scenario A2 & A7; confirm it flags per-step (D) but exposes
#   no blast_radius / upstream / root-cause across A2->A3->A5->A7 (C1-C3).

# 2. Rail reuse (NeMo-Guardrails) — confirm it blocks A7 inline (R)
pip install nemoguardrails
#   author an execution rail denying transfer_funds; confirm no lineage/replay.

# 3. Substrate (Phoenix) — confirm spans persist but causal queries don't exist
pip install arize-phoenix
#   push the 7 incident spans; confirm trace view but no native blast_radius/
#   backward-attribution/what-if.

# 4. TrinityGuard — resolve what it refers to, then repeat C1-C5 against it.
```

Acceptance: the wedge is *live*-confirmed iff, for at least one dangerous
multi-agent chain, none of the tools answer C1–C5 (durable graph, backward
root-cause, blast radius, reversible context-remediation, what-if) out of the box.
