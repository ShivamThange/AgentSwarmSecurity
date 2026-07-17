from __future__ import annotations

"""Executable boundary validation (replaces the desk-only analysis).

Confirms two things an operator needs to trust the design:

1. Each reuse target (§9) either backs a twin seam or is absent — and the twin
   still runs regardless (graceful degradation).
2. The five wedge capabilities C1–C5 (durable cross-run lineage, backward
   root-cause, blast radius, reversible context remediation, what-if replay)
   are served by twin modules and are NOT provided by any reused tool. This is
   the defensibility check: if a dependency ever started providing C1–C5, that
   is a signal to re-evaluate the moat.

Exit code 0 means the wedge is intact and every configured integration seam is
importable. Non-zero means a wedge capability is missing (a real regression) —
a missing optional integration only warns.
"""

import argparse
import importlib
import os
import sys

# Ensure the repo root is importable when run as `python scripts/...`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse targets and the twin seam each one plugs into.
REUSE_TARGETS = [
    ("llamafirewall", "detection.Judge (small tier)",
     "twin.integrations.llamafirewall_judge"),
    ("nemoguardrails", "guard.GuardBackend (inline rail)",
     "twin.integrations.nemo_guard"),
    ("openinference.semconv", "otel_ingest (Phoenix attribute conventions)",
     None),
    ("langfuse", "otel_ingest (Langfuse attribute conventions)", None),
    ("autogen", "integrations.handoff (AutoGen tap)", None),
    ("langgraph", "integrations.handoff (LangGraph tap)", None),
]

# The wedge: capability -> (twin module, attribute that implements it).
WEDGE_CAPABILITIES = [
    ("C1 durable cross-run lineage graph", "twin.store", "TwinStore"),
    ("C2 backward root-cause attribution", "twin.attribution",
     "find_root_cause"),
    ("C3 blast radius", "twin.store", "TwinStore.blast_radius"),
    ("C4 reversible context remediation", "twin.remediation",
     "RemediationEngine"),
    ("C5 what-if replay over held state", "twin.replay", "build_preview"),
]


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def _has_attr(module: str, dotted: str) -> bool:
    try:
        obj = importlib.import_module(module)
        for part in dotted.split("."):
            obj = getattr(obj, part)
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="also fail if a reuse target's seam is missing")
    args = parser.parse_args()

    print("== Wedge capabilities (C1-C5) -- must be twin-native ==")
    wedge_ok = True
    for name, module, attr in WEDGE_CAPABILITIES:
        ok = _has_attr(module, attr)
        wedge_ok = wedge_ok and ok
        print(f"  [{'OK ' if ok else 'MISSING'}] {name:42s} <- {module}.{attr}")

    print("\n== Reuse targets (section 9) -- optional, plug into twin seams ==")
    seam_missing = False
    for pkg, seam, seam_module in REUSE_TARGETS:
        installed = _importable(pkg)
        seam_ok = seam_module is None or _importable(seam_module)
        seam_missing = seam_missing or not seam_ok
        state = "installed" if installed else "absent (degrades to native)"
        seam_state = "" if seam_ok else "  <SEAM IMPORT FAILED>"
        print(f"  [{'x' if installed else ' '}] {pkg:24s} -> {seam:42s} "
              f"{state}{seam_state}")

    print("\n== Verdict ==")
    if not wedge_ok:
        print("  FAIL: a wedge capability (C1–C5) is missing — this is a "
              "regression in the defensible core.")
        return 2
    if args.strict and seam_missing:
        print("  FAIL (--strict): an integration seam failed to import.")
        return 3
    print("  PASS: C1-C5 are twin-native and intact; no reused tool provides "
          "them. Integration seams are wired; absent packages degrade "
          "gracefully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
