"""Counterfactual "what-if" remediation preview (Section 14.1).

Public entry point. The projection is produced by a *real forward replay* over the
persisted twin (see `replay.py`): corrected context is substituted at the root and
the same detection code is re-run node-by-node. This is not a re-execution of the
agents and spends no new inference — it re-derives drift from the twin's own state
under corrected input. Advisory only: it informs the human decision, it never
replaces the approval step (14.1 guardrail).
"""
from __future__ import annotations

from . import replay
from .graph import TwinStore
from .models import WhatIfPreview


def build_preview(store: TwinStore, root_id: str) -> WhatIfPreview | None:
    return replay.build_preview(store, root_id)
