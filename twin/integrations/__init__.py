from __future__ import annotations

"""Adapters that wire real external tools behind the twin's internal seams.

Every adapter here is optional at runtime: the heavy third-party package is
imported lazily and, if it is missing or fails, the adapter degrades to the
native deterministic component it replaces (and logs the degradation). This is
the same contract used by the sentence-transformers and OpenAI backends, so the
service always boots and always produces a verdict — with or without the
external dependency installed.
"""

__all__ = [
    "llamafirewall_judge",
    "nemo_guard",
    "handoff",
]
