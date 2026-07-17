from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from ..config import Settings
from ..detection import DetectionPolicy, _behaviour_text
from ..guard import GuardDecision, NativeGuard
from ..models import Span

log = logging.getLogger(__name__)


class NeMoGuard:
    """Guard backend that layers NVIDIA NeMo Guardrails over the native rail.

    The native deterministic rail always runs first and its denials are
    authoritative — NeMo can only *add* advisory denials, never clear one. NeMo
    is loaded lazily from a Colang config directory (``TWIN_NEMO_CONFIG_PATH``);
    if the package or config is missing, or a rails call fails, this backend is
    transparently equivalent to :class:`NativeGuard`. NeMo denials are marked
    ``reversible``-neutral via a distinct rule so operators can tell a
    model-based block from a deterministic one.
    """

    backend = "nemo"

    def __init__(self, settings: Settings, policy: DetectionPolicy,
                 native: Optional[NativeGuard] = None,
                 rails: Optional[Any] = None) -> None:
        self.settings = settings
        self.policy = policy
        self.native = native or NativeGuard(policy)
        self._rails = rails
        self._built = rails is not None
        self._unavailable = False
        self._lock = threading.Lock()

    def _build_rails(self) -> Optional[Any]:
        config_path = self.settings.nemo_config_path
        if not config_path:
            log.warning("guard backend is 'nemo' but TWIN_NEMO_CONFIG_PATH is "
                        "unset; running the native rail only")
            return None
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except ImportError:
            log.warning("guard backend is 'nemo' but 'nemoguardrails' is not "
                        "installed; run pip install -r "
                        "requirements-integrations.txt — using the native rail")
            return None
        try:
            config = RailsConfig.from_path(config_path)
            return LLMRails(config)
        except Exception as exc:  # pragma: no cover - environment dependent
            log.warning("failed to load NeMo Guardrails config from %s (%s: "
                        "%s); using the native rail", config_path,
                        type(exc).__name__, exc)
            return None

    def _get_rails(self) -> Optional[Any]:
        if self._built:
            return self._rails
        with self._lock:
            if not self._built:
                self._rails = self._build_rails()
                self._built = True
                self._unavailable = self._rails is None
        return self._rails

    def _nemo_denials(self, span: Span) -> list[GuardDecision]:
        rails = self._get_rails()
        if rails is None:
            return []
        try:
            messages = [
                {"role": "user", "content": span.task_spec
                 or span.declared_intent},
                {"role": "assistant", "content": _behaviour_text(span)},
            ]
            result = rails.generate(messages=messages)
            content = ""
            if isinstance(result, dict):
                content = str(result.get("content", ""))
            else:
                content = str(getattr(result, "content", result) or "")
            blocked = ("i'm not able" in content.lower()
                       or "cannot comply" in content.lower()
                       or "blocked" in content.lower())
            if not blocked:
                return []
            tools = [c.name for c in span.tool_calls] or ["<output>"]
            return [GuardDecision(
                tool=t, allowed=False, rule="nemo_guardrails",
                reason=f"NeMo Guardrails rail refused this step: "
                       f"{content[:200]}")
                for t in tools]
        except Exception as exc:
            log.warning("NeMo Guardrails evaluation failed (%s: %s); native "
                        "rail decision stands", type(exc).__name__, exc)
            return []

    def check_span(self, span: Span) -> list[GuardDecision]:
        decisions = self.native.check_span(span)
        gated = {d.tool for d in decisions}
        for d in self._nemo_denials(span):
            if d.tool not in gated:
                decisions.append(d)
        return decisions

    def blocked_decisions(self, span: Span) -> list[GuardDecision]:
        return [d for d in self.check_span(span) if not d.allowed]

    def info(self) -> dict:
        return {"backend": self.backend,
                "config_path": self.settings.nemo_config_path,
                "available": not self._unavailable if self._built else None}
