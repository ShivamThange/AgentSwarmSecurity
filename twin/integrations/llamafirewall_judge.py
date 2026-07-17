from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from ..config import Settings
from ..detection import DetectionPolicy, StubJudge, _behaviour_text
from ..models import DriftSignals, JudgeVerdict, Span

log = logging.getLogger(__name__)

_SCANNER_NAMES = {
    "prompt_guard": "PROMPT_GUARD",
    "alignment_check": "ALIGNMENT_CHECK",
    "pii_detection": "PII_DETECTION",
    "hidden_ascii": "HIDDEN_ASCII",
    "code_shield": "CODE_SHIELD",
}

# Decisions LlamaFirewall can return; anything not ALLOW is treated as the
# behaviour no longer serving the goal.
_ALLOW = "ALLOW"


class LlamaFirewallJudge:
    """`detection.Judge` implementation backed by Meta's LlamaFirewall.

    Runs the configured scanners (PromptGuard for injection, AlignmentCheck for
    stated-vs-revealed behaviour) over a reconstructed conversation trace and
    maps the worst scan decision onto a :class:`JudgeVerdict`. If the
    ``llamafirewall`` package is not installed, or a scan raises, it falls back
    to the deterministic :class:`StubJudge` so callers always get a verdict.

    The firewall handle is built lazily and cached; construction is the
    expensive step (it loads local models), so it is done once under a lock on
    first use. Pass ``firewall=`` to inject a pre-built handle (used in tests).
    """

    def __init__(self, settings: Settings, policy: DetectionPolicy,
                 firewall: Optional[Any] = None) -> None:
        self.settings = settings
        self.policy = policy
        self.scanners = [s for s in settings.llamafirewall_scanners
                         if s in _SCANNER_NAMES]
        self.model = "llamafirewall[" + ",".join(self.scanners) + "]"
        self._fallback = StubJudge(policy)
        self._fw = firewall
        self._built = firewall is not None
        self._unavailable = False
        self._lock = threading.Lock()

    # --- firewall construction -------------------------------------------

    def _build_firewall(self) -> Optional[Any]:
        try:
            from llamafirewall import LlamaFirewall, Role, ScannerType
        except ImportError:
            log.warning(
                "judge backend is 'llamafirewall' but the 'llamafirewall' "
                "package is not installed; run "
                "pip install -r requirements-integrations.txt — falling back "
                "to the deterministic judge")
            return None
        try:
            user_scanners = []
            assistant_scanners = []
            for name in self.scanners:
                scanner = getattr(ScannerType, _SCANNER_NAMES[name], None)
                if scanner is None:
                    continue
                # AlignmentCheck judges the assistant's trajectory; the rest
                # screen incoming/foreign content on the user side.
                if name == "alignment_check":
                    assistant_scanners.append(scanner)
                else:
                    user_scanners.append(scanner)
            scanners: dict[Any, list] = {}
            if user_scanners:
                scanners[Role.USER] = user_scanners
            if assistant_scanners:
                scanners[Role.ASSISTANT] = assistant_scanners
            if not scanners:
                log.warning("no valid llamafirewall scanners configured (%s); "
                            "falling back to the deterministic judge",
                            self.settings.llamafirewall_scanners)
                return None
            return LlamaFirewall(scanners=scanners)
        except Exception as exc:  # pragma: no cover - environment dependent
            log.warning("failed to initialise LlamaFirewall (%s: %s); falling "
                        "back to the deterministic judge",
                        type(exc).__name__, exc)
            return None

    def _get_firewall(self) -> Optional[Any]:
        if self._built:
            return self._fw
        with self._lock:
            if not self._built:
                self._fw = self._build_firewall()
                self._built = True
                self._unavailable = self._fw is None
        return self._fw

    # --- scanning ---------------------------------------------------------

    @staticmethod
    def _message_factories():
        try:
            from llamafirewall import AssistantMessage, UserMessage
            return UserMessage, AssistantMessage
        except ImportError:
            # A firewall handle can be injected (tests) without the package on
            # the path; a lightweight shim carrying .content is enough for it.
            class _Msg:
                def __init__(self, content: str) -> None:
                    self.content = content

            return _Msg, _Msg

    def _trace(self, span: Span, signals: DriftSignals) -> list:
        user_msg, assistant_msg = self._message_factories()
        goal = (span.task_spec + "\n" + span.declared_intent).strip()
        trace: list = [user_msg(content=goal or span.task_spec or " ")]
        # Foreign / injected content surfaced by the cheap signals is fed in as
        # additional user turns so PromptGuard can score it.
        for ent in signals.foreign_entities[:4]:
            trace.append(user_msg(content=ent))
        trace.append(assistant_msg(content=_behaviour_text(span) or " "))
        return trace

    @staticmethod
    def _worst(results: list) -> Optional[Any]:
        worst = None
        worst_rank = -1
        for r in results:
            decision = getattr(r, "decision", None)
            name = getattr(decision, "name", str(decision)).upper()
            rank = 0 if name == _ALLOW else 1
            score = float(getattr(r, "score", 0.0) or 0.0)
            if (rank, score) > (worst_rank, float(getattr(worst, "score", 0.0)
                                                 or 0.0) if worst else -1.0):
                worst, worst_rank = r, rank
        return worst

    def _scan(self, trace: list) -> Optional[Any]:
        fw = self._fw
        # scan_replay evaluates a whole trace; fall back to scanning the final
        # assistant turn if the installed version lacks it.
        replay = getattr(fw, "scan_replay", None)
        if callable(replay):
            return replay(trace)
        scan = getattr(fw, "scan", None)
        if callable(scan):
            results = [scan(msg) for msg in trace]
            return self._worst([r for r in results if r is not None])
        return None

    def assess(self, span: Span, signals: DriftSignals) -> JudgeVerdict:
        fw = self._get_firewall()
        if fw is None:
            v = self._fallback.assess(span, signals)
            v.rationale = "[llamafirewall unavailable] " + v.rationale
            return v
        try:
            result = self._scan(self._trace(span, signals))
            if result is None:
                raise RuntimeError("scan produced no result")
            decision = getattr(result, "decision", None)
            name = getattr(decision, "name", str(decision)).upper()
            score = float(getattr(result, "score", 0.0) or 0.0)
            score = max(0.0, min(1.0, score))
            reason = str(getattr(result, "reason", "") or "").strip()
            serves_goal = name == _ALLOW
            # A high score means high confidence in the flagged decision; when
            # the step is allowed, invert so confidence still reads as "how sure
            # are we of this verdict".
            confidence = score if not serves_goal else max(0.5, 1.0 - score)
            rationale = (f"LlamaFirewall {name.lower()}"
                         + (f": {reason}" if reason else "")
                         + f" (score={score:.2f})")
            return JudgeVerdict(serves_goal=serves_goal,
                                confidence=max(0.0, min(1.0, confidence)),
                                rationale=rationale[:500])
        except Exception as exc:
            log.warning("LlamaFirewall scan failed (%s: %s); using "
                        "deterministic fallback", type(exc).__name__, exc)
            v = self._fallback.assess(span, signals)
            v.rationale = f"[llamafirewall scan error: {type(exc).__name__}] " \
                          + v.rationale
            return v

    def info(self) -> dict:
        return {"backend": "llamafirewall", "scanners": self.scanners,
                "available": not self._unavailable if self._built else None}
