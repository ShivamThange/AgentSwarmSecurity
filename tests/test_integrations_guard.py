from __future__ import annotations

from twin.detection import DEFAULT_POLICY
from twin.guard import NativeGuard, build_guard
from twin.integrations.nemo_guard import NeMoGuard
from twin.models import Privilege, Span, ToolCall

from .conftest import make_settings


def _forbidden_transfer() -> Span:
    return Span(
        span_id="g1", trace_id="t", agent_id="payments", privilege=Privilege.HIGH,
        task_spec="Prepare a payment summary. Do NOT move funds.",
        declared_intent="Prepare the payment summary for review.",
        tool_calls=[ToolCall(name="transfer_funds", args={"amount": 1})],
    )


def _benign() -> Span:
    return Span(
        span_id="g2", trace_id="t", agent_id="notifier",
        task_spec="Post a progress update.",
        declared_intent="Post a short progress update.",
        tool_calls=[ToolCall(name="post_status")],
    )


def test_build_guard_defaults_to_native(tmp_path):
    guard = build_guard(make_settings(tmp_path), DEFAULT_POLICY)
    assert isinstance(guard, NativeGuard)
    assert guard.info()["backend"] == "native"


def test_native_blocks_forbidden_transfer(tmp_path):
    guard = build_guard(make_settings(tmp_path), DEFAULT_POLICY)
    denials = guard.blocked_decisions(_forbidden_transfer())
    assert len(denials) == 1
    assert denials[0].tool == "transfer_funds"
    assert denials[0].rule == "sensitive_action_gate"


def test_nemo_backend_degrades_to_native_without_config(tmp_path):
    settings = make_settings(tmp_path, guard_backend="nemo")
    guard = build_guard(settings, DEFAULT_POLICY)
    assert isinstance(guard, NeMoGuard)
    # No config path and no package -> behaves exactly like the native rail.
    assert len(guard.blocked_decisions(_forbidden_transfer())) == 1
    assert guard.blocked_decisions(_benign()) == []


class _FakeRails:
    def __init__(self, content: str) -> None:
        self.content = content

    def generate(self, messages):  # noqa: ANN001
        return {"role": "assistant", "content": self.content}


def test_nemo_adds_advisory_denial_on_top_of_native(tmp_path):
    guard = NeMoGuard(make_settings(tmp_path), DEFAULT_POLICY,
                      rails=_FakeRails("I'm not able to comply with that."))
    denials = guard.blocked_decisions(_benign())
    assert any(d.rule == "nemo_guardrails" for d in denials)


def test_nemo_never_clears_a_native_block(tmp_path):
    # Even if NeMo would allow, the native deterministic block stands.
    guard = NeMoGuard(make_settings(tmp_path), DEFAULT_POLICY,
                      rails=_FakeRails("Sure, that's fine."))
    denials = guard.blocked_decisions(_forbidden_transfer())
    assert any(d.rule == "sensitive_action_gate" and not d.allowed
               for d in denials)


def test_nemo_rails_error_falls_back_to_native(tmp_path):
    class _Boom:
        def generate(self, messages):  # noqa: ANN001
            raise RuntimeError("rails server down")

    guard = NeMoGuard(make_settings(tmp_path), DEFAULT_POLICY, rails=_Boom())
    # Native decision must still be produced despite the NeMo failure.
    assert len(guard.blocked_decisions(_forbidden_transfer())) == 1
