from collections.abc import Callable
from typing import Any

import pytest
from livekit.agents import AgentStateChangedEvent, ToolExecutionUpdatedEvent, UserStateChangedEvent
from livekit.agents.llm import FunctionCall
from livekit.agents.voice.events import AgentState, ToolCallEnded, ToolCallStarted, UserState

from unmute_core.livekit import interruption, yield_to_caller


class FakeSpeech:
    """What the guard reads of a reply, and whether it was interrupted."""

    def __init__(self, *, allow_interruptions: bool = True, id: str = "speech_1") -> None:
        self.allow_interruptions = allow_interruptions
        self.interrupted = False
        self.id = id

    def done(self) -> bool:
        return False

    def interrupt(self) -> None:
        self.interrupted = True


class FakeSession:
    """A session's states, its current reply, and its event handlers."""

    def __init__(self, speech: FakeSpeech) -> None:
        self.current_speech = speech
        self.agent_state: AgentState = "listening"
        self.user_state: UserState = "listening"
        self.handlers: dict[str, Callable[[Any], None]] = {}

    def on(self, name: str, handler: Callable[[Any], None]) -> None:
        self.handlers[name] = handler

    def caller(self, state: UserState) -> None:
        old, self.user_state = self.user_state, state
        self.handlers["user_state_changed"](UserStateChangedEvent(old_state=old, new_state=state))

    def agent(self, state: AgentState) -> None:
        old, self.agent_state = self.agent_state, state
        self.handlers["agent_state_changed"](AgentStateChangedEvent(old_state=old, new_state=state))

    def tool(self, update: ToolCallStarted | ToolCallEnded) -> None:
        self.handlers["tool_execution_updated"](ToolExecutionUpdatedEvent(update=update))


def _session(*, allow_interruptions: bool = True) -> tuple[FakeSession, FakeSpeech]:
    reply = FakeSpeech(allow_interruptions=allow_interruptions)
    session = FakeSession(reply)
    yield_to_caller(session)  # ty: ignore[invalid-argument-type]
    return session, reply


def test_a_caller_who_resumes_while_the_agent_thinks_drops_the_reply() -> None:
    session, reply = _session()
    session.agent("thinking")
    session.caller("speaking")
    assert reply.interrupted


def test_a_turn_committed_while_the_caller_speaks_drops_the_reply() -> None:
    session, reply = _session()
    session.caller("speaking")
    session.agent("thinking")
    assert reply.interrupted


def test_audio_that_plays_is_left_to_livekit() -> None:
    session, reply = _session()
    session.agent("speaking")
    session.caller("speaking")
    assert not reply.interrupted


def test_a_reply_that_runs_a_tool_is_never_cut() -> None:
    session, reply = _session()
    call = FunctionCall(call_id="c1", name="book", arguments="{}")
    session.tool(ToolCallStarted(function_call=call))
    session.agent("thinking")
    session.caller("speaking")
    assert not reply.interrupted
    session.tool(ToolCallEnded(id="c1", call_id="c1", status="done"))
    session.caller("listening")
    session.caller("speaking")
    assert reply.interrupted


def test_a_tool_running_in_another_reply_does_not_keep_this_one() -> None:
    session, reply = _session()
    session.current_speech = FakeSpeech(id="main")
    call = FunctionCall(call_id="c1", name="book", arguments="{}")
    session.tool(ToolCallStarted(function_call=call))
    session.current_speech = reply
    session.agent("thinking")
    session.caller("speaking")
    assert reply.interrupted


def test_a_reply_that_must_not_stop_is_kept() -> None:
    session, reply = _session(allow_interruptions=False)
    session.agent("thinking")
    session.caller("speaking")
    assert not reply.interrupted


def test_min_speech_sets_the_interruption_minimum_duration_for_stop() -> None:
    assert interruption("stop", min_speech_s=0.8) == {
        "enabled": True,
        "mode": "vad",
        "min_duration": 0.8,
    }
    assert "min_duration" not in interruption("stop")


@pytest.mark.parametrize("backchannels", ["keep", "drop"])
def test_min_speech_is_refused_beside_a_word_gate(backchannels: Any) -> None:
    with pytest.raises(ValueError, match="stop only"):
        interruption(backchannels, min_speech_s=0.8)


@pytest.mark.parametrize(
    ("backchannels", "warned"), [("keep", True), ("drop", True), ("stop", False)]
)
def test_a_rule_on_the_adaptive_detector_warns_when_no_livekit_key_is_set(
    backchannels: Any,
    warned: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_INFERENCE_API_KEY", raising=False)
    interruption(backchannels)
    assert ("needs LIVEKIT_API_KEY" in caplog.text) is warned
    caplog.clear()
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    interruption(backchannels)
    assert "needs LIVEKIT_API_KEY" not in caplog.text
