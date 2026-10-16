"""Opening a LiveKit call: what the SIP call carries, and the greeting."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from livekit import rtc
from livekit.agents import AgentSession
from pydantic import BaseModel

from tests.hooks.helpers import checks, on_end
from tests.livekit.test_agent_lifecycle import Scripted
from unmute_core.agent import AgentSpec
from unmute_core.call import FAILURE_LINE, CallEnd, CallHooks
from unmute_core.greeting import GREETING_INSTRUCTION, Greeting
from unmute_core.hooks import Hooks
from unmute_core.livekit import (
    UnmuteAgent,
    _close,
    attach_call,
    greet,
    interruption,
    max_duration,
    open_call,
    sip_call_info,
)
from unmute_core.livekit._close import CloseGuard
from unmute_core.model import ModelRequest, TextDelta
from unmute_core.state import CallState, HeldSave

CALLER = "+34600111222"
TRUNK = "+34910000000"


def _sip(**attributes: str) -> Any:
    return SimpleNamespace(kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP, attributes=attributes)


def _participant(**attributes: str) -> Any:
    return _sip(**{"sip.ruleID": "SDR_1", **attributes})


def test_an_inbound_sip_call_reads_its_numbers_from_the_participant():
    participant = _participant(
        **{"sip.phoneNumber": CALLER, "sip.trunkPhoneNumber": TRUNK, "sip.callID": "SCL_1"}
    )
    call = sip_call_info(participant, {}, agent_id="salon", session_id="s-1")
    assert call.model_dump(exclude_none=True) == {
        "agent_id": "salon",
        "session_id": "s-1",
        "channel": "phone",
        "direction": "inbound",
        "calling_number": CALLER,
        "called_number": TRUNK,
        "remote_number": CALLER,
        "carrier_call_id": "SCL_1",
    }


def test_an_outbound_call_dials_the_metadata_number_from_the_trunk():
    call = sip_call_info(None, {"phone_number": CALLER}, trunk_number=TRUNK)
    assert (call.direction, call.calling_number, call.called_number, call.remote_number) == (
        "outbound",
        TRUNK,
        CALLER,
        CALLER,
    )


@pytest.mark.parametrize(
    ("participant", "metadata", "given", "direction"),
    [
        (_sip(**{"sip.ruleID": "SDR_1"}), {"phone_number": TRUNK}, None, "inbound"),
        (_sip(), {"phone_number": CALLER}, None, "outbound"),
        (_sip(), {}, None, "inbound"),
        (None, {"direction": "outbound"}, None, "outbound"),
        (_sip(**{"sip.ruleID": "SDR_1"}), {}, "outbound", "outbound"),
    ],
)
def test_the_direction_comes_from_the_argument_the_metadata_or_the_sip_participant(
    participant, metadata, given, direction
):
    assert sip_call_info(participant, metadata, direction=given).direction == direction


class Session:
    """Records what the greeting asked the session for."""

    llm = None
    tts = None
    current_agent = SimpleNamespace(llm=None, tts=None)

    def __init__(self) -> None:
        self.said: list[str] = []
        self.generated: list[str] = []
        self.protected: list[bool] = []

    def say(self, text: str, *, allow_interruptions: object, **_options: object) -> str:
        self.said.append(text)
        self.protected.append(allow_interruptions is False)
        return "speech"

    def generate_reply(self, *, instructions: str, allow_interruptions: object) -> str:
        self.generated.append(instructions)
        self.protected.append(allow_interruptions is False)
        return "speech"


class Customer(BaseModel):
    name: str | None = None


def test_greet_says_the_line_words_one_or_waits():
    state = CallState(Customer)
    state.save({"name": "Ana"})
    session = Session()
    assert greet(cast(Any, session), Greeting(text="Hi {{name}}!"), state) == "speech"
    assert greet(cast(Any, session), Greeting(), state) == "speech"
    assert greet(cast(Any, session), Greeting(speaks_first="user"), state) is None
    assert session.said == ["Hi Ana!"]
    assert session.generated == [GREETING_INSTRUCTION]
    greet(cast(Any, session), Greeting(text="Hello.", interruptible=False), state)
    assert session.protected == [False, False, True]


class Closing:
    """A session that can be shut down and closed."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.shut = 0

    def on(self, name: str, handler: Any) -> None:
        self.handlers[name] = handler

    def shutdown(self) -> None:
        self.shut += 1


async def test_max_duration_shuts_the_session_down_unless_it_closed_first():
    session = Closing()
    max_duration(cast(Any, session), 0.01)
    await asyncio.sleep(0.05)
    assert session.shut == 1
    closed = Closing()
    max_duration(cast(Any, closed), 0.01)
    closed.handlers["close"](None)
    await asyncio.sleep(0.05)
    assert closed.shut == 0


def test_unmutes_interruption_settings_map_onto_livekits_options():
    assert interruption("keep", enabled=False) == {"enabled": False}
    assert interruption("stop", min_words=2) == {"enabled": True, "mode": "vad", "min_words": 2}
    assert "min_words" not in interruption("drop")
    with pytest.raises(ValueError, match="0 or more"):
        interruption("drop", min_words=-1)


async def test_the_jobs_shutdown_waits_for_the_held_saves_once_the_session_closed(monkeypatch):
    state = CallState(Customer)
    saved: list[object] = []

    async def slow_decision() -> dict[str, str]:
        await asyncio.sleep(0.05)
        return {"name": "Ana"}

    state.hold("greet", HeldSave(saved.append, lambda: True, slow_decision))
    agent = UnmuteAgent(AgentSpec("Hi."), state=state)
    agent._closed(cast(Any, None))
    await agent._guard.wait()
    assert saved == [{"name": "Ana"}]

    monkeypatch.setattr(_close, "CLOSE_WAIT_SECONDS", 0.01)
    never_closed = UnmuteAgent(AgentSpec("Hi."), state=CallState(Customer))
    await never_closed._guard.wait()


class Handle:
    """A greeting's speech that ends when told."""

    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def add_done_callback(self, callback: Any) -> None:
        self.callbacks.append(callback)

    def finish(self) -> None:
        for callback in self.callbacks:
            callback(self)

    async def wait_for_playout(self) -> None:
        return None


class Live:
    """A session that greets, holds the caller's audio, fails and closes."""

    llm = None
    tts = None
    current_agent = SimpleNamespace(llm=None, tts=None)

    def __init__(self) -> None:
        self.audio: list[bool] = []
        self.input = SimpleNamespace(set_audio_enabled=self.audio.append)
        self.handlers: dict[str, list[Any]] = {}
        self.said: list[str] = []
        self.shut = 0
        self.handle = Handle()
        self.history = SimpleNamespace(
            to_provider_format=lambda _format: (
                [
                    {"role": "system", "content": "prompt"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "tool_calls": []},
                    {"role": "assistant", "content": "hello"},
                ],
                None,
            )
        )

    def on(self, name: str, handler: Any) -> None:
        self.handlers.setdefault(name, []).append(handler)

    once = on

    def off(self, name: str, handler: Any) -> None:
        if handler in self.handlers.get(name, []):
            self.handlers[name].remove(handler)

    def emit(self, name: str, event: Any) -> None:
        for handler in list(self.handlers.get(name, [])):
            handler(event)

    def say(self, text: str, *, allow_interruptions: object, **_options: object) -> Handle:
        self.said.append(text)
        return self.handle

    def shutdown(self) -> None:
        self.shut += 1


async def test_open_call_holds_the_caller_until_the_greeting_starts():
    session = Live()
    call = CallHooks(CallState(Customer))
    handle = await open_call(cast(Any, session), Greeting(text="Hello."), call)
    assert handle is session.handle
    assert session.audio == [False]
    session.emit("agent_state_changed", SimpleNamespace(new_state="thinking"))
    assert session.audio == [False]
    session.emit("agent_state_changed", SimpleNamespace(new_state="speaking"))
    assert session.audio == [False, True]


async def test_a_protected_greeting_holds_the_caller_until_it_ends():
    session = Live()
    call = CallHooks(CallState(Customer))
    await open_call(cast(Any, session), Greeting(text="Hello.", interruptible=False), call)
    session.emit("agent_state_changed", SimpleNamespace(new_state="speaking"))
    assert session.audio == [False]
    session.handle.finish()
    assert session.audio == [False, True]


async def test_a_failed_check_says_the_failure_line_and_keeps_the_audio_off():
    session = Live()

    async def broken() -> None:
        raise RuntimeError("down")

    call = CallHooks(CallState(Customer), hooks=checks(broken))
    assert await open_call(cast(Any, session), Greeting(text="Hello."), call) is None
    assert session.said == [FAILURE_LINE]
    assert session.shut == 1
    assert session.audio == [False]


async def test_an_unrecoverable_model_error_fails_the_call_once_and_a_tts_error_does_not():
    session = Live()
    call = CallHooks()
    attach_call(cast(Any, session), call)
    attach_call(cast(Any, session), call)
    error = SimpleNamespace(type="llm_error", recoverable=True, error=RuntimeError("x"))
    session.emit("error", SimpleNamespace(error=error))
    await asyncio.sleep(0)
    assert session.said == []
    speech = SimpleNamespace(type="tts_error", recoverable=False, error=RuntimeError("x"))
    session.emit("error", SimpleNamespace(error=speech))
    await asyncio.sleep(0.01)
    assert session.said == []
    assert session.shut == 0
    fatal = SimpleNamespace(type="llm_error", recoverable=False, error=RuntimeError("x"))
    session.emit("error", SimpleNamespace(error=fatal))
    await asyncio.sleep(0.01)
    assert session.said == [FAILURE_LINE]
    assert session.shut == 1


async def test_the_close_ends_the_call_with_the_transcript():
    session = Live()
    ended: list[CallEnd] = []
    call = CallHooks(hooks=on_end(ended.append))
    attach_call(cast(Any, session), call)
    session.emit("close", SimpleNamespace(reason=SimpleNamespace(value="participant_disconnected")))
    await asyncio.sleep(0.01)
    assert ended[0].reason == "caller_hung_up"
    assert ended[0].transcript == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


async def test_open_call_gives_the_audio_back_when_the_greeting_fails():
    session = Live()

    def broken_say(text: str, *, allow_interruptions: object, **_options: object) -> Handle:
        raise RuntimeError("no voice")

    session.say = broken_say  # ty: ignore[invalid-assignment]
    with pytest.raises(RuntimeError, match="no voice"):
        await open_call(cast(Any, session), Greeting(text="Hello."), CallHooks(CallState(Customer)))
    assert session.audio == [False, True]


def test_attach_call_binds_a_call_once():
    session = Live()
    call = CallHooks()
    attach_call(cast(Any, session), call)
    attach_call(cast(Any, Live()), call)
    assert len(session.handlers["error"]) == 1


async def test_the_models_close_after_the_calls_end_has_decided_the_held_saves():
    class Router(Scripted):
        """A model whose request fails once it is closed, as a real client does."""

        async def stream(self, request):
            await asyncio.sleep(0.01)
            if self.closed:
                raise RuntimeError("Cannot send a request, as the client has been closed.")
            yield TextDelta("ok")

    model = Router()
    state = CallState(Customer)
    saved: list[object] = []

    async def decision() -> dict[str, str]:
        async for _ in model.stream(ModelRequest(messages=[])):
            pass
        return {"name": "Ana"}

    state.hold("greet", HeldSave(saved.append, lambda: True, decision))
    call = CallHooks(state)
    agent = UnmuteAgent(AgentSpec("Hi.", model=model), state=state, call=call)
    agent._held_session = cast(Any, Handle())
    ending = asyncio.ensure_future(call.end("ended"))
    agent._closed(cast(Any, None))
    await agent._guard.wait()
    assert saved == [{"name": "Ana"}]
    assert model.closed == 1
    assert ending.done()


async def test_only_the_trace_guard_raises_a_close_failure_into_the_shutdown() -> None:
    async def fail() -> None:
        raise RuntimeError("flush broke")

    task = asyncio.create_task(fail())
    quiet, loud = CloseGuard(lambda: [task]), CloseGuard(lambda: [task], reraise=True)
    quiet.closed()
    loud.closed()
    await quiet.wait()
    with pytest.raises(RuntimeError, match="flush broke"):
        await loud.wait()


async def test_open_call_returns_none_when_the_session_closed_during_the_checks():
    hooks = Hooks()
    session = AgentSession()

    @hooks.before_call
    async def slow_check() -> None:
        await asyncio.sleep(0.01)
        await session.aclose()

    await session.start(UnmuteAgent(AgentSpec("Help.")))
    assert await open_call(session, Greeting(text="Hello."), CallHooks(hooks=hooks)) is None
