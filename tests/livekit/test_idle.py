"""``idle_nudges`` feeds a LiveKit session's events to the controller and speaks its lines."""

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

from livekit.agents import AgentSession, AgentStateChangedEvent, UserInputTranscribedEvent

from unmute_core.idle import IdleNudges
from unmute_core.livekit import idle_nudges

CONFIG = IdleNudges(
    first_nudge_text="Are you still there?",
    second_nudge_text="Still here when you are.",
    final_hangup_text="Goodbye.",
)


class Said:
    """A spoken line that has already played."""

    interrupted = False

    def __await__(self) -> Any:
        yield from asyncio.sleep(0).__await__()
        return self


class Session:
    """The events ``idle_nudges`` reads, and the say and shutdown it calls."""

    llm = None
    tts = None
    current_agent = SimpleNamespace(llm=None, tts=None)

    def __init__(self) -> None:
        self.handlers: dict[str, list[Callable[[Any], object]]] = {}
        self.said: list[str] = []
        self.closed = False

    def on(self, event: str, handler: Callable[[Any], object]) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, data: Any) -> None:
        for handler in self.handlers.get(event, []):
            handler(data)

    def say(self, text: str, **_options: object) -> Said:
        self.said.append(text)
        return Said()

    def shutdown(self) -> None:
        self.closed = True


async def _no_wait(_seconds: float) -> None:
    await asyncio.sleep(0)


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def test_a_quiet_caller_hears_the_three_lines_then_the_session_closes() -> None:
    session = Session()
    controller = idle_nudges(cast(AgentSession, cast(Any, session)), CONFIG)
    controller._sleep = _no_wait

    session.emit(
        "agent_state_changed", AgentStateChangedEvent(old_state="speaking", new_state="listening")
    )
    await _settle()

    assert session.said == ["Are you still there?", "Still here when you are.", "Goodbye."]
    assert session.closed


async def test_the_callers_words_start_the_wait_over() -> None:
    session = Session()
    gate = asyncio.Event()
    controller = idle_nudges(cast(AgentSession, cast(Any, session)), CONFIG)

    async def held(_seconds: float) -> None:
        await gate.wait()

    controller._sleep = held
    session.emit(
        "agent_state_changed", AgentStateChangedEvent(old_state="speaking", new_state="listening")
    )
    await _settle()
    session.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="hang on", is_final=False)
    )
    gate.set()
    await _settle()

    assert session.said == []
    session.emit("close", None)
