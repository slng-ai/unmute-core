"""Ending beside another tool, protected lines, the greeting's hold and the plan's lookups."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from livekit.agents import NOT_GIVEN, llm
from livekit.agents.voice import events

from tests.livekit.test_agent_lifecycle import UnmuteAgent
from tests.livekit.test_call import Customer, Live
from tests.outputs import AnyResult
from tests.tasks.job import Canned, job, plan, state
from unmute_core.agent import AgentSpec
from unmute_core.call import CallHooks
from unmute_core.greeting import Greeting
from unmute_core.livekit import SipLine, open_call
from unmute_core.livekit._calls import announce, end_call_now
from unmute_core.state import CallState
from unmute_core.tools import ToolSpec


class Speaker:
    """A session that records what each line was told about interruptions."""

    llm = None
    tts = None
    current_agent = SimpleNamespace(llm=None, tts=None)

    def __init__(self) -> None:
        self.allowed: list[object] = []

    def say(self, text: str, *, allow_interruptions: object, **_: object) -> Any:
        self.allowed.append(allow_interruptions)
        return SimpleNamespace(wait_for_playout=_played)


async def _played() -> None:
    return None


def _tool(*, wait: bool) -> ToolSpec:
    return ToolSpec(
        "check",
        "Check.",
        {"type": "object", "properties": {}},
        AnyResult,
        Canned({}),
        announce_wait=wait,
    )


async def test_a_line_the_tool_waits_for_cannot_be_cut_off_and_another_can() -> None:
    session = Speaker()
    await cast(Any, announce(cast(Any, session), _tool(wait=True), "One moment."))
    assert announce(cast(Any, session), _tool(wait=False), "One moment.") is None
    assert session.allowed == [False, NOT_GIVEN]


async def test_a_sip_lines_own_line_cannot_be_cut_off() -> None:
    session = Speaker()
    await SipLine(cast(Any, session)).say("Connecting you.")
    assert session.allowed == [False]


def test_ending_the_call_cancels_the_reply_to_the_tools_beside_the_ending_one() -> None:
    handlers: list[Any] = []
    shut: list[str] = []
    session = SimpleNamespace(
        once=lambda name, handler: handlers.append((name, handler)),
        shutdown=lambda: shut.append("shutdown"),
    )
    end_call_now(cast(Any, session))
    assert shut == ["shutdown"]
    ((name, handler),) = handlers
    assert name == "function_tools_executed"
    output = llm.FunctionCallOutput(call_id="c1", name="a", output="x", is_error=False)
    event = events.FunctionToolsExecutedEvent(
        function_calls=[llm.FunctionCall(call_id="c1", name="a", arguments="{}")],
        function_call_outputs=[output],
    )
    assert event.has_tool_reply
    handler(event)
    assert not event.has_tool_reply


async def test_a_greeting_that_never_finishes_gives_the_audio_back_after_the_hold() -> None:
    session = Live()
    call = CallHooks(CallState(Customer))
    await open_call(
        cast(Any, session), Greeting(text="Hello.", interruptible=False), call, max_hold_s=0.02
    )
    assert session.audio == [False]
    await asyncio.sleep(0.06)
    assert session.audio == [False, True]


async def test_a_caller_who_speaks_first_is_heard() -> None:
    session = Live()
    call = CallHooks(CallState(Customer))
    assert await open_call(cast(Any, session), Greeting(speaks_first="user"), call) is None
    assert session.audio == [False, True]


async def test_a_greeting_that_ends_in_time_cancels_the_hold() -> None:
    session = Live()
    call = CallHooks(CallState(Customer))
    await open_call(
        cast(Any, session), Greeting(text="Hello.", interruptible=False), call, max_hold_s=0.02
    )
    session.handle.finish()
    await asyncio.sleep(0.06)
    assert session.audio == [False, True]


def test_a_task_reading_a_tool_fails_at_build_when_the_call_has_no_prefetch() -> None:
    payload = job()
    payload["tasks"][1]["inputs"] = [{"kind": "tool", "name": "load_customer"}]
    tasks = plan(payload)
    with pytest.raises(ValueError, match="'load_customer', which no prefetch Lookup runs"):
        UnmuteAgent(
            AgentSpec("Orders.", tasks=tasks), state=state(tasks), call=CallHooks(state(tasks))
        )


def test_a_task_reading_a_tool_builds_without_the_calls_hooks() -> None:
    payload = job()
    payload["tasks"][1]["inputs"] = [{"kind": "tool", "name": "load_customer"}]
    tasks = plan(payload)
    UnmuteAgent(AgentSpec("Orders.", tasks=tasks), state=state(tasks))
