"""A reply the caller cuts off while its tool still runs.

LiveKit 1.8.6 stopped dropping such a tool's result: once the interruption timeout stops waiting
for the tool, the tool's first update is a placeholder and its result arrives later as a deferred
reply (voice/generation.py ``_release_running_tools``). These tests pin what a library agent
does with that, and that the tools which must not be cut off never reach it.
"""

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from livekit.agents import AgentSession, llm
from livekit.agents.llm._realtime.openai import RealtimeModel
from livekit.agents.voice import SpeechHandle, speech_handle

from tests.conformance.livekit_driver import Silence
from tests.hooks.helpers import on_end
from tests.livekit.fake import ScriptedLLM, call
from tests.livekit.test_agent_lifecycle import Scripted
from tests.livekit.test_trace import Memory
from tests.outputs import AnyResult
from tests.realtime.fake_server import FakeRealtimeServer
from tests.tasks.job import job, plan, registry, state
from unmute_core.agent import AgentSpec
from unmute_core.call import AGENT_ENDED, CallEnd, CallHooks
from unmute_core.livekit import UnmuteAgent
from unmute_core.model import TextDelta, ToolCall
from unmute_core.telemetry import Telemetry
from unmute_core.tools import ToolEffect, ToolInterruption, ToolRegistry, ToolSpec

OPEN = {"type": "object", "properties": {}, "additionalProperties": True}
FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}


class Gate:
    """A tool executor that runs until the test lets it finish. It notes a cancel."""

    def __init__(self, result: object = None) -> None:
        self.result = {"done": True} if result is None else result
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.runs = 0
        self.cancelled = 0

    async def __aenter__(self) -> "Gate":
        """Let a test that fails before it lets the tool go still close its session."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Let the tool go."""
        self.release.set()

    async def run(self, arguments: dict[str, Any], state: object) -> object:
        """Start, then wait for ``release``."""
        self.runs += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return self.result


@pytest.fixture(autouse=True)
def short_interruption_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make LiveKit's interruption timeout (5 s) short enough for a test."""
    monkeypatch.setattr(speech_handle, "INTERRUPTION_TIMEOUT", 0.05)


async def until(done: Callable[[], bool]) -> None:
    """Wait until ``done``, or fail the test after 3 s."""
    async with asyncio.timeout(3):
        while not done():
            await asyncio.sleep(0.01)


def tool_items(session: AgentSession, name: str) -> list[tuple[str, str]]:
    """The calls of a tool in the session's history, and its outputs, in order."""
    return [
        ("call", item.arguments) if isinstance(item, llm.FunctionCall) else ("out", item.output)
        for item in session.history.items
        if isinstance(item, llm.FunctionCall | llm.FunctionCallOutput) and item.name == name
    ]


def replies(session: AgentSession) -> list[str | None]:
    """The agent's replies in the session's history, in order."""
    return [
        item.text_content
        for item in session.history.items
        if isinstance(item, llm.ChatMessage) and item.role == "assistant"
    ]


def slow_tool(
    gate: Gate,
    *,
    interruption: ToolInterruption = "provider_default",
    effect: ToolEffect = "returns_data",
) -> ToolRegistry:
    """A registry of one tool, ``slow``, that runs on ``gate``."""
    return ToolRegistry(
        [
            ToolSpec(
                "slow",
                "A slow tool.",
                OPEN,
                AnyResult,
                gate,
                interruption=interruption,
                effect=effect,
            )
        ]
    )


async def ask(session: AgentSession, gate: Gate) -> SpeechHandle:
    """Send the caller's turn and return its reply, once the reply's tool runs."""
    handle = session.generate_reply(user_input="Go on.")
    async with asyncio.timeout(3):
        await gate.started.wait()
    return handle


async def test_a_tool_cut_off_for_longer_than_the_timeout_answers_the_model_once_it_ends() -> None:
    gate = Gate()
    sink = Memory()
    model = ScriptedLLM(call("slow", "One moment."), "It is done.")
    agent = UnmuteAgent(
        AgentSpec("Help.", tools=slow_tool(gate)),
        telemetry=Telemetry(sink, agent="orders", call_id="RM_1"),
    )
    async with AgentSession(llm=model) as session, gate:
        await session.start(agent)
        reply = await ask(session, gate)
        assert reply.allow_interruptions
        session.interrupt()
        await until(reply.done)
        await until(lambda: len(tool_items(session, "slow")) == 2)
        assert len(model.requests) == 1
        [(_, started), (_, released)] = tool_items(session, "slow")
        assert started == "{}"
        assert "still in progress" in released

        gate.release.set()
        await until(lambda: bool(replies(session)))
        await until(lambda: session.agent_state == "listening")
        assert len(model.requests) == 2
        assert gate.runs == 1
        assert gate.cancelled == 0
        assert tool_items(session, "slow")[-1] == ("out", '{"done": true}')
        assert [text for role, text in model.requests[-1].texts if role == "assistant"] == []
        assert replies(session) == ["It is done."]
    assert agent._trace is not None and agent._trace.closing is not None
    await agent._trace.closing
    [span] = [s for s in sink.spans.get_finished_spans() if s.name == "llm_tool_call"]
    assert (span.attributes or {})["gen_ai.tool.call.result"] == '{"done": true}'


async def test_a_realtime_tool_cut_off_answers_the_model_once_it_ends() -> None:
    gate = Gate()
    model = Scripted([TextDelta("One moment."), ToolCall("c1", "slow", "{}")], [TextDelta("Done.")])
    async with FakeRealtimeServer(model) as server:
        live = RealtimeModel(
            base_url=server.http_url, api_key="test", modalities=["audio"], turn_detection=None
        )
        agent = UnmuteAgent(AgentSpec("Help.", tools=slow_tool(gate)))
        async with AgentSession(llm=live) as session, gate:
            session.output.audio = Silence()
            await session.start(agent)
            await server.caller_says("Go on.")
            async with asyncio.timeout(3):
                await gate.started.wait()
            session.interrupt()
            await server.wait_idle()
            assert len(server.requests) == 1
            await until(lambda: "still in progress" in tool_items(session, "slow")[-1][1])
            gate.release.set()
            await server.wait_for(lambda: len(server.requests) == 2)
            await server.wait_idle()
            assert gate.runs == 1
            assert gate.cancelled == 0
            assert tool_items(session, "slow")[-1] == ("out", '{"done": true}')
        await live.aclose()


async def test_a_continue_tool_keeps_its_reply_from_being_cut_off() -> None:
    gate = Gate()
    model = ScriptedLLM(call("slow", "One moment."), "It is done.")
    agent = UnmuteAgent(AgentSpec("Help.", tools=slow_tool(gate, interruption="continue")))
    async with AgentSession(llm=model) as session, gate:
        await session.start(agent)
        reply = await ask(session, gate)
        assert not reply.allow_interruptions
        with pytest.raises(RuntimeError, match="does not allow interruptions"):
            session.interrupt()
        await asyncio.sleep(0.2)
        assert not reply.interrupted
        assert len(model.requests) == 1
        gate.release.set()
        await until(lambda: len(model.requests) == 2)
        await until(reply.done)
        assert tool_items(session, "slow") == [("call", "{}"), ("out", '{"done": true}')]
        assert gate.cancelled == 0


async def test_an_ending_tool_cut_off_still_ends_the_call_once_with_no_reply() -> None:
    gate = Gate()
    ends: list[CallEnd] = []
    hooks = CallHooks(hooks=on_end(ends.append))
    model = ScriptedLLM(call("slow", "Goodbye."), "Never said.")
    agent = UnmuteAgent(
        AgentSpec("Help.", tools=slow_tool(gate, effect="ends_conversation")), call=hooks
    )
    closed = asyncio.Event()
    async with AgentSession(llm=model) as session, gate:
        session.on("close", lambda _event: closed.set())
        await session.start(agent)
        reply = await ask(session, gate)
        session.interrupt()
        await until(reply.done)
        assert not closed.is_set()
        gate.release.set()
        async with asyncio.timeout(3):
            await closed.wait()
        await until(lambda: bool(ends))
    assert [end.reason for end in ends] == [AGENT_ENDED]
    assert len(model.requests) == 1


async def test_a_delegate_holds_its_reply_and_a_step_tool_cut_off_ends_the_step_once() -> None:
    gate = Gate(FOUND)
    tasks = plan(job(), tools=registry(lookup_order=gate))
    model = ScriptedLLM(call("find_order"), call("lookup_order"), "It is open.", "Never.")
    agent = UnmuteAgent(AgentSpec("Orders.", tasks=tasks), state=state(tasks))
    async with AgentSession(llm=model) as session, gate:
        await session.start(agent)
        reply = await ask(session, gate)
        assert not reply.allow_interruptions
        with pytest.raises(RuntimeError, match="does not allow interruptions"):
            reply.interrupt()
        session.interrupt()
        await asyncio.sleep(0.3)
        assert not reply.interrupted
        assert agent.tasks is not None and agent.tasks.active is not None
        assert "still in progress" in " ".join(o for _, o in tool_items(session, "lookup_order"))
        gate.release.set()
        await until(reply.done)
        await until(lambda: session.agent_state == "listening")
        assert agent.tasks.active is None
        assert agent.state is not None and agent.state.plain("order_id") == "A-12"
        assert gate.runs == 1
        assert len(model.requests) == 3
        main_answer = model.requests[-1]
        assert main_answer.calls == ["find_order"]
        assert "still in progress" not in main_answer.prompt
        assert main_answer.texts == [("user", "Go on.")]


@pytest.mark.parametrize("effect", ["returns_data", "ends_conversation"])
async def test_a_call_asked_to_close_while_a_tool_runs_waits_for_the_tool(
    effect: ToolEffect,
) -> None:
    gate = Gate()
    model = ScriptedLLM(call("slow", "One moment."), "Never said.")
    agent = UnmuteAgent(AgentSpec("Help.", tools=slow_tool(gate, effect=effect)))
    closed = asyncio.Event()
    async with AgentSession(llm=model) as session, gate:
        session.on("close", lambda _event: closed.set())
        await session.start(agent)
        await ask(session, gate)
        session.shutdown(drain=False)
        await asyncio.sleep(0.3)
        assert not closed.is_set()
        gate.release.set()
        async with asyncio.timeout(3):
            await closed.wait()
    assert gate.cancelled == 0
    assert len(model.requests) == 1
