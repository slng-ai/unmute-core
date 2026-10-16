"""Live realtime checks against OpenAI. Skipped unless you set their variables.

Each one opens a realtime session, has the model say a fixed greeting, then asks a
question it answers with a tool. The caller types: no audio goes in. The tool's announce
line is not said: a realtime model words its own. Run them with::

    OPENAI_API_KEY=... UNMUTE_LIVE_REALTIME=1 uv run pytest -m live tests/live -k realtime -s
"""

import asyncio
import os
from typing import Any

import pytest

pytest.importorskip("livekit.agents")

from tests.conformance.livekit_driver import Lines, Silence
from tests.outputs import AnyResult
from tests.tasks.job import State
from unmute_core.agent import AgentSpec
from unmute_core.call import CallHooks
from unmute_core.greeting import Greeting
from unmute_core.state import CallState
from unmute_core.tools import ToolRegistry, local_tool

pytestmark = pytest.mark.live

GREETING = Greeting(text="Hello, this is the corner shop.")
PROMPT = "You answer calls for a corner shop. Use the closing_time tool for opening hours."
QUESTION = "What time do you close today?"


def need() -> str:
    """Return the OpenAI key, or skip the test when it is unset."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or os.environ.get("UNMUTE_LIVE_REALTIME") != "1":
        pytest.skip("set OPENAI_API_KEY and UNMUTE_LIVE_REALTIME=1 to run this live test")
    if not key.startswith("sk-"):
        pytest.fail("OPENAI_API_KEY is not an OpenAI key: one starts with sk-", pytrace=False)
    return key


def answered(lines: list[str]) -> bool:
    """Return True when a line after the greeting gives the closing time."""
    return any("6" in line or "six" in line.lower() for line in lines[1:])


def tools(calls: list[str]) -> ToolRegistry:
    """The one tool the model is asked to call."""

    def closing_time() -> dict[str, str]:
        """Return today's closing time."""
        calls.append("closing_time")
        return {"closes": "18:00"}

    return ToolRegistry([local_tool(closing_time, output=AnyResult, announce="Let me check.")])


async def test_livekit_realtime_greets_and_calls_a_tool() -> None:
    from livekit.agents import AgentSession
    from livekit.agents.llm._realtime.openai import RealtimeModel

    from unmute_core.livekit import UnmuteAgent, open_call

    key = need()
    calls: list[str] = []
    state = CallState(State)
    call = CallHooks(state)
    lines = Lines()
    model = RealtimeModel(api_key=key)
    try:
        async with AgentSession(llm=model) as session:
            session.output.transcription = lines
            session.output.audio = Silence()
            await session.start(
                UnmuteAgent(AgentSpec(PROMPT, tools=tools(calls)), state=state, call=call)
            )
            handle = await open_call(session, GREETING, call)
            assert handle is not None
            await handle.wait_for_playout()
            await session.run(user_input=QUESTION)
    finally:
        await model.aclose()
    print(lines.lines)
    assert "corner shop" in lines.lines[0].lower()
    assert calls == ["closing_time"]
    assert answered(lines.lines)


async def test_pipecat_realtime_greets_and_calls_a_tool() -> None:
    from pipecat.frames.frames import EndFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    from pipecat.services.openai.realtime import events
    from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
    from pipecat.workers.runner import WorkerRunner

    from tests.conformance.pipecat_driver import Speaker
    from unmute_core.pipecat import UnmuteFlows, open_call

    key = need()
    calls: list[str] = []
    state = CallState(State)
    call = CallHooks(state)
    llm = OpenAIRealtimeLLMService(api_key=key)
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)
    flows = UnmuteFlows(
        AgentSpec(PROMPT, tools=tools(calls)),
        llm=llm,
        context=context,
        aggregators=aggregators,
        call=call,
    )
    flows.install()
    speaker = Speaker(realtime=True)
    worker = PipelineWorker(Pipeline([aggregators.user(), llm, speaker, aggregators.assistant()]))
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        await open_call(worker, GREETING, call, context=context, llm=llm)
        async with asyncio.timeout(30):
            while not speaker.lines:
                await asyncio.sleep(0.1)
        question: Any = events.ConversationItem(
            type="message",
            role="user",
            content=[events.ItemContent(type="input_text", text=QUESTION)],
        )
        await llm.send_client_event(events.ConversationItemCreateEvent(item=question))
        await llm.send_client_event(events.ResponseCreateEvent())
        async with asyncio.timeout(30):
            while not answered(speaker.lines):
                await asyncio.sleep(0.1)
    finally:
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(running, 10)
    print(speaker.lines)
    assert "corner shop" in speaker.lines[0].lower()
    assert calls == ["closing_time"]
