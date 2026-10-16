"""The fake Realtime server talks to LiveKit's OpenAI engine and to Pipecat's service."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents import Agent, AgentSession, function_tool
from livekit.agents.llm._realtime.openai import RealtimeModel
from livekit.agents.voice.io import TextOutput
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import EndFrame, LLMRunFrame, LLMTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.workers.runner import WorkerRunner

from tests.realtime.fake_server import FakeRealtimeServer
from unmute_core.model import ModelEvent, ModelRequest, TextDelta, ToolCall, Usage
from unmute_core.realtime import say_line

INSTRUCTIONS = "You help."


class Script:
    """A ``ModelCall`` that answers each request with the next scripted list of events."""

    def __init__(self, *replies: list[ModelEvent]) -> None:
        self.replies = list(replies)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Stream the next reply."""
        self.requests.append(request)
        for event in self.replies.pop(0):
            yield event


class Lines(TextOutput):
    """Collects the text the session would show or speak."""

    def __init__(self) -> None:
        super().__init__(label="test", next_in_chain=None)
        self.text = ""

    async def capture_text(self, text: str) -> None:
        """Add text to what was said."""
        self.text += text

    def flush(self) -> None:
        """Do nothing: the text is kept whole."""


def _script() -> Script:
    return Script(
        [ToolCall("call_1", "lookup", '{"order": "A-12"}')],
        [TextDelta("It is "), TextDelta("open."), Usage(7, 3)],
    )


async def _until(check: Any) -> None:
    async with asyncio.timeout(5):
        while not check():
            await asyncio.sleep(0.01)


async def test_livekit_engine_runs_a_tool_and_speaks_the_answer() -> None:
    calls: list[str] = []

    @function_tool
    async def lookup(order: str) -> str:
        """Look up an order."""
        calls.append(order)
        return "open"

    model = _script()
    lines = Lines()
    async with FakeRealtimeServer(model) as server:
        realtime = RealtimeModel(
            base_url=server.http_url, api_key="test", modalities=["audio"], turn_detection=None
        )
        async with AgentSession(llm=realtime) as session:
            session.output.transcription = lines
            await session.start(Agent(instructions=INSTRUCTIONS, tools=[lookup]))
            await server.caller_says("hello")
            await _until(lambda: "It is open." in lines.text)
            await server.wait_idle()

            assert calls == ["A-12"]
            first, second = server.requests
            assert first.messages[0] == {"role": "system", "content": INSTRUCTIONS}
            assert first.messages[1] == {"role": "user", "content": "hello"}
            assert [tool["function"]["name"] for tool in first.tools] == ["lookup"]
            assert [m["role"] for m in second.messages] == ["system", "user", "assistant", "tool"]
            assert second.messages[2]["tool_calls"][0]["id"] == "call_1"
            assert second.messages[3] == {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "open",
            }
            texts = [m.text_content for m in session.history.messages()]
            assert "hello" in texts
            assert "It is open." in texts

            await session.generate_reply(instructions=say_line("Hi there"))
            await _until(lambda: "Hi there" in lines.text)
            assert len(server.requests) == 2
        await realtime.aclose()


class _Capture(FrameProcessor):
    """Keeps the text frames that pass."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
        """Keep an ``LLMTextFrame``'s text, and pass every frame on."""
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self.texts.append(frame.text)
        await self.push_frame(frame, direction)


async def test_pipecat_service_runs_a_tool_and_streams_the_answer() -> None:
    calls: list[dict[str, Any]] = []
    model = Script(
        [TextDelta("One moment."), ToolCall("call_1", "lookup", '{"order": "A-12"}')],
        [TextDelta("It is open.")],
        [TextDelta("You are welcome.")],
    )

    async def lookup(params: FunctionCallParams) -> None:
        calls.append(dict(params.arguments))
        await params.result_callback("open")

    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="lookup",
                description="Look up an order.",
                properties={"order": {"type": "string"}},
                required=["order"],
            )
        ]
    )
    async with FakeRealtimeServer(model) as server:
        service = OpenAIRealtimeLLMService(
            api_key="test",
            base_url=server.ws_url,
            settings=OpenAIRealtimeLLMService.Settings(
                session_properties=events.SessionProperties(
                    output_modalities=["audio"],
                    audio=events.AudioConfiguration(input=events.AudioInput(turn_detection=False)),
                ),
                system_instruction=INSTRUCTIONS,
            ),
        )
        service.register_function("lookup", lookup)
        context = LLMContext(tools=tools)
        aggregators = LLMContextAggregatorPair(context)
        capture = _Capture()
        worker = PipelineWorker(
            Pipeline([aggregators.user(), service, capture, aggregators.assistant()])
        )
        running = asyncio.create_task(WorkerRunner(handle_sigint=False).run(worker))
        try:
            await worker.queue_frame(LLMRunFrame())
            await server.wait_for(lambda: len(server.requests) == 2)
            await server.wait_idle()
            await _until(lambda: "It is open." in "".join(capture.texts))

            assert calls == [{"order": "A-12"}]
            first, second = server.requests
            assert first.messages[0] == {"role": "system", "content": INSTRUCTIONS}
            assert [tool["function"]["name"] for tool in first.tools] == ["lookup"]
            assert [m["role"] for m in second.messages][-2:] == ["assistant", "tool"]
            assert "One moment." in "".join(capture.texts)

            await service.send_client_event(
                events.ResponseCreateEvent(
                    response=events.ResponseProperties(instructions=say_line("Hi there"))
                )
            )
            await _until(lambda: "Hi there" in "".join(capture.texts))
            assert len(server.requests) == 2

            await server.caller_says("thanks")
            await _until(lambda: "You are welcome." in "".join(capture.texts))
            assert server.requests[-1].messages[-1] == {"role": "user", "content": "thanks"}
        finally:
            await worker.queue_frame(EndFrame())
            await asyncio.wait_for(running, 5)
