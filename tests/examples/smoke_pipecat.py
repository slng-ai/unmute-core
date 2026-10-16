"""Run one text turn through a Pipecat example's own ``bot()``.

``scripts/check-examples.sh`` runs it inside the example's venv, from the project folder::

    .venv/bin/python -I ../../../tests/examples/smoke_pipecat.py tools [--live]

``bot()`` runs as written, on a fake transport. The transport says a caller connected,
then types one caller line. Speech services pass frames through, so the call is text
only. Without ``--live`` a scripted model answers and no request leaves the machine.
With ``--live`` the example's own model answers.
"""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path[:0] = [os.getcwd(), str(Path(__file__).resolve().parents[2])]

import httpx2  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    EndWorkerFrame,
    Frame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    StartFrame,
    TTSSpeakFrame,
)
from pipecat.processors.filters.identity_filter import IdentityFilter  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.runner.types import RunnerArguments  # noqa: E402
from pipecat.services.openai.llm import OpenAILLMService  # noqa: E402
from pipecat.transports.base_transport import BaseTransport  # noqa: E402

from tests.examples.offline import (  # noqa: E402
    CALLER,
    PROMPTS,
    REPLY,
    check,
    go_offline,
    system_text,
)

SPEECH = ("OpenAISTTService", "OpenAITTSService", "SlngSTTService", "SlngTTSService")


def passthrough(*args: object, **kwargs: object) -> FrameProcessor:
    """Stand in for a speech service: frames pass through untouched."""
    return IdentityFilter()


def openai_reply(request: httpx2.Request) -> httpx2.Response:
    """Keep the request's system text, then answer with the reply, streamed as OpenAI does."""
    PROMPTS.append(system_text(json.loads(request.content)["messages"]))
    chunks = [{"role": "assistant", "content": REPLY}, {}]
    events = [
        {
            "id": "smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "scripted",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None if delta else "stop"}],
        }
        for delta in chunks
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})


def scripted_client(self: OpenAILLMService, *args: object, **kwargs: object) -> AsyncOpenAI:
    """Give the example's real OpenAI service a client that never leaves the machine."""
    return AsyncOpenAI(
        api_key="smoke",
        base_url="https://openai.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(openai_reply)),
    )


def scripted_llm() -> OpenAILLMService:
    """Stand in for a model built by a helper, like the salon's Gemini on Vertex."""
    return OpenAILLMService(settings=OpenAILLMService.Settings(model="scripted"))


class CallerInput(FrameProcessor):
    """Says a caller connected when the pipeline starts, then types the caller line."""

    def __init__(self, transport: "FakeTransport") -> None:
        """Keep the transport whose event this fires."""
        super().__init__()
        self.transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame on. After the start frame, connect and speak."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.transport.connected()
            caller = LLMMessagesAppendFrame([{"role": "user", "content": CALLER}], run_llm=True)
            await self.push_frame(caller)


class AgentOutput(FrameProcessor):
    """Keeps what the agent says, and ends the call after its first full reply.

    A reply is the model's text, or a line code speaks after a tool, like the opening
    line of a step that listens. A line spoken before any tool, like a greeting, is not.
    """

    def __init__(self) -> None:
        """Start with nothing said."""
        super().__init__()
        self.replies: list[str] = []
        self._text = ""
        self._tool_answered = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame on. Collect reply text and end once a reply is complete."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self._text += frame.text
        elif isinstance(frame, FunctionCallResultFrame):
            self._tool_answered = True
        elif isinstance(frame, TTSSpeakFrame) and self._tool_answered:
            self._text += frame.text
            await self._reply()
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._reply()

    async def _reply(self) -> None:
        """Keep the reply so far and end the call, when there is one."""
        if self._text.strip():
            self.replies.append(self._text)
            self._text = ""
            await self.push_frame(EndWorkerFrame(reason="smoke test done"))


class FakeTransport(BaseTransport):
    """A transport with one caller who types instead of speaking."""

    def __init__(self) -> None:
        """Register the events a web transport fires."""
        super().__init__(name="smoke")
        self.caller = CallerInput(self)
        self.agent = AgentOutput()
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    async def connected(self) -> None:
        """Fire the caller-connected event, as a web transport does."""
        await self._call_event_handler("on_client_connected", "smoke-client")

    def input(self) -> FrameProcessor:
        """The caller's side."""
        return self.caller

    def output(self) -> FrameProcessor:
        """The agent's side."""
        return self.agent


async def smoke(name: str, live: bool) -> str:
    """Run the module's ``bot()`` until the agent answers the caller. Return the reply."""
    if not live:
        go_offline()
    module: Any = importlib.import_module(name)
    transport = FakeTransport()

    async def create_transport(*args: object, **kwargs: object) -> FakeTransport:
        """Hand the bot the fake transport."""
        return transport

    shared = [module, *(m for n, m in list(sys.modules.items()) if n.startswith("voice_stack"))]
    for loaded in shared:
        if hasattr(loaded, "create_transport"):
            setattr(loaded, "create_transport", create_transport)  # noqa: B010
        for service in SPEECH:
            if hasattr(loaded, service):
                setattr(loaded, service, passthrough)
        if not live and hasattr(loaded, "build_llm"):
            setattr(loaded, "build_llm", lambda *args, **kwargs: scripted_llm())  # noqa: B010
    if not live:
        OpenAILLMService.create_client = scripted_client  # ty: ignore[invalid-assignment]
    await module.bot(RunnerArguments())
    return check(name, transport.agent.replies, live=live)


def main() -> None:
    """Run the smoke for the module named on the command line."""
    name, *flags = sys.argv[1:]
    reply = asyncio.run(asyncio.wait_for(smoke(name, "--live" in flags), timeout=30))
    print(f"ok: {name} answered: {reply[:80]!r}")


if __name__ == "__main__":
    main()
