"""``voicemail_tool``: the model calls it at a voicemail, and the call ends after its message."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
)
from pipecat.services.openai.realtime import events

from tests.pipecat import test_realtime
from tests.pipecat.test_tools import Params
from unmute_core.call import AGENT_ENDED, CallHooks
from unmute_core.pipecat import _voicemail as voicemail
from unmute_core.pipecat import voicemail_tool
from unmute_core.telephony import VOICEMAIL_MESSAGE


@pytest.fixture(autouse=True)
def no_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voicemail, "VOICEMAIL_SETTLE_SECONDS", 0.0)


class Line(Params):
    """Tool call parameters with a pipeline that queues frames, and a bot that can stop."""

    def __init__(self) -> None:
        super().__init__()
        self.queued: list[Any] = []
        self.handlers: list[Any] = []
        self.order: list[str] = []

    def add_event_handler(self, name: str, handler: Any) -> None:
        self.handlers.append(handler)

    async def queue_frame(self, frame: Any) -> None:
        self.order.append("queued")
        self.queued.append(frame)

    async def end(self) -> None:
        self.order.append("ended")
        self.ended = True

    async def bot_stopped(self) -> None:
        for handler in self.handlers:
            await handler(self, BotStoppedSpeakingFrame())


class Idle:
    """The idle observer's controller, which records what holds it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.controller = SimpleNamespace(
            suspend=lambda reason: self.calls.append(("suspend", reason)),
            resume=lambda reason: self.calls.append(("resume", reason)),
        )


def test_the_tool_has_the_reference_name_description_and_no_parameters() -> None:
    tool = voicemail_tool()
    assert tool.name == "detected_answering_machine"
    assert tool.description == (
        "Call this if you detect a voicemail system, after hearing the voicemail greeting."
    )
    assert tool.properties == {} and tool.required == []


async def test_a_hangup_answers_with_no_reply_and_ends_the_call_noted_as_the_agents() -> None:
    call, idle, line = CallHooks(), Idle(), Line()
    tool = voicemail_tool(call=call, idle=cast(Any, idle))
    assert tool.handler is not None
    await tool.handler(cast(Any, line))
    [(result, properties)] = line.results
    assert result == {"status": "completed"} and properties.run_llm is False
    assert line.queued == []
    assert line.ended
    assert idle.calls == [("suspend", "voicemail"), ("resume", "voicemail")]
    assert (await call.end("ended")).reason == AGENT_ENDED


async def test_leaving_a_message_has_the_model_say_it_and_waits_until_it_has_played() -> None:
    call, line = CallHooks(), Line()
    tool = voicemail_tool(on_voicemail="leave_message", message="Call me back.", call=call)
    assert tool.handler is not None
    running = asyncio.ensure_future(tool.handler(cast(Any, line)))
    async with asyncio.timeout(1):
        while not line.queued:
            await asyncio.sleep(0)
    await asyncio.sleep(0.01)
    assert not line.ended
    [frame] = line.queued
    assert isinstance(frame, LLMMessagesAppendFrame) and frame.run_llm
    assert frame.messages == [{"role": "system", "content": "Call me back."}]
    await line.bot_stopped()
    await running
    assert line.order == ["queued", "ended"]
    assert (await call.end("ended")).reason == AGENT_ENDED


async def test_the_default_message_is_the_voicemail_message() -> None:
    line = Line()
    tool = voicemail_tool(on_voicemail="leave_message")
    assert tool.handler is not None
    running = asyncio.ensure_future(tool.handler(cast(Any, line)))
    async with asyncio.timeout(1):
        while not line.queued:
            await asyncio.sleep(0)
    await line.bot_stopped()
    await running
    assert line.queued[0].messages[0]["content"] == VOICEMAIL_MESSAGE


async def test_a_message_that_cannot_be_said_is_logged_and_the_call_still_ends(
    caplog: pytest.LogCaptureFixture,
) -> None:
    line, idle = Line(), Idle()

    async def refuse(frame: Any) -> None:
        raise RuntimeError("the pipeline is gone")

    line.queue_frame = refuse  # ty: ignore[invalid-assignment]
    tool = voicemail_tool(on_voicemail="leave_message", idle=cast(Any, idle))
    assert tool.handler is not None
    await tool.handler(cast(Any, line))
    assert line.ended
    assert "the voicemail message could not be left" in caplog.text
    assert idle.calls[-1] == ("resume", "voicemail")


async def test_a_realtime_model_says_the_message_itself_and_is_sent_no_result() -> None:
    llm = test_realtime.realtime()
    line = Line()
    line.llm = llm  # ty: ignore[invalid-assignment]
    tool = voicemail_tool(on_voicemail="leave_message", message="Call me back.")
    assert tool.handler is not None
    running = asyncio.ensure_future(tool.handler(cast(Any, line)))
    async with asyncio.timeout(1):
        while not test_realtime.sent(llm):
            await asyncio.sleep(0)
    [reply] = test_realtime.sent(llm)
    assert isinstance(reply, events.ResponseCreateEvent)
    assert reply.response is not None and "Call me back." in (reply.response.instructions or "")
    assert "call_1" in llm._completed_tool_calls
    assert not line.ended
    await llm.push_frame(LLMFullResponseStartFrame())
    await llm.push_frame(LLMFullResponseEndFrame())
    await running
    assert line.ended
    assert line.queued == []
