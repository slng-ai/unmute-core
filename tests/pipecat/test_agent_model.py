"""Each agent of a call has its own model and voice on Pipecat.

That the next model request goes to the next agent's model, and that its first reply offers
no handoff tool, is in tests/conformance (``handoff_own_model.json``). These cases cover the
Pipecat side: the voice switch on a real pipeline, its order, and what ``hand_off`` refuses.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any, cast

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    ManuallySwitchServiceFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.service_switcher import ServiceSwitcher
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.workers.runner import WorkerRunner

from tests.pipecat.harness import Harness
from tests.tasks.job import Canned, specialist_plan, state
from unmute_core.agent import AgentSpec
from unmute_core.model import ModelEvent, ModelRequest, TextDelta, ToolCall
from unmute_core.pipecat import ModelCallLLMService, UnmuteFlows, VoiceSwitcher

NOTES_PROMPT = "You take notes."
FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}


class Turns:
    """A model call that answers each request with its next scripted events."""

    def __init__(self, name: str, *replies: list[ModelEvent]) -> None:
        self.model = name
        self.replies = list(replies)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Answer with the next reply's events."""
        self.requests.append(request)
        for event in self.replies.pop(0) if self.replies else []:
            yield event


class Voice(FrameProcessor):
    """Stands in for a TTS service: keeps each line it is given and plays it at once."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.lines: list[str] = []
        self._reply: str | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Keep the replies and the fixed lines this voice speaks."""
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._reply = ""
        elif isinstance(frame, LLMTextFrame) and self._reply is not None:
            self._reply += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._reply:
                self.lines.append(self._reply)
            self._reply = None
        elif isinstance(frame, TTSSpeakFrame):
            self.lines.append(frame.text)
            await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            return
        await self.push_frame(frame, direction)


async def _until(check: Callable[[], bool]) -> None:
    """Wait until ``check`` holds, for at most three seconds."""
    async with asyncio.timeout(3):
        while not check():
            await asyncio.sleep(0)


async def test_the_next_agent_says_the_line_and_replies_in_its_own_voice_and_model() -> None:
    """The handoff's pipeline flushes cross the switcher at once: the test waits 3 seconds."""
    first = Turns("gpt-4o-mini", [ToolCall("call-1", "to_notes", "{}")])
    notes = Turns("gpt-4o", [TextDelta("Notes desk.")])
    front, back = Voice("front"), Voice("back")
    voices = VoiceSwitcher([front, back])
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)
    llm = ModelCallLLMService(
        first,
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini", system_instruction="You help with orders."
        ),
    )

    async def to_notes(params: FunctionCallParams) -> None:
        """Hand the call to the notes desk, on its own model and voice."""
        await flows.hand_off(
            params,
            AgentSpec(NOTES_PROMPT, tasks=specialist_plan(), model=notes),
            announce="One moment.",
            tts=back,
        )

    flows = UnmuteFlows(
        AgentSpec("You help with orders."),
        llm=llm,
        context=context,
        aggregators=aggregators,
        state=state(specialist_plan()),
        native_tools=[FunctionSchema("to_notes", "To notes.", {}, [], handler=to_notes)],
        tts=voices,
    )
    flows.install()
    worker = PipelineWorker(Pipeline([aggregators.user(), llm, voices, aggregators.assistant()]))
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        caller = [{"role": "user", "content": "I need a note taken."}]
        await worker.queue_frame(LLMMessagesAppendFrame(cast(Any, caller), run_llm=True))
        await _until(lambda: back.lines == ["One moment.", "Notes desk."])
    finally:
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(running, 5)

    assert front.lines == []
    assert len(first.requests) == 1
    assert [request.model for request in notes.requests] == ["gpt-4o"]
    assert llm.model_call is notes
    assert flows.model is notes


class Speaker(TTSService):
    """Pipecat's TTS service with silence for a voice, after a provider's delay."""

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            sample_rate=16000,
            push_start_frame=True,
            push_stop_frames=True,
            stop_frame_timeout_s=0.05,
            settings=TTSSettings(model=None, voice=None, language=None),
        )
        self.lines: list[str] = []

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None]:
        self.lines.append(text.strip())
        await asyncio.sleep(0.05)
        yield TTSAudioRawFrame(b"\x00" * 640, 16000, 1, context_id=context_id)


class Output(FrameProcessor):
    """Plays the speech that leaves the voices, as an output transport does."""

    def __init__(self) -> None:
        super().__init__()
        self.heard: list[str] = []
        self._speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            if not self._speaking:
                self._speaking = True
                await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            return
        if isinstance(frame, TTSTextFrame):
            self.heard.append(frame.text)
        await self.push_frame(frame, direction)
        if isinstance(frame, TTSStoppedFrame) and self._speaking:
            self._speaking = False
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)


SPANISH_LINES = ("Un momento.", "Un segundo.")


async def test_the_handoff_line_is_said_in_the_next_voice_after_the_old_reply_and_rotates() -> None:
    model = Turns(
        "gpt-4o-mini",
        [TextDelta("Sure."), ToolCall("call-1", "to_spanish", "{}")],
        [TextDelta("Hola.")],
        [ToolCall("call-2", "to_english", "{}")],
        [TextDelta("Hello.")],
        [ToolCall("call-3", "to_spanish", "{}")],
        [TextDelta("Hola.")],
    )
    english, spanish, output = Speaker("english"), Speaker("spanish"), Output()
    voices = VoiceSwitcher([english, spanish])
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)
    llm = ModelCallLLMService(
        model,
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini", system_instruction="You speak English."
        ),
    )

    async def to_spanish(params: FunctionCallParams) -> None:
        await flows.hand_off(
            params,
            AgentSpec("Hablas español.", handoffs=("to_english",)),
            native_tools=[back],
            announce=SPANISH_LINES,
            tts=spanish,
        )

    async def to_english(params: FunctionCallParams) -> None:
        await flows.hand_off(
            params,
            AgentSpec("You speak English.", handoffs=("to_spanish",)),
            native_tools=[there],
            tts=english,
        )

    there = FunctionSchema("to_spanish", "To Spanish.", {}, [], handler=to_spanish)
    back = FunctionSchema("to_english", "To English.", {}, [], handler=to_english)
    flows = UnmuteFlows(
        AgentSpec("You speak English.", handoffs=("to_spanish",)),
        llm=llm,
        context=context,
        aggregators=aggregators,
        native_tools=[there],
        tts=voices,
    )
    flows.install()
    worker = PipelineWorker(
        Pipeline([aggregators.user(), llm, voices, output, aggregators.assistant()])
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        for caller, heard in [("Español.", 3), ("English.", 4), ("Español otra vez.", 6)]:
            message = [{"role": "user", "content": caller}]
            await worker.queue_frame(LLMMessagesAppendFrame(cast(Any, message), run_llm=True))
            await _until(lambda heard=heard: len(output.heard) == heard)
    finally:
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(running, 5)

    first, hola, second, hola_again = spanish.lines
    assert {first, second} == set(SPANISH_LINES)
    assert (hola, hola_again) == ("Hola.", "Hola.")
    assert english.lines == ["Sure.", "Hello."]
    assert output.heard == ["Sure.", first, "Hola.", "Hello.", second, "Hola."]


def _desk(hand_off: Callable[[UnmuteFlows, FunctionCallParams], Any], **kwargs: Any) -> Harness:
    """A call whose ``to_notes`` runs ``hand_off`` on the call's flows."""
    h: Harness

    async def handler(params: FunctionCallParams) -> None:
        """Hand the call off."""
        await hand_off(h.flows, params)

    tool = FunctionSchema("to_notes", "The to_notes handoff.", {}, [], handler=handler)
    h = Harness(native_tools=[tool], lookup_order=Canned(FOUND), **kwargs)
    return h


async def test_the_voice_switches_before_the_next_agents_first_reply() -> None:
    front, back = FrameProcessor(), FrameProcessor()

    async def hand_off(flows: UnmuteFlows, params: FunctionCallParams) -> None:
        await flows.hand_off(params, AgentSpec(NOTES_PROMPT), tts=back)

    h = _desk(hand_off, tts=VoiceSwitcher([front, back]))
    h.user("I need a note taken.")
    await h.call("to_notes")

    kinds = [type(frame) for frame in h.worker.log if not isinstance(frame, str)]
    assert kinds[-3:] == [ManuallySwitchServiceFrame, LLMUpdateSettingsFrame, LLMRunFrame]
    switch = next(f for f in h.worker.log if isinstance(f, ManuallySwitchServiceFrame))
    assert switch.service is back


async def test_a_handoff_without_a_voice_goes_back_to_the_calls_first_voice() -> None:
    front, back = FrameProcessor(), FrameProcessor()

    async def hand_off(flows: UnmuteFlows, params: FunctionCallParams) -> None:
        await flows.hand_off(params, AgentSpec(NOTES_PROMPT))

    h = _desk(hand_off, tts=VoiceSwitcher([front, back]))
    h.user("I need a note taken.")
    await h.call("to_notes")

    [switch] = [f for f in h.worker.log if isinstance(f, ManuallySwitchServiceFrame)]
    assert switch.service is front


@pytest.mark.parametrize(
    ("spec", "kwargs", "match"),
    [
        (
            AgentSpec(NOTES_PROMPT, model=Turns("gpt-4o")),
            {},
            "needs the call's LLM to be a ModelCallLLMService",
        ),
        (
            AgentSpec(NOTES_PROMPT),
            {"tts": FrameProcessor()},
            "needs that TTS service in the VoiceSwitcher",
        ),
        (
            AgentSpec(NOTES_PROMPT, handoffs=("to_orders",)),
            {},
            "names 'to_orders', which is not one of the new",
        ),
    ],
)
async def test_a_handoff_refuses_what_the_call_cannot_give(
    spec: AgentSpec, kwargs: dict[str, Any], match: str
) -> None:
    async def hand_off(flows: UnmuteFlows, params: FunctionCallParams) -> None:
        await flows.hand_off(params, spec, **kwargs)

    h = _desk(hand_off)
    h.user("I need a note taken.")
    with pytest.raises(ValueError, match=match):
        await h.call("to_notes")


def test_the_calls_voices_need_a_voice_switcher() -> None:
    with pytest.raises(ValueError, match="takes a VoiceSwitcher"):
        Harness(tts=ServiceSwitcher([FrameProcessor(), FrameProcessor()]))


async def test_an_opening_cut_off_before_its_request_leaves_the_handoffs_in() -> None:
    back_tool = FunctionSchema("to_orders", "Back to orders.", {}, [], handler=None)

    async def hand_off(flows: UnmuteFlows, params: FunctionCallParams) -> None:
        await flows.hand_off(
            params,
            AgentSpec(NOTES_PROMPT, tasks=specialist_plan(), handoffs=("to_orders",)),
            native_tools=[back_tool],
        )

    h = _desk(hand_off)
    h.user("I need a note taken.")
    await h.call("to_notes")
    h.user("Actually, call me back.")
    await h.reply("Noted.")

    assert h.tool_names() == ["search_web", "take_note", "to_orders"]
