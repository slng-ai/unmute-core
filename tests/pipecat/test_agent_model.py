"""Each agent of a call has its own model, voice and STT on Pipecat.

That the next model request goes to the next agent's model, and that its first reply offers
no handoff tool, is in tests/conformance (``handoff_own_model.json``). These cases cover the
Pipecat side: the voice and STT switch on a real pipeline, its order, and what ``hand_off``
refuses.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
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
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.service_switcher import ServiceSwitcher
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.workers.runner import WorkerRunner

from tests.pipecat.harness import Harness
from tests.tasks.job import Canned, specialist_plan, state
from unmute_core.agent import AgentSpec
from unmute_core.model import ModelEvent, ModelRequest, TextDelta, ToolCall
from unmute_core.pipecat import ModelCallLLMService, STTSwitcher, UnmuteFlows, VoiceSwitcher

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


async def test_the_next_agent_speaks_and_listens_on_its_own_services_after_the_line() -> None:
    """The handoff's pipeline flushes cross the switcher at once: the test waits 3 seconds."""
    first = Turns("gpt-4o-mini", [ToolCall("call-1", "to_notes", "{}")])
    notes = Turns("gpt-4o", [TextDelta("Notes desk.")])
    front, back = Voice("front"), Voice("back")
    voices = VoiceSwitcher([front, back])
    front_stt, back_stt = IdentityFilter(), IdentityFilter()
    stts = STTSwitcher([front_stt, back_stt])
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)
    llm = ModelCallLLMService(
        first,
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini", system_instruction="You help with orders."
        ),
    )

    async def to_notes(params: FunctionCallParams) -> None:
        """Hand the call to the notes desk, on its own model, voice and STT."""
        await flows.hand_off(
            params,
            AgentSpec(NOTES_PROMPT, tasks=specialist_plan(), model=notes),
            announce="One moment.",
            tts=back,
            stt=back_stt,
        )

    flows = UnmuteFlows(
        AgentSpec("You help with orders."),
        llm=llm,
        context=context,
        aggregators=aggregators,
        state=state(specialist_plan()),
        native_tools=[FunctionSchema("to_notes", "To notes.", {}, [], handler=to_notes)],
        tts=voices,
        stt=stts,
    )
    flows.install()
    worker = PipelineWorker(
        Pipeline([stts, aggregators.user(), llm, voices, aggregators.assistant()])
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        caller = [{"role": "user", "content": "I need a note taken."}]
        await worker.queue_frame(LLMMessagesAppendFrame(cast(Any, caller), run_llm=True))
        await _until(lambda: back.lines == ["Notes desk."])
    finally:
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(running, 5)

    assert front.lines == ["One moment."]
    assert len(first.requests) == 1
    assert [request.model for request in notes.requests] == ["gpt-4o"]
    assert llm.model_call is notes
    assert flows.model is notes
    assert stts.strategy.active_service is back_stt


def _desk(hand_off: Callable[[UnmuteFlows, FunctionCallParams], Any], **kwargs: Any) -> Harness:
    """A call whose ``to_notes`` runs ``hand_off`` on the call's flows."""
    h: Harness

    async def handler(params: FunctionCallParams) -> None:
        """Hand the call off."""
        await hand_off(h.flows, params)

    tool = FunctionSchema("to_notes", "The to_notes handoff.", {}, [], handler=handler)
    h = Harness(native_tools=[tool], lookup_order=Canned(FOUND), **kwargs)
    return h


async def test_the_voice_and_stt_switch_before_the_next_agents_first_reply() -> None:
    front, back = FrameProcessor(), FrameProcessor()
    front_stt, back_stt = FrameProcessor(), FrameProcessor()

    async def hand_off(flows: UnmuteFlows, params: FunctionCallParams) -> None:
        await flows.hand_off(params, AgentSpec(NOTES_PROMPT), tts=back, stt=back_stt)

    h = _desk(hand_off, tts=VoiceSwitcher([front, back]), stt=STTSwitcher([front_stt, back_stt]))
    h.user("I need a note taken.")
    await h.call("to_notes")

    kinds = [type(frame) for frame in h.worker.log if not isinstance(frame, str)]
    assert kinds[-4:] == [
        ManuallySwitchServiceFrame,
        ManuallySwitchServiceFrame,
        LLMUpdateSettingsFrame,
        LLMRunFrame,
    ]
    switches = [f.service for f in h.worker.log if isinstance(f, ManuallySwitchServiceFrame)]
    assert switches == [back_stt, back]


async def test_a_handoff_without_a_voice_or_stt_goes_back_to_the_calls_first_ones() -> None:
    front, back = FrameProcessor(), FrameProcessor()
    front_stt, back_stt = FrameProcessor(), FrameProcessor()

    async def hand_off(flows: UnmuteFlows, params: FunctionCallParams) -> None:
        await flows.hand_off(params, AgentSpec(NOTES_PROMPT))

    h = _desk(hand_off, tts=VoiceSwitcher([front, back]), stt=STTSwitcher([front_stt, back_stt]))
    h.user("I need a note taken.")
    await h.call("to_notes")

    switches = [f.service for f in h.worker.log if isinstance(f, ManuallySwitchServiceFrame)]
    assert switches == [front_stt, front]


async def test_a_task_step_listens_and_speaks_on_its_owners_services() -> None:
    h = _desk(
        lambda flows, params: None,
        tts=VoiceSwitcher([FrameProcessor(), FrameProcessor()]),
        stt=STTSwitcher([FrameProcessor(), FrameProcessor()]),
    )
    h.user("Where is my order?")
    await h.call("find_order")

    assert h.flows.site == "task:find_order"
    assert not [f for f in h.worker.log if isinstance(f, ManuallySwitchServiceFrame)]


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
            AgentSpec(NOTES_PROMPT),
            {"stt": FrameProcessor()},
            "needs that STT service in the STTSwitcher",
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


@pytest.mark.parametrize(
    ("kind", "match"), [("tts", "takes a VoiceSwitcher"), ("stt", "takes an STTSwitcher")]
)
def test_the_calls_services_need_unmutes_switcher(kind: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        switcher: dict[str, Any] = {kind: ServiceSwitcher([FrameProcessor(), FrameProcessor()])}
        Harness(**switcher)


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
