"""``TelemetryObserver``: pipeline frames become the shared spans."""

import json
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    AggregationType,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from tests.pipecat.harness import Harness
from tests.pipecat.test_model import Scripted, service
from unmute_core.call import CallHooks
from unmute_core.hooks import Hooks
from unmute_core.model import FallbackModelCall, TextDelta
from unmute_core.pipecat import TelemetryObserver

SECOND = 1_000_000_000


async def test_the_observer_stamps_the_framework_on_the_call_span(spans: Any) -> None:
    telemetry, exported = spans
    TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    await telemetry.close()

    [call] = [span for span in exported() if span.name == "conversation"]
    assert dict(call.attributes or {})["unmute.call.framework"] == "pipecat"


async def test_a_turn_with_a_model_request_and_a_tool_call(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    assert not observer.observe_every_push
    llm = service(Scripted())
    other = FrameProcessor()

    async def push(frame: Any, at: float, source: Any = other) -> None:
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=other,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=int(at * SECOND),
            )
        )

    await push(UserStartedSpeakingFrame(), 0.0)
    await push(VADUserStoppedSpeakingFrame(), 1.0)
    await push(TranscriptionFrame("Where is my order?", "caller", "now"), 1.25)
    await push(LLMFullResponseStartFrame(), 1.3, llm)
    usage = LLMTokenUsage(prompt_tokens=80, completion_tokens=5, total_tokens=85)
    await push(MetricsFrame(data=[TTFBMetricsData(processor="llm", value=0.2)]), 1.5, llm)
    await push(LLMTextFrame("Let me check."), 1.5, llm)
    await push(MetricsFrame(data=[LLMUsageMetricsData(processor="llm", value=usage)]), 1.6, llm)
    await push(LLMFullResponseEndFrame(), 1.6, llm)
    await push(BotStartedSpeakingFrame(), 2.0)
    await push(FunctionCallInProgressFrame("lookup", "c1", {"id": "A-12"}), 2.1)
    await push(FunctionCallResultFrame("lookup", "c1", {"id": "A-12"}, {"found": True}), 2.2)
    await push(EndFrame(), 3.0)

    by_name = {span.name: dict(span.attributes or {}) for span in exported()}
    assert set(by_name) == {"turn", "llm", "llm_tool_call"}
    turn = by_name["turn"]
    assert turn["input.value"] == "Where is my order?"
    assert turn["output.value"] == "Let me check."
    assert turn["turn.e2e_seconds"] == 1.0
    assert turn["turn.stt_final_seconds"] == 0.25
    assert turn["turn.llm_ttft_seconds"] == 0.2
    llm_span = by_name["llm"]
    assert llm_span["gen_ai.request.model"] == "gpt-4.1"
    assert llm_span["gen_ai.usage.input_tokens"] == 80
    assert llm_span["output.value"] == "Let me check."
    assert by_name["llm_tool_call"]["gen_ai.tool.call.result"] == '{"found": true}'


class SlngSTTService(STTService):
    """A stand-in STT service: only its class name and settings are read."""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None]:
        yield None


class SlngTTSService(TTSService):
    """A stand-in TTS service: only its class name and settings are read."""

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None]:
        yield None


def _pusher(observer: TelemetryObserver) -> Any:
    """Return a function that shows the observer one frame, as the worker would."""
    other = FrameProcessor()

    async def push(frame: Any, at: float, source: Any = other, destination: Any = other) -> None:
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=destination,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=int(at * SECOND),
            )
        )

    return push


async def test_a_model_request_shows_its_prompt_tools_and_the_calls_chosen(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    llm = service(Scripted(), prompt="You help with orders.")
    lookup = FunctionSchema("lookup", "Find an order.", {"id": {"type": "string"}}, ["id"])
    context = LLMContext(
        messages=[{"role": "user", "content": "Where is A-12?"}], tools=ToolsSchema([lookup])
    )

    await push(LLMContextFrame(context), 1.0, destination=llm)
    await push(LLMFullResponseStartFrame(), 1.1, llm)
    call = FunctionCallFromLLM("lookup", "c1", {"id": "A-12"}, context)
    await push(FunctionCallsStartedFrame(function_calls=[call]), 1.2, llm)
    await push(LLMFullResponseEndFrame(), 1.3, llm)
    await push(EndFrame(), 2.0)

    [request] = [dict(s.attributes or {}) for s in exported() if s.name == "llm"]
    assert json.loads(request["input.value"]) == [
        {"role": "system", "content": "You help with orders."},
        {"role": "user", "content": "Where is A-12?"},
    ]
    [tool] = json.loads(request["gen_ai.tool.definitions"])
    assert tool["function"]["name"] == "lookup"
    chose = json.loads(request["output.value"])
    assert chose["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"id": "A-12"}'},
        }
    ]


class Named(Scripted):
    """A model call whose name changes, like a fallback chain that moved to its next call."""

    model = "first"


async def test_a_model_request_names_the_model_that_answered_at_its_end(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    model = Named()
    llm = service(model)

    await push(LLMFullResponseStartFrame(), 1.0, llm)
    await push(LLMTextFrame("Hi."), 1.1, llm)
    model.model = "backup"
    await push(LLMFullResponseEndFrame(), 1.2, llm)
    await push(EndFrame(), 2.0)

    [request] = [dict(s.attributes or {}) for s in exported() if s.name == "llm"]
    assert request["gen_ai.response.model"] == "backup"
    assert request["gen_ai.request.model"] == "first"


class Answers(Scripted):
    """A named model call that fails, or answers."""

    def __init__(self, model: str, *, fails: bool = False) -> None:
        super().__init__([TextDelta("Hi.")])
        self.model = model
        self.fails = fails

    async def stream(self, request: Any) -> AsyncGenerator[Any]:
        if self.fails:
            raise RuntimeError("down")
        async for event in super().stream(request):
            yield event


async def test_a_fallback_chain_names_the_call_that_answered(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="first", provider="openai")
    push = _pusher(observer)
    chain = FallbackModelCall([Answers("first", fails=True), Answers("second")])
    llm = service(cast(Any, chain))

    await push(LLMFullResponseStartFrame(), 1.0, llm)
    _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext())]
    await push(LLMFullResponseEndFrame(), 1.2, llm)
    await push(EndFrame(), 2.0)

    [request] = [dict(s.attributes or {}) for s in exported() if s.name == "llm"]
    assert request["gen_ai.response.model"] == "second"


async def test_a_sticky_chain_request_names_the_model_first_asked_not_the_one_that_answered(
    spans: Any,
) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    chain = FallbackModelCall([Answers("first", fails=True), Answers("second")], sticky=True)
    llm = service(cast(Any, chain))

    await push(LLMFullResponseStartFrame(), 1.0, llm)
    _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext())]
    await push(LLMFullResponseEndFrame(), 1.2, llm)
    await push(EndFrame(), 2.0)

    [request] = [dict(s.attributes or {}) for s in exported() if s.name == "llm"]
    assert request["gen_ai.request.model"] == "first"
    assert request["gen_ai.response.model"] == "second"


async def test_an_error_from_the_model_service_goes_on_its_request_span(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    llm = service(Scripted())

    await push(LLMFullResponseStartFrame(), 1.0, llm)
    await push(LLMTextFrame("Let me check"), 1.1, llm)
    error = ErrorFrame("failed", exception=RuntimeError("dropped"), processor=llm)
    await push(error, 1.2, llm)
    await push(LLMFullResponseEndFrame(), 1.3, llm)
    await push(EndFrame(), 2.0)

    [request] = [s for s in exported() if s.name == "llm"]
    assert dict(request.attributes or {})["error.type"] == "RuntimeError"
    assert dict(request.attributes or {})["output.value"] == "Let me check"


async def test_an_error_from_another_processor_stays_off_the_request_span(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    llm = service(Scripted())
    tts = FrameProcessor()

    await push(LLMFullResponseStartFrame(), 1.0, llm)
    await push(LLMTextFrame("Hello"), 1.1, llm)
    error = ErrorFrame("tts down", exception=ValueError("tts"), processor=tts)
    await push(error, 1.2, tts)
    await push(error, 1.21, llm)
    await push(LLMFullResponseEndFrame(), 1.3, llm)
    await push(EndFrame(), 2.0)

    [request] = [s for s in exported() if s.name == "llm"]
    assert "error.type" not in (request.attributes or {})


async def test_a_chain_that_starts_with_its_first_call_each_request_says_so(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="first", provider="openai")
    push = _pusher(observer)
    chain = FallbackModelCall([Answers("first", fails=True), Answers("second")])
    llm = service(cast(Any, chain))
    await llm.get_chat_completions(LLMContext())

    for at in (1.0, 3.0):
        await push(LLMFullResponseStartFrame(), at, llm)
        _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext())]
        await push(LLMFullResponseEndFrame(), at + 0.2, llm)
    await push(EndFrame(), 5.0)

    requests = [dict(s.attributes or {}) for s in exported() if s.name == "llm"]
    assert [r["gen_ai.response.model"] for r in requests] == ["second", "second"]
    assert [r["gen_ai.request.model"] for r in requests] == ["first", "first"]


async def test_a_prompt_suffix_reads_the_model_the_request_starts_with() -> None:
    seen: list[str] = []

    hooks = Hooks()

    @hooks.before_model
    async def probe(request: Any) -> Any:
        seen.append(request.model)
        return request

    chain = FallbackModelCall([Answers("first", fails=True), Answers("second")])
    llm = service(cast(Any, chain))
    llm.call = CallHooks(hooks=hooks)
    for _ in range(2):
        _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext())]
    assert seen == ["first", "first"]


async def test_stt_and_tts_spans_sit_in_the_turn(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    stt = SlngSTTService(settings=STTSettings(model="rt-v5", language=Language.EN))
    tts = SlngTTSService(settings=TTSSettings(model="sonic", voice="v1", language=None))

    await push(UserStartedSpeakingFrame(), 0.0)
    await push(VADUserStoppedSpeakingFrame(), 1.0)
    await push(TranscriptionFrame("Where is my order?", "caller", "now"), 1.25, stt)
    await push(TTSStartedFrame(context_id="t1"), 2.0, tts)
    for word in ("It", "is", "open."):
        await push(TTSTextFrame(word, AggregationType.WORD, context_id="t1"), 2.1, tts)
    await push(TTSAudioRawFrame(b"\0\0", 24000, 1, context_id="t1"), 2.15, tts)
    await push(BotStartedSpeakingFrame(), 2.2)
    await push(TTSStoppedFrame(context_id="t1"), 2.5, tts)
    await push(BotStoppedSpeakingFrame(), 2.9)
    await push(TTSStartedFrame(context_id="t2"), 3.0, tts)
    await push(UserStartedSpeakingFrame(), 3.1)
    await push(EndFrame(), 4.0)

    finished = exported()
    by_id = {span.context.span_id: span.name for span in finished}
    named = {(s.name, by_id.get(s.parent.span_id if s.parent else 0)) for s in finished}
    assert {("stt", "turn"), ("tts", "turn")} <= named
    [heard] = [dict(s.attributes or {}) for s in finished if s.name == "stt"]
    assert (heard["output.value"], heard["gen_ai.request.model"]) == ("Where is my order?", "rt-v5")
    assert heard["gen_ai.provider.name"] == "slng"
    assert heard["metrics.ttfb"] == 0.25
    assert heard["stt.language"] == "en"
    said, cut = [dict(s.attributes or {}) for s in finished if s.name == "tts"]
    assert (said["input.value"], said["tts.voice"], said["metrics.ttfb"]) == (
        "It is open.",
        "v1",
        pytest.approx(0.15),
    )
    assert said["tts.interrupted"] is False
    assert cut["tts.interrupted"] is True


async def test_two_replies_in_one_turn_read_as_two_sentences(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    llm = service(Scripted())
    for at, text in ((1.0, "One sec."), (2.0, "It is open.")):
        await push(LLMFullResponseStartFrame(), at, llm)
        await push(LLMTextFrame(text), at, llm)
        await push(LLMFullResponseEndFrame(), at, llm)
    await push(EndFrame(), 3.0)

    [turn] = [dict(s.attributes or {}) for s in exported() if s.name == "turn"]
    assert turn["output.value"] == "One sec. It is open."


def _turns(exported: Any) -> list[dict[str, Any]]:
    """Each finished turn's attributes, in order."""
    spans = sorted(exported(), key=lambda span: span.start_time or 0)
    return [dict(span.attributes or {}) for span in spans if span.name == "turn"]


async def test_a_caller_who_starts_after_the_agent_finished_did_not_interrupt(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    llm = service(Scripted())
    for at, line in ((0.0, "Book me in."), (5.0, "Friday.")):
        await push(UserStartedSpeakingFrame(), at)
        await push(TranscriptionFrame(line, "caller", "now"), at + 0.5)
        await push(LLMFullResponseStartFrame(), at + 1, llm)
        await push(LLMTextFrame("Which day?"), at + 1, llm)
        await push(LLMFullResponseEndFrame(), at + 1, llm)
        await push(BotStartedSpeakingFrame(), at + 2)
        if at == 0.0:
            await push(BotStoppedSpeakingFrame(), at + 3)
    await push(UserStartedSpeakingFrame(), 7.5)
    await push(TranscriptionFrame("Sorry, Saturday.", "caller", "now"), 8.0)
    await push(EndFrame(), 9.0)

    turns = _turns(exported)
    assert [t["turn.interrupted"] for t in turns] == [False, True, False]


async def test_stt_time_runs_from_the_end_of_speech_in_the_same_turn(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    llm = service(Scripted())
    await push(VADUserStartedSpeakingFrame(), 0.0)
    await push(UserStartedSpeakingFrame(), 0.1)
    await push(VADUserStoppedSpeakingFrame(), 1.0)
    await push(TranscriptionFrame("Book me in.", "caller", "now"), 1.3)
    await push(LLMFullResponseStartFrame(), 1.5, llm)
    await push(LLMTextFrame("Which day?"), 1.5, llm)
    await push(LLMFullResponseEndFrame(), 1.5, llm)
    await push(VADUserStartedSpeakingFrame(), 10.0)
    await push(TranscriptionFrame("Friday.", "caller", "now"), 10.5)
    await push(EndFrame(), 12.0)

    first, second = _turns(exported)
    assert first["turn.stt_final_seconds"] == pytest.approx(0.3)
    assert "turn.stt_final_seconds" not in second


async def test_a_cut_off_request_keeps_the_callers_words_in_one_turn(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    llm = service(Scripted())
    await push(UserStartedSpeakingFrame(), 0.0)
    await push(TranscriptionFrame("Yeah, it", "caller", "now"), 0.5)
    await push(LLMFullResponseStartFrame(), 0.6, llm)
    await push(UserStartedSpeakingFrame(), 0.7)
    await push(TranscriptionFrame("is +34600111222.", "caller", "now"), 1.5)
    await push(LLMFullResponseStartFrame(), 1.6, llm)
    await push(LLMTextFrame("Is that right?"), 1.8, llm)
    await push(LLMFullResponseEndFrame(), 1.9, llm)
    await push(EndFrame(), 3.0)

    [turn] = _turns(exported)
    assert turn["input.value"] == "Yeah, it is +34600111222."
    assert turn["output.value"] == "Is that right?"
    cut, answered = [
        dict(span.attributes or {})
        for span in sorted(exported(), key=lambda span: span.start_time or 0)
        if span.name == "llm"
    ]
    assert cut["langfuse.observation.level"] == "WARNING"
    assert cut["llm.interrupted"] is True
    assert "langfuse.observation.level" not in answered


def _named(exported: Any, name: str) -> list[dict[str, Any]]:
    """The attributes of each finished span called ``name``, in start order."""
    spans = sorted(exported(), key=lambda span: span.start_time or 0)
    return [dict(span.attributes or {}) for span in spans if span.name == name]


async def test_a_tts_span_ends_when_playback_stops_not_when_audio_is_made(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    tts = SlngTTSService(settings=TTSSettings(model="sonic", voice="v1", language=None))

    await push(TTSStartedFrame(context_id="t1"), 1.0, tts)
    await push(TTSTextFrame("Which day?", AggregationType.SENTENCE, context_id="t1"), 1.0, tts)
    await push(TTSAudioRawFrame(b"\0\0", 24000, 1, context_id="t1"), 1.2, tts)
    await push(BotStartedSpeakingFrame(), 1.25)
    await push(TTSStoppedFrame(context_id="t1"), 1.3, tts)
    assert _named(exported, "tts") == []

    await push(BotStoppedSpeakingFrame(), 2.5)
    [said] = _named(exported, "tts")
    assert said["input.value"] == "Which day?"
    assert said["tts.interrupted"] is False


async def test_a_line_the_caller_cuts_is_interrupted_and_judged_an_interruption(
    spans: Any,
) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    tts = SlngTTSService(settings=TTSSettings(model="sonic", voice="v1", language=None))

    await push(TTSStartedFrame(context_id="t1"), 1.0, tts)
    await push(
        TTSTextFrame("Shall I book it?", AggregationType.SENTENCE, context_id="t1"), 1.0, tts
    )
    await push(TTSAudioRawFrame(b"\0\0", 24000, 1, context_id="t1"), 1.1, tts)
    await push(BotStartedSpeakingFrame(), 1.2)
    await push(TTSStoppedFrame(context_id="t1"), 1.3, tts)
    await push(TranscriptionFrame("No wait, Saturday please", "caller", "now"), 2.0)
    await push(UserStartedSpeakingFrame(), 2.05)
    await push(BotStoppedSpeakingFrame(), 2.1)
    await push(EndFrame(), 3.0)

    [cut] = _named(exported, "tts")
    assert cut["tts.interrupted"] is True
    [verdict] = _named(exported, "caller.over_agent")
    assert json.loads(verdict["input.value"]) == {"interruption": True}


async def test_eou_and_a_kept_answer_over_the_agent(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    llm = service(Scripted())

    await push(VADUserStartedSpeakingFrame(), 0.0)
    await push(UserStartedSpeakingFrame(), 0.1)
    await push(VADUserStoppedSpeakingFrame(), 1.0)
    await push(TranscriptionFrame("Friday morning.", "caller", "now"), 1.1)
    await push(UserStoppedSpeakingFrame(), 1.4)
    await push(LLMFullResponseStartFrame(), 1.5, llm)
    await push(LLMTextFrame("Shall I book nine?"), 1.6, llm)
    await push(LLMFullResponseEndFrame(), 1.7, llm)
    await push(BotStartedSpeakingFrame(), 2.0)
    await push(VADUserStartedSpeakingFrame(), 3.0)
    await push(TranscriptionFrame("Yes, please.", "caller", "now"), 3.3)
    await push(VADUserStoppedSpeakingFrame(), 3.4)
    await push(BotStoppedSpeakingFrame(), 3.6)
    await push(UserStartedSpeakingFrame(), 3.61)
    await push(UserStoppedSpeakingFrame(), 3.62)
    await push(EndFrame(), 5.0)

    first, kept = _turns(exported)
    assert first["turn.eou_seconds"] == pytest.approx(0.4)
    assert kept["input.value"] == "Yes, please."
    assert kept["turn.eou_seconds"] == pytest.approx(0.22)
    [verdict] = _named(exported, "caller.over_agent")
    assert json.loads(verdict["input.value"]) == {"interruption": False}


async def test_a_request_the_caller_interrupts_is_cut_even_when_its_end_arrives(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-4.1", provider="openai")
    push = _pusher(observer)
    llm = service(Scripted())
    await push(UserStartedSpeakingFrame(), 0.0)
    await push(TranscriptionFrame("The number is plus three four", "caller", "now"), 0.5)
    await push(LLMFullResponseStartFrame(), 0.6, llm)
    await push(InterruptionFrame(), 0.9)
    await push(LLMFullResponseEndFrame(), 0.95, llm)
    await push(TranscriptionFrame("six seven one.", "caller", "now"), 1.0)
    await push(LLMFullResponseStartFrame(), 1.1, llm)
    await push(LLMTextFrame("Is that right?"), 1.2, llm)
    await push(LLMFullResponseEndFrame(), 1.3, llm)
    await push(EndFrame(), 2.0)

    cut, answered = _named(exported, "llm")
    assert cut["llm.interrupted"] is True
    assert "llm.interrupted" not in answered


async def test_the_requests_trace_headers_carry_the_open_turns_trace_id(spans: Any) -> None:
    telemetry, _ = spans
    llm = service(Scripted([]))
    h = Harness(llm=llm)
    observer = TelemetryObserver(telemetry, model="m", provider="openai", flows=h.flows)
    assert llm.trace_headers == observer.trace_headers
    assert observer.trace_headers() == {}
    other = FrameProcessor()
    await observer.on_push_frame(
        FramePushed(
            source=other,
            destination=other,
            frame=UserStartedSpeakingFrame(),
            direction=FrameDirection.DOWNSTREAM,
            timestamp=0,
        )
    )
    sent = observer.trace_headers()
    _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext())]
    [request] = cast(Scripted, llm.model_call).requests
    assert request.headers == sent
    trace_id = sent["traceparent"].split("-")[1]
    assert telemetry._call is not None
    assert trace_id == f"{telemetry._call.get_span_context().trace_id:032x}"


@pytest.mark.parametrize(("metrics", "warned"), [(False, True), (True, False)])
async def test_an_llm_with_metrics_off_warns_once_that_spans_miss_tokens(
    metrics: bool, warned: bool, spans: Any, caplog: pytest.LogCaptureFixture
) -> None:
    class Measured(OpenAILLMService):
        metrics_enabled = metrics
        usage_metrics_enabled = metrics

    llm = Measured(api_key="k", settings=OpenAILLMService.Settings(model="m"))
    push = _pusher(TelemetryObserver(spans[0], model="m", provider="p"))
    with caplog.at_level("WARNING"):
        await push(LLMFullResponseStartFrame(), 1.0, source=llm)
        await push(LLMFullResponseEndFrame(), 1.1, source=llm)
    assert caplog.text.count("enable_usage_metrics=True") == (1 if warned else 0)
