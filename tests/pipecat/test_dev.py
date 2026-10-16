"""``DevObserver``: pipeline frames become the records ``unmute dev`` reads."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from pipecat.frames.frames import (
    AggregationType,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    ProcessingMetricsData,
    TextAggregationMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from tests.dev.contract import problems
from tests.pipecat.test_model import Scripted, service
from tests.tasks.job import job, plan, registry, state
from unmute_core.dev import DevRecorder
from unmute_core.pipecat import DevObserver
from unmute_core.tasks import CallTasks, DelegateRun

SECOND = 1_000_000_000


class SlngSTTService(STTService):
    """A stand-in STT service: only its class name and settings are read."""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None]:
        yield None


class SlngTTSService(TTSService):
    """A stand-in TTS service: only its class name and settings are read."""

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None]:
        yield None


class Call:
    """A scripted call: frames pushed at chosen times, and the records that came out."""

    def __init__(self, **kwargs: Any) -> None:
        self.records: list[dict[str, Any]] = []
        self.recorder = DevRecorder("call-1", self.records.append)
        self.observer = DevObserver(self.recorder, **kwargs)
        self.llm = service(Scripted())
        self.stt = SlngSTTService(settings=STTSettings(model="rt-v5", language=None))
        self.tts = SlngTTSService(settings=TTSSettings(model="sonic", voice="v1", language=None))
        self.other = FrameProcessor()

    async def push(self, frame: Any, at: float, source: Any = None) -> None:
        """Show the observer one frame, as the worker would, and let its event tasks run."""
        await self.observer.on_push_frame(
            FramePushed(
                source=source or self.other,
                destination=self.other,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=int(at * SECOND),
            )
        )
        await asyncio.sleep(0)

    def latest(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Return each record's last payload, after checking every record against the contract."""
        assert [problems(json.dumps(r)) for r in self.records] == [[] for _ in self.records]
        return {(r["kind"], r["id"]): r[r["kind"]] for r in self.records}

    def of(self, kind: str) -> list[dict[str, Any]]:
        """Return the last payload of each record of a kind, in order."""
        return [p for (k, _), p in self.latest().items() if k == kind]


async def a_spoken_turn(call: Call) -> None:
    """Script one turn: the caller speaks, the model answers in a sentence, the agent says it.

    The caller fell silent half a second ago by the wall clock, which Pipecat's latency
    observer reads, so the wait for the reply has parts to show.
    """
    push = call.push
    await push(UserStartedSpeakingFrame(), 0.0)
    await push(InterimTranscriptionFrame("Where", "caller", "now"), 0.5, call.stt)
    await push(TranscriptionFrame("Where is my order?", "caller", "now"), 1.0, call.stt)
    await push(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=time.time() - 0.3), 1.0)
    await push(UserStoppedSpeakingFrame(), 1.1)
    await push(LLMContextFrame(LLMContext(messages=[])), 1.2)
    await push(LLMFullResponseStartFrame(), 1.3, call.llm)
    await push(
        MetricsFrame(data=[TTFBMetricsData(processor=call.llm.name, value=0.2)]), 1.5, call.llm
    )
    await push(LLMTextFrame("It is "), 1.5, call.llm)
    await push(LLMTextFrame("on its way."), 1.6, call.llm)
    await push(LLMFullResponseEndFrame(), 1.7, call.llm)
    await push(TTSStartedFrame(context_id="t1"), 1.8, call.tts)
    await push(TTSTextFrame("on", AggregationType.WORD, context_id="t1"), 1.9, call.tts)
    await push(TTSAudioRawFrame(b"\0\0", 24000, 1, context_id="t1"), 2.0, call.tts)
    await push(BotStartedSpeakingFrame(), 2.1)
    await push(TTSStoppedFrame(context_id="t1"), 2.5, call.tts)
    await push(BotStoppedSpeakingFrame(), 3.0)


async def test_a_spoken_turn_gives_its_exchanges_text_operations_and_timings() -> None:
    call = Call()
    await a_spoken_turn(call)
    await call.push(EndFrame(), 4.0)
    latest = call.latest()

    assert latest["call", "call"]["state"] == "ended"
    assert latest["exchange", "input-1"] == {
        "role": "input",
        "state": "ended",
        "link_status": "known",
    }
    [reply] = call.of("exchange")[1:]
    assert reply == {
        "role": "response",
        "state": "ended",
        "playback": "ended",
        "link_status": "known",
        "input_id": "input-1",
    }
    users = [t for t in call.of("text") if t["speaker"] == "user"]
    assert [(t["text"], t["state"], t["origin"]) for t in users] == [
        ("Where is my order?", "final", "recognition")
    ]
    [answer] = [t for t in call.of("text") if t["speaker"] == "assistant"]
    assert (answer["text"], answer["state"], answer["origin"]) == (
        "It is on its way.",
        "final",
        "generated",
    )
    ops = {(o["type"], o["state"]) for o in call.of("operation")}
    assert ops == {("stt", "ended"), ("llm", "ended"), ("tts", "ended")}
    llm = next(o for o in call.of("operation") if o["type"] == "llm")
    assert llm["model"] == "m"
    stt = next(o for o in call.of("operation") if o["type"] == "stt")
    assert (stt["model"], stt["provider"], stt["exchange_id"]) == ("rt-v5", "slng", "input-1")
    measured = {
        (m["metric"], m["scope"]): m["value"]
        for m in call.of("measurement")
        if m["state"] == "measured"
    }
    assert measured[("first_response", "operation")] == 0.2
    assert measured[("request_duration", "operation")] == pytest.approx(0.4)
    assert measured[("reply_latency", "response")] >= 0


async def test_the_caller_text_is_provisional_then_final_in_one_row() -> None:
    call = Call()
    await call.push(UserStartedSpeakingFrame(), 0.0)
    await call.push(InterimTranscriptionFrame("Where", "caller", "now"), 0.5, call.stt)
    assert [t["state"] for t in call.of("text")] == ["provisional"]
    await call.push(TranscriptionFrame("Where is my order?", "caller", "now"), 1.0, call.stt)
    await call.push(TranscriptionFrame("Thanks", "caller", "now"), 1.5, call.stt)
    rows = call.of("text")
    assert [(t["text"], t["state"]) for t in rows] == [
        ("Where is my order?", "final"),
        ("Thanks", "final"),
    ]
    assert len({t["message_id"] for t in rows}) == 1
    assert [r["revision"] for r in call.records if r["kind"] == "text"] == [1, 2, 1]


async def test_typed_text_opens_an_input_and_a_greeting_opens_a_response() -> None:
    call = Call()
    await call.push(TTSSpeakFrame("Hello, welcome."), 0.1)
    await call.push(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Book me in"}]), 1.0
    )
    texts = call.of("text")
    assert [(t["speaker"], t["text"], t["state"]) for t in texts] == [
        ("assistant", "Hello, welcome.", "final"),
        ("user", "Book me in", "final"),
    ]
    greeting, typed = call.of("exchange")
    assert greeting["role"] == "response" and greeting["link_status"] == "unavailable"
    assert typed["role"] == "input"


async def test_a_reply_is_linked_to_the_input_it_follows_and_a_second_reply_is_not() -> None:
    call = Call()
    await a_spoken_turn(call)
    await call.push(LLMFullResponseStartFrame(), 5.0, call.llm)
    first, second = [e for e in call.of("exchange") if e["role"] == "response"]
    assert first["link_status"] == "known"
    assert second["link_status"] == "unavailable" and "input_id" not in second


async def test_a_caller_who_speaks_as_the_greeting_ends_gets_the_reply_linked() -> None:
    call = Call()
    await call.push(TTSSpeakFrame("Hello, welcome."), 0.1)
    await call.push(BotStartedSpeakingFrame(), 0.2)
    await call.push(TranscriptionFrame("Hi there", "caller", "now"), 0.3, call.stt)
    await call.push(BotStoppedSpeakingFrame(), 0.4)
    await call.push(
        LLMContextFrame(LLMContext(messages=[])), 0.4, LLMAssistantAggregator(LLMContext())
    )
    await call.push(UserStartedSpeakingFrame(), 0.5)
    await call.push(LLMContextFrame(LLMContext(messages=[])), 0.6)
    await call.push(LLMFullResponseStartFrame(), 0.7, call.llm)
    inputs = [e for e in call.of("exchange") if e["role"] == "input"]
    replies = [e for e in call.of("exchange") if e["role"] == "response"]
    assert len(inputs) == 1
    assert replies[-1]["link_status"] == "known"


async def test_a_tool_call_is_an_operation_under_its_model_request_and_is_timed() -> None:
    call = Call()
    await call.push(UserStartedSpeakingFrame(), 0.0)
    await call.push(LLMFullResponseStartFrame(), 1.0, call.llm)
    await call.push(FunctionCallInProgressFrame("lookup", "c1", {"id": "A-12"}), 1.2)
    await call.push(FunctionCallInProgressFrame("slow", "c2", {}), 1.2)
    await call.push(FunctionCallInProgressFrame("bad", "c3", {}), 1.2)
    await call.push(FunctionCallInProgressFrame("hung", "c4", {}), 1.2)
    await call.push(LLMFullResponseEndFrame(), 1.3, call.llm)
    await call.push(FunctionCallResultFrame("lookup", "c1", {"id": "A-12"}, {"found": True}), 1.7)
    await call.push(FunctionCallResultFrame("slow", "c2", {}, "x", properties=None), 2.2)
    await call.push(
        FunctionCallResultFrame("bad", "c3", {}, "x", error="KeyError: 'secret order'"), 1.4
    )
    await call.push(FunctionCallCancelFrame("hung", "c4", run_llm=True), 3.2)
    ops = {o["name"]: o for o in call.of("operation") if o["type"] == "tool"}
    assert {name: o["state"] for name, o in ops.items()} == {
        "lookup": "returned",
        "slow": "returned",
        "bad": "failed",
        "hung": "timed_out",
    }
    assert ops["bad"]["reason"] == "The handler raised KeyError."
    assert "secret" not in json.dumps(call.records)
    assert ops["lookup"]["parent_operation_id"] == "llm-3"
    durations = {
        m["operation_id"]: m["value"]
        for m in call.of("measurement")
        if m["metric"] == "tool_duration"
    }
    assert durations["tool-4"] == pytest.approx(0.5)
    assert durations["tool-7"] == pytest.approx(2.0)


async def test_a_tool_that_starts_a_task_has_no_tool_operation() -> None:
    call = Call(handoffs={"book"})
    await call.push(FunctionCallInProgressFrame("book", "c1", {}), 1.0)
    await call.push(FunctionCallResultFrame("book", "c1", {}, {"status": "ok"}), 1.1)
    assert call.of("operation") == []


async def test_the_caller_speaking_over_the_agent_cuts_its_reply_off() -> None:
    call = Call()
    await call.push(UserStartedSpeakingFrame(), 0.0)
    await call.push(LLMFullResponseStartFrame(), 1.0, call.llm)
    await call.push(
        LLMTextFrame(
            "It is on",
        ),
        1.1,
        call.llm,
    )
    await call.push(TTSStartedFrame(context_id="t1"), 1.2, call.tts)
    await call.push(BotStartedSpeakingFrame(), 1.3)
    await call.push(UserStartedSpeakingFrame(), 1.5)
    await call.push(InterruptionFrame(), 1.5)
    await call.push(LLMFullResponseEndFrame(), 1.6, call.llm)
    ops = {o["type"]: o["state"] for o in call.of("operation")}
    assert ops == {"llm": "cancelled", "tts": "cancelled"}
    [answer] = [t for t in call.of("text") if t["speaker"] == "assistant"]
    assert answer["state"] == "incomplete"
    reply = next(e for e in call.of("exchange") if e["role"] == "response")
    assert (reply["state"], reply["playback"]) == ("interrupted", "interrupted")
    unfinished = [m for m in call.of("measurement") if m["state"] == "unavailable"]
    assert {m["metric"] for m in unfinished} == {"first_response", "request_duration"}


async def test_a_fatal_error_and_a_cancel_end_the_call_as_an_error() -> None:
    call = Call()
    await call.push(ErrorFrame("boom: the prompt said X", fatal=True), 1.0, call.llm)
    await call.push(CancelFrame(), 2.0)
    final = call.latest()["call", "call"]
    assert final["state"] == "error"
    assert final["reason"] == f"{call.llm.name} reported a fatal error."
    assert "prompt" not in json.dumps(call.records)
    count = len(call.records)
    await call.push(UserStartedSpeakingFrame(), 3.0)
    assert len(call.records) == count


async def test_the_time_from_connecting_to_the_first_speech_is_a_call_measurement() -> None:
    call = Call()
    await call.push(ClientConnectedFrame(), 0.0)
    await call.push(TTSSpeakFrame("Hello."), 0.2)
    await call.push(BotStartedSpeakingFrame(), 0.5)
    first = [m for m in call.of("measurement") if m["metric"] == "first_speech"]
    assert [(m["scope"], m["state"]) for m in first] == [("call", "measured")]


async def test_processing_time_a_service_reports_settles_the_request_duration() -> None:
    call = Call()
    await call.push(UserStartedSpeakingFrame(), 0.0)
    await call.push(LLMFullResponseStartFrame(), 1.0, call.llm)
    await call.push(
        MetricsFrame(data=[ProcessingMetricsData(processor=call.llm.name, value=0.9, model="gpt")]),
        1.2,
        call.llm,
    )
    await call.push(MetricsFrame(data=[TTFBMetricsData(processor="other", value=3)]), 1.2, call.llm)
    await call.push(LLMFullResponseEndFrame(), 5.0, call.llm)
    duration = next(m for m in call.of("measurement") if m["metric"] == "request_duration")
    assert (duration["value"], duration["source"]) == (0.9, "pipecat-llm-processing")
    assert next(o for o in call.of("operation") if o["type"] == "llm")["model"] == "gpt"


class Notes(Scripted):
    """The model call of an agent that took the call with a model of its own."""

    model = "notes-model"
    provider = "slng"


async def test_an_llm_operation_names_the_model_call_a_handoff_switched_to() -> None:
    call = Call()
    call.llm.model_call = Notes()
    await call.push(UserStartedSpeakingFrame(), 0.0)
    await call.push(LLMFullResponseStartFrame(), 1.0, call.llm)
    await call.push(
        MetricsFrame(data=[ProcessingMetricsData(processor=call.llm.name, value=0.9, model="m")]),
        1.2,
        call.llm,
    )
    await call.push(LLMFullResponseEndFrame(), 5.0, call.llm)
    operation = next(o for o in call.of("operation") if o["type"] == "llm")
    assert (operation["model"], operation["provider"]) == ("notes-model", "slng")
    duration = next(m for m in call.of("measurement") if m["metric"] == "request_duration")
    assert (duration["model"], duration["provider"]) == ("notes-model", "slng")


async def test_an_llm_operation_names_the_model_that_answered_at_its_end() -> None:
    call = Call()
    model = Notes()
    model.model = "first"
    call.llm.model_call = model
    await call.push(UserStartedSpeakingFrame(), 0.0)
    await call.push(LLMFullResponseStartFrame(), 1.0, call.llm)
    model.model = "backup"
    await call.push(LLMFullResponseEndFrame(), 2.0, call.llm)
    operation = next(o for o in call.of("operation") if o["type"] == "llm")
    assert operation["model"] == "backup"


async def test_a_whole_call_with_state_and_tasks_matches_the_contract() -> None:
    task_plan = plan(job(), tools=registry())
    tasks = CallTasks(task_plan, state(task_plan))
    call = Call()
    call.recorder.watch_state(tasks.state)
    tasks.listen(call.recorder.task_event)
    await a_spoken_turn(call)
    run = tasks.start("take_note", caller_turns=2)
    assert isinstance(run, DelegateRun)
    step = run.next_step()
    assert step is not None
    step.opened(caller_turns=2)
    await step.finish({"note": "call back"}, caller_turns=3)
    run.ended(step)
    run.result(caller_turns=3)
    await call.push(EndFrame(), 5.0)
    handoff = next(o for o in call.of("operation") if o["type"] == "handoff")
    assert (handoff["name"], handoff["state"]) == ("take_note", "returned")
    [row] = call.of("state")
    assert row["origin"] == {"kind": "step", "name": "take_note"}
    assert row["changes"][0]["field"] == "note"
    assert {r["kind"] for r in call.records} == {
        "call",
        "exchange",
        "text",
        "operation",
        "measurement",
        "breakdown",
        "state",
    }


async def test_every_record_has_the_shape_of_the_unmute_cli_samples() -> None:
    samples: dict[str, set[str]] = {}
    for path in sorted((Path(__file__).parent.parent / "data" / "devmetrics").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line.removeprefix("UNMUTE_METRIC "))
            samples.setdefault(record["kind"], set(record))
    call = Call()
    task_plan = plan(job(), tools=registry())
    call.recorder.watch_state(state(task_plan))
    await a_spoken_turn(call)
    await call.push(EndFrame(), 4.0)
    for record in call.records:
        assert set(record) == samples[record["kind"]]
        assert list(record)[:2] == ["version", "kind"]


async def test_a_spoken_turn_gives_its_turn_decision_speech_length_and_breakdown() -> None:
    call = Call()
    await a_spoken_turn(call)
    await call.push(
        MetricsFrame(data=[TextAggregationMetricsData(processor=call.tts.name, value=0.15)]),
        3.1,
        call.tts,
    )
    await call.push(EndFrame(), 4.0)
    measured = {m["metric"]: m for m in call.of("measurement") if m["state"] == "measured"}
    [reply_id] = [i for (k, i), p in call.latest().items() if p.get("role") == "response"]
    assert measured["speech_duration"]["value"] == pytest.approx(0.9)
    assert measured["turn_detection"]["value"] >= 0
    assert measured["text_aggregation"]["value"] == 0.15
    for metric in ("speech_duration", "turn_detection"):
        assert (measured[metric]["scope"], measured[metric]["exchange_id"]) == (
            "response",
            reply_id,
        )
    [breakdown] = call.of("breakdown")
    assert breakdown["measured_from"] == "user_silence"
    assert breakdown["parts"]
    assert sum(p["duration_secs"] for p in breakdown["parts"]) == pytest.approx(
        breakdown["total_secs"]
    )


async def test_a_control_tool_is_a_handoff_operation_with_no_tool_duration() -> None:
    call = Call(controls={"transfer_to_sales"})
    await call.push(UserStartedSpeakingFrame(), 0.0)
    await call.push(LLMFullResponseStartFrame(), 1.0, call.llm)
    await call.push(FunctionCallInProgressFrame("transfer_to_sales", "c1", {}), 1.2)
    await call.push(FunctionCallInProgressFrame("lookup", "c2", {}), 1.2)
    await call.push(FunctionCallResultFrame("transfer_to_sales", "c1", {}, {"ok": True}), 4.2)
    await call.push(FunctionCallResultFrame("lookup", "c2", {}, {"ok": True}), 1.5)
    ops = {o["name"]: o for o in call.of("operation") if o["type"] in ("tool", "handoff")}
    assert (ops["transfer_to_sales"]["type"], ops["transfer_to_sales"]["state"]) == (
        "handoff",
        "returned",
    )
    assert ops["transfer_to_sales"]["parent_operation_id"] == "llm-3"
    assert ops["lookup"]["type"] == "tool"
    timed = {m["operation_id"] for m in call.of("measurement") if m["metric"] == "tool_duration"}
    assert len(timed) == 1
    assert timed == {
        i for (k, i), p in call.latest().items() if k == "operation" and p["type"] == "tool"
    }
