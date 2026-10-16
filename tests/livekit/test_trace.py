"""A LiveKit session gives the call's spans, with the parents docs/core/observe/telemetry.mdx lists."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest
from livekit import rtc
from livekit.agents import NOT_GIVEN, AgentSession, inference, llm
from livekit.agents.voice import Agent, ModelSettings
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.livekit.fake import ScriptedLLM, call
from tests.tasks.job import Canned, job, plan, registry, state
from unmute_core.agent import AgentSpec
from unmute_core.livekit import UnmuteAgent
from unmute_core.livekit._model import ModelNode
from unmute_core.livekit._trace import SessionTrace, traced_tts
from unmute_core.model import ModelEvent, ModelRequest, NativeModelCall, TextDelta
from unmute_core.telemetry import Sink, Telemetry

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}


@dataclass(kw_only=True)
class Memory(Sink):
    """Keeps the spans in memory."""

    spans: InMemorySpanExporter = field(default_factory=InMemorySpanExporter)

    def exporter(self) -> SpanExporter:
        return self.spans


async def test_a_session_gives_the_call_turn_model_tool_and_step_spans() -> None:
    sink = Memory()
    task_plan = plan(job(mode="status"), tools=registry(lookup_order=Canned(FOUND)))
    model = ScriptedLLM(call("find_order"), call("lookup_order"), "It is open.")
    agent = UnmuteAgent(
        AgentSpec("Orders.", tasks=task_plan),
        state=state(task_plan),
        telemetry=Telemetry(sink, agent="orders", call_id="RM_1"),
    )
    async with AgentSession(llm=model) as session:
        await session.start(agent)
        await session.run(user_input="Where is order A-12?")
    assert agent._trace is not None and agent._trace.closing is not None
    assert await agent._trace.closing

    spans = sink.spans.get_finished_spans()
    by_id = {span.context.span_id: span for span in spans}

    def parent(name: str) -> str:
        span = next(span for span in spans if span.name == name)
        return by_id[span.parent.span_id].name if span.parent else ""

    assert sorted({span.name for span in spans}) == [
        "conversation",
        "llm",
        "llm_tool_call",
        "task.completed",
        "task.ended_on_tool",
        "task.started",
        "task_step",
        "turn",
    ]
    assert parent("turn") == "conversation"
    assert parent("llm") == "turn"
    assert parent("llm_tool_call") == "turn"
    assert parent("task_step") == "conversation"
    attributes = [(span.name, dict(span.attributes or {})) for span in spans]
    assert "task:find_order" in [a.get("unmute.prompt.site") for n, a in attributes if n == "llm"]
    [call_span] = [a for n, a in attributes if n == "conversation"]
    assert call_span["unmute.call.framework"] == "livekit"
    [turn] = [a for n, a in attributes if n == "turn"]
    assert turn["input.value"] == "Where is order A-12?"
    assert turn["output.value"] == "Let me look. It is open."
    [step] = [a for n, a in attributes if n == "task_step"]
    assert step["unmute.task.outcome"] == "completed"
    assert json.loads(str(step["output.value"]))["outcome"] == "completed"


async def test_each_model_request_shows_its_prompt_tools_and_choice() -> None:
    sink = Memory()
    task_plan = plan(job(mode="status"), tools=registry(lookup_order=Canned(FOUND)))
    model = ScriptedLLM(call("find_order"), call("lookup_order"), "It is open.")
    agent = UnmuteAgent(
        AgentSpec("Orders for {{order_id}}.", tasks=task_plan),
        state=state(task_plan),
        telemetry=Telemetry(sink, agent="orders", call_id="RM_1"),
    )
    async with AgentSession(llm=model) as session:
        await session.start(agent)
        await session.run(user_input="Where is order A-12?")
    assert agent._trace is not None and agent._trace.closing is not None
    assert await agent._trace.closing

    spans = sink.spans.get_finished_spans()
    requests = [_attrs(span) for span in spans if span.name == "llm"]
    first = requests[0]
    sent = json.loads(first["input.value"])
    assert sent[0]["role"] == "system" and "{{" not in sent[0]["content"]
    assert sent[-1] == {"role": "user", "content": "Where is order A-12?"}
    offered = [tool["function"]["name"] for tool in json.loads(first["gen_ai.tool.definitions"])]
    assert offered == model.requests[0].tools
    chose = json.loads(first["output.value"])
    assert [c["function"]["name"] for c in chose["tool_calls"]] == ["find_order"]
    assert "langfuse.observation.completion_start_time" in first
    call_span = next(_attrs(s) for s in spans if s.name == "conversation")
    assert json.loads(call_span["input.value"])[0] == {
        "role": "user",
        "content": "Where is order A-12?",
    }
    assert json.loads(call_span["output.value"])["state"]["status"] == "open"


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    """A finished span's attributes as a plain dict."""
    return dict(span.attributes or {})


def _trace(sink: Memory) -> SessionTrace:
    """A trace attached to a stand-in session that emits nothing."""
    trace = SessionTrace(Telemetry(sink, agent="orders", call_id="RM_1"))
    trace.attach(cast(Any, SimpleNamespace(on=lambda *args: None)))
    return trace


async def test_the_callers_spoken_message_gives_an_stt_span() -> None:
    sink = Memory()
    trace = _trace(sink)
    spoken = llm.ChatMessage(
        role="user",
        content=["Where is my order?"],
        metrics={
            "started_speaking_at": 1_000.0,
            "transcription_delay": 0.25,
            "stt_metadata": {"model_name": "nova-3", "model_provider": "deepgram"},
        },
    )
    trace._transcribed(cast(Any, SimpleNamespace(is_final=True, language="en")))
    trace._item_added(cast(Any, SimpleNamespace(item=spoken)))
    trace._item_added(cast(Any, SimpleNamespace(item=llm.ChatMessage(role="user", content=["hi"]))))
    await trace.telemetry.close()

    [stt] = [span for span in sink.spans.get_finished_spans() if span.name == "stt"]
    assert stt.start_time == 1_000 * 10**9
    attributes = dict(stt.attributes or {})
    assert attributes["output.value"] == "Where is my order?"
    assert attributes["gen_ai.request.model"] == "nova-3"
    assert attributes["metrics.ttfb"] == 0.25
    assert attributes["stt.language"] == "en"
    assert attributes["gen_ai.provider.name"] == "deepgram"


@pytest.mark.parametrize("cut", [False, True])
async def test_tts_node_gives_a_tts_span_per_piece_of_speech(
    monkeypatch: pytest.MonkeyPatch, cut: bool
) -> None:
    sink = Memory()
    trace = _trace(sink)

    async def node(agent: Any, text: Any, settings: Any) -> AsyncIterator[rtc.AudioFrame]:
        async for _ in text:
            yield rtc.AudioFrame.create(24000, 1, 240)

    monkeypatch.setattr(Agent.default, "tts_node", node)
    plugin = SimpleNamespace(model="sonic", provider="cartesia", _opts=SimpleNamespace(voice="v1"))
    agent = cast(Any, SimpleNamespace(tts=NOT_GIVEN, session=SimpleNamespace(tts=plugin)))

    async def text() -> AsyncIterator[str]:
        for chunk in ("Hello ", "there."):
            yield chunk

    frames = cast(AsyncGenerator[rtc.AudioFrame], traced_tts(agent, trace, text(), cast(Any, None)))
    if cut:
        await anext(frames)
        await frames.aclose()
    else:
        assert len([frame async for frame in frames]) == 2
    await trace.telemetry.close()

    [tts] = [_attrs(s) for s in sink.spans.get_finished_spans() if s.name == "tts"]
    assert tts["input.value"] == ("Hello " if cut else "Hello there.")
    assert (tts["gen_ai.request.model"], tts["tts.voice"]) == ("sonic", "v1")
    assert tts["tts.interrupted"] is cut
    assert tts["metrics.ttfb"] >= 0


def _add(trace: SessionTrace, item: llm.ChatMessage) -> None:
    """Hand the trace one item as LiveKit's ``conversation_item_added`` does."""
    trace._item_added(cast(Any, SimpleNamespace(item=item)))


def _say(text: str, *, interrupted: bool = False) -> llm.ChatMessage:
    """One line the agent said."""
    return llm.ChatMessage(role="assistant", content=[text], interrupted=interrupted)


async def _turns(trace: SessionTrace, sink: Memory) -> list[dict[str, Any]]:
    """Close the session and return each turn's attributes, in order."""
    trace._closed(cast(Any, SimpleNamespace(reason=SimpleNamespace(value="user_initiated"))))
    assert trace.closing is not None
    await trace.closing
    spans = sorted(sink.spans.get_finished_spans(), key=lambda span: span.start_time or 0)
    return [_attrs(span) for span in spans if span.name == "turn"]


async def test_a_turn_holds_its_caller_and_every_line_until_the_next_caller() -> None:
    sink = Memory()
    trace = _trace(sink)
    booking = llm.ChatMessage(role="user", content=["Book me in."])
    trace.open_turn(booking)
    _add(trace, _say("One sec."))
    trace.open_turn(booking)
    _add(trace, booking)
    _add(trace, _say("You are booked."))
    thanks = llm.ChatMessage(role="user", content=["Thanks."])
    trace.open_turn(thanks)
    _add(trace, thanks)
    _add(trace, _say("Bye."))

    turns = await _turns(trace, sink)

    assert [(t["input.value"], t["output.value"]) for t in turns] == [
        ("Book me in.", "One sec. You are booked."),
        ("Thanks.", "Bye."),
    ]


async def test_the_agent_speaking_first_gets_its_own_turn() -> None:
    sink = Memory()
    trace = _trace(sink)
    _add(trace, _say("Hello, how can I help?"))
    question = llm.ChatMessage(role="user", content=["Is it open?"])
    _add(trace, question)
    trace.open_turn(question)
    _add(trace, _say("Yes."))

    turns = await _turns(trace, sink)

    assert [(t.get("input.value"), t["output.value"]) for t in turns] == [
        (None, "Hello, how can I help?"),
        ("Is it open?", "Yes."),
    ]


async def test_a_caller_who_speaks_over_the_agent_starts_the_next_turn() -> None:
    sink = Memory()
    trace = _trace(sink)
    first = llm.ChatMessage(role="user", content=["Where is my order?"])
    trace.open_turn(first)
    _add(trace, first)
    _add(trace, _say("Let me", interrupted=True))
    second = llm.ChatMessage(role="user", content=["It is A-12."])
    trace.open_turn(second)
    _add(trace, second)
    _add(trace, _say("A-12 is open."))

    turns = await _turns(trace, sink)

    assert [t["input.value"] for t in turns] == ["Where is my order?", "It is A-12."]
    assert [t["turn.interrupted"] for t in turns] == [True, False]


async def test_a_spoken_line_kept_out_of_the_chat_still_counts() -> None:
    sink = Memory()
    trace = _trace(sink)
    done = llm.ChatMessage(role="user", content=["Yes, save it."])
    trace.open_turn(done)
    _add(trace, done)
    trace.said("Saved.")
    _add(trace, _say("Anything else?"))

    [turn] = await _turns(trace, sink)

    assert turn["output.value"] == "Saved. Anything else?"


async def test_the_rest_of_a_split_caller_line_joins_the_turn() -> None:
    sink = Memory()
    trace = _trace(sink)
    first = llm.ChatMessage(role="user", content=["Yeah, it"])
    trace.open_turn(first, answering=False)
    _add(trace, first)
    rest = llm.ChatMessage(role="user", content=["'s +34600111222."])
    trace.open_turn(rest, answering=False)
    _add(trace, rest)
    _add(trace, _say("Is that right?"))

    [turn] = await _turns(trace, sink)

    assert turn["input.value"] == "Yeah, it 's +34600111222."
    assert turn["output.value"] == "Is that right?"


async def test_a_caller_message_that_replaces_a_piece_of_it_is_said_once() -> None:
    sink = Memory()
    trace = _trace(sink)
    piece = llm.ChatMessage(role="user", content=["Great, thank you."])
    trace.open_turn(piece, answering=False)
    _add(trace, piece)
    whole = llm.ChatMessage(role="user", content=["Great, thank you. Nothing else."])
    trace.open_turn(whole, answering=False)
    _add(trace, whole)
    _add(trace, _say("Goodbye."))

    [turn] = await _turns(trace, sink)

    assert turn["input.value"] == "Great, thank you. Nothing else."


async def test_speech_over_the_agent_shows_how_livekit_judged_it() -> None:
    sink = Memory()
    trace = _trace(sink)
    trace.open_turn()
    trace.over_agent(inference.OverlappingSpeechEvent(is_interruption=False, probability=0.123))

    await _turns(trace, sink)

    [event] = [_attrs(s) for s in sink.spans.get_finished_spans() if s.name == "caller.over_agent"]
    assert json.loads(event["input.value"]) == {
        "interruption": False,
        "agent_ended": False,
        "probability": 0.12,
        "detection_delay": 0.0,
    }


async def test_turn_latencies_come_from_the_first_message_that_has_them() -> None:
    sink = Memory()
    trace = _trace(sink)
    booking = llm.ChatMessage(
        role="user",
        content=["Book me in."],
        metrics={"transcription_delay": 0.3, "end_of_turn_delay": 1.6},
    )
    trace.open_turn(booking)
    _add(trace, booking)
    _add(trace, _say("One sec."))
    reply = _say("Which day?")
    reply.metrics = {"e2e_latency": 1.2, "llm_node_ttft": 0.6, "tts_node_ttfb": 0.2}
    _add(trace, reply)

    [turn] = await _turns(trace, sink)

    assert (turn["turn.stt_final_seconds"], turn["turn.e2e_seconds"]) == (0.3, 1.2)
    assert turn["turn.eou_seconds"] == 1.6
    assert (turn["turn.llm_ttft_seconds"], turn["turn.tts_ttfb_seconds"]) == (0.6, 0.2)


async def test_a_reply_cut_off_before_it_ends_is_a_warning() -> None:
    sink = Memory()
    trace = _trace(sink)

    async def endless(request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield TextDelta("Let me")
        await asyncio.Event().wait()

    node = ModelNode(state=None, model=NativeModelCall(endless), trace=trace)
    stream = cast(
        AsyncGenerator[Any],
        node.stream(
            Agent(instructions="Orders."),
            llm.ChatContext(),
            [],
            ModelSettings(),
            prompt="Orders.",
            site="",
        ),
    )
    await anext(stream)
    await stream.aclose()
    await trace.telemetry.close()

    [request] = [_attrs(span) for span in sink.spans.get_finished_spans() if span.name == "llm"]
    assert request["langfuse.observation.level"] == "WARNING"
    assert request["llm.interrupted"] is True
    assert request["output.value"] == "Let me"
