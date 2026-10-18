"""``DevObserver`` on LiveKit: session events become the records ``unmute dev`` reads.

The scripted cases emit LiveKit's own events on a session that never starts, as a voice call
would send them. The others run a real text-only session with a scripted model. The
conformance scenarios check the same records on both adapters (``Run.dev_problems``).
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from livekit.agents import Agent, AgentSession, function_tool, llm, stt
from livekit.agents.metrics import LLMMetrics, TTSMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.voice import (
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    MetricsCollectedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit.agents.voice.events import (
    AgentState,
    CloseReason,
    EventTypes,
    ToolCallEnded,
    ToolCallStarted,
    ToolExecutionUpdatedEvent,
)
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities

from tests.dev.contract import problems
from tests.livekit.fake import ScriptedLLM, call
from tests.tasks.job import Canned, job, plan, registry, state
from unmute_core.agent import AgentSpec
from unmute_core.call import TRANSFERRED, CallHooks
from unmute_core.dev import DevRecorder
from unmute_core.handoff import HandoffContext
from unmute_core.livekit import DevObserver, LiveKitModel, UnmuteAgent

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}
DATA = Path(__file__).parent.parent / "data" / "devmetrics"


class Ears(stt.STT):
    """A stand-in STT: only its label, model and provider are read."""

    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))

    @property
    def model(self) -> str:
        return "nova-3"

    @property
    def provider(self) -> str:
        return "deepgram"

    async def _recognize_impl(self, *args: Any, **kwargs: Any) -> stt.SpeechEvent:
        raise NotImplementedError


class Records:
    """The records one call sent."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.recorder = DevRecorder("call-1", self.sent.append, target="livekit")

    def latest(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Return each record's last payload, after checking every record against the contract."""
        assert [problems(json.dumps(r)) for r in self.sent] == [[] for _ in self.sent]
        return {(r["kind"], r["id"]): r[r["kind"]] for r in self.sent}

    def of(self, kind: str) -> list[dict[str, Any]]:
        """Return the last payload of each record of a kind, in order."""
        return [p for (k, _), p in self.latest().items() if k == kind]

    def ids(self, kind: str, **fields: Any) -> list[str]:
        """Return the ids of the records of a kind whose payload has these fields."""
        return [
            i
            for (k, i), p in self.latest().items()
            if k == kind and all(p.get(key) == value for key, value in fields.items())
        ]

    def measured(self) -> dict[str, dict[str, Any]]:
        """Return each measured measurement by metric, the first of each."""
        found: dict[str, dict[str, Any]] = {}
        for m in self.of("measurement"):
            if m["state"] == "measured":
                found.setdefault(m["metric"], m)
        return found


class Voice:
    """A voice call scripted as LiveKit's session events, on a session that never starts."""

    def __init__(self) -> None:
        self.records = Records()
        self.session = AgentSession(stt=Ears())
        self.observer = DevObserver(self.records.recorder)
        self.observer.attach(self.session)
        self.agent: AgentState = "listening"

    def emit(self, name: EventTypes, event: Any) -> None:
        """Send one event, as the session would."""
        self.session.emit(name, event)

    def state(self, new: AgentState) -> None:
        """Move the agent to a new state."""
        self.emit(
            "agent_state_changed", AgentStateChangedEvent(old_state=self.agent, new_state=new)
        )
        self.agent = new

    def caller(self, *transcripts: str, metrics: dict[str, float] | None = None) -> None:
        """The caller speaks, is heard word by word, and their turn is committed."""
        self.emit(
            "user_state_changed", UserStateChangedEvent(old_state="listening", new_state="speaking")
        )
        for text in transcripts[:-1]:
            self.emit(
                "user_input_transcribed", UserInputTranscribedEvent(transcript=text, is_final=False)
            )
        self.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript=transcripts[-1], is_final=True),
        )
        self.emit(
            "user_state_changed", UserStateChangedEvent(old_state="speaking", new_state="listening")
        )
        message = llm.ChatMessage(role="user", content=[transcripts[-1]], metrics=metrics or {})
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=message))

    def said(self, text: str, *, interrupted: bool = False, **metrics: float) -> None:
        """The agent's reply joins the chat, with the timings LiveKit puts on it."""
        message = llm.ChatMessage(
            role="assistant", content=[text], interrupted=interrupted, metrics=metrics
        )
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=message))

    def metric(self, metrics: Any) -> None:
        """LiveKit reports one request's metrics."""
        self.emit("metrics_collected", MetricsCollectedEvent(metrics=metrics))


def _llm_metrics(**fields: Any) -> LLMMetrics:
    """One model request's metrics, as LiveKit reports them once it is over."""
    values: dict[str, Any] = {
        "label": "openai.LLM",
        "request_id": "req_1",
        "timestamp": 0.0,
        "duration": 0.8,
        "ttft": 0.3,
        "cancelled": False,
        "completion_tokens": 5,
        "prompt_tokens": 50,
        "prompt_cached_tokens": 0,
        "total_tokens": 55,
        "tokens_per_second": 10.0,
        "metadata": Metadata(model_name="gpt-4.1-mini", model_provider="openai"),
    }
    return LLMMetrics(**(values | fields))


def _tts_metrics(**fields: Any) -> TTSMetrics:
    """One piece of synthesis's metrics, as LiveKit reports them once it is over."""
    values: dict[str, Any] = {
        "label": "cartesia.TTS",
        "request_id": "tts_1",
        "timestamp": 0.0,
        "ttfb": 0.12,
        "duration": 0.6,
        "audio_duration": 1.5,
        "cancelled": False,
        "characters_count": 17,
        "streamed": True,
        "metadata": Metadata(model_name="sonic-2", model_provider="cartesia"),
    }
    return TTSMetrics(**(values | fields))


def a_spoken_turn(call: Voice) -> None:
    """Script one turn: the caller speaks, the model answers, the agent says it."""
    call.caller(
        "Where",
        "Where is my order?",
        metrics={"end_of_turn_delay": 0.4, "transcription_delay": 0.2},
    )
    call.state("thinking")
    call.metric(_llm_metrics())
    call.state("speaking")
    call.metric(_tts_metrics())
    call.said(
        "It is on its way.",
        e2e_latency=1.1,
        started_speaking_at=100.0,
        stopped_speaking_at=101.5,
        playback_latency=0.05,
        llm_node_ttft=0.3,
        llm_node_ttfs=0.45,
        tts_node_ttfb=0.2,
    )
    call.state("listening")


async def test_a_spoken_turn_gives_its_exchanges_text_operations_and_timings() -> None:
    call = Voice()
    a_spoken_turn(call)
    call.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED))
    records = call.records
    latest = records.latest()

    assert latest["call", "call"]["state"] == "ended"
    assert latest["call", "call"]["reason"] == "caller_hung_up"
    assert latest["exchange", "input-1"] == {
        "role": "input",
        "state": "ended",
        "link_status": "known",
    }
    [reply_id] = records.ids("exchange", role="response")
    assert latest["exchange", reply_id] == {
        "role": "response",
        "state": "ended",
        "playback": "ended",
        "link_status": "known",
        "input_id": "input-1",
    }
    texts = records.of("text")
    assert [(t["speaker"], t["text"], t["state"], t["origin"]) for t in texts] == [
        ("user", "Where is my order?", "final", "recognition"),
        ("assistant", "It is on its way.", "final", "generated"),
    ]
    ops = {o["type"]: o for o in records.of("operation")}
    assert {kind: o["state"] for kind, o in ops.items()} == {
        "stt": "ended",
        "llm": "ended",
        "tts": "ended",
    }
    assert (ops["stt"]["model"], ops["stt"]["provider"]) == ("nova-3", "deepgram")
    assert ops["stt"]["exchange_id"] == "input-1"
    assert (ops["llm"]["model"], ops["llm"]["source_request_id"]) == ("gpt-4.1-mini", "req_1")
    assert ops["tts"]["exchange_id"] == reply_id
    measured = records.measured()
    reply_scoped = {
        "turn_detection": 0.4,
        "transcription_delay": 0.2,
        "reply_latency": 1.1,
        "speech_duration": 1.5,
        "playback_delay": 0.05,
        "text_aggregation": pytest.approx(0.15),
    }
    assert {metric: measured[metric]["value"] for metric in reply_scoped} == reply_scoped
    assert {measured[metric]["exchange_id"] for metric in reply_scoped} == {reply_id}
    firsts = [
        m
        for m in records.of("measurement")
        if m["metric"] == "first_response" and m["scope"] == "operation"
    ]
    assert {(m["operation_id"], m["value"]) for m in firsts} == {
        (records.ids("operation", type="llm")[0], 0.3),
        (records.ids("operation", type="tts")[0], 0.12),
    }
    assert measured["first_speech"]["scope"] == "call"
    nodes = {
        key[1].rsplit(":", 1)[1]: p
        for key, p in records.latest().items()
        if key[0] == "measurement" and p["scope"] == "response" and p["metric"] == "first_response"
    }
    assert {field: (m["value"], m["exchange_id"]) for field, m in nodes.items()} == {
        "llm_node_ttft": (0.3, reply_id),
        "tts_node_ttfb": (0.2, reply_id),
    }


async def test_a_tool_reply_spoken_with_say_still_gets_its_reply_latency() -> None:
    call = Voice()
    call.caller("Yes, book it.", metrics={"end_of_turn_delay": 0.4, "stopped_speaking_at": 99.0})
    call.state("thinking")
    call.metric(_llm_metrics())
    call.state("speaking")
    call.said(
        "Lovely, your haircut is booked.", started_speaking_at=100.2, stopped_speaking_at=101.5
    )
    call.state("listening")
    latency = call.records.measured()["reply_latency"]
    assert latency["value"] == pytest.approx(1.2)
    assert latency["source"] == "livekit-chat-message.started_speaking_at"


async def test_the_caller_speaking_over_the_agent_cuts_its_reply_off() -> None:
    call = Voice()
    call.caller("Where is my order?")
    call.state("thinking")
    call.state("speaking")
    call.emit(
        "user_state_changed", UserStateChangedEvent(old_state="listening", new_state="speaking")
    )
    call.said("It is on", interrupted=True)
    call.state("listening")
    records = call.records
    [reply_id] = records.ids("exchange", role="response")
    reply = records.latest()["exchange", reply_id]
    assert (reply["state"], reply["playback"]) == ("interrupted", "interrupted")
    [answer] = [t for t in records.of("text") if t["speaker"] == "assistant"]
    assert answer["state"] == "incomplete"
    inputs = records.ids("exchange", role="input")
    assert [records.latest()["exchange", i]["state"] for i in inputs] == ["ended", "open"]


async def test_a_pause_in_the_callers_speech_keeps_one_input() -> None:
    call = Voice()
    call.emit(
        "user_state_changed", UserStateChangedEvent(old_state="listening", new_state="speaking")
    )
    call.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="Where is", is_final=True)
    )
    call.emit(
        "user_state_changed", UserStateChangedEvent(old_state="speaking", new_state="listening")
    )
    call.caller("my order?")
    rows = [t for t in call.records.of("text") if t["speaker"] == "user"]
    assert [(t["text"], t["separator_before"]) for t in rows] == [
        ("Where is", ""),
        ("my order?", " "),
    ]
    assert len(call.records.ids("exchange", role="input")) == 1


async def test_a_fatal_error_ends_the_call_as_an_error_with_its_name_only() -> None:
    call = Voice()
    error = llm.LLMError(
        timestamp=0.0,
        label="openai.LLM",
        error=RuntimeError("the prompt said X"),
        recoverable=False,
    )
    call.emit("error", ErrorEvent(error=error, source=SimpleNamespace(label="openai.LLM")))
    call.emit("close", CloseEvent(reason=CloseReason.ERROR))
    final = call.records.latest()["call", "call"]
    assert (final["state"], final["reason"]) == ("error", "openai.LLM reported a fatal error.")
    assert "prompt" not in json.dumps(call.records.sent)
    count = len(call.records.sent)
    call.caller("Hello?")
    assert len(call.records.sent) == count


async def test_the_observer_needs_a_livekit_recorder_and_one_session() -> None:
    with pytest.raises(ValueError, match="target='livekit'"):
        DevObserver(DevRecorder("call-1", lambda record: None))
    call = Voice()
    with pytest.raises(ValueError, match="one DevObserver"):
        DevObserver(DevRecorder("call-2", lambda r: None, target="livekit")).attach(call.session)


class Orders(Agent):
    """A plain LiveKit agent with one tool."""

    @function_tool
    async def lookup(self, order_id: str) -> str:
        """Look an order up."""
        return "open"


async def test_a_plain_agent_gets_its_requests_from_livekits_metrics() -> None:
    records = Records()
    model = ScriptedLLM(call("lookup", "Let me look.", order_id="A-12"), "It is open.")
    async with AgentSession(llm=model) as session:
        DevObserver(records.recorder).attach(session)
        await session.start(Orders(instructions="Orders."))
        await session.say("Hello, how can I help?")
        await session.run(user_input="Where is order A-12?")
    latest = records.latest()
    assert latest["call", "call"]["state"] == "ended"
    assert latest["call", "call"]["model_calls"] == "partial"
    greeting, reply = records.ids("exchange", role="response")
    assert latest["exchange", greeting]["link_status"] == "unavailable"
    assert latest["exchange", reply]["input_id"] == records.ids("exchange", role="input")[0]
    texts = records.of("text")
    assert [(t["speaker"], t["text"], t["exchange_id"]) for t in texts] == [
        ("assistant", "Hello, how can I help?", greeting),
        ("user", "Where is order A-12?", records.ids("exchange", role="input")[0]),
        ("assistant", "Let me look.", reply),
        ("assistant", "It is open.", reply),
    ]
    first, second = records.ids("operation", type="llm")
    [tool] = [o for o in records.of("operation") if o["type"] == "tool"]
    assert (tool["name"], tool["state"], tool["parent_operation_id"]) == (
        "lookup",
        "returned",
        first,
    )
    timed = {
        (m["operation_id"], m["metric"])
        for m in records.of("measurement")
        if m["state"] == "measured" and m.get("operation_id")
    }
    assert {(first, "first_response"), (second, "request_duration")} <= timed
    assert ("measurement", f"{first}:first_response") in latest


async def test_unmute_agent_reports_each_request_and_its_task() -> None:
    records = Records()
    task_plan = plan(job(mode="status"), tools=registry(lookup_order=Canned(FOUND)))
    call_state = state(task_plan)
    records.recorder.watch_state(call_state)
    model = ScriptedLLM(call("find_order"), call("lookup_order"), "It is open.")
    agent = UnmuteAgent(AgentSpec("Orders.", tasks=task_plan), state=call_state)
    async with AgentSession(llm=model) as session:
        DevObserver(records.recorder).attach(session)
        await session.start(agent)
        await session.run(user_input="Where is order A-12?")
    latest = records.latest()
    [caller] = records.ids("exchange", role="input")
    [reply] = records.ids("exchange", role="response")
    assert latest["exchange", reply] == {
        "role": "response",
        "state": "ended",
        "playback": "ended",
        "link_status": "known",
        "input_id": caller,
    }
    llms = records.ids("operation", type="llm")
    assert len(llms) == 3
    assert {latest["operation", o]["state"] for o in llms} == {"ended"}
    assert {latest["operation", o]["exchange_id"] for o in llms} == {reply}
    for operation in llms:
        for metric in ("first_response", "request_duration"):
            timing = latest["measurement", f"{operation}:{metric}"]
            assert (timing["state"], timing["source"]) == ("measured", "livekit-llm-node")
    [handoff] = [o for o in records.of("operation") if o["type"] == "handoff"]
    assert (handoff["name"], handoff["state"]) == ("find_order", "returned")
    [tool] = [o for o in records.of("operation") if o["type"] == "tool"]
    assert (tool["name"], tool["parent_operation_id"]) == ("lookup_order", llms[1])
    answers = [t for t in records.of("text") if t["speaker"] == "assistant"]
    assert ("It is open.", "final") in [(t["text"], t["state"]) for t in answers]
    assert [t["text"] for t in answers].count("It is open.") == 1
    [row] = records.of("state")
    assert row["origin"] == {"kind": "step", "name": "find_order"}


async def test_every_record_has_the_shape_of_the_unmute_cli_samples() -> None:
    samples: dict[str, set[str]] = {}
    for path in sorted(DATA.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line.removeprefix("UNMUTE_METRIC "))
            samples.setdefault(record["kind"], set(record))
    call = Voice()
    a_spoken_turn(call)
    call.emit("close", CloseEvent(reason=CloseReason.USER_INITIATED))
    for record in call.records.sent:
        assert set(record) == samples[record["kind"]]
        assert list(record)[:2] == ["version", "kind"]


class Desk(UnmuteAgent):
    """Orders, on a plugin of its own, handing off with a summary that plugin writes."""

    @function_tool
    async def to_notes(self) -> UnmuteAgent | str:
        """Hand the caller to the notes desk."""
        assert self.state is not None
        notes = UnmuteAgent(AgentSpec("You take notes."), state=self.state)
        return await self.hand_off(notes, context=HandoffContext(history="summary"))


async def test_a_plugin_model_gives_one_llm_operation_per_request_and_its_name() -> None:
    records = Records()
    own = ScriptedLLM(call("to_notes"), "The caller wants a note.", name="gpt-4o-mini")
    session_llm = ScriptedLLM("Notes desk.", name="gpt-4o")
    desk = Desk(AgentSpec("You help with orders.", model=LiveKitModel(own)), state=state(plan()))
    async with AgentSession(llm=session_llm) as session:
        DevObserver(records.recorder).attach(session)
        await session.start(desk)
        await session.run(user_input="I need a note taken.")
    assert len(own.requests) == 2, "the handoff and its summary"
    latest = records.latest()
    llms = records.ids("operation", type="llm")
    assert [latest["operation", o]["model"] for o in llms] == ["gpt-4o-mini", "gpt-4o"]
    assert {latest["operation", o]["state"] for o in llms} == {"ended"}
    timings = {latest["measurement", f"{o}:request_duration"]["source"] for o in llms}
    assert timings == {"livekit-llm-node"}


async def test_a_control_tool_is_a_handoff_operation_and_a_hang_up_is_not_a_failure() -> None:
    records = Records()
    session = AgentSession()
    DevObserver(records.recorder, controls={"transfer_to_sales"}).attach(session)

    def tool(name: str, call_id: str, at: float) -> None:
        session.emit(
            "tool_execution_updated",
            ToolExecutionUpdatedEvent(
                update=ToolCallStarted(
                    function_call=llm.FunctionCall(call_id=call_id, name=name, arguments="{}")
                ),
                created_at=at,
            ),
        )

    def end(call_id: str, status: Any, at: float) -> None:
        session.emit(
            "tool_execution_updated",
            ToolExecutionUpdatedEvent(
                update=ToolCallEnded(id=call_id, call_id=call_id, status=status), created_at=at
            ),
        )

    tool("transfer_to_sales", "c1", 1.0)
    tool("lookup", "c2", 1.0)
    end("c2", "error", 2.0)
    session._closing = True
    end("c1", "error", 5.0)
    ops = {o["name"]: o for o in records.of("operation")}
    assert (ops["transfer_to_sales"]["type"], ops["transfer_to_sales"]["state"]) == (
        "handoff",
        "cancelled",
    )
    assert (ops["lookup"]["type"], ops["lookup"]["state"]) == ("tool", "failed")
    timed = [m["operation_id"] for m in records.of("measurement") if m["metric"] == "tool_duration"]
    assert timed == [records.ids("operation", type="tool")[0]]


class Playback(AudioOutput):
    """An audio output that sends no audio and says when playback starts."""

    def __init__(self) -> None:
        super().__init__(label="test", capabilities=AudioOutputCapabilities(pause=False))

    async def capture_frame(self, frame: Any) -> None:
        return None

    def flush(self) -> None:
        return None

    def clear_buffer(self) -> None:
        return None


async def test_first_speech_is_the_earliest_playback_start_of_the_audio_output() -> None:
    records = Records()
    session = AgentSession()
    output = Playback()
    session.output.audio = output
    observer = DevObserver(records.recorder)
    observer.attach(session)
    base = observer._timings.attached_at
    output.on_playback_started(created_at=base + 3.0)
    output.on_playback_started(created_at=base + 1.5)
    output.on_playback_started(created_at=base + 2.0)
    session.emit(
        "agent_state_changed",
        AgentStateChangedEvent(old_state="listening", new_state="speaking", created_at=base + 0.5),
    )
    firsts = [m for m in records.of("measurement") if m["metric"] == "first_speech"]
    assert [(m["source"], pytest.approx(m["value"])) for m in firsts] == [
        ("livekit-audio-output", pytest.approx(1.5))
    ]


async def test_unseen_requests_and_unclaimed_text_limit_the_call_record() -> None:
    call = Voice()
    assert call.records.latest()["call", "call"]["generated_text"] == "available"
    call.caller("Hi")
    call.said("Welcome.")
    assert call.records.latest()["call", "call"]["generated_text"] == "limited"
    assert call.records.latest()["call", "call"]["model_calls"] == "known"
    call.metric(_llm_metrics())
    assert call.records.latest()["call", "call"]["model_calls"] == "partial"


async def test_the_dev_record_ends_with_the_reason_the_call_asked_for() -> None:
    call = Voice()
    hooks = CallHooks()
    call.observer.entered(SimpleNamespace(call=hooks, handoffs=()))
    hooks.end_requested(TRANSFERRED)
    call.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED))
    assert call.records.latest()["call", "call"]["reason"] == "transferred"
