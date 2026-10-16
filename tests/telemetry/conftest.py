"""In-memory stand-ins for the network exporter, and one scripted call that uses every span."""

from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from unmute_core.state import CallState
from unmute_core.telemetry import Telemetry, _coval, _sinks

TOOLS = [{"type": "function", "function": {"name": "find_slots", "parameters": {}}}]
CALL = {"id": "call_1", "type": "function", "function": {"name": "find_slots", "arguments": "{}"}}
SPANS = {"conversation", "turn", "stt", "llm", "llm_tool_call", "tts", "task.started", "task_step"}


class Booking(BaseModel):
    """The state of the scripted call."""

    phone: str | None = None


class RecordingExporter(InMemorySpanExporter):
    """Keeps the spans in memory and the settings the sink built the OTLP exporter with."""

    def __init__(self, **settings: Any) -> None:
        super().__init__()
        self.settings = settings


@pytest.fixture
def exporters(monkeypatch: pytest.MonkeyPatch) -> list[RecordingExporter]:
    """Every OTLP exporter a sink builds during the test, in order."""
    created: list[RecordingExporter] = []

    def build(**settings: Any) -> RecordingExporter:
        exporter = RecordingExporter(**settings)
        created.append(exporter)
        return exporter

    monkeypatch.setattr(_sinks, "OTLPSpanExporter", build)
    monkeypatch.setattr(_coval, "OTLPSpanExporter", build)
    return created


async def run_call(telemetry: Telemetry) -> None:
    """One call with every span: a step, a turn with speech, a model request and a tool call."""
    telemetry.start_call()
    step = telemetry.start_task_step(task="booking", step="collect_phone")
    turn = telemetry.start_turn(1)
    telemetry.stt(
        text="My number is 555 0100", model="nova-3", provider="deepgram", language="en", ttfb=0.2
    )
    telemetry.event("task.started", {"task": "booking", "step": 0})
    llm = telemetry.start_model_request(
        model="gpt-4.1",
        provider="openai",
        site="task:booking",
        messages=[{"role": "user", "content": "My number is 555 0100"}],
        tools=TOOLS,
        parent=turn,
    )
    telemetry.end_model_request(
        llm,
        text="Let me check.",
        tool_calls=[CALL],
        input_tokens=12,
        output_tokens=4,
        finish_reason="tool_calls",
        ttfb=0.3,
    )
    tts = telemetry.start_tts(model="sonic", provider="cartesia", voice="v1")
    telemetry.end_tts(tts, text="Let me check.", ttfb=0.1)
    tool = telemetry.start_tool_call(
        name="find_slots", call_id="call_1", arguments='{"day": "2026-10-07"}', parent=turn
    )
    telemetry.end_tool_call(tool, error="TimeoutError")
    telemetry.end_turn(
        turn,
        user_text="My number is 555 0100",
        agent_lines=["Let me check."],
        e2e=1.1,
        stt_final=0.29,
        llm_ttft=0.2,
        tts_ttfb=0.34,
    )
    telemetry.end_task_step(step, outcome="completed", saved={"phone": "555 0100"})
    state = CallState(Booking, confirm={"phone": "collect_phone"})
    state.save({"phone": "555 0100"}, step="collect_phone")
    assert await telemetry.close(state=state, reason="user_initiated")


def attrs(span: ReadableSpan) -> dict[str, Any]:
    """Return a finished span's attributes as a plain dict."""
    return dict(span.attributes or {})


def parent_id(span: ReadableSpan) -> int | None:
    """Return the span ID of a finished span's parent."""
    return span.parent.span_id if span.parent else None


def by_name(spans: tuple[ReadableSpan, ...] | list[ReadableSpan]) -> dict[str, ReadableSpan]:
    """Index finished spans by name. Each name appears once in ``run_call``."""
    return {span.name: span for span in spans}
