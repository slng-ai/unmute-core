"""Telemetry emits the shared span tree on a TracerProvider of its own."""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime

import pytest
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from tests.telemetry.conftest import CALL, SPANS, TOOLS, attrs, by_name, parent_id, run_call
from unmute_core.telemetry import Sink, Telemetry
from unmute_core.telemetry._telemetry import upstream_provider


@dataclass(kw_only=True)
class Memory(Sink):
    spans: InMemorySpanExporter = field(default_factory=InMemorySpanExporter)

    def exporter(self) -> SpanExporter:
        return self.spans


async def test_span_tree_and_call_identity_on_every_span():
    sink = Memory()
    await run_call(Telemetry(sink, agent="salon", call_id="RM_abc"))

    spans = by_name(sink.spans.get_finished_spans())
    assert set(spans) == SPANS
    call = spans["conversation"]
    assert call.parent is None
    for name in ("turn", "task_step"):
        assert parent_id(spans[name]) == call.context.span_id
    for name in ("llm", "llm_tool_call", "stt", "tts", "task.started"):
        assert parent_id(spans[name]) == spans["turn"].context.span_id
    assert {span.context.trace_id for span in spans.values()} == {call.context.trace_id}
    for span in spans.values():
        assert attrs(span)["session.id"] == "RM_abc"
        assert attrs(span)["gen_ai.conversation.id"] == "RM_abc"
        assert attrs(span)["agent.name"] == "salon"
        assert span.resource.attributes["service.name"] == "salon"


async def test_span_attributes_follow_the_contract():
    sink = Memory()
    await run_call(Telemetry(sink, agent="salon", call_id="RM_abc"))
    spans = by_name(sink.spans.get_finished_spans())

    assert attrs(spans["conversation"])["gen_ai.operation.name"] == "invoke_workflow"
    assert attrs(spans["conversation"])["langfuse.trace.name"] == "salon"
    turn = attrs(spans["turn"])
    assert turn["turn.number"] == 1
    assert turn["gen_ai.operation.name"] == "invoke_agent"
    assert turn["input.value"] == "My number is 555 0100"
    assert turn["output.value"] == "Let me check."
    assert turn["turn.interrupted"] is False
    assert turn["turn.e2e_seconds"] == 1.1
    assert turn["turn.tts_ttfb_seconds"] == 0.34
    llm = attrs(spans["llm"])
    assert llm["gen_ai.operation.name"] == "chat"
    assert llm["gen_ai.request.model"] == "gpt-4.1"
    assert llm["gen_ai.provider.name"] == "openai"
    assert llm["unmute.prompt.site"] == "task:booking"
    assert '"My number is 555 0100"' in llm["input.value"]
    assert llm["gen_ai.usage.input_tokens"] == 12
    assert llm["gen_ai.response.finish_reasons"] == ("tool_calls",)
    assert llm["metrics.ttfb"] == 0.3
    tool = spans["llm_tool_call"]
    assert attrs(tool)["gen_ai.operation.name"] == "execute_tool"
    assert attrs(tool)["gen_ai.tool.name"] == attrs(tool)["function.name"] == "find_slots"
    assert attrs(tool)["gen_ai.tool.call.id"] == attrs(tool)["tool_call_id"] == "call_1"
    assert attrs(tool)["tool.error"] == 1
    assert attrs(tool)["error.type"] == "TimeoutError"
    assert tool.status.status_code is StatusCode.ERROR
    step = attrs(spans["task_step"])
    assert (step["unmute.task.name"], step["unmute.task.step"]) == ("booking", "collect_phone")
    assert step["unmute.task.outcome"] == "completed"
    assert json.loads(step["output.value"]) == {
        "outcome": "completed",
        "saved": {"phone": "555 0100"},
    }


async def test_a_model_request_shows_the_tools_offered_and_the_calls_chosen():
    sink = Memory()
    await run_call(Telemetry(sink, agent="salon", call_id="RM_abc"))
    span = by_name(sink.spans.get_finished_spans())["llm"]
    llm = attrs(span)

    assert json.loads(llm["gen_ai.tool.definitions"]) == TOOLS
    assert json.loads(llm["output.value"]) == {
        "role": "assistant",
        "content": "Let me check.",
        "tool_calls": [CALL],
    }
    started = datetime.fromisoformat(llm["langfuse.observation.completion_start_time"])
    assert started.timestamp() == pytest.approx((span.start_time or 0) / 1e9 + 0.3)


async def test_speech_spans_and_events_sit_in_the_turn():
    sink = Memory()
    await run_call(Telemetry(sink, agent="salon", call_id="RM_abc"))
    spans = by_name(sink.spans.get_finished_spans())

    stt = attrs(spans["stt"])
    assert stt["langfuse.observation.type"] == "generation"
    assert (stt["gen_ai.request.model"], stt["gen_ai.provider.name"]) == ("nova-3", "deepgram")
    assert (stt["output.value"], stt["stt.language"], stt["metrics.ttfb"]) == (
        "My number is 555 0100",
        "en",
        0.2,
    )
    tts = attrs(spans["tts"])
    assert (tts["input.value"], tts["tts.voice"], tts["metrics.character_count"]) == (
        "Let me check.",
        "v1",
        13,
    )
    assert json.loads(tts["langfuse.observation.usage_details"]) == {"characters": 13}
    assert tts["tts.interrupted"] is False
    event = spans["task.started"]
    assert attrs(event)["langfuse.observation.type"] == "event"
    assert json.loads(attrs(event)["input.value"]) == {"task": "booking", "step": 0}
    assert event.start_time == event.end_time


async def test_the_call_carries_the_transcript_state_and_end_reason():
    sink = Memory()
    await run_call(Telemetry(sink, agent="salon", call_id="RM_abc"))
    call = attrs(by_name(sink.spans.get_finished_spans())["conversation"])

    assert json.loads(call["input.value"]) == [
        {"role": "user", "content": "My number is 555 0100"},
        {"role": "assistant", "content": "Let me check."},
    ]
    assert json.loads(call["output.value"]) == {
        "end_reason": "user_initiated",
        "state": {"phone": "555 0100"},
        "confirmed": ["phone"],
    }
    assert call["unmute.call.end_reason"] == "user_initiated"


async def test_each_line_of_a_turn_is_its_own_transcript_entry():
    sink = Memory()
    telemetry = Telemetry(sink, agent="salon", call_id="RM_abc")
    telemetry.start_call()
    telemetry.end_turn(
        telemetry.start_turn(1), user_text="Book me in.", agent_lines=["One sec.", "Done."]
    )
    await telemetry.close()
    spans = by_name(sink.spans.get_finished_spans())

    assert attrs(spans["turn"])["output.value"] == "One sec. Done."
    assert json.loads(attrs(spans["conversation"])["input.value"]) == [
        {"role": "user", "content": "Book me in."},
        {"role": "assistant", "content": "One sec."},
        {"role": "assistant", "content": "Done."},
    ]


async def test_a_long_prompt_is_kept_whole():
    sink = Memory()
    telemetry = Telemetry(sink, agent="salon", call_id="RM_abc")
    telemetry.start_call()
    prompt = [{"role": "system", "content": "x" * 40_000}]
    telemetry.end_model_request(
        telemetry.start_model_request(model="m", provider="p", messages=prompt)
    )
    await telemetry.close()

    assert (
        json.loads(attrs(by_name(sink.spans.get_finished_spans())["llm"])["input.value"]) == prompt
    )


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, None])
async def test_bad_latencies_are_left_out(bad):
    sink = Memory()
    telemetry = Telemetry(sink, agent="salon", call_id="RM_abc")
    telemetry.start_call()
    telemetry.end_turn(telemetry.start_turn(1), e2e=bad, llm_ttft=0.2)
    await telemetry.close()

    turn = attrs(by_name(sink.spans.get_finished_spans())["turn"])
    assert "turn.e2e_seconds" not in turn
    assert turn["turn.llm_ttft_seconds"] == 0.2


async def test_framework_spans_on_the_provider_get_the_call_identity():
    sink = Memory()
    telemetry = Telemetry(
        sink, agent="salon", call_id="RM_abc", resource={"deployment.environment.name": "prod"}
    )
    telemetry.provider.get_tracer("livekit-agents").start_span("agent_turn").end()
    await telemetry.close()

    (span,) = sink.spans.get_finished_spans()
    assert attrs(span)["session.id"] == "RM_abc"
    assert span.resource.attributes["deployment.environment.name"] == "prod"


async def test_headers_carry_the_span_as_traceparent():
    telemetry = Telemetry(Memory(), agent="salon", call_id="RM_abc")
    telemetry.start_call()
    llm = telemetry.start_model_request(model="m", provider="p")

    traceparent = telemetry.headers(llm)["traceparent"]
    assert traceparent.split("-")[2] == format(llm.get_span_context().span_id, "016x")
    await telemetry.close()


async def test_spans_need_the_call_span_first():
    telemetry = Telemetry(Memory(), agent="salon", call_id="RM_abc")
    with pytest.raises(RuntimeError, match="start_call"):
        telemetry.start_turn(1)
    telemetry.start_call()
    with pytest.raises(RuntimeError, match="one Telemetry per call"):
        telemetry.start_call()


async def test_close_ends_the_call_span_and_flushes():
    sink = Memory()
    telemetry = Telemetry(sink, agent="salon", call_id="RM_abc")
    call = telemetry.start_call()

    assert await telemetry.close()
    assert not call.is_recording()
    assert [span.name for span in sink.spans.get_finished_spans()] == ["conversation"]


def test_the_call_identity_processor_reports_its_flush_done():
    from unmute_core.telemetry._telemetry import _CallAttributes

    assert _CallAttributes({}).force_flush() is True


def test_needs_agent_and_call_id():
    with pytest.raises(ValueError, match="call_id"):
        Telemetry(agent="salon", call_id="")


@pytest.mark.parametrize(
    ("provider", "model", "upstream"),
    [
        ("slng", "cartesia/sonic:3.5", "cartesia"),
        ("SLNG", "soniox/speech-ai:rt-v5", "soniox"),
        ("slng", "slng/deepgram/nova:3-en", "deepgram"),
        ("slng", "SLNG/Cartesia/sonic", "cartesia"),
        ("slng", "slng", None),
        ("deepgram", "nova-3", None),
        ("deepgram", "deepgram/nova-3", None),
        ("openrouter", "anthropic/claude", "anthropic"),
        ("my-gateway", "acme/voice-1", None),
    ],
)
def test_a_gateway_names_the_upstream_from_the_model_id(provider, model, upstream):
    assert upstream_provider(provider, model) == upstream


def test_a_gateway_of_your_own_names_its_upstream():
    assert upstream_provider("my-gateway", "acme/voice-1", {"my-gateway"}) == "acme"


async def test_slng_speech_spans_name_their_upstream():
    sink = Memory()
    telemetry = Telemetry(sink, agent="salon", call_id="RM_abc")
    telemetry.start_call()
    telemetry.stt(text="Hi", model="soniox/speech-ai:rt-v5", provider="SLNG")
    telemetry.end_tts(telemetry.start_tts(model="cartesia/sonic:3.5", provider="slng"), text="Hi")
    await telemetry.close()

    spans = by_name(sink.spans.get_finished_spans())
    assert attrs(spans["stt"])["unmute.upstream.provider"] == "soniox"
    assert attrs(spans["tts"])["unmute.upstream.provider"] == "cartesia"
    assert attrs(spans["tts"])["gen_ai.provider.name"] == "slng"


async def test_describe_call_before_start_waits_for_the_call_span():
    sink = Memory()
    telemetry = Telemetry(sink, agent="salon", call_id="RM_abc")
    telemetry.describe_call(origin="phone")
    telemetry.describe_call(framework="livekit")
    await run_call(telemetry)

    call = attrs(by_name(sink.spans.get_finished_spans())["conversation"])
    assert call["unmute.call.origin"] == "phone"
    assert call["unmute.call.framework"] == "livekit"


async def test_describe_call_after_start_sets_the_attributes_and_a_later_value_wins():
    sink = Memory()
    telemetry = Telemetry(sink, agent="salon", call_id="RM_abc")
    telemetry.start_call()
    telemetry.describe_call(origin="browser", framework="pipecat")
    telemetry.describe_call(origin="phone")
    await telemetry.close()

    call = attrs(by_name(sink.spans.get_finished_spans())["conversation"])
    assert call["unmute.call.origin"] == "phone"
    assert call["unmute.call.framework"] == "pipecat"


async def test_describe_call_after_close_does_nothing():
    sink = Memory()
    telemetry = Telemetry(sink, agent="salon", call_id="RM_abc")
    telemetry.start_call()
    await telemetry.close()
    telemetry.describe_call(origin="phone")

    call = attrs(by_name(sink.spans.get_finished_spans())["conversation"])
    assert "unmute.call.origin" not in call


async def test_describe_call_with_nothing_sets_nothing():
    sink = Memory()
    telemetry = Telemetry(sink, agent="salon", call_id="RM_abc")
    telemetry.describe_call()
    await run_call(telemetry)

    call = attrs(by_name(sink.spans.get_finished_spans())["conversation"])
    assert not [
        key for key in call if key.startswith("unmute.call.") and key != "unmute.call.end_reason"
    ]
