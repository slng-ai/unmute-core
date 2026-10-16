"""Coval's sink: ID lookup, filing a call no simulation placed, and Coval's own span names."""

import json
import logging
from typing import Any, cast

import httpx
import pytest
from opentelemetry.sdk.trace import ReadableSpan

from tests.telemetry.conftest import attrs, by_name, run_call
from unmute_core.telemetry import (
    Coval,
    CovalSettings,
    Logfire,
    MetadataOnly,
    Otlp,
    Telemetry,
    _coval,
    coval_simulation_id,
)
from unmute_core.telemetry._coval import (
    COVAL_SUBMIT_ENDPOINT,
    COVAL_SUBMIT_TIMEOUT_SECONDS,
    MAX_PROMPT_CHARS,
    MAX_TEXT_CHARS,
    _Correlation,
    _coval_shape,
    _submit,
    _transcript_of,
)

TRANSCRIPT = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]


def shaped(
    name: str, attributes: dict, start_time: int | None = None, end_time: int | None = None
) -> dict:
    """Return the attributes Coval would get for a span with these attributes."""
    span = ReadableSpan(name=name, attributes=attributes, start_time=start_time, end_time=end_time)
    return attrs(_coval_shape(span, None))


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (None, None),
        ({}, None),
        ({"coval.simulation_id": "sim-1"}, "sim-1"),
        ({"sip.h.X-Coval-Simulation-Id": "sim-2"}, "sim-2"),
        ({"x-coval-simulation-id": "sim-3"}, "sim-3"),
        ({"coval.simulation_id": "first", "x-coval-simulation-id": "second"}, "first"),
        ({"coval.simulation_id": "", "x-coval-simulation-id": "second"}, "second"),
        ({"x-coval-simulation-id": "", "other": "x"}, None),
        ({"x-other": "sim"}, None),
    ],
)
def test_simulation_id_lookup(values, expected):
    assert coval_simulation_id(values) == expected


def test_settings_read_the_simulation_id(monkeypatch):
    monkeypatch.setenv("COVAL_SIMULATION_ID", "sim-env")
    monkeypatch.setenv("COVAL_API_KEY", "k")
    assert Coval.from_env().simulation_id == "sim-env"
    assert Coval.from_settings(CovalSettings(api_key="k")).file_calls is True


def test_from_settings_logs_that_tracing_is_on(caplog):
    with caplog.at_level(logging.INFO):
        Coval.from_settings(CovalSettings(api_key="cv-key"))
        Coval.from_settings(CovalSettings())
    assert (
        "Coval tracing on: traces to https://api.coval.dev/v1/traces, COVAL_API_KEY present"
        in caplog.text
    )
    assert "COVAL_API_KEY MISSING" in caplog.text


def submitted(**options):
    """Run ``_submit`` against a mock transport. Return the id and the request it sent."""
    sent: list[httpx.Request] = []
    handler = options.pop("handler", None) or (
        lambda request: httpx.Response(200, json={"conversation": {"conversation_id": "conv-7"}})
    )

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return handler(request)

    conversation_id = _submit(
        "cv-key",
        TRANSCRIPT,
        agent="salon",
        session_id="RM_abc",
        origin=options.pop("origin", "phone"),
        transport=httpx.MockTransport(record),
    )
    return conversation_id, sent


def test_submit_sends_the_transcript_and_returns_the_id():
    conversation_id, (request,) = submitted()

    assert conversation_id == "conv-7"
    assert str(request.url) == COVAL_SUBMIT_ENDPOINT
    assert request.method == "POST"
    assert request.headers["x-api-key"] == "cv-key"
    assert json.loads(request.content) == {
        "transcript": TRANSCRIPT,
        "metadata": {
            "agent": "salon",
            "source": "unmute",
            "session_id": "RM_abc",
            "origin": "phone",
        },
        "tags": ["salon"],
        "external_conversation_id": "RM_abc",
    }
    assert request.extensions["timeout"]["read"] == COVAL_SUBMIT_TIMEOUT_SECONDS == 5


def test_submit_leaves_out_an_unknown_origin():
    _, (request,) = submitted(origin=None)
    assert "origin" not in json.loads(request.content)["metadata"]


def test_submit_warns_when_coval_is_too_slow(caplog):
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    with caplog.at_level(logging.WARNING):
        conversation_id, _ = submitted(handler=slow)
    assert conversation_id is None
    assert "5 s budget" in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(httpx.Response(500, text="boom"), id="server-error"),
        pytest.param(httpx.Response(200, text="not json"), id="not-json"),
        pytest.param(httpx.Response(200, json={}), id="no-conversation"),
        pytest.param(httpx.Response(200, json={"conversation": None}), id="null-conversation"),
        pytest.param(
            httpx.Response(200, json={"conversation": {"conversation_id": ""}}), id="empty"
        ),
    ],
)
def test_submit_returns_none_and_warns_on_a_bad_answer(response, caplog):
    with caplog.at_level(logging.WARNING):
        conversation_id, _ = submitted(handler=lambda request: response)
    assert conversation_id is None
    assert "Could not register this call with Coval" in caplog.text


def test_transcript_merges_consecutive_lines_of_one_speaker():
    lines = [
        {"role": "assistant", "content": "Hello"},
        {"role": "assistant", "content": " How are you? "},
        {"role": "user", "content": "Fine"},
        {"role": "user", "content": ""},
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "thanks"},
    ]
    assert _transcript_of(json.dumps(lines)) == [
        {"role": "assistant", "content": "Hello How are you?"},
        {"role": "user", "content": "Fine thanks"},
    ]
    assert _transcript_of("not json") == []
    assert _transcript_of(None) == []


@pytest.fixture
def filed(monkeypatch):
    """Replace ``_submit``. Its calls are kept, and it answers ``conv-1``."""
    calls: list[tuple] = []

    def fake(api_key, transcript, **details):
        calls.append((api_key, transcript, details))
        return "conv-1"

    monkeypatch.setattr(_coval, "_submit", fake)
    return calls


async def test_a_call_no_simulation_placed_is_filed_when_it_closes(exporters, filed, caplog):
    coval = Coval(api_key="cv-key")
    telemetry = Telemetry(coval, agent="salon", call_id="RM_abc")
    telemetry.describe_call(origin="phone", framework="livekit")
    with caplog.at_level(logging.INFO):
        await run_call(telemetry)

    ((api_key, transcript, details),) = filed
    assert api_key == "cv-key"
    assert transcript == [
        {"role": "user", "content": "My number is 555 0100"},
        {"role": "assistant", "content": "Let me check."},
    ]
    assert details == {"agent": "salon", "session_id": "RM_abc", "origin": "phone"}
    (exporter,) = exporters
    assert exporter.settings["headers"] == {"x-api-key": "cv-key", "X-Conversation-Id": "conv-1"}
    spans = by_name(exporter.get_finished_spans())
    assert attrs(spans["conversation"])["coval.conversation_id"] == "conv-1"
    assert attrs(spans["conversation"])["coval.correlation.method"] == "conversation_submit"
    assert "Filed this call as Coval conversation conv-1 with 8 span(s)" in caplog.text


async def test_a_bound_call_is_not_filed(exporters, filed, caplog):
    coval = Coval(api_key="cv-key")
    telemetry = Telemetry(coval, agent="salon", call_id="RM_abc")
    coval.activate(simulation_id="sim-1", method="dispatch_metadata")
    with caplog.at_level(logging.INFO):
        await run_call(telemetry)

    assert filed == []
    assert "Coval already correlated this call as sim-1" in caplog.text
    root = attrs(by_name(exporters[0].get_finished_spans())["conversation"])
    assert root["coval.simulation_id"] == "sim-1"
    assert root["coval.correlation.method"] == "dispatch_metadata"


@pytest.mark.parametrize(
    ("sink", "reason"),
    [
        pytest.param(Coval(api_key=None), "no API key", id="no-key"),
        pytest.param(Coval(api_key="k", file_calls=False), "file_calls=False", id="file-calls-off"),
        pytest.param(Coval(api_key="k", filter=MetadataOnly()), "metadata only", id="filter"),
    ],
)
async def test_a_call_is_not_filed_when(exporters, filed, caplog, sink, reason):
    telemetry = Telemetry(sink, agent="salon", call_id="RM_abc")
    with caplog.at_level(logging.INFO):
        await run_call(telemetry)

    assert filed == []
    assert exporters == []
    assert reason in caplog.text
    assert "held span(s) are dropped" in caplog.text


async def test_a_call_without_a_transcript_is_not_filed(exporters, filed, caplog):
    telemetry = Telemetry(Coval(api_key="k"), agent="salon", call_id="RM_abc")
    telemetry.start_call()
    with caplog.at_level(logging.WARNING):
        await telemetry.close()

    assert filed == []
    assert "This call produced no transcript" in caplog.text
    assert "1 held span(s) are dropped" in caplog.text


async def test_a_failed_submit_never_breaks_close(exporters, monkeypatch, caplog):
    real = httpx.Client
    monkeypatch.setattr(
        _coval.httpx,
        "Client",
        lambda **kw: real(transport=httpx.MockTransport(lambda r: httpx.Response(500)), **kw),
    )
    telemetry = Telemetry(Coval(api_key="k"), agent="salon", call_id="RM_abc")
    with caplog.at_level(logging.WARNING):
        await run_call(telemetry)

    assert exporters == []
    assert "Could not register this call with Coval" in caplog.text


def test_activation_is_refused_twice(exporters):
    coval = Coval(api_key="k")
    Telemetry(coval, agent="salon", call_id="RM_abc")
    coval.activate(simulation_id="a")
    with pytest.raises(RuntimeError, match="already bound"):
        coval.activate(simulation_id="b")


def test_the_call_span_names_origin_framework_and_binding():
    span = ReadableSpan(
        name="conversation",
        attributes={"unmute.call.origin": "phone", "unmute.call.framework": "livekit"},
    )
    result = attrs(_coval_shape(span, _Correlation("X-Simulation-Id", "sim-1", "environment")))

    assert result["coval.call.origin"] == "phone"
    assert result["agent.provider"] == "livekit"
    assert result["coval.agent_type"] == "sip_inbound_voice"
    assert result["coval.simulation_id"] == "sim-1"
    assert result["coval.correlation.method"] == "environment"
    assert result["unmute.call.origin"] == "phone"


def test_the_call_span_names_a_conversation_binding_and_a_pipecat_agent():
    span = ReadableSpan(name="conversation", attributes={"unmute.call.framework": "pipecat"})
    result = attrs(_coval_shape(span, _Correlation("X-Conversation-Id", "c-1", None)))

    assert result["agent.provider"] == "pipecat"
    assert "coval.agent_type" not in result
    assert result["coval.conversation_id"] == "c-1"
    assert "coval.correlation.method" not in result


def test_the_turn_names():
    result = shaped(
        "turn",
        {
            "turn.number": 3,
            "input.value": "Book me in",
            "output.value": "Which day?",
            "turn.interrupted": True,
            "turn.e2e_seconds": 1.2,
            "turn.eou_seconds": 0.4,
        },
    )

    assert result["turn.index"] == 3
    assert result["turn.user_transcript"] == "Book me in"
    assert result["turn.agent_transcript"] == "Which day?"
    assert result["turn.interrupted"] == 1
    assert result["metrics.e2e_latency"] == 1.2
    assert result["vad.end_of_turn_delay"] == 0.4
    assert "input.value" not in result
    assert result["turn.number"] == 3


def test_a_turn_without_latencies_has_no_zero_ones():
    result = shaped("turn", {"turn.interrupted": False, "turn.e2e_seconds": 0.0})
    assert result["turn.interrupted"] == 0
    assert "metrics.e2e_latency" not in result
    assert "vad.end_of_turn_delay" not in result


def test_the_stt_names():
    result = shaped(
        "stt",
        {"output.value": "hello", "gen_ai.provider.name": "deepgram", "metrics.ttfb": 0.2},
    )
    assert result["transcript"] == "hello"
    assert result["stt.providerName"] == "deepgram"
    assert result["stt.transcription_delay"] == 0.2
    assert result["metrics.ttfb"] == 0.2
    assert "output.value" not in result


@pytest.mark.parametrize("name", ["stt", "llm", "tts"])
def test_a_zero_ttfb_is_dropped(name):
    assert "metrics.ttfb" not in shaped(name, {"metrics.ttfb": 0.0})
    assert "metrics.ttfb" not in shaped(name, {"metrics.ttfb": -1})
    assert shaped(name, {"metrics.ttfb": 0.1})["metrics.ttfb"] == 0.1


def llm_attributes(**more):
    """The attributes of a model request that chose to call a tool."""
    messages = [
        {"role": "system", "content": "You book salon visits."},
        {"role": "user", "content": "Book me in"},
    ]
    tools = [{"type": "function", "function": {"name": "find_slots"}}, {"function": {"name": "a"}}]
    return {
        "input.value": json.dumps(messages),
        "gen_ai.tool.definitions": json.dumps(tools),
        "output.value": "Which day?",
        "gen_ai.response.finish_reasons": ("stop",),
        **more,
    }


def test_the_llm_names():
    result = shaped("llm", llm_attributes(**{"llm.interrupted": True}))

    assert result["gen_ai.system_instructions"] == "You book salon visits."
    assert json.loads(result["input"])[1]["content"] == "Book me in"
    assert result["prompt.message_count"] == 2
    assert result["tools"] == ["a", "find_slots"]
    assert result["tool_count"] == 2
    assert result["output"] == "Which day?"
    assert result["response.length"] == 10
    assert result["llm.finish_reason"] == "stop"
    assert result["llm.cancelled"] == 1
    assert not {"input.value", "output.value", "gen_ai.tool.definitions"} & set(result)


def test_an_llm_reply_with_tool_calls_has_no_output_and_finishes_with_tool_calls():
    reply = {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]}
    result = shaped("llm", {"output.value": json.dumps(reply)})

    assert "output" not in result
    assert result["llm.finish_reason"] == "tool_calls"
    assert result["llm.cancelled"] == 0


def test_a_lone_surrogate_in_a_message_keeps_the_message_and_the_tools():
    messages = [
        {"role": "system", "content": "You book salon visits."},
        {"role": "user", "content": "caf\ud800"},
    ]
    tools = [{"function": {"name": "a\ud800"}}]
    result = shaped(
        "llm",
        {"input.value": json.dumps(messages), "gen_ai.tool.definitions": json.dumps(tools)},
    )

    assert result["prompt.message_count"] == 2
    assert result["gen_ai.system_instructions"] == "You book salon visits."
    assert result["tool_count"] == 1


def test_an_llm_finish_reason_defaults_to_stop():
    assert shaped("llm", {"output.value": "Hi"})["llm.finish_reason"] == "stop"


def test_the_tool_names():
    result = shaped(
        "llm_tool_call",
        {"gen_ai.tool.call.result": '{"slots": 3}', "function.name": "find_slots"},
        start_time=1_000_000_000,
        end_time=1_250_000_000,
    )
    assert result["tool.result"] == '{"slots": 3}'
    assert result["tool.result.length"] == 12
    assert result["tool.latency_ms"] == 250.0
    assert "gen_ai.tool.call.result" not in result


def test_the_tts_names():
    result = shaped("tts", {"input.value": "Which day?", "metrics.character_count": 10})
    assert result["transcript"] == "Which day?"
    assert result["tts.characters_count"] == 10
    assert "input.value" not in result


def test_text_is_cut_to_coval_limits():
    long = "x" * 5000
    turn = shaped("turn", {"input.value": long, "output.value": long})
    assert len(turn["turn.user_transcript"]) == MAX_TEXT_CHARS + 1
    assert turn["turn.user_transcript"].endswith("…")
    assert len(shaped("tts", {"input.value": long})["transcript"]) == MAX_TEXT_CHARS + 1
    assert (
        len(shaped("llm_tool_call", {"gen_ai.tool.call.result": long})["tool.result"])
        == MAX_TEXT_CHARS + 1
    )
    assert shaped("llm_tool_call", {"gen_ai.tool.call.result": long})["tool.result.length"] == 5000
    assert len(shaped("task_step", {"input.value": long})["input.value"]) == MAX_TEXT_CHARS + 1
    assert (
        len(shaped("conversation", {"output.value": long * 3})["output.value"])
        == MAX_PROMPT_CHARS + 1
    )


def test_a_prompt_keeps_the_system_message_and_the_newest_messages():
    history = [{"role": "user", "content": f"line {i} " + "y" * 500} for i in range(40)]
    messages = [{"role": "system", "content": "SYSTEM"}, *history]
    result = shaped("llm", {"input.value": json.dumps(messages)})

    assert len(result["input"]) <= MAX_PROMPT_CHARS + 1
    assert result["prompt.message_count"] == 41
    assert 1 < result["prompt.messages_traced"] < 41
    assert "SYSTEM" in result["input"]
    assert "line 39 " in result["input"]
    assert "line 0 " not in result["input"]
    assert result["gen_ai.system_instructions"] == "SYSTEM"


async def test_coval_shapes_a_copy_and_the_other_sinks_get_the_neutral_names(exporters):
    coval = Coval(api_key="cv-key")
    telemetry = Telemetry(
        Otlp(endpoint="http://x"),
        Logfire(token="pylf_v1_us_a"),
        coval,
        agent="salon",
        call_id="RM_abc",
    )
    telemetry.describe_call(origin="browser", framework="pipecat")
    coval.activate(simulation_id="sim-1")
    await run_call(telemetry)

    otlp, logfire, coval_exporter = exporters
    for plain in (otlp, logfire):
        for span in plain.get_finished_spans():
            assert not [key for key in attrs(span) if key.startswith(("coval.", "agent.provider"))]
    for span in otlp.get_finished_spans():
        assert not [
            key
            for key in attrs(span)
            if key.startswith("logfire.") or "gen_ai.input.messages" in key
        ]
        assert not [key for key in attrs(span) if "message" in key and key.startswith("gen_ai.")]
    otlp_llm = attrs(by_name(otlp.get_finished_spans())["llm"])
    assert "input.value" in otlp_llm
    assert "input" not in otlp_llm
    coval_llm = attrs(by_name(coval_exporter.get_finished_spans())["llm"])
    assert "input.value" not in coval_llm
    assert coval_llm["tools"] == ["find_slots"]
    root = attrs(by_name(coval_exporter.get_finished_spans())["conversation"])
    assert root["coval.call.origin"] == "browser"
    assert root["agent.provider"] == "pipecat"


def test_a_transcript_line_that_is_not_text_files_nothing_and_raises_nothing() -> None:
    line = {"role": "user", "content": cast(Any, 7)}
    assert _submit("cv-key", [line], agent="salon", session_id="RM_abc", origin=None) is None
