"""Logfire's names: a row label and the GenAI message attributes, added on a copy of each span."""

import json
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from tests.telemetry.conftest import attrs, by_name, run_call
from unmute_core.telemetry import Logfire, MetadataOnly, Telemetry
from unmute_core.telemetry._logfire import genai_messages, logfire_message, logfire_shape


def loads(value: object) -> Any:
    """Parse a JSON text attribute."""
    return json.loads(str(value))


MESSAGES = [
    {"role": "system", "content": "You book salon visits."},
    {"role": "user", "content": "Book me in"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": "find_slots", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "c1", "content": '{"slots": 3}'},
]


@pytest.mark.parametrize(
    ("name", "attributes", "expected"),
    [
        ("turn", {"input.value": "x" * 100}, "turn: " + "x" * 80),
        ("turn", {}, "turn"),
        ("turn", {"input.value": '"quoted"'}, "turn: quoted"),
        (
            "llm",
            {"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4.1"},
            "chat gpt-4.1",
        ),
        (
            "llm_tool_call",
            {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "find_slots"},
            "tool find_slots",
        ),
        ("stt", {"gen_ai.operation.name": "stt", "gen_ai.request.model": "nova-3"}, "stt nova-3"),
        ("tts", {"gen_ai.operation.name": "tts", "gen_ai.request.model": "sonic"}, "tts sonic"),
        ("conversation", {"session.id": "RM_abc"}, "call RM_abc"),
        ("conversation", {}, "call"),
        ("task_step", {}, None),
    ],
)
def test_logfire_message(name, attributes, expected):
    assert logfire_message(name, attributes) == expected


def test_messages_are_mapped_to_the_genai_names():
    mapped = genai_messages(
        {
            "gen_ai.operation.name": "chat",
            "input.value": json.dumps(MESSAGES),
            "output.value": "Which day?",
        }
    )

    assert loads(mapped["gen_ai.system_instructions"]) == [
        {"type": "text", "content": "You book salon visits."}
    ]
    assert loads(mapped["gen_ai.input.messages"]) == [
        {"role": "user", "parts": [{"type": "text", "content": "Book me in"}]},
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "name": "find_slots", "arguments": "{}"}],
        },
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "c1", "response": '{"slots": 3}'}],
        },
    ]
    assert loads(mapped["gen_ai.output.messages"]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Which day?"}]}
    ]


def test_an_output_with_tool_calls_is_parsed():
    reply = {"role": "assistant", "content": "One moment", "tool_calls": MESSAGES[2]["tool_calls"]}
    mapped = genai_messages({"gen_ai.operation.name": "chat", "output.value": json.dumps(reply)})

    assert loads(mapped["gen_ai.output.messages"])[0]["parts"] == [
        {"type": "text", "content": "One moment"},
        {"type": "tool_call", "name": "find_slots", "arguments": "{}"},
    ]


@pytest.mark.parametrize(
    "attributes",
    [
        {"gen_ai.operation.name": "stt", "input.value": json.dumps(MESSAGES)},
        {"gen_ai.operation.name": "chat", "gen_ai.input.messages": "[]", "input.value": "[]"},
        {"gen_ai.operation.name": "chat"},
    ],
)
def test_nothing_is_mapped_when_there_are_no_messages_to_map(attributes):
    assert genai_messages(attributes) == {}


def test_shaping_leaves_the_original_span_alone():
    span = ReadableSpan(name="turn", attributes={"input.value": "hi"})
    shaped = logfire_shape(span)

    assert attrs(shaped)["logfire.msg"] == "turn: hi"
    assert attrs(span) == {"input.value": "hi"}


async def test_the_exporter_adds_the_names_to_a_copy(exporters):
    telemetry = Telemetry(Logfire(token="pylf_v1_us_a"), agent="salon", call_id="RM_abc")
    await run_call(telemetry)

    spans = by_name(exporters[0].get_finished_spans())
    assert attrs(spans["conversation"])["logfire.msg"] == "call RM_abc"
    assert attrs(spans["turn"])["logfire.msg"] == "turn: My number is 555 0100"
    llm = attrs(spans["llm"])
    assert llm["logfire.msg"] == "chat gpt-4.1"
    assert loads(llm["gen_ai.input.messages"])[0]["role"] == "user"
    assert "input.value" in llm
    assert "logfire.msg" not in attrs(spans["task_step"])


async def test_metadata_only_runs_first_so_only_labels_go_out(exporters):
    telemetry = Telemetry(
        Logfire(token="pylf_v1_us_a", filter=MetadataOnly()), agent="salon", call_id="RM_abc"
    )
    await run_call(telemetry)

    spans = by_name(exporters[0].get_finished_spans())
    llm = attrs(spans["llm"])
    assert llm["logfire.msg"] == "chat gpt-4.1"
    assert not {"input.value", "gen_ai.input.messages", "gen_ai.output.messages"} & set(llm)
    assert attrs(spans["turn"])["logfire.msg"] == "turn"
