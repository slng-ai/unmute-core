"""Locked values sent as their parameter's type, read from a field, or from a call fact.

The cases include the rule for which locked value fits which parameter type.
"""

import json
import math
import re
from typing import Any

import pytest

from tests.outputs import AnyResult
from unmute_core.state import CallInfo, CallState, StateSchema
from unmute_core.tools import TRANSCRIPT_MAX_BYTES, ToolError, ToolRegistry, ToolSpec, Transcript
from unmute_core.tools._spec import coerce_locked, locked_value

SCHEMA = StateSchema(
    [
        {"name": "customer_id", "type": "str"},
        {"name": "count", "type": "int | None"},
        {"name": "day", "type": "Date | None"},
        {"name": "status", "type": 'Literal["1", "2"] | None'},
        {"name": "contact", "type": "Contact | None", "confirm": "take_details"},
        {"name": "bookings", "type": "list[str]"},
    ],
    [{"name": "Contact", "fields": [{"name": "name", "type": "str | None"}]}],
)


class Recorder:
    """An executor that keeps the arguments it was given."""

    def __init__(self) -> None:
        self.arguments: dict[str, Any] | None = None

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        self.arguments = arguments
        return {"ok": True}


def _parameters(kind: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "customer_id": {"type": kind},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["customer_id"],
        "additionalProperties": False,
    }


def _tool(kind: str, source: str, recorder: Recorder | None = None) -> ToolSpec:
    return ToolSpec(
        "lookup",
        "Look up.",
        _parameters(kind),
        AnyResult,
        recorder or Recorder(),
        {"customer_id": source},
    )


def _state(**values: object) -> CallState:
    state = CallState(SCHEMA.model, confirm=SCHEMA.confirm)
    state.save(values, step="take_details")
    return state


@pytest.mark.parametrize(
    ("kind", "saved", "sent"),
    [
        ("string", "42", "42"),
        ("integer", "42", 42),
        ("number", "2.5", 2.5),
        ("boolean", "true", True),
    ],
)
async def test_a_locked_value_is_sent_as_its_parameters_type(kind, saved, sent):
    recorder = Recorder()
    registry = ToolRegistry([_tool(kind, "{{customer_id}}", recorder)])
    await registry.call("lookup", {"tags": ["one"]}, _state(customer_id=saved))
    assert recorder.arguments == {"customer_id": sent, "tags": ["one"]}


async def test_a_number_fills_a_string_as_its_text():
    recorder = Recorder()
    await ToolRegistry([_tool("string", "{{count}}", recorder)]).call("lookup", {}, _state(count=7))
    assert recorder.arguments == {"customer_id": "7"}


@pytest.mark.parametrize("saved", ["1", "0", "True", "null", "1.0"])
async def test_a_boolean_takes_only_true_or_false(saved):
    with pytest.raises(ToolError, match="is not a valid boolean"):
        await ToolRegistry([_tool("boolean", "{{customer_id}}")]).call(
            "lookup", {}, _state(customer_id=saved)
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, "Infinity"])
def test_a_number_must_be_finite(value):
    with pytest.raises(ToolError, match="is not a valid number"):
        coerce_locked(value, "number")


async def test_the_whole_arguments_are_checked_after_filling():
    parameters = _parameters("string")
    parameters["properties"]["customer_id"]["minLength"] = 5
    tool = ToolSpec(
        "lookup",
        "Look up.",
        parameters,
        AnyResult,
        Recorder(),
        {"customer_id": "{{customer_id}}"},
    )
    with pytest.raises(ToolError, match=r"arguments\.customer_id: 'C-1' is too short"):
        await ToolRegistry([tool]).call("lookup", {}, _state(customer_id="C-1"))


@pytest.mark.parametrize("form", ["{{contact.name}}", "{{contact__name}}"])
async def test_a_locked_value_reads_a_field_inside_a_shape(form):
    recorder = Recorder()
    await ToolRegistry([_tool("string", form, recorder)]).call(
        "lookup", {}, _state(contact={"name": "Robin Vega"})
    )
    assert recorder.arguments == {"customer_id": "Robin Vega"}


@pytest.mark.parametrize("saved", [{}, {"contact": {"name": None}}])
async def test_an_absent_field_is_unset_and_names_the_task_that_confirms_it(saved):
    state = CallState(SCHEMA.model, confirm=SCHEMA.confirm)
    state.save(saved, step="other")
    refused = await ToolRegistry([_tool("string", "{{contact.name}}")]).call("lookup", {}, state)
    assert refused == {
        "refused": "contact.name not set yet. Run take_details first: it confirms contact "
        "with the caller, then call this tool again."
    }


async def test_a_call_value_fills_a_locked_value():
    recorder = Recorder()
    registry = ToolRegistry([_tool("string", "{{call.carrier_call_id}}", recorder)])
    await registry.call("lookup", {}, CallState(call=CallInfo(carrier_call_id="call-9")))
    assert recorder.arguments == {"customer_id": "call-9"}
    refused = await registry.call("lookup", {}, CallState())
    assert refused == {
        "refused": "this tool reads call.carrier_call_id, which this call does not carry"
    }


@pytest.mark.parametrize(
    ("source", "new"),
    [
        ("{{@call_id}}", "{{call.carrier_call_id}}"),
        ("{{@phone_number}}", "{{call.remote_number}}"),
        ("{{@first_user_message}}", "{{call.first_user_message}}"),
        ("{{@transcript}}", "Transcript()"),
        ("{{@weather}}", "{{call.<field>}}"),
    ],
)
def test_an_old_call_fact_fails_when_built_naming_the_new_spelling(source, new):
    with pytest.raises(ValueError, match=re.escape(new)):
        _tool("string", source)


@pytest.mark.parametrize(
    ("kind", "source", "message"),
    [
        ("string", "{{ghost}}", "names no value of the State: 'ghost'"),
        ("string", "{{call.nothing}}", "shape 'CallInfo' declares no field 'nothing'"),
        ("string", "{{contact.phone}}", "shape 'Contact' declares no field 'phone'"),
        ("string", "{{customer_id.x}}", "which has no fields to name"),
        ("string", "{{contact}}", "names a shape, so parameter 'customer_id' must be an object"),
        ("string", "{{bookings}}", "names a list, so parameter 'customer_id' must be an array"),
        ("array", "{{contact}}", "names a shape, so parameter 'customer_id' must be an object"),
        ("object", "{{bookings}}", "names a list, so parameter 'customer_id' must be an array"),
        ("object", "see {{contact}}", "inside a longer text. Lock it whole"),
        ("string", "all {{bookings}}", "inside a longer text. Lock it whole"),
        ("integer", "{{day}}", "cannot be sent as the integer parameter 'customer_id'"),
        ("boolean", "{{count}}", "cannot be sent as the boolean parameter"),
        ("boolean", "{{status}}", "cannot be sent as the boolean parameter"),
    ],
)
def test_a_registry_built_with_the_state_checks_each_locked_path(kind, source, message):
    with pytest.raises(ValueError, match=message):
        ToolRegistry([_tool(kind, source)], state=SCHEMA.model)


@pytest.mark.parametrize("source", ["{{contact.}}", "{{contact..name}}", "{{@}}", "x }}"])
def test_a_broken_placeholder_is_refused_when_the_tool_is_declared(source):
    with pytest.raises(ValueError, match="holds a placeholder that is not"):
        _tool("string", source)


@pytest.mark.parametrize(
    ("kind", "source"),
    [
        ("integer", "{{count}}"),
        ("string", "{{count}}"),
        ("integer", "{{status}}"),
        ("boolean", "{{customer_id}}"),
        ("string", "{{day}}"),
        ("string", "{{contact.name}}"),
        ("string", "the customer {{customer_id}}"),
    ],
)
def test_a_locked_path_that_fits_its_parameter_is_accepted(kind, source):
    ToolRegistry([_tool(kind, source)], state=SCHEMA.model)


@pytest.mark.parametrize(
    ("declared", "source"),
    [
        ({"type": "object"}, "{{contact}}"),
        ({}, "{{contact}}"),
        ({"type": "array"}, "{{bookings}}"),
        ({}, "{{bookings}}"),
    ],
)
async def test_a_shape_or_a_list_locked_whole_is_sent_as_it_is(declared, source):
    recorder = Recorder()
    parameters = {"type": "object", "properties": {"customer_id": declared}}
    tool = ToolSpec("lookup", "Look up.", parameters, AnyResult, recorder, {"customer_id": source})
    registry = ToolRegistry([tool], state=SCHEMA.model)
    state = _state(contact={"name": "Ana"}, bookings=["a", "b"])
    await registry.call("lookup", {}, state)
    expected = {"name": "Ana"} if "contact" in source else ["a", "b"]
    assert recorder.arguments == {"customer_id": expected}


def test_a_transcript_reads_the_newest_messages() -> None:
    messages = [{"role": "user", "content": str(n)} for n in range(5)]
    context = {"transcript": messages}
    assert locked_value(Transcript(max_messages=2), None, context) == messages[-2:]
    assert locked_value(Transcript(max_messages=2, text=True), None, context) == "user: 3\nuser: 4"
    assert locked_value(Transcript(), None, None) == []
    odd = {"transcript": ({"role": "user"}, "not a message")}
    assert locked_value(Transcript(text=True), None, odd) == "user: "


def test_a_long_transcript_keeps_the_newest_messages_within_the_byte_budget() -> None:
    messages = [{"role": "user", "content": f"{n:03d} " + "x" * 400} for n in range(200)]
    context = {"transcript": messages}

    sent = locked_value(Transcript(), None, context)
    assert isinstance(sent, list)
    assert len(json.dumps(sent).encode()) <= TRANSCRIPT_MAX_BYTES
    assert sent[-1] == messages[-1]
    assert sent == messages[-len(sent) :]
    assert 0 < len(sent) < 200

    as_text = locked_value(Transcript(text=True), None, context)
    assert isinstance(as_text, str)
    assert len(as_text.encode()) <= TRANSCRIPT_MAX_BYTES
    assert as_text.endswith(str(messages[-1]["content"]))


def test_a_transcript_in_arabic_keeps_its_budget_as_sent() -> None:
    messages = [{"role": "user", "content": f"{n:03d} " + "مرحبا " * 70} for n in range(200)]
    context = {"transcript": messages}

    as_text = locked_value(Transcript(text=True), None, context)
    assert isinstance(as_text, str)
    assert len(json.dumps(as_text).encode()) <= TRANSCRIPT_MAX_BYTES
    assert as_text.endswith(str(messages[-1]["content"]))


def test_one_message_over_the_budget_still_sends_the_newest() -> None:
    huge = [{"role": "user", "content": "x" * (TRANSCRIPT_MAX_BYTES * 2)}]
    assert locked_value(Transcript(), None, {"transcript": huge}) == huge
