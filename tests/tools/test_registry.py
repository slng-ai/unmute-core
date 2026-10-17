"""The registry: model-facing schemas, argument checks, locked values and error wording.

The locked-value cases check that a missing source refuses the call, the
model's value is never used, and the value never reaches the model's schema.
"""

import threading
from datetime import date
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from tests.outputs import AnyResult
from unmute_core.state import CallState
from unmute_core.tools import (
    Done,
    ParameterSchemaError,
    ToolError,
    ToolRegistry,
    ToolSpec,
    local_tool,
    replies_to,
)


class Customer(BaseModel):
    phone: str | None = None
    tier: int | None = None
    profile: dict[str, str] | None = None


class Recorder:
    """An executor that keeps the arguments it was given."""

    def __init__(self) -> None:
        self.arguments: dict[str, Any] | None = None

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        self.arguments = arguments
        return {"ok": True}


BOOK = {
    "type": "object",
    "properties": {
        "day": {"type": "string", "format": "date"},
        "phone": {"type": "string"},
        "party": {"type": "integer", "minimum": 1},
    },
    "required": ["day", "phone"],
}


def _spec(locked=None, executor=None) -> ToolSpec:
    return ToolSpec("book", "Book a table.", BOOK, AnyResult, executor or Recorder(), locked or {})


def _state(**values) -> CallState:
    state = CallState(Customer, confirm={"phone": "verify"})
    if values:
        state.save(values, step="verify")
    return state


def test_schemas_are_openai_function_tools_without_locked_arguments():
    registry = ToolRegistry([_spec(locked={"phone": "{{phone}}"})])
    assert registry.schemas() == [
        {
            "type": "function",
            "function": {
                "name": "book",
                "description": "Book a table.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "day": {
                            "type": "string",
                            "description": "Expected a date written YYYY-MM-DD.",
                        },
                        "party": {"type": "integer", "minimum": 1},
                    },
                    "required": ["day"],
                },
            },
        }
    ]


def _shown(node: dict[str, Any], *, required: bool = False, defs: dict | None = None) -> object:
    parameters = {
        "type": "object",
        "properties": {"x": node},
        "required": ["x"] if required else [],
    }
    if defs:
        parameters["$defs"] = defs
    spec = ToolSpec("t", "T.", parameters, AnyResult, Recorder())
    return spec.model_parameters["properties"]["x"]


@pytest.mark.parametrize(
    ("node", "required", "shown"),
    [
        (
            {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Why.",
            },
            False,
            {"type": "string", "description": "Why."},
        ),
        ({"type": ["integer", "null"], "default": None}, False, {"type": "integer"}),
        (
            {"anyOf": [{"enum": ["a", "b"], "type": "string"}, {"type": "null"}], "default": None},
            False,
            {"enum": ["a", "b"], "type": "string"},
        ),
        (
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            True,
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
        ),
        (
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            False,
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
        ),
        (
            {"type": "string", "format": "birth-date"},
            True,
            {"type": "string", "description": "Expected a date of birth written YYYY-MM-DD."},
        ),
        (
            {"type": "string", "format": "spelled"},
            True,
            {
                "type": "string",
                "description": "Expected the code as one word of letters and digits, like KP482913.",
            },
        ),
        (
            {"type": "string", "format": "phone", "description": "Their phone."},
            True,
            {
                "type": "string",
                "description": "Their phone. Expected a phone number with its country code: a "
                "plus, the country code, then the number.",
            },
        ),
        (
            {"type": "string", "pattern": "^KP[0-9]{6}$"},
            True,
            {"type": "string", "description": "Expected text matching ^KP[0-9]{6}$."},
        ),
        (
            {"anyOf": [{"format": "date", "type": "string"}, {"type": "null"}], "default": None},
            False,
            {"type": "string", "description": "Expected a date written YYYY-MM-DD."},
        ),
    ],
)
def test_the_model_sees_a_plain_schema(node, required, shown):
    assert _shown(node, required=required) == shown


def test_the_model_sees_a_shape_inlined():
    defs = {"Car": {"type": "object", "title": "Car", "properties": {"plate": {"type": "string"}}}}
    shown = _shown({"$ref": "#/$defs/Car", "description": "Their car."}, required=True, defs=defs)
    assert shown == {
        "type": "object",
        "properties": {"plate": {"type": "string"}},
        "description": "Their car.",
    }


async def test_an_optional_argument_sent_as_null_is_still_accepted():
    recorder = Recorder()
    parameters = {
        "type": "object",
        "properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}},
    }
    registry = ToolRegistry([ToolSpec("t", "T.", parameters, AnyResult, recorder)])
    assert await registry.call("t", {"note": None}) == {"ok": True}
    assert recorder.arguments == {"note": None}


async def test_locked_value_replaces_what_the_model_sent():
    recorder = Recorder()
    registry = ToolRegistry([_spec({"phone": "{{phone}}"}, recorder)])
    await registry.call(
        "book", '{"day": "2026-10-07", "phone": "+1 invented"}', _state(phone="+34600111222")
    )
    assert recorder.arguments == {"day": "2026-10-07", "phone": "+34600111222"}


async def test_whole_placeholder_keeps_the_type_and_mixed_text_fills_as_text():
    recorder = Recorder()
    spec = ToolSpec(
        "book",
        "Book a table.",
        {"type": "object", "properties": {}},
        AnyResult,
        recorder,
        {
            "tier": "{{tier}}",
            "note": "tier {{tier}}, city {{profile__city}}",
            "source": "voice",
            "n": 3,
        },
    )
    state = _state(tier=2, profile={"city": "Madrid"})
    await ToolRegistry([spec]).call("book", {}, state)
    assert recorder.arguments == {
        "tier": 2,
        "note": "tier 2, city Madrid",
        "source": "voice",
        "n": 3,
    }


@pytest.mark.parametrize("state", [lambda: _state(), lambda: _state(phone="")])
async def test_missing_locked_value_refuses_and_never_uses_the_model_value(state):
    recorder = Recorder()
    registry = ToolRegistry([_spec({"phone": "{{phone}}"}, recorder)])
    refused = await registry.call("book", {"day": "2026-10-07", "phone": "+1"}, state())
    assert refused == {
        "refused": "phone not set yet. Run verify first: it confirms phone with the caller, "
        "then call this tool again."
    }
    assert recorder.arguments is None


async def test_a_locked_value_on_an_agent_without_state_is_an_error():
    recorder = Recorder()
    registry = ToolRegistry([_spec({"phone": "{{phone}}"}, recorder)])
    with pytest.raises(ToolError, match="keeps no state"):
        await registry.call("book", {"day": "2026-10-07"}, None)
    assert recorder.arguments is None


async def test_value_waiting_for_the_caller_is_not_sent():
    state = _state()
    state.save({"phone": "+34600111222"}, step="other")
    assert not state.is_confirmed("phone")
    refused = await ToolRegistry([_spec({"phone": "{{phone}}"})]).call("book", {"day": "x"}, state)
    assert isinstance(refused, dict)
    assert "phone not set yet" in refused["refused"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ({"day": "x", "phone": "+1", "party": 0}, "party: 0 is less than the minimum of 1"),
        ({"day": 5, "phone": "+1"}, "day: 5 is not of type 'string'. Call book again"),
    ],
)
async def test_bad_arguments_name_what_to_fix(arguments, message):
    with pytest.raises(ToolError, match=message):
        await ToolRegistry([_spec()]).call("book", arguments)


@pytest.mark.parametrize(
    ("arguments", "missing"),
    [
        ({"phone": "+1"}, ["day"]),
        ({"day": "", "phone": "+1"}, ["day"]),
        ({"day": " ", "phone": "+1", "party": 2}, ["day"]),
        ({"day": ""}, ["day", "phone"]),
    ],
)
async def test_a_required_argument_left_out_or_empty_is_missing(arguments, missing):
    recorder = Recorder()
    refused = await ToolRegistry([_spec(executor=recorder)]).call("book", arguments)
    one = len(missing) == 1
    assert refused == {
        "refused": f"{' and '.join(missing)} {'is' if one else 'are'} missing. Ask the caller "
        f"for {'it' if one else 'them'}, then call book again.",
        "missing": missing,
    }
    assert recorder.arguments is None


@pytest.mark.parametrize(
    ("result", "spoke", "replies"),
    [
        ({"refused": "day is missing.", "missing": ["day"]}, True, False),
        ({"refused": "day is missing.", "missing": ["day"]}, False, True),
        ({"refused": "phone not set yet."}, True, True),
        ({"error": "day: bad"}, True, True),
        ({"missing": []}, True, True),
        (None, True, True),
    ],
)
def test_the_model_replies_to_a_result_unless_its_reply_asked_for_the_missing_argument(
    result, spoke, replies
):
    assert replies_to(result, spoke=spoke) is replies


async def test_an_argument_the_schema_does_not_name_is_refused():
    with pytest.raises(ToolError, match=r"arguments: bogus is not an argument of book\. Call book"):
        await ToolRegistry([_spec()]).call("book", {"day": "x", "phone": "+1", "bogus": 1})
    with pytest.raises(
        ToolError, match=r"a, b are not arguments of book\. Call book again without"
    ):
        await ToolRegistry([_spec()]).call("book", {"day": "x", "phone": "+1", "b": 1, "a": 1})

    @local_tool
    def echo(day: str) -> AnyResult:
        """Echo."""
        return AnyResult(day=day)

    with pytest.raises(ToolError, match="bogus is not an argument of echo"):
        await ToolRegistry([echo]).call("echo", {"day": "x", "bogus": 1})


@pytest.mark.parametrize(
    "extra",
    [
        {"additionalProperties": True},
        {"additionalProperties": {"type": "integer"}},
    ],
)
async def test_an_open_schema_passes_extra_arguments_through(extra):
    recorder = Recorder()
    parameters = {"type": "object", "properties": {"day": {"type": "string"}}, **extra}
    spec = ToolSpec("book", "Book.", parameters, AnyResult, recorder)
    await ToolRegistry([spec]).call("book", {"day": "x", "xtra": 1})
    assert recorder.arguments == {"day": "x", "xtra": 1}


async def test_a_schema_without_properties_is_open():
    recorder = Recorder()
    spec = ToolSpec("ping", "Ping.", {"type": "object"}, AnyResult, recorder)
    await ToolRegistry([spec]).call("ping", {"anything": 1})
    assert recorder.arguments == {"anything": 1}


async def test_empty_arguments_read_as_an_empty_object():
    recorder = Recorder()
    spec = ToolSpec("ping", "Ping.", {"type": "object", "properties": {}}, AnyResult, recorder)
    await ToolRegistry([spec]).call("ping", "")
    assert recorder.arguments == {}


async def test_unknown_tool_lists_the_tools():
    with pytest.raises(ToolError, match=r"no tool 'nope'. The tools are: book"):
        await ToolRegistry([_spec()]).call("nope", {})


async def test_only_keeps_the_named_tools_and_refuses_an_unknown_one():
    other = ToolSpec("cancel", "Cancel a table.", BOOK, AnyResult, Recorder())
    step = ToolRegistry([_spec(), other]).only(["cancel"])
    assert [schema["function"]["name"] for schema in step.schemas()] == ["cancel"]
    with pytest.raises(ToolError, match="there is no tool 'book'"):
        await step.call("book", {"day": "2026-10-02", "phone": "+34600111222"})
    with pytest.raises(ValueError, match="there is no tool 'ghost'"):
        ToolRegistry([_spec()]).only(["ghost"])


def test_two_tools_with_one_name_are_refused():
    with pytest.raises(ValueError, match="two tools are named 'book'"):
        ToolRegistry([_spec(), _spec()])


@pytest.mark.parametrize("parameters", [{"type": "string"}, {"type": "object", "$id": "x"}])
def test_bad_spec_fails_at_start(parameters):
    with pytest.raises(ParameterSchemaError):
        ToolSpec("book", "Book.", parameters, AnyResult, Recorder())


async def test_handler_bug_tells_the_model_not_to_claim_success(caplog):
    @local_tool
    def explode() -> Done:
        """Fail."""
        raise RuntimeError("database password in here")

    with pytest.raises(ToolError) as caught:
        await ToolRegistry([explode]).call("explode", {})
    assert caught.value.message == (
        "explode failed before completing. Do not claim success; retry only with corrected input."
    )
    assert "tool explode failed" in caplog.text


async def test_tool_error_from_a_handler_reaches_the_model_unchanged():
    @local_tool
    async def full() -> Done:
        """Say the day is full."""
        raise ToolError("that day is full")

    with pytest.raises(ToolError, match=r"^that day is full$"):
        await ToolRegistry([full]).call("full", {})


async def test_local_tool_reads_signature_and_docstring_and_runs_sync_code_in_a_thread():
    threads = []

    @local_tool
    def find_slots(
        day: date, party: Annotated[int, Field(description="How many people.", ge=1)] = 2
    ) -> AnyResult:
        """Return the free times on a day.

        Args:
            day: The day.
        """
        threads.append(threading.current_thread())
        return AnyResult(times=[f"{day.isoformat()} 10:00 for {party}"])

    function = ToolRegistry([find_slots]).schemas()[0]["function"]
    assert function["description"] == "Return the free times on a day."
    assert function["parameters"]["required"] == ["day"]
    assert function["parameters"]["properties"]["party"]["description"] == "How many people."
    result = await ToolRegistry([find_slots]).call("find_slots", {"day": "2026-10-07"})
    assert result == {"times": ["2026-10-07 10:00 for 2"]}
    assert threads != [threading.main_thread()]


async def test_local_tool_gets_locked_values_as_keywords():
    async def save(note: str, phone: str) -> dict:
        return {"note": note, "phone": phone}

    tool = local_tool(
        save, output=AnyResult, description="Save a note.", locked={"phone": "{{phone}}"}
    )
    assert "phone" not in tool.model_parameters["properties"]
    result = await ToolRegistry([tool]).call("save", {"note": "hi"}, _state(phone="+34600111222"))
    assert result == {"note": "hi", "phone": "+34600111222"}


class Day(BaseModel):
    day: date


async def test_local_tool_refuses_a_value_pydantic_cannot_read():
    @local_tool
    def on(day: date) -> Day:
        """Echo a day."""
        return Day(day=day)

    with pytest.raises(ToolError, match=r"^day: Input should be a valid date"):
        await ToolRegistry([on]).call("on", {"day": "tomorrow"})


def _untyped(day, party=2) -> Day:
    """No hint."""
    return Day(day=day)


def _many(*args: int) -> Day:
    """Many."""
    return Day(day=args[0])


def _no_output(day: date) -> dict:
    """Returns a dict and names no output."""
    return {"day": day}


def _two_outputs(day: date) -> Day:
    """Returns one model and names another."""
    return Day(day=day)


@pytest.mark.parametrize(
    ("handler", "options", "message"),
    [
        (_untyped, {}, "give the parameter 'day' a type"),
        (_many, {}, r"\*args and \*\*kwargs cannot be tool arguments"),
        (_no_output, {}, "name its output: return a Pydantic model"),
        (_two_outputs, {"output": AnyResult}, "returns Day but output= is AnyResult"),
    ],
)
def test_local_tool_refuses_a_function_it_cannot_type(
    handler: Any, options: dict[str, Any], message: str
):
    with pytest.raises(TypeError, match=message):
        local_tool(handler, **options)


async def test_a_lone_model_parameter_is_the_whole_input_and_the_return_is_the_output():
    class Output(BaseModel):
        day: date
        free: bool

    def check_day(value: Day) -> Output:
        """Check a day."""
        return Output(day=value.day, free=True)

    tool = local_tool(check_day)
    assert tool.parameters["required"] == ["day"]
    assert tool.output is Output
    result = await ToolRegistry([tool]).call("check_day", '{"day": "2026-10-07"}')
    assert result == {"day": "2026-10-07", "free": True}


class _SyncRun:
    def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        return {}


@pytest.mark.parametrize("executor", [object(), _SyncRun()])
def test_a_tool_whose_executor_has_no_async_run_is_refused_when_built(executor: Any):
    with pytest.raises(ValueError, match="needs a ToolExecutor with an async run"):
        ToolSpec("t", "T.", {"type": "object", "properties": {}}, AnyResult, executor)


def test_announce_lines_are_kept_stripped_and_one_line_is_one_alternative():
    assert _spec_with(announce=" One moment. ").announce == ("One moment.",)
    assert _spec_with(announce=["A.", "B."]).announce == ("A.", "B.")
    assert _spec_with().announce == ()


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"announce": [" "]}, "an announce line must not be blank"),
        ({"announce": "Hi {{name"}, "not {{name}} or {{name.field}}"),
    ],
)
def test_a_bad_announce_line_fails_at_start(fields, error):
    with pytest.raises(ValueError, match=error):
        _spec_with(**fields)


async def test_started_runs_once_the_call_is_checked_and_never_for_a_refused_one():
    started: list[str] = []
    registry = ToolRegistry([_spec_with(announce="One moment.")])

    async def note(tool: ToolSpec) -> None:
        started.append(tool.name)

    with pytest.raises(ToolError):
        await registry.call("book", {"day": 5, "phone": "+34600111222"}, started=note)
    assert await registry.call("book", {"day": "2026-10-12"}, started=note) == {
        "refused": "phone is missing. Ask the caller for it, then call book again.",
        "missing": ["phone"],
    }
    assert started == []
    await registry.call("book", {"day": "2026-10-12", "phone": "+34600111222"}, started=note)
    assert started == ["book"]


def test_only_a_successful_call_of_a_tool_that_ends_the_conversation_ends_the_call():
    registry = ToolRegistry(
        [
            _spec_with(effect="ends_conversation"),
            local_tool(lambda: {}, output=AnyResult, name="other", description="x"),
        ]
    )
    assert registry.ends_call("book", {"ok": True})
    assert registry.ends_call("book", "done")
    assert not registry.ends_call("book", {"error": "no slot"})
    assert not registry.ends_call("book", {"ok": False})
    assert not registry.ends_call("other", {"ok": True})
    assert not registry.ends_call("missing", {"ok": True})


def _spec_with(**fields: Any) -> ToolSpec:
    return ToolSpec("book", "Book a table.", BOOK, AnyResult, Recorder(), **fields)


def test_a_blank_description_becomes_the_tool_name():
    spec = ToolSpec("book", " ", BOOK, AnyResult, Recorder())
    assert spec.description == "book"
