"""What a call does around the tool: its filled announce line, refusals, hidden results, events."""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from tests.outputs import AnyResult, NoArguments
from unmute_core.state import CallInfo, CallState
from unmute_core.tools import Done, ToolError, ToolRegistry, ToolSpec, local_tool, webhook_tool

NO_ARGUMENTS: dict[str, Any] = {"type": "object", "properties": {}}


class Booking(BaseModel):
    name: str | None = None
    day: str | None = None


class Answer:
    """An executor that returns a fixed result and counts its runs."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.runs = 0

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        self.runs += 1
        return self.result


def _tool(result: object = None, **options: Any) -> ToolSpec:
    return ToolSpec(
        "check",
        "Check the booking.",
        NO_ARGUMENTS,
        AnyResult,
        Answer(result or {"slots": 3}),
        **options,
    )


async def test_an_announce_line_is_filled_from_state_and_call_values():
    state = CallState(Booking, call=CallInfo(carrier_call_id="c-7"))
    state.save({"name": "Ana"})
    heard: list[object] = []
    tool = _tool(announce="One moment, {{name}}, checking call {{call.carrier_call_id}}.")

    await ToolRegistry([tool]).call("check", {}, state, started=lambda t: heard.append(t.announce))

    assert heard == [("One moment, Ana, checking call c-7.",)]


async def test_an_unset_value_in_the_line_refuses_before_anything_is_said_or_run():
    tool = _tool(announce="Checking {{day}} for you.")
    heard: list[ToolSpec] = []

    result = await ToolRegistry([tool]).call("check", {}, CallState(Booking), started=heard.append)

    assert result == {
        "refused": "day not set yet. Get it from the caller, then call this tool again."
    }
    assert heard == []
    assert isinstance(tool.executor, Answer)
    assert tool.executor.runs == 0


async def test_an_unset_url_value_refuses_before_the_announce_line():
    tool = webhook_tool(
        "lookup",
        "Look up the booking.",
        "https://bookings.example.com/days/{{day}}",
        input=NoArguments,
        output=AnyResult,
        announce="Let me look.",
    )
    heard: list[ToolSpec] = []

    result = await ToolRegistry([tool]).call("lookup", {}, CallState(Booking), started=heard.append)

    assert isinstance(result, dict)
    assert result["refused"].startswith("day not set yet")
    assert heard == []


@pytest.mark.parametrize(
    ("result", "shown"),
    [
        ({"card": "4111"}, {"ok": True}),
        ({"error": "card declined"}, {"error": "card declined"}),
        ({"ok": False, "reason": "x"}, {"ok": False, "reason": "x"}),
        ({"refused": "day not set yet"}, {"refused": "day not set yet"}),
    ],
)
def test_a_hidden_result_reads_ok_and_a_failure_is_always_shown(result, shown):
    registry = ToolRegistry([_tool(show_result=False)])
    assert registry.for_model("check", result) == shown


@pytest.mark.parametrize(
    ("result", "shown"),
    [
        ({"done": True, "read_back": "Shall I file it?"}, {"done": True}),
        ({"error": "not yet", "read_back": "x"}, {"error": "not yet", "read_back": "x"}),
    ],
)
def test_a_hidden_field_is_left_out_of_what_the_model_reads(result, shown):
    registry = ToolRegistry([_tool(hidden=("read_back",))])
    assert registry.for_model("check", result) == shown


def test_a_hidden_field_the_output_does_not_have_is_refused_when_built():
    class Checked(BaseModel):
        done: bool

    with pytest.raises(ValueError, match="hides 'read_bak'"):
        ToolSpec("check", "Check.", NO_ARGUMENTS, Checked, Answer({}), hidden=("read_bak",))


def test_a_shown_result_reaches_the_model_unchanged():
    assert ToolRegistry([_tool()]).for_model("check", {"slots": 3}) == {"slots": 3}


def test_result_instructions_follow_the_description():
    registry = ToolRegistry([_tool(result_instructions="Read out the number of slots only.")])
    assert registry.schemas()[0]["function"]["description"] == (
        "Check the booking.\n\nUse the result like this: Read out the number of slots only."
    )


async def test_events_name_each_stage_and_a_broken_sink_never_breaks_the_call(caplog):
    @local_tool
    def fail() -> Done:
        """Fail."""
        raise ToolError("no slots")

    seen: list[tuple[str, dict[str, Any]]] = []
    later = ToolSpec("later", "Later.", NO_ARGUMENTS, AnyResult, Answer({}), announce="On {{day}}.")
    registry = ToolRegistry([_tool(), fail, later])
    registry.listen(lambda stage, data: seen.append((stage, data)))

    def broken(stage: str, data: dict[str, Any]) -> None:
        raise RuntimeError(stage)

    registry.listen(broken)

    await registry.call("check", {})
    with pytest.raises(ToolError):
        await registry.call("fail", {})
    await registry.call("later", {}, CallState(Booking))

    stages = [(stage, data["name"]) for stage, data in seen]
    assert stages == [
        ("tool.started", "check"),
        ("tool.succeeded", "check"),
        ("tool.started", "fail"),
        ("tool.failed", "fail"),
        ("tool.refused", "later"),
    ]
    assert seen[3][1]["error"] == "no slots"
    assert isinstance(seen[1][1]["ms"], int)
    assert "tool event sink failed" in caplog.text


async def test_a_result_that_reports_a_failure_is_a_failed_event():
    seen: list[str] = []
    registry = ToolRegistry([_tool({"ok": False})])
    registry.listen(lambda stage, data: seen.append(stage))
    await registry.call("check", {})
    assert seen == ["tool.started", "tool.failed"]


async def test_only_and_with_tools_keep_the_sinks():
    seen: list[str] = []
    registry = ToolRegistry([_tool()])
    registry.listen(lambda stage, data: seen.append(data["name"]))
    more = registry.with_tools(
        local_tool(lambda: {}, output=AnyResult, name="noop", description="Nothing.")
    )
    await more.only(["noop"]).call("noop", {})
    assert seen == ["noop", "noop"]


class Shape(BaseModel):
    meta: dict[str, object] | None = None
    name: str | None = None


@pytest.mark.parametrize(
    ("values", "line", "said"),
    [
        ({"meta": {"a": {"b": 1}}}, "Checking {{meta}}.", ('Checking {"a":{"b":1}}.',)),
        ({"name": "  "}, "{{name}}", ()),
        ({"name": "{{meta}}"}, "Hi {{name}}.", ("Hi {{meta}}.",)),
        ({"name": "Ana"}, "You said {{call.first_user_message}}.", ("You said use {{name}}.",)),
    ],
)
async def test_a_filled_line_is_never_checked_again_or_filled_twice(values, line, said):
    state = CallState(Shape, call=CallInfo(first_user_message="use {{name}}"))
    state.save(values)
    heard: list[object] = []
    tool = ToolSpec("check", "Check.", NO_ARGUMENTS, AnyResult, Answer({}), announce=line)
    await ToolRegistry([tool]).call("check", {}, state, started=lambda t: heard.append(t.announce))
    assert heard == [said]


async def test_a_cancelled_call_ends_its_events():
    import asyncio

    seen: list[str] = []
    began = asyncio.Event()

    class Hang:
        async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
            began.set()
            await asyncio.Event().wait()
            return {}

    registry = ToolRegistry([ToolSpec("hang", "Hang.", NO_ARGUMENTS, AnyResult, Hang())])
    registry.listen(lambda stage, data: seen.append(f"{stage} {data.get('error', '')}".strip()))
    task = asyncio.create_task(registry.call("hang", {}))
    await began.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == ["tool.started", "tool.failed cancelled"]


def test_a_refused_field_that_is_false_is_no_failure():
    registry = ToolRegistry([_tool(show_result=False)])
    assert registry.for_model("check", {"refused": False, "card": "x"}) == {"ok": True}


class Order(BaseModel):
    order_id: str | None = Field(default=None, description="The number on the receipt")
    note: str | None = None


async def test_a_refusal_names_the_task_that_saves_the_value():
    state = CallState(Order)
    state.suppliers["order_id"] = "find_order"
    tool = _tool(announce="Checking {{order_id}}.")

    result = await ToolRegistry([tool]).call("check", {}, state)

    assert result == {
        "refused": "order_id not set yet. Run find_order first: it saves order_id, "
        "then call this tool again."
    }


async def test_a_refusal_quotes_the_description_when_no_task_saves_the_value():
    tool = _tool(announce="Checking {{order_id}}.")

    result = await ToolRegistry([tool]).call("check", {}, CallState(Order))

    assert result == {
        "refused": "order_id not set yet. Get it from the caller (The number on the receipt), "
        "then call this tool again."
    }


async def test_a_confirming_task_comes_before_the_supplier():
    state = CallState(Order, confirm={"order_id": "confirm_order"})
    state.suppliers["order_id"] = "find_order"
    state.save({"order_id": "A1"})

    result = await ToolRegistry([_tool(announce="Checking {{order_id}}.")]).call("check", {}, state)

    assert "Run confirm_order first: it confirms order_id" in str(result)


def test_only_reads_a_generator_once_and_the_registry_knows_its_names():
    registry = ToolRegistry(
        [_tool(), local_tool(lambda: {}, output=AnyResult, name="noop", description="Nothing.")]
    )
    only = registry.only(name for name in ["noop"])
    assert only.names == ("noop",)
    assert "noop" in only
    assert "check" not in only
    with pytest.raises(ValueError, match="there is no tool 'ghost'"):
        registry.only(name for name in ["ghost"])
