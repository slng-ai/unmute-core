"""What the conformance scenario ``tool_results`` cannot reach: build checks and save rules."""

from datetime import date
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from unmute_core.state import CallState
from unmute_core.tools import (
    Assign,
    Done,
    SpokenLines,
    ToolRegistry,
    ToolReply,
    local_tool,
)


class Order(BaseModel):
    number: str
    status: Literal["shipped", "held"]
    due: date


class Other(BaseModel):
    number: str


class State(BaseModel):
    order: Order | None = None
    other: Other | None = None
    count: int | None = None
    order_number: str | None = None
    numbers: list[str] = []


GOOD = {"number": "A-12", "status": "shipped", "due": "2026-10-16"}


def _returning(result: object):
    def find_order(order_id: str) -> dict:
        """Find an order."""
        return result  # ty: ignore[invalid-return-type]

    return local_tool(find_order, output=Order)


def _registry(**rows: Any) -> ToolRegistry:
    bye = local_tool(lambda: {}, output=Done, name="bye", effect="ends_conversation")
    return ToolRegistry([_returning(GOOD), bye], **rows)


def _assign(*rows: Assign) -> dict[str, Any]:
    return {"state": State, "assign": {"find_order": list(rows)}}


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            {"assign": {"find_order": [Assign(variable="order", path=())]}},
            "Build the ToolRegistry with state=YourState",
        ),
        (
            {"state": State, "assign": {"ghost": [Assign(variable="order", path=())]}},
            "there is no tool 'ghost'",
        ),
        (_assign(Assign(variable="order", path=("dew",))), "reads result.dew"),
        (
            _assign(Assign(variable="order_number", path=("order_id",), source="input")),
            "a tool's own save reads its result",
        ),
        (
            _assign(Assign(variable="count", path=("due",))),
            "the result holds Date and the State value holds int",
        ),
        (_assign(Assign(variable="other", path=())), "that State value is not typed as Order"),
        (
            _assign(Assign(variable="order_number", path=("number",), append=True)),
            "that State value is not a list",
        ),
        (
            {"replies": [ToolReply(tool="find_order", when={"colour": ["red"]}, say=["Red."])]},
            "reads result.colour",
        ),
        (
            {"state": State, "replies": [ToolReply(tool="find_order", say=["Hi {{nobody}}."])]},
            "the State has no value 'nobody'",
        ),
        (
            {"state": State, "replies": [ToolReply(tool="find_order", say=["{{other.numbr}}"])]},
            "'numbr'",
        ),
        (
            {"replies": [ToolReply(tool="bye", say=["Goodbye."])]},
            "bye ends the call, so the call is over before a line could play",
        ),
        (
            {"replies": [ToolReply(tool="find_order", when={"status": ["lost"]}, say=["Lost."])]},
            "matches result.status on 'lost', which Order never holds there. It holds held, "
            "shipped",
        ),
        (
            {"replies": [ToolReply(when={"status": ["held"]}, say=["Held."])]},
            "a ToolReply on the ToolRegistry names its tool",
        ),
        (
            {
                "state": State,
                "replies": [
                    ToolReply(
                        tool="find_order",
                        say=["Found it."],
                        assign=(Assign(variable="count", path=("due",)),),
                    )
                ],
            },
            "the result holds Date and the State value holds int",
        ),
    ],
)
def test_a_save_or_a_reply_that_does_not_fit_is_refused_when_built(
    rows: dict[str, Any], message: str
):
    with pytest.raises(ValueError, match=message):
        _registry(**rows)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"replies": [ToolReply(tool="bye", say=["Held."])]},
            "tool find_order has a reply for 'bye'. Leave tool out",
        ),
        (
            {"replies": [ToolReply(when={"status": ["lost"]}, say=["Lost."])]},
            "matches result.status on 'lost'",
        ),
        (
            {"replies": [ToolReply(say=["{{result.colour}}"])]},
            "reads result.colour",
        ),
        (
            {"replies": [ToolReply(say=["Bye."])], "effect": "ends_conversation"},
            "tool find_order ends the call",
        ),
    ],
)
def test_a_tools_own_reply_that_does_not_fit_is_refused_when_built(
    options: dict[str, Any], message: str
):
    def find_order(order_id: str) -> Order:
        """Find an order."""
        return Order.model_validate(GOOD)

    with pytest.raises(ValueError, match=message):
        local_tool(find_order, **options)


@pytest.mark.parametrize(
    ("status", "line"),
    [("held", "The registry's line."), ("shipped", "The tool's line.")],
)
async def test_a_tools_own_reply_follows_the_registrys(status: str, line: str):
    def find_order(order_id: str) -> Order:
        """Find an order."""
        return Order.model_validate({**GOOD, "status": status})

    tool = local_tool(find_order, replies=[ToolReply(say=["The tool's line."])])
    registry = ToolRegistry(
        [tool],
        replies=[
            ToolReply(tool="find_order", when={"status": ["held"]}, say=["The registry's line."])
        ],
    )
    result = await registry.call("find_order", {"order_id": "A-12"}, None)
    assert registry.use_result("find_order", result, None, SpokenLines()) == line


@pytest.mark.parametrize(
    ("order_id", "result", "ran"),
    [("A-1", {**GOOD, "status": "held"}, []), ("A-12", GOOD, ["A-12"])],
)
async def test_a_precheck_outcome_is_the_result_and_the_tool_does_not_run(
    order_id: str, result: dict[str, Any], ran: list[str]
):
    runs: list[str] = []
    started: list[str] = []

    def find_order(order_id: str) -> Order:
        """Find an order."""
        runs.append(order_id)
        return Order.model_validate(GOOD)

    def too_old(order_id: str) -> Order | None:
        return Order.model_validate({**GOOD, "status": "held"}) if order_id == "A-1" else None

    registry = ToolRegistry([local_tool(find_order, precheck=too_old, announce="One moment.")])
    called = await registry.call(
        "find_order", {"order_id": order_id}, None, started=lambda tool: started.append(tool.name)
    )
    assert (called, runs, bool(started)) == (result, ran, bool(ran))
    assert registry.precheck("find_order", {"order_id": order_id}) == (None if ran else result)


async def _async_check(order_id: str) -> None:
    return None


@pytest.mark.parametrize(
    ("precheck", "message"),
    [
        (_async_check, "make it sync"),
        (lambda colour: None, "the precheck takes 'colour', which find_order does not"),
    ],
)
def test_a_precheck_that_cannot_run_before_the_read_back_is_refused_when_built(
    precheck: Any, message: str
):
    def find_order(order_id: str) -> Order:
        """Find an order."""
        return Order.model_validate(GOOD)

    with pytest.raises(TypeError, match=message):
        local_tool(find_order, precheck=precheck)


@pytest.mark.parametrize(
    ("row", "saved"),
    [
        (Assign(variable="order", path=()), {"order": GOOD}),
        (Assign(variable="numbers", path=("number",), append=True), {"numbers": ["A-1", "A-12"]}),
    ],
)
async def test_a_result_saves_the_whole_output_or_appends(row: Assign, saved: dict[str, Any]):
    registry = ToolRegistry([_returning(GOOD)], state=State, assign={"find_order": [row]})
    state = CallState(State)
    state.save({"numbers": ["A-1"]})
    result = await registry.call("find_order", {"order_id": "A-12"}, state)
    registry.use_result("find_order", result, state, SpokenLines())
    assert {name: state.plain(name) for name in saved} == saved


async def test_an_error_result_is_not_checked_and_saves_nothing():
    registry = ToolRegistry(
        [_returning({"error": "no such order"})],
        **_assign(Assign(variable="order", path=())),
    )
    state = CallState(State)
    result = await registry.call("find_order", {"order_id": "A-12"}, state)
    assert result == {"error": "no such order"}
    registry.use_result("find_order", result, state, SpokenLines())
    assert state.plain("order") is None


async def test_a_save_the_state_refuses_is_told_to_the_sinks_and_the_line_still_plays():
    registry = ToolRegistry(
        [_returning({**GOOD, "number": "A" * 70_000})],
        replies=[ToolReply(tool="find_order", say=["Found it."])],
        **_assign(Assign(variable="order_number", path=("number",))),
    )
    seen: list[str] = []
    registry.listen(lambda stage, data: seen.append(stage))
    state = CallState(State)
    result = await registry.call("find_order", {"order_id": "A-12"}, state)
    assert registry.use_result("find_order", result, state, SpokenLines()) == "Found it."
    assert seen[-2:] == ["tool.save_refused", "tool.replied"]
    assert state.plain("order_number") is None


async def test_a_reply_line_reads_a_field_of_a_shape_in_the_state():
    registry = ToolRegistry(
        [_returning(GOOD)],
        state=State,
        replies=[
            ToolReply(tool="find_order", say=["{{result.number}} replaces {{other.number}}."])
        ],
    )
    state = CallState(State)
    state.save({"other": {"number": "B-7"}})
    result = await registry.call("find_order", {"order_id": "A-12"}, state)
    assert registry.use_result("find_order", result, state, SpokenLines()) == "A-12 replaces B-7."
