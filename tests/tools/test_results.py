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
    ],
)
def test_a_save_or_a_reply_that_does_not_fit_is_refused_when_built(
    rows: dict[str, Any], message: str
):
    with pytest.raises(ValueError, match=message):
        _registry(**rows)


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
    assert seen[-1] == "tool.save_refused"
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
