"""The save tool for values no task saves, and the block that shows their current text."""

import pytest
from pydantic import BaseModel, Field

from unmute_core.state import CallState
from unmute_core.tools import ToolError, ToolRegistry, values_block, values_tool


class Caller(BaseModel):
    name: str | None = Field(None, description="The caller's name")
    party: int = 2
    booking: str | None = None


OWNERS = {"booking": "book_table"}


def _registry(state: CallState) -> ToolRegistry:
    return ToolRegistry([values_tool(state, owners=OWNERS)])


def test_the_schema_offers_every_value_no_task_saves():
    [schema] = _registry(CallState(Caller)).schemas()
    parameters = schema["function"]["parameters"]
    assert schema["function"]["name"] == "save_values"
    assert set(parameters["properties"]) == {"name", "party"}
    assert parameters["properties"]["name"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "description": "The caller's name",
    }
    assert parameters["properties"]["party"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    assert "required" not in parameters


async def test_a_save_lands_in_state_and_null_clears_to_the_default():
    state = CallState(Caller)
    events = []
    state.listen(events.append)
    registry = _registry(state)

    saved = await registry.call("save_values", {"name": "Ana", "party": 4}, state)
    cleared = await registry.call("save_values", {"party": None}, state)

    assert saved == {"ok": True, "saved": {"name": "Ana", "party": 4}}
    assert cleared == {"ok": True, "saved": {"party": 2}}
    assert state.values.name == "Ana"
    assert events[0].origin.kind == "tool"
    assert events[0].origin.name == "save_values"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"booking": "x"}, "booking is saved by the task book_table. Start that task to change it"),
        ({"mood": "good"}, "mood is not a value of this call. The values are: name, party"),
        ({"party": "many"}, "party: 'many' is not valid"),
    ],
)
async def test_a_value_a_task_owns_or_an_unknown_one_is_refused(arguments, message):
    state = CallState(Caller)
    with pytest.raises(ToolError, match=message):
        await _registry(state).call("save_values", arguments, state)
    assert state.dump() == Caller().model_dump(mode="json", by_alias=True)


def test_a_state_every_task_owns_has_nothing_to_save():
    with pytest.raises(ValueError, match="nothing to save"):
        values_tool(CallState(Caller), owners=dict.fromkeys(["name", "party", "booking"], "t"))


def test_the_block_lists_each_value_its_owner_and_its_text():
    state = CallState(Caller)
    state.save({"name": "Ana {{party}}"})
    assert values_block(state, owners=OWNERS) == (
        "Values of this call. Save new ones with save_values.\n"
        "- name (The caller's name): Ana { {party} }\n"
        "- party: 2\n"
        "- booking, saved by the task book_table: none recorded yet."
    )


def test_the_block_names_no_save_tool_when_every_value_is_owned():
    state = CallState(Caller)
    owners = dict.fromkeys(["name", "party", "booking"], "t")
    assert values_block(state, owners=owners).splitlines()[0] == "Values of this call."


async def test_a_value_the_call_starts_with_is_shown_but_never_saved() -> None:
    class Shop(BaseModel):
        brand: str | None = None
        note: str | None = None

    state = CallState(Shop, call_start={"brand": "optional"})
    state.save({"brand": "Corner Shop"})
    tool = values_tool(state)
    assert list(tool.parameters["properties"]) == ["note"]
    with pytest.raises(ToolError, match="brand is set when the call starts"):
        await ToolRegistry([tool]).call(tool.name, {"brand": "Other"}, state)
    assert "- brand, set when the call starts: Corner Shop" in values_block(state)
