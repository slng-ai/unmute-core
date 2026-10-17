"""The plan: what it indexes for the adapters, and the checks against the State and tools.

The cases cover the plan, entry history, providers, via tools, confirmation and skips.
"""

import re
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from tests.outputs import AnyResult
from tests.tasks.job import OPEN, Canned, State, job, plan, registry, state
from unmute_core.model import variable_key_max
from unmute_core.model._prompt import check_keys, render_prompt
from unmute_core.state import CallState
from unmute_core.tasks import TaskPlan, TasksSpec, entry_history, tool_succeeded
from unmute_core.tools import ToolRegistry, ToolSpec


def test_plan_indexes_delegates_steps_and_hidden_tools() -> None:
    task_plan = plan()
    assert [(row.kind, row.name) for row in task_plan.delegates()] == [
        ("task", "find_order"),
        ("group", "order_flow"),
    ]
    assert [task.name for task in task_plan.steps("order_flow")] == ["find_order", "take_note"]
    assert not task_plan.visible_to_main("lookup_order")
    assert task_plan.visible_to_main("search_web")
    assert task_plan.ends_on_tool("order_flow") and not task_plan.ends_on_tool("take_note")
    assert task_plan.record and not plan(job(mode="status")).record
    assert TaskPlan(TasksSpec(), state=task_plan.state, tools=task_plan.tools).delegates() == []


def test_entry_history_is_the_task_own_alone_and_in_a_shared_group() -> None:
    task = plan().tasks["find_order"]
    assert entry_history(task, group_scope=None) == "messages"
    assert entry_history(task, group_scope="shared") == "messages"
    full = task.model_copy(update={"history": "full"})
    assert entry_history(full, group_scope="shared") == "full"
    assert entry_history(task, group_scope="isolated") == "reset"
    reset = task.model_copy(update={"history": "reset"})
    assert entry_history(reset, group_scope="shared") == "reset"
    assert entry_history(task, group_scope="isolated", reentry=True) == "caller_turn"
    assert entry_history(reset, group_scope="shared", reentry=True) == "caller_turn"
    assert entry_history(task, group_scope="shared", reentry=True) == "messages"
    assert entry_history(task, group_scope=None, reentry=True) == "messages"


def test_tool_success_is_an_answer_without_an_error() -> None:
    assert tool_succeeded({"found": True})
    assert tool_succeeded('{"ok": true}')
    assert tool_succeeded("plain text")
    assert tool_succeeded({"error": None, "booking_id": "B-7"})
    assert tool_succeeded('{"error": ""}')
    assert not tool_succeeded({"error": "http_error"})
    assert not tool_succeeded({"ok": False})
    assert not tool_succeeded('{"error": "timeout"}')


def _refund(**fields: Any) -> dict[str, Any]:
    return {
        "name": "refund",
        "when": "The caller wants a refund.",
        "instructions": "Refund.",
        "requires": ["order_id"],
    } | fields


def test_a_task_waits_for_its_one_provider() -> None:
    payload = job()
    payload["tasks"].append(_refund())
    task_plan = plan(payload)
    assert task_plan.providers == {"refund": "find_order"}
    refund = task_plan.tasks["refund"]
    call_state = state(task_plan)
    assert task_plan.missing(refund, call_state) == ["order_id"]
    prerequisite = task_plan.prerequisite(refund, call_state)
    assert prerequisite is not None and prerequisite.name == "find_order"
    call_state.save({"order_id": "A-1"})
    assert task_plan.prerequisite(refund, call_state) is None


def test_via_tools_index_the_task_they_enter() -> None:
    payload = job()
    payload["tasks"][1] |= {"tools": ["search_web"], "via": ["search_web"]}
    assert plan(payload).via_tasks == {"search_web": "take_note"}


def test_a_step_is_skipped_only_while_the_value_it_confirms_is_confirmed() -> None:
    payload = job()
    payload["groups"][0]["skip_when_confirmed"] = {"find_order": "order_id"}
    task_plan = plan(payload, confirm={"order_id": "find_order"})
    assert task_plan.withdrawing == {"find_order"}
    call_state = state(task_plan)
    assert task_plan.skipped_steps("order_flow", call_state.is_confirmed) == frozenset()
    call_state.save({"order_id": "A-12"}, step="find_order")
    assert task_plan.skipped_steps("order_flow", call_state.is_confirmed) == {0}


def test_a_step_takes_its_groups_inputs_then_its_own_once_each() -> None:
    payload = job()
    payload["groups"][0]["inputs"] = [
        {"kind": "tool", "name": "normalize"},
        {"kind": "input", "name": "salon_name", "value": "Luna Salon"},
    ]
    payload["tasks"][1]["inputs"] = [
        {"kind": "memory", "name": "order_id"},
        {"kind": "input", "name": "salon_name", "value": "Luna Salon"},
    ]
    task_plan = plan(payload)
    step = task_plan.tasks["take_note"]
    assert [(row.kind, row.name) for row in task_plan.inputs_for(step, "order_flow")] == [
        ("tool", "normalize"),
        ("input", "salon_name"),
        ("memory", "order_id"),
    ]
    assert [row.name for row in task_plan.inputs_for(step)] == ["order_id", "salon_name"]
    assert task_plan.inputs_for(task_plan.tasks["find_order"]) == []


def _bad(path: list[Any], value: Any) -> dict[str, Any]:
    """The job with one field changed, by a path of keys and indexes."""
    payload = job()
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return payload


def _opening_reads(value: str) -> dict[str, Any]:
    """The job with ``find_order`` opening on a search for ``value``."""
    payload = job()
    payload["tasks"][0]["tools"] = ["lookup_order", "search_web"]
    payload["tasks"][0]["opens_with"] = {
        "tool": "search_web",
        "arguments": {"q": value},
        "say": ["Hi."],
    }
    return payload


@pytest.mark.parametrize(
    ("payload", "confirm", "message"),
    [
        (
            _bad(["tasks", 1, "tools"], ["ghost"]),
            None,
            "task 'take_note': there is no tool 'ghost'",
        ),
        (
            _bad(["groups", 0, "arguments"], ["ghost"]),
            None,
            "group 'order_flow' takes 'ghost', which the State does not have",
        ),
        (
            _bad(["tasks", 0, "arguments"], ["ghost"]),
            None,
            "task 'find_order' takes 'ghost', which the State does not have",
        ),
        (
            _opening_reads("ghost"),
            None,
            "task 'find_order' reads 'ghost', which the State does not have",
        ),
        (
            _bad(["tasks", 1, "assign"], [{"variable": "ghost", "path": ["g"]}]),
            None,
            "assigns 'ghost', which the State does not have",
        ),
        (
            _bad(["tasks", 1, "assign"], [{"variable": "note", "path": ["n"], "append": True}]),
            None,
            "appends to 'note': make it a list",
        ),
        (_bad(["tasks", 1, "instructions"], "Read {{ghost}}."), None, "reads 'ghost'"),
        (_bad(["tasks", 1, "instructions"], "Read {{ghost__name}}."), None, "reads 'ghost'"),
        (
            _bad(["tasks", 1, "instructions"], "Read {{tags.first}}."),
            None,
            "instructions {{tags.first}}: tags is a list, and a path cannot name a field",
        ),
        (
            _bad(["tasks", 1, "instructions"], "Read {{note.first}}."),
            None,
            "note is str, which has no fields to name",
        ),
        (_bad(["groups", 0, "name"], "search_web"), None, "names both a task or group and a tool"),
        (
            _bad(["tasks", 1, "inputs"], [{"kind": "memory", "name": "ghost"}]),
            None,
            "reads 'ghost'",
        ),
        (
            _bad(["groups", 0, "inputs"], [{"kind": "memory", "name": "ghost"}]),
            None,
            "reads 'ghost'",
        ),
        (
            _bad(
                ["tasks", 0, "finish"],
                [
                    {
                        "tool": "lookup_order",
                        "success": {"found": [True]},
                        "say": ["Saved {{nobody}}."],
                    }
                ],
            ),
            None,
            "reads 'nobody'",
        ),
        (
            _bad(["tasks", 1, "assign"], [{"variable": "note", "path": ["n"], "source": "input"}]),
            None,
            "saves 'note' from a tool's input, but no finish rule names a tool",
        ),
        (job(), {"note": "take_note"}, "which the agent does not run directly"),
        (job(), {"note": "find_order"}, "must assign 'note', the value it confirms"),
        (job(), {"note": "ghost"}, "which the agent does not run directly"),
        (
            _bad(["groups", 0, "skip_when_confirmed"], {"take_note": "order_id"}),
            {"order_id": "find_order"},
            "Name a value that step confirms",
        ),
    ],
)
def test_a_plan_refuses_names_its_state_or_tools_do_not_have(
    payload: dict[str, Any], confirm: dict[str, str] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        plan(payload, confirm=confirm)


def _typed_lookup() -> ToolRegistry:
    """The job's tools, with ``lookup_order`` taking one typed input, ``order_id``."""
    schema = {"type": "object", "properties": {"order_id": {"type": "string"}}}
    tools = registry()
    return ToolRegistry(
        [
            ToolSpec("lookup_order", "Find an order.", schema, AnyResult, Canned()),
            tools.get("search_web"),
            tools.get("normalize"),
        ]
    )


def test_a_confirm_line_reads_only_inputs_the_tool_takes() -> None:
    payload = job()
    payload["tasks"][0]["finish"][0]["confirm"] = ["Close {{input.order_number}}?"]
    with pytest.raises(ValueError, match="asks about 'order_number' before 'lookup_order' runs"):
        plan(payload, tools=_typed_lookup())
    payload["tasks"][0]["finish"][0]["confirm"] = ["Close {{input.order_id}}?"]
    assert plan(payload, tools=_typed_lookup())


def test_a_step_saves_only_inputs_its_closing_tool_takes() -> None:
    payload = job()
    payload["tasks"][0]["assign"][0] = {
        "variable": "order_id",
        "path": ["order_number"],
        "source": "input",
    }
    with pytest.raises(ValueError, match="input 'order_number', which no closing tool takes"):
        plan(payload, tools=_typed_lookup())
    payload["tasks"][0]["assign"][0]["path"] = ["order_id"]
    assert plan(payload, tools=_typed_lookup())


class _Order(BaseModel):
    id: str


class _Found(BaseModel):
    found: bool
    status: Literal["open", "closed"] | None = None
    order: _Order | None = None


def _typed() -> ToolRegistry:
    tools = registry()
    lookup = ToolSpec("lookup_order", "Find an order.", OPEN, _Found, Canned())
    return ToolRegistry([lookup, tools.get("search_web"), tools.get("normalize")])


def _step(**changes: Any) -> dict[str, Any]:
    payload = job()
    payload["tasks"][0] |= changes
    return payload


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            _step(finish=[{"tool": "lookup_order", "success": {"booked": [True]}}]),
            "reads result.booked, which _Found, the output of lookup_order, does not have",
        ),
        (
            _step(replies=[{"tool": "lookup_order", "when": {"colour": ["red"]}, "say": ["Red."]}]),
            "reads result.colour",
        ),
    ],
)
def test_a_step_reads_only_result_fields_its_tool_returns(
    payload: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        plan(payload, tools=_typed())


def test_a_confirming_task_that_runs_directly_and_saves_the_value_is_accepted() -> None:
    assert plan(confirm={"order_id": "find_order"}).confirm == {"order_id": "find_order"}


def test_a_plan_with_braces_in_instructions_renders_them_unchanged() -> None:
    text = 'Send {"a": {"b": 1}} and close with }}.'
    payload = _bad(["tasks", 1, "instructions"], text)
    task_plan = plan(payload)
    rendered = render_prompt(task_plan.tasks["take_note"].instructions, CallState(State))
    assert rendered == text


def test_a_tool_input_must_name_a_prefetch_lookup() -> None:
    payload = job()
    payload["groups"][0]["inputs"] = [{"kind": "tool", "name": "load_group"}]
    payload["tasks"][1]["inputs"] = [
        {"kind": "tool", "name": "load_customr"},
        {"kind": "memory", "name": "order_id"},
    ]
    task_plan = plan(payload)
    task_plan.check_lookups(["load_customr", "load_group"])
    with pytest.raises(ValueError, match=r"task take_note reads the tool 'load_customr', which no"):
        task_plan.check_lookups(["load_customer", "load_group"])
    with pytest.raises(ValueError, match=r"group order_flow reads the tool 'load_group'"):
        task_plan.check_lookups(["load_customr"])


def test_a_tool_input_with_no_lookups_at_all_is_refused() -> None:
    payload = job()
    payload["tasks"][1]["inputs"] = [{"kind": "tool", "name": "load_customer"}]
    with pytest.raises(ValueError, match=r"reads the tool 'load_customer', which no prefetch"):
        plan(payload).check_lookups([])


class _KeyLimit:
    variable_key_max = 64


def test_a_placeholder_key_is_checked_only_against_a_model_call_that_has_a_limit() -> None:
    long = "a" * 70
    assert variable_key_max(object()) is None
    assert variable_key_max(object(), _KeyLimit()) == 64
    check_keys(f"Hi {{{{{long}}}}}", "the main prompt", 70)
    with pytest.raises(ValueError, match=r"longer than the 64 characters the model call takes"):
        check_keys(f"Hi {{{{{long}}}}}", "the main prompt", 64)
    plan(job()).check_keys(None)
