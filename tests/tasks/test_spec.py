"""The spec models: they round-trip, and they refuse bad input.

The refused cases cover finish parameters, non-blocking steps, inputs, requirements, via
tools and spoken lines.
"""

import re
from typing import Any

import pytest
from pydantic import ValidationError

from tests.tasks.job import job, plan
from unmute_core.tasks import TasksSpec


def test_a_spec_survives_json_unchanged() -> None:
    spec = TasksSpec.model_validate(job())
    again = TasksSpec.model_validate_json(spec.model_dump_json())
    assert again == spec
    short = spec.model_dump(mode="json", exclude_defaults=True)
    assert "version" not in short and short["tasks"][1] == {
        "name": "take_note",
        "instructions": "Take a note.",
        "assign": [{"variable": "note", "path": ["note"], "optional": True}],
    }


def _task(**fields: Any) -> dict[str, Any]:
    return {"name": "x", "when": "w", "instructions": "x"} | fields


def _with(**fields: Any) -> dict[str, Any]:
    """The job with one extra task."""
    payload = job()
    payload["tasks"].append(_task(**fields))
    return payload


def _group(**fields: Any) -> dict[str, Any]:
    payload = job()
    payload["groups"][0].update(fields)
    return payload


def _task_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = job()
    payload["tasks"][1]["inputs"] = rows
    return payload


REFUSED = [
    # One-part paths are named after the field, longer ones after the variable.
    (
        _with(
            assign=[{"variable": "order", "path": ["a", "b"]}, {"variable": "s", "path": ["order"]}]
        ),
        "the finish parameter 'order'",
    ),
    (
        _with(assign=[{"variable": "a", "path": ["ref"]}, {"variable": "b", "path": ["ref"]}]),
        "the finish parameter 'ref'",
    ),
    (_with(assign=[{"variable": "note", "path": ["unserved_request"]}]), "keeps for a hand-back"),
    (_with(assign=[{"variable": "unserved_request", "path": ["a", "b"]}]), "keeps for a hand-back"),
    (
        _with(assign=[{"variable": "a", "path": ["x"]}, {"variable": "a", "path": ["y"]}]),
        "assigns 'a' twice",
    ),
    (_with(name="finish"), "reserved for the generated task tool"),
    (_with(name="summary"), "a handoff summary uses the prompt site task:summary"),
    (_with(tools=["finish"]), "the name of the step's own tool"),
    (_with(tools=["lookup_order", "lookup_order"]), "lists the tool 'lookup_order' twice"),
    (_with(instructions="Hi {{#if x}}"), "take only {{name}} placeholders"),
    (_with(instructions="Hi {{ not a name }}"), "take only {{name}} placeholders"),
    (_with(instructions="Hi {{name"), "take only {{name}} placeholders"),
    (_with(instructions="   "), "must not be blank"),
    (_with(when="  "), "omit it instead"),
    (_with(announce=["Hi {{name}}"]), "takes no placeholders"),
    (
        _with(finish=[{"tool": "lookup_order", "success": {"found": [True]}}]),
        "does not list in tools",
    ),
    (
        _with(tools=["lookup_order"], finish=[{"tool": "lookup_order", "success": {"a": [1, 1]}}]),
        "lists a value twice",
    ),
    (
        _with(
            tools=["lookup_order"],
            finish=[{"tool": "lookup_order", "success": {"a": [1]}, "say": ["Hi {{#if x}}"]}],
        ),
        "only {{result.<field>}} and {{variable}} placeholders",
    ),
    (
        _with(
            tools=["lookup_order"],
            finish=[
                {"tool": "lookup_order", "success": {"a": [1]}, "confirm": ["{{result.ref}}?"]}
            ],
        ),
        "a confirm line takes only {{input.<field>}} and {{variable}} placeholders",
    ),
    (
        _with(
            tools=["lookup_order"],
            finish=[{"tool": "lookup_order", "success": {"a": [1]}, "say": ["{{input.ref}}."]}],
        ),
        "a spoken line takes only {{result.<field>}} and {{variable}} placeholders",
    ),
    (
        _with(
            tools=["lookup_order"],
            finish=[
                {"tool": "lookup_order", "success": {"a": [1]}, "confirm": ["Close it?"]},
                {"tool": "lookup_order", "success": {"a": [2]}},
            ],
        ),
        "gives 'lookup_order' two different confirm lines",
    ),
    (_with(history="last_n"), "max_messages is required with"),
    (_with(max_messages=3), "max_messages is required with"),
    (_with(history="summary", max_messages=3), "max_messages is required with"),
    (_with(handoffs=["to_notes", "to_notes"]), "lists the handoff 'to_notes' twice"),
    (_with(tools=["lookup_order"], handoffs=["lookup_order"]), "as a handoff and as one"),
    (_with(handoffs=["finish"]), "as a handoff and as one"),
    (_with(via=["search_web"]), "which it does not list in tools"),
    (
        _with(requires=["note"], assign=[{"variable": "note", "path": ["n"]}]),
        "a value it saves itself",
    ),
    (_with(name="order_flow"), "names both a task and a group"),
    (_with(name="take_note"), "two tasks are named 'take_note'"),
    (_with(when=None), "nothing can start it"),
    (_group(name="summary"), "a handoff summary uses the prompt site task:summary"),
    (_group(steps=["find_order", "ghost"]), "names task 'ghost', which does not exist"),
    (_group(steps=["find_order", "find_order"]), "runs 'find_order' twice"),
    (_group(non_blocking=["ghost"]), "the non-blocking step 'ghost' is not one of its steps"),
    (_group(skip_when_confirmed={"ghost": "order_id"}), "the skipped step 'ghost'"),
    (job(task_only_tools=["search_web"]), "listed by no task"),
    (
        _task_inputs([{"kind": "memory", "name": "order_id", "value": "A-12"}]),
        "only an input of kind input",
    ),
    (
        _task_inputs([{"kind": "tool", "name": "normalize", "value": "x"}]),
        "only an input of kind input",
    ),
    (_task_inputs([{"kind": "input", "name": "salon_name"}]), "only an input of kind input"),
    (
        _task_inputs(
            [{"kind": "memory", "name": "order_id"}, {"kind": "memory", "name": "order_id"}]
        ),
        "takes an input twice",
    ),
    (_with(arguments=["note", "note"]), "takes the argument 'note' twice"),
    (_group(arguments=["note", "note"]), "takes the argument 'note' twice"),
    (_with(when=None, via=["search_web"], tools=["search_web"], arguments=["note"]), "no when"),
    (_with(opens_with={"tool": "search_web", "say": ["Hi."]}), "which it does not list in tools"),
    (
        _with(
            tools=["lookup_order"],
            finish=[{"tool": "lookup_order", "success": {"found": [True]}}],
            opens_with={"tool": "lookup_order", "say": ["Hi."]},
        ),
        "a tool it ends on",
    ),
    (
        _with(
            tools=["lookup_order"],
            finish=[
                {
                    "tool": "lookup_order",
                    "success": {"found": [True]},
                    "confirm": ["Look up {{input.order_id}}?"],
                }
            ],
            opens_with={
                "tool": "lookup_order",
                "arguments": {"order_id": "order_id"},
                "say": ["Hi."],
            },
        ),
        "Remove the opening's say lines",
    ),
    (
        _with(replies=[{"tool": "search_web", "when": {"ok": [True]}, "say": ["Hi."]}]),
        "replies to 'search_web', which it does not list in tools",
    ),
    (
        _with(tools=["lookup_order"], finish=[{"tool": "lookup_order", "success": {}}]),
        "has no success fields, so every result would complete the step",
    ),
    (
        _with(
            tools=["lookup_order"],
            finish=[
                {
                    "tool": "lookup_order",
                    "success": {"found": [True]},
                    "assign": [{"variable": "note", "path": ["n"]}],
                }
            ],
        ),
        "saves 'note', which the task's assign does not declare",
    ),
    (
        _with(
            tools=["lookup_order"],
            assign=[{"variable": "tags", "path": ["t"]}],
            finish=[
                {
                    "tool": "lookup_order",
                    "success": {},
                    "ends": "failed",
                    "assign": [{"variable": "tags", "path": ["t"], "append": True}],
                }
            ],
        ),
        "saves 'tags' with append set differently from the task's assign",
    ),
    (_group(then="transfer"), "then_target is required with transfer. Name the agent"),
    (_group(then_target="billing"), "then_target is legal with transfer only. Remove it"),
    (_group(then="end", then_target="billing"), "then_target is legal with transfer only"),
]


def test_a_group_transfers_the_call_to_the_agent_it_names() -> None:
    task_plan = plan(_group(then="transfer", then_target="billing"))
    group = task_plan.groups["order_flow"]
    assert (group.then, group.then_target) == ("transfer", "billing")
    assert task_plan.then_targets == {"order_flow": "billing"}
    task_plan.check_then_targets({"billing", "notes"})
    with pytest.raises(ValueError, match="transfers the call to 'billing', which then_targets"):
        task_plan.check_then_targets({"notes"})


@pytest.mark.parametrize(("payload", "message"), REFUSED)
def test_a_spec_that_breaks_a_rule_is_refused(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(message)):
        TasksSpec.model_validate(payload)


def test_a_requirement_has_exactly_one_provider_that_requires_nothing() -> None:
    TasksSpec.model_validate(_with(requires=["order_id"]))
    nobody = _with(requires=["note"])
    nobody["tasks"][1]["assign"] = []
    with pytest.raises(
        ValidationError, match="exactly one other task must save all of them, found: none"
    ):
        TasksSpec.model_validate(nobody)
    chained = job(groups=[])
    chained["tasks"].append(_task(name="refund", requires=["order_id"]))
    chained["tasks"][0]["requires"] = ["note"]
    chained["tasks"][1]["when"] = "The caller leaves a note."
    with pytest.raises(ValidationError, match="so it cannot require anything itself"):
        TasksSpec.model_validate(chained)


def test_a_group_steps_requirement_is_saved_by_an_earlier_step() -> None:
    payload = job()
    payload["tasks"][1]["requires"] = ["order_id"]
    TasksSpec.model_validate(payload)
    payload["groups"][0]["steps"] = ["take_note", "find_order"]
    with pytest.raises(ValidationError, match="runs 'take_note' before any step saves 'order_id'"):
        TasksSpec.model_validate(payload)


def test_a_via_tool_enters_one_task() -> None:
    payload = job()
    payload["tasks"][1] |= {"tools": ["search_web"], "via": ["search_web"]}
    TasksSpec.model_validate(payload)
    payload["tasks"][0] |= {"tools": ["lookup_order", "search_web"], "via": ["search_web"]}
    with pytest.raises(ValidationError, match="the tool 'search_web' enters two tasks"):
        TasksSpec.model_validate(payload)


def test_owners_name_the_first_task_that_saves_each_value():
    spec = TasksSpec.model_validate(job())
    owners = spec.owners()
    for task in spec.tasks:
        for row in task.assign:
            assert owners[row.variable] == next(
                t.name for t in spec.tasks if any(r.variable == row.variable for r in t.assign)
            )
    assert set(owners) == {row.variable for task in spec.tasks for row in task.assign}


def test_starters_name_what_the_main_agent_can_start():
    spec = TasksSpec.model_validate(
        {
            "mode": "record",
            "tasks": [
                {
                    "name": "verify",
                    "instructions": "Verify.",
                    "assign": [{"variable": "a", "path": ["a"]}],
                },
                {
                    "name": "lookup",
                    "when": "Asked.",
                    "instructions": "Look.",
                    "assign": [{"variable": "b", "path": ["b"]}],
                },
                {
                    "name": "hidden",
                    "instructions": "Hidden.",
                    "tools": ["search_web"],
                    "via": ["search_web"],
                    "assign": [{"variable": "c", "path": ["c"]}],
                },
            ],
            "groups": [
                {
                    "name": "booking",
                    "when": "Booking.",
                    "steps": ["verify", "lookup"],
                    "context_scope": "shared",
                }
            ],
        }
    )
    assert spec.starters() == {"a": "booking", "b": "lookup"}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"native_tools": ["mcp", "mcp"]}, "lists the native tool 'mcp' twice"),
        ({"native_tools": ["finish"]}, "'finish' as a native tool and as one of its own"),
        ({"handoffs": ["to_sales"], "native_tools": ["to_sales"]}, "as a native tool"),
    ],
)
def test_native_tools_are_unique_and_never_a_tool_of_the_task(fields, message):
    with pytest.raises(ValidationError, match=message):
        TasksSpec.model_validate(_with(**fields))


@pytest.mark.parametrize("text", ["Close it with }} when done.", 'Send {"a": {"b": 1}} as is.'])
def test_instructions_may_hold_braces_that_are_not_placeholders(text: str) -> None:
    spec = TasksSpec.model_validate(_with(instructions=text))
    assert spec.tasks[-1].instructions == text
