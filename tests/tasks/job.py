"""The test job for the task tests.

Two tasks and a group over three values: ``find_order`` looks an order up with a tool that
ends the step, ``take_note`` saves a note, and ``order_flow`` runs both. A value left out of
a finish is saved as empty, so its rows are ``optional``.
"""

import copy
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field

from tests.outputs import AnyResult
from unmute_core.model import fill_messages, template_variables
from unmute_core.state import CallState, Id
from unmute_core.tasks import TaskPlan, TasksSpec
from unmute_core.tools import ToolExecutor, ToolRegistry, ToolSpec


class State(BaseModel):
    """What the test job remembers."""

    order_id: Id | None = None
    status: Literal["open", "closed"] | None = None
    note: str | None = None
    tags: list[str] = Field(default_factory=list)


class Canned:
    """A tool executor that answers with the next canned result and keeps the arguments."""

    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        self.calls.append(arguments)
        return self.results.pop(0) if self.results else {"ok": True}


OPEN = {"type": "object", "properties": {}, "additionalProperties": True}


def registry(**executors: ToolExecutor) -> ToolRegistry:
    """The job's tools: ``lookup_order`` (task-only), ``search_web`` and ``normalize``."""
    names = ("lookup_order", "search_web", "normalize")
    return ToolRegistry(
        ToolSpec(name, f"The {name} tool.", OPEN, AnyResult, executors.get(name) or Canned())
        for name in names
    )


JOB: dict[str, Any] = {
    "mode": "record",
    "tasks": [
        {
            "name": "find_order",
            "when": "The caller asks about an order.",
            "instructions": "Find order {{order_id}}.",
            "tools": ["lookup_order"],
            "assign": [
                {"variable": "order_id", "path": ["order", "id"], "optional": True},
                {"variable": "status", "path": ["status"], "optional": True},
            ],
            "finish": [{"tool": "lookup_order", "success": {"found": [True]}}],
            "announce": ["Let me look."],
        },
        {
            "name": "take_note",
            "instructions": "Take a note.",
            "assign": [{"variable": "note", "path": ["note"], "optional": True}],
        },
    ],
    "groups": [
        {
            "name": "order_flow",
            "when": "The caller wants help with an order.",
            "steps": ["find_order", "take_note"],
            "context_scope": "shared",
        }
    ],
    "task_only_tools": ["lookup_order"],
}


def job(**overrides: Any) -> dict[str, Any]:
    """A fresh copy of the job, with top-level keys replaced."""
    return copy.deepcopy(JOB) | overrides


def plan(
    payload: Mapping[str, Any] | None = None,
    *,
    confirm: Mapping[str, str] | None = None,
    tools: ToolRegistry | None = None,
) -> TaskPlan:
    """The job's plan, checked against ``State``."""
    spec = TasksSpec.model_validate(payload if payload is not None else job())
    return TaskPlan(spec, state=State, confirm=confirm, tools=tools or registry())


SPECIALIST: dict[str, Any] = {
    "mode": "record",
    "tasks": [
        {
            "name": "take_note",
            "when": "The caller wants a note taken.",
            "instructions": "Take a note.",
            "assign": [{"variable": "note", "path": ["note"], "optional": True}],
        }
    ],
}


def specialist_plan(tools: ToolRegistry | None = None) -> TaskPlan:
    """The plan of a second main agent on the same ``State``: ``take_note`` and ``search_web``.

    A handoff moves the call between it and the job's agent.
    """
    spec = TasksSpec.model_validate(copy.deepcopy(SPECIALIST))
    return TaskPlan(spec, state=State, tools=(tools or registry()).only(["search_web"]))


def state(task_plan: TaskPlan) -> CallState:
    """A fresh call state for a plan."""
    return CallState(task_plan.state, confirm=task_plan.confirm)


def filled(text: str, call_state: CallState, site: str) -> str:
    """A prompt as the model reads it: its placeholders filled at ``site``."""
    messages = [{"role": "system", "content": text}]
    variables = template_variables(messages, call_state, site)
    return fill_messages(messages, variables)[0]["content"]


class Sending(Canned):
    """A tool executor whose sends finish only when the call's end drains them."""

    def __init__(self) -> None:
        super().__init__({})
        self.drained = False

    async def drain(self, timeout: float = 5.0) -> None:
        self.drained = True
