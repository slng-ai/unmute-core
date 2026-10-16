"""The order a group's steps run in, with going back and non-blocking steps.

Going back is LiveKit ``TaskGroup``'s ``out_of_scope`` tool rebuilt so every
framework runs the same one: while a step runs, the caller can go back to steps this run
already entered and is not about to resume. Those run again in the order asked, then the
interrupted step, then the rest. It is offered only in ``record`` mode.

A step entered again is told why (``reentry_note``), and cannot end unserved until the
caller says something new in it (``refuses_unserved``). A non-blocking step that fails
lets the group carry on. An unserved step always stops the group: the caller asked for
something else.
"""

import json
from collections import deque
from collections.abc import Sequence
from dataclasses import field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.tasks._spec import GO_BACK_TOOL_NAME, TaskGroupSpec, TaskSpec
from unmute_core.tasks._texts import (
    GO_BACK_DESCRIPTION,
    REENTRY_REDO,
    REENTRY_RESUME,
    ToolDeclaration,
)
from unmute_core.tasks._values import SaveOutcome

GO_BACK_PARAMETER = "steps"


def _step_names(raw: object) -> list[str]:
    """Read the model's ``steps``: one name, or a list of names with anything else dropped."""
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [name for name in raw if isinstance(name, str)]
    return []


class GoBackArguments(BaseModel):
    """The arguments of a go-back call, as the model wrote them.

    Attributes:
        steps: The step names it asked for, in order. A key it did not send reads as none.
    """

    steps: Annotated[list[str], BeforeValidator(_step_names)] = []


def step_description(task: TaskSpec) -> str:
    """Name a step for the go-back tool: when it runs, and what it saves."""
    described = task.when or task.name.replace("_", " ")
    saves = list(dict.fromkeys(row.variable for row in task.assign))
    if not saves:
        return described
    if not described.endswith((".", "!", "?")):
        described += "."
    return f"{described} It saves: {', '.join(saves)}."


@original_errors
@dataclass(config=VALUE)
class GroupProgress:
    """One group run's step order. Indexes are positions in ``steps``.

    Attributes:
        group: The group.
        steps: Its tasks, in order.
        skipped: Steps left out as the group starts (``TaskPlan.skipped_steps``).
        current: The running step, or None.
        reentered: The running step was entered before in this run.
        follows_step: The running step is entered for the first time after another step,
            so the caller's last turn answered that step.
        stopped: Why the group stopped, or None.
    """

    group: TaskGroupSpec
    steps: Sequence[TaskSpec]
    skipped: frozenset[int] = frozenset()
    current: int | None = field(init=False, default=None)
    reentered: bool = field(init=False, default=False)
    follows_step: bool = field(init=False, default=False)
    stopped: Literal["unserved", "failed"] | None = field(init=False, default=None)
    _queue: deque[int] = field(init=False)
    _entered: set[int] = field(init=False, default_factory=set)
    _entry_turns: int = field(init=False, default=0)
    _redo: set[int] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        """Queue every step that is not skipped."""
        self._queue = deque(index for index in range(len(self.steps)) if index not in self.skipped)

    def next_step(self) -> int | None:
        """Return the index of the step to run now, or None when the group is done."""
        if self.stopped is not None or not self._queue:
            self.current = None
            return None
        index = self._queue.popleft()
        self.current = index
        self.reentered = index in self._entered
        self.follows_step = not self.reentered and bool(self._entered)
        if not self.reentered:
            self._entered.add(index)
        self._entry_turns = 0
        return index

    def mark_opened(self, *, caller_turns: int) -> None:
        """Note how many caller turns the running step's context holds as it opens."""
        self._entry_turns = caller_turns

    @property
    def next_task(self) -> TaskSpec | None:
        """The step queued after the running one, if any."""
        return self.steps[self._queue[0]] if self._queue else None

    @property
    def task(self) -> TaskSpec:
        """The running step.

        Raises:
            RuntimeError: If no step is running.
        """
        if self.current is None:
            raise RuntimeError("no step is running")
        return self.steps[self.current]

    def is_blocking(self, index: int) -> bool:
        """Return True when the group stops if this step fails."""
        return self.steps[index].name not in self.group.non_blocking

    def record(self, index: int, outcome: SaveOutcome) -> Literal["continue", "stop"]:
        """Record how a step ended, and say whether the group carries on."""
        self._redo.discard(index)
        if outcome.status == "unserved":
            self.stopped = "unserved"
            return "stop"
        if outcome.status == "failed" and self.is_blocking(index):
            self.stopped = "failed"
            return "stop"
        return "continue"

    def status(self) -> str:
        """Return how the group ended: completed, unserved or failed."""
        return self.stopped or "completed"

    def revisitable(self) -> list[TaskSpec]:
        """Return the steps entered in this run, other than the running or queued ones."""
        queued = set(self._queue)
        return [
            self.steps[index]
            for index in sorted(self._entered)
            if index != self.current and index not in queued
        ]

    def may_go_back(self, caller_turns: int) -> bool:
        """Return True once the caller has spoken since the running step opened."""
        return caller_turns > self._entry_turns

    def reentry_note(self) -> str | None:
        """Return why the running step runs again, or None on its first entry."""
        if not self.reentered:
            return None
        return REENTRY_REDO if self.current in self._redo else REENTRY_RESUME

    def refuses_unserved(self, caller_turns: int) -> bool:
        """Return True while a step entered again may not end unserved yet."""
        return self.reentered and caller_turns <= self._entry_turns

    def go_back(self, names: Sequence[str]) -> list[int]:
        """Queue the named earlier steps, then the interrupted one, then the rest.

        Raises:
            ValueError: For a name this run cannot go back to, or none. Nothing changes.
        """
        indexes = {task.name: index for index, task in enumerate(self.steps)}
        revisitable = [task.name for task in self.revisitable()]
        # the model reads these refusals, so they name what it can still go back to
        if (stray := next((name for name in names if name not in revisitable), None)) is not None:
            raise ValueError(
                f"cannot go back to {stray!r}. Go back to one of: "
                + (", ".join(revisitable) or "none, no step ran before this one")
            )
        targets = [indexes[name] for name in dict.fromkeys(names)]
        if not targets:
            raise ValueError("name at least one earlier step")
        resume = [] if self.current is None else [self.current]
        rest = [index for index in self._queue if index not in (*targets, *resume)]
        self._queue = deque([*targets, *resume, *rest])
        self._redo.update(targets)
        self.current = None
        return targets

    def go_back_tool(self) -> dict[str, Any] | None:
        """Return the go-back tool for the running step, or None with nothing to go back to."""
        steps = self.revisitable()
        if not steps:
            return None
        described = {task.name: step_description(task) for task in steps}
        return ToolDeclaration(
            name=GO_BACK_TOOL_NAME,
            description=GO_BACK_DESCRIPTION + json.dumps(described, ensure_ascii=False),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    GO_BACK_PARAMETER: {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "enum": list(described)},
                        "description": "The earlier steps to run again, in order.",
                    }
                },
                "required": [GO_BACK_PARAMETER],
            },
        ).to_dict()


def go_back_targets(arguments: object) -> list[str]:
    """Return the step names a go-back call asked for, or an empty list."""
    if not isinstance(arguments, dict):
        return []
    return GoBackArguments.model_validate(arguments).steps
