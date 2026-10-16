"""What a run hands the main agent, and what later steps read of it.

In ``record`` mode a run hands back a record: its status first, then each step in the
order it ran:

* ``completed``: ``saved`` holds its values as the main agent reads them now, an
  unconfirmed one as ``UNSET``. ``confirmed`` names the ones it confirmed.
* ``handed back``: ``not_done`` is what it could not serve, in the caller's words.
  ``not_saved`` names values given with it that did not fit.
* ``skipped``: its value was already confirmed, or a prerequisite was already met.
* ``failed``: it broke, or could not start without a value it needs.

``said`` is what the caller heard in the reply that finished the step. ``not_heard`` is a
tool's result the step finished on without telling the caller.
"""

import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import field
from types import MappingProxyType
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, SkipValidation
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.state import CallState
from unmute_core.tasks._progress import GroupProgress
from unmute_core.tasks._spec import TaskSpec
from unmute_core.tasks._texts import SERVED_HEADING, SERVED_RULE
from unmute_core.tasks._values import SaveOutcome

UNSET = "UNSET"

StepOutcome = Literal["completed", "handed back", "skipped", "failed"]
_OUTCOMES: Mapping[str, StepOutcome] = MappingProxyType(
    {
        "completed": "completed",
        "unserved": "handed back",
        "failed": "failed",
    }
)

TaskStage = Literal[
    "task.started",
    "task.completed",
    "task.unserved",
    "task.refused_rerun",
    "task.refused_busy",
    "task.value_refused",
    "task.ended_on_tool",
    "task.handed_off",
    "task.failed",
    "task.requires",
    "task.entered_via",
    "group.started",
    "group.step_started",
    "group.step_skipped",
    "group.went_back",
    "group.stopped",
    "group.completed",
]
TaskEventSink = Callable[[str, dict[str, Any]], None]

logger = logging.getLogger(__name__)
"""Receives each task event: its stage and its data. Names only, never values."""


def new_run_id() -> str:
    """Return a short id for one delegate run."""
    return uuid.uuid4().hex[:16]


def task_event(
    stage: TaskStage,
    *,
    task: str | None = None,
    group: str | None = None,
    run_id: str | None = None,
    step: int | None = None,
    saved: Sequence[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build one task event: names, a run id, a step index and saved names, never values.

    Both adapters report the same sequence:

    * a task: ``task.started``, any ``task.value_refused``, then ``task.completed``,
      ``task.unserved`` or ``task.failed``. A step that ends on its own tool reports
      ``task.ended_on_tool`` right before ``task.completed``.
    * a group: ``group.started``, ``group.step_skipped`` per skipped step, then per step
      ``group.step_started`` and the task events, then ``group.completed`` or
      ``group.stopped``.
    * going back: ``group.went_back`` for the interrupted step, then ``group.step_started``
      for each step run again.
    * a task waiting for values: ``task.requires``, then its provider's events.
    * a task entered through a tool: ``task.entered_via`` before ``task.started``.
    * a step that hands the call to another agent: ``task.handed_off``, which ends its
      task or group, with no ``group.completed`` or ``group.stopped``.
    * a refused re-run: only ``task.refused_rerun``.
    * a start refused while another run is active: only ``task.refused_busy``.
    """
    event = _TaskEvent(task=task, group=group, task_run_id=run_id, step=step, saved=saved)
    return stage, event.model_dump(exclude_none=True)


class _TaskEvent(BaseModel):
    """The data of one task event. A field that is None stays out of the dict the sink gets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str | None = None
    group: str | None = None
    task_run_id: str | None = None
    step: int | None = None
    saved: list[str] | None = None


@original_errors
@dataclass(frozen=True, config=VALUE)
class _Entry:
    """One step's line in the record."""

    step: str
    outcome: StepOutcome
    saved: tuple[str, ...] = ()
    not_saved: tuple[str, ...] = ()
    request: str = ""
    said: str = ""
    not_heard: str = ""


@original_errors
@dataclass(config=VALUE)
class RunRecord:
    """The steps one delegate run went through, in the order they ran.

    Attributes:
        name: The task or group.
        skipped: Group step index to name, for steps left out as it started. Each is
            listed where the group would have run it.
    """

    name: str
    skipped: Mapping[int, str] = field(default_factory=dict)
    _entries: list[_Entry] = field(init=False, default_factory=list)
    _pending: dict[int, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Hold the skipped steps until the group goes past them."""
        self._pending = dict(sorted(self.skipped.items()))

    @classmethod
    def of_task(cls, name: str, *, provider: str | None, provider_runs: bool) -> Self:
        """Start a task's record. A provider that need not run is listed as skipped."""
        record = cls(name)
        if provider is not None and not provider_runs:
            record.skip(provider)
        return record

    @classmethod
    def of_group(cls, progress: GroupProgress) -> Self:
        """Start a group's record, with the steps it skips as it starts."""
        return cls(
            progress.group.name,
            skipped={index: progress.steps[index].name for index in progress.skipped},
        )

    def skip(self, step: str) -> None:
        """List a step left out outside a group's order."""
        self._entries.append(_Entry(step=step, outcome="skipped"))

    def ended(
        self,
        step: str,
        outcome: SaveOutcome,
        *,
        index: int | None = None,
        said: str = "",
        not_heard: str = "",
    ) -> None:
        """Record one step run that ended.

        Args:
            step: The task.
            outcome: How it ended.
            index: Its place in a group, or None outside one.
            said: What the caller heard in the reply that finished it.
            not_heard: A tool's result it finished on without telling the caller.
        """
        if index is not None:
            passed = [position for position in self._pending if position < index]
            self._entries.extend(
                _Entry(step=self._pending.pop(position), outcome="skipped") for position in passed
            )
        self._entries.append(
            _Entry(
                step=step,
                outcome=_OUTCOMES[outcome.status],
                saved=outcome.saved,
                not_saved=outcome.refused,
                request=outcome.unserved_request,
                said=said,
                not_heard=not_heard,
            )
        )

    def served(self) -> list[tuple[str, tuple[str, ...]]]:
        """Return each step that did its work, with what it saved, in the order it ran.

        A hand-back that saved did its own work. Only its request is undone.
        """
        return [
            (entry.step, entry.saved)
            for entry in self._entries
            if entry.outcome == "completed" or (entry.outcome == "handed back" and entry.saved)
        ]

    def build(self, status: str, state: CallState) -> dict[str, Any]:
        """Return the record the main agent reads. Call it as the run returns."""
        entries = list(self._entries)
        if status == "completed":
            entries.extend(_Entry(step=name, outcome="skipped") for name in self._pending.values())
        result = _RunResult(
            status=status, task=self.name, steps=[_step_line(entry, state) for entry in entries]
        )
        return result.model_dump(exclude_none=True)


class _StepLine(BaseModel):
    """One step's line of the record. A field that is None stays out of the dict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: str
    outcome: StepOutcome
    saved: dict[str, str] | None = None
    confirmed: list[str] | None = None
    not_saved: list[str] | None = None
    not_done: str | None = None
    said: str | None = None
    not_heard: str | None = None


class _RunResult(BaseModel):
    """What the main agent reads of a run in ``record`` mode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    task: str
    steps: list[_StepLine]


def _step_line(entry: _Entry, state: CallState) -> _StepLine:
    """Return one step's line, its values read now."""
    confirmed = [
        name
        for name in entry.saved
        if state.confirm.get(name) == entry.step and state.is_confirmed(name)
    ]
    said = entry.said.strip()
    return _StepLine(
        step=entry.step,
        outcome=entry.outcome,
        saved={
            name: state.render(name) if state.is_confirmed(name) else UNSET for name in entry.saved
        }
        or None,
        confirmed=confirmed or None,
        not_saved=list(entry.not_saved) or None,
        not_done=entry.request if entry.outcome == "handed back" else None,
        said=said or None,
        not_heard=None if said else entry.not_heard or None,
    )


@original_errors
@dataclass(eq=False, config=VALUE)
class ServedRequests:
    """The steps that did their work in this call's earlier runs (``record`` mode).

    A step reads them after its inputs, so it does not take a request an earlier run
    already served as still open. Names only, never values.

    Attributes:
        tasks: The agent's tasks by name, for their ``when`` sentences.
    """

    tasks: SkipValidation[Mapping[str, TaskSpec]]
    _saved: dict[str, list[str]] = field(init=False, default_factory=dict)

    def record(self, record: RunRecord) -> None:
        """Add a run that has returned, whatever its status."""
        for step, saved in record.served():
            names = self._saved.setdefault(step, [])
            names.extend(name for name in saved if name not in names)

    def section(self, task: str) -> str:
        """Return the section ``task``'s prompt reads, or "" when nothing else was done."""
        lines = [
            _line(step, self._when(step), saved)
            for step, saved in self._saved.items()
            if step != task
        ]
        if not lines:
            return ""
        return "\n".join([SERVED_HEADING, *lines, SERVED_RULE])

    def _when(self, step: str) -> str:
        """Return a step's ``when`` sentence, or ""."""
        task = self.tasks.get(step)
        return (task.when or "").strip() if task is not None else ""


def _line(step: str, when: str, saved: Sequence[str]) -> str:
    """Return one served step as a line: its name, what it serves, what it saved."""
    text = f"- {step}" + (f" ({when.rstrip('.')})" if when else "")
    if not saved:
        return text + ": done."
    names = saved[0] if len(saved) == 1 else ", ".join(saved[:-1]) + " and " + saved[-1]
    return text + f": saved {names}."


class TaskEvents:
    """The event bus of one call's tasks: the sinks that hear each task event."""

    def __init__(self) -> None:
        """Start with no sink."""
        self._sinks: list[TaskEventSink] = []

    def listen(self, sink: TaskEventSink) -> None:
        """Send every later task event to ``sink`` too. Adding the same sink again does nothing.

        Args:
            sink: Receives each task event, like ``Telemetry.event``.
        """
        if sink not in self._sinks:
            self._sinks.append(sink)

    def emit(
        self,
        stage: TaskStage,
        *,
        task: str | None = None,
        group: str | None = None,
        run_id: str | None = None,
        step: int | None = None,
        saved: Sequence[str] | None = None,
    ) -> None:
        """Send one task event to every sink. A sink that fails never stops the call."""
        name, data = task_event(
            stage, task=task, group=group, run_id=run_id, step=step, saved=saved
        )
        for sink in self._sinks:
            try:
                sink(name, data)
            except Exception:
                logger.debug("task event %s could not be reported", name, exc_info=True)
