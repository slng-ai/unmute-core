"""An agent's tasks, checked against its State and tools, and indexed for the adapters.

These are the checks that need the State or the tools. The checks that need neither run
in ``TasksSpec``.
"""

import copy
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from functools import cached_property
from typing import Any, Literal

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.handoff import History
from unmute_core.model._prompt import check_keys
from unmute_core.state import CallState
from unmute_core.state._call_info import CALL, call_model, reserved_name_error
from unmute_core.state._paths import PLACEHOLDER, ValuePathError, parse_path, path_annotation
from unmute_core.tasks._spec import (
    FinishRule,
    TaskGroupSpec,
    TaskInput,
    TaskSpec,
    TasksSpec,
    input_fields,
)
from unmute_core.tasks._values import (
    StepResult,
    StepValues,
    appends_to_list,
    argument_parameters,
    finish_parameters,
    result_model,
    state_fields,
)
from unmute_core.tools import ToolRegistry
from unmute_core.tools._lines import result_paths
from unmute_core.tools._spec import check_saves

EntryHistory = History | Literal["caller_turn"]
RESUMED_SCOPE = "shared"
"""A task that runs after its prerequisite opens like the next step of a shared group."""


@original_errors
@dataclass(frozen=True, config=VALUE)
class Delegate:
    """One tool the main agent gains: it starts a task or a group.

    Attributes:
        kind: ``task`` or ``group``.
        name: The task or group, and the tool's name.
        when: Its ``when`` sentence.
    """

    kind: Literal["task", "group"]
    name: str
    when: str | None


def entry_history(
    task: TaskSpec, *, group_scope: str | None, reentry: bool = False
) -> EntryHistory:
    """Return what a step hears of the call as it opens. Every adapter follows this rule.

    A task applies its own ``history``, alone and in a shared group. An isolated group's
    step starts fresh, and so does a step with ``reset`` in any group. A step that would
    start fresh but is entered again after going back opens on the caller's newest turn
    alone (``caller_turn``): that turn is what they want changed.

    Args:
        task: The task.
        group_scope: The group's ``context_scope``, or None outside a group.
        reentry: The step is entered again after going back.
    """
    if group_scope == "isolated" or (group_scope is not None and task.history == "reset"):
        return "caller_turn" if reentry else "reset"
    return task.history


class TaskPlan:
    """The tasks and groups of one agent, checked once, read by every call.

    Attributes:
        spec: The checked spec.
        record: The mode is ``record``.
        tasks: Task name to task.
        groups: Group name to group.
        state: The State model.
        confirm: State value to the step that confirms it.
        tools: Every tool of the agent.
        providers: Task to the one task that saves all it requires.
        via_tasks: Tool name to the task its success enters.
        withdrawing: Steps a group may skip once their value is confirmed. Entering one
            withdraws that confirmation, because it asks the caller again.
        then_targets: Each group whose ``then`` is ``transfer``, to the agent it hands the
            call to. The adapter refuses a target it cannot hand the call to, as it is built.
    """

    def __init__(
        self,
        spec: TasksSpec,
        *,
        state: type[BaseModel],
        confirm: Mapping[str, str] | None = None,
        tools: ToolRegistry,
    ) -> None:
        """Check the spec against the State, its confirmations and the tools.

        Args:
            spec: The tasks and groups.
            state: The State model the call's ``CallState`` runs on. A task may read
                ``call``, what the call carries, but never save it.
            confirm: State value to the step that confirms it, as given to ``CallState``.
            tools: Every tool of the agent, task-only ones included.

        Raises:
            ReservedNameError: If the State declares ``call``, or a task saves it.
            ValueError: Naming the first task, group or value that does not fit.
        """
        self.spec = spec
        self.record = spec.mode == "record"
        self.tasks = {task.name: task for task in spec.tasks}
        self.groups = {group.name: group for group in spec.groups}
        self.state = call_model(state)
        self.confirm = dict(confirm or {})
        self.tools = tools
        self._check()
        self.providers = {
            task.name: task.providers(spec.tasks)[0].name for task in spec.tasks if task.requires
        }
        self.via_tasks = {tool: task.name for task in spec.tasks for tool in task.via}
        self.withdrawing = frozenset(
            step for group in spec.groups for step in group.skip_when_confirmed
        )
        self.then_targets = {
            group.name: group.then_target
            for group in spec.groups
            if group.then == "transfer" and group.then_target is not None
        }

    def _check(self) -> None:
        """Check every name a task or group reads against the State and the tools."""
        values = state_fields(self.state)
        if clash := sorted((self.tasks.keys() | self.groups.keys()) & set(self.tools.names)):
            raise ValueError(
                f"'{clash[0]}' names both a task or group and a tool. The main agent calls "
                "both by name, so rename one"
            )
        for task in self.spec.tasks:
            self._check_task(task, values)
        for group in self.spec.groups:
            self._check_group(group, values)
        self._check_confirms()
        try:
            self.tools.only(self.spec.task_only_tools)
        except ValueError as error:  # the registry words it; this names where it came from
            raise ValueError(f"task_only_tools: {error}") from None

    def _check_task(self, task: TaskSpec, values: Mapping[str, str]) -> None:
        """Check one task's tools, assignments, reads and confirm lines."""
        where = f"task '{task.name}'"
        try:
            self.tools.only(task.tools)
        except ValueError as error:  # the registry words it; this names the task
            raise ValueError(f"{where}: {error}") from None
        for row in task.assign:
            if row.variable == CALL:
                raise reserved_name_error(where)
            if row.variable not in values:  # needs the State, which the spec has not
                raise ValueError(
                    f"{where} assigns '{row.variable}', which the State does not have. Name a "
                    "value of the State"
                )
            if row.append and not appends_to_list(self.state, row):
                raise ValueError(
                    f"{where} appends to '{row.variable}': make it a list, or do not append"
                )
        self._check_reads(task, where, values)
        self._check_results(task, where)
        self._check_inputs(task, where)
        self._check_outcome_rows(task, where)
        for rule in task.finish:
            schema = self.tools.get(rule.tool).model_parameters
            if schema.get("additionalProperties") is not False and not schema.get("properties"):
                continue
            if stray := sorted(input_fields(rule.confirm) - schema.get("properties", {}).keys()):
                raise ValueError(
                    f"{where} asks about '{stray[0]}' before '{rule.tool}' runs, which is not "
                    "an input the model gives it. Ask about an input the tool takes"
                )

    def _check_inputs(self, task: TaskSpec, where: str) -> None:
        """Check that every ``assign`` row read from an input names one a closing tool takes.

        One closing tool is enough: the row is saved when that tool ends the step.

        Raises:
            ValueError: Naming the task, the value and the tools.
        """
        for row in (row for row in task.assign if row.source == "input"):
            if not task.finish:
                raise ValueError(
                    f"{where} saves '{row.variable}' from a tool's input, but no finish rule "
                    "names a tool. Add a finish rule, or read the value from a result"
                )
            if not row.path or any(self._takes(rule.tool, row.path[0]) for rule in task.finish):
                continue
            tools = ", ".join(f"'{rule.tool}'" for rule in task.finish)
            raise ValueError(
                f"{where} saves '{row.variable}' from the input '{row.path[0]}', which no "
                f"closing tool takes ({tools}). Name an input one of them takes"
            )

    def _check_outcome_rows(self, task: TaskSpec, where: str) -> None:
        """Check the rows a finish rule or a reply saves against its tool and the State.

        A rule's row reads what its tool returns or takes. A reply's row saves a result
        into a value of the State.

        Raises:
            ValueError: Naming the task, the tool and the value.
        """
        for rule in task.finish:
            tool = self.tools.get(rule.tool)
            for row in rule.assign:
                here = f"{where}: a finish rule on '{rule.tool}' saves '{row.variable}'"
                if row.source == "result":
                    tool.check_result_path(row.path, f"{here}, but it")
                elif row.path and not self._takes(rule.tool, row.path[0]):
                    raise ValueError(
                        f"{here} from the input '{row.path[0]}', which the tool does not take. "
                        "Name an input it takes"
                    )
        for reply in task.replies:
            if reply.tool is None:  # TaskSpec refuses a reply with no tool
                continue
            if any(row.variable == CALL for row in reply.assign):
                raise reserved_name_error(where)
            here = f"{where}: a reply to '{reply.tool}'"
            check_saves(self.tools.get(reply.tool), reply.assign, self.state, here)

    def _takes(self, tool: str, field: str) -> bool:
        """Return True when the tool's input has the field, or is open to any field."""
        schema = self.tools.get(tool).model_parameters
        inputs = schema.get("properties", {})
        return field in inputs or (schema.get("additionalProperties") is not False and not inputs)

    def _check_results(self, task: TaskSpec, where: str) -> None:
        """Check every result field a task's lines and rules read against its tool's output.

        A line, a ``when`` field and a success field must be in the output of the tool they
        name. An ``assign`` row is not checked: the model's finish call may fill it too.

        Raises:
            ValueError: Naming the task, the tool and the field.
        """
        reads = [
            *((rule.tool, rule.say, rule.success) for rule in task.finish),
            *(
                (reply.tool, reply.say, reply.when)
                for reply in task.replies
                if reply.tool is not None  # TaskSpec refuses a reply with no tool
            ),
            *([(task.opens_with.tool, task.opens_with.say, {})] if task.opens_with else []),
        ]
        for name, lines, fields in reads:
            tool = self.tools.get(name)
            for path in result_paths(lines):
                tool.check_result_path(path, where)
            tool.check_match(fields, where)

    def _check_reads(self, task: TaskSpec, where: str, values: Mapping[str, str]) -> None:
        """Check every State value a task reads or takes exists."""
        opening = task.opens_with
        for name in (
            *task.requires,
            *(rule_name for rule in task.finish for rule_name in rule.say_variables),
            *(row.name for row in task.inputs if row.kind == "memory"),
            *(opening.arguments.values() if opening else ()),
            *(opening.say_variables if opening else ()),
            *(name for reply in task.replies for name in reply.say_variables),
        ):
            root = name.split(".")[0]
            if root not in values:  # needs the State, which the spec has not
                raise ValueError(
                    f"{where} reads '{root}', which the State does not have. Name a value of "
                    "the State"
                )
            if root != name:
                try:
                    path_annotation(self.state, parse_path(name, values))
                except ValuePathError as error:
                    raise ValueError(f"{where}: {{{{{name}}}}}: {error}") from None
        for match in PLACEHOLDER.finditer(task.instructions):
            try:
                path = parse_path(match[1], values)
                if path.root not in values:
                    raise ValueError(
                        f"{where} reads '{path.root}', which the State does not have. Name a "
                        "value of the State"
                    )
                path_annotation(self.state, path)
            except ValuePathError as error:
                raise ValueError(f"{where}: instructions {match[0]}: {error}") from None
        for name in task.arguments:
            if name not in values:  # needs the State, which the spec has not
                raise ValueError(
                    f"{where} takes '{name}', which the State does not have. Name a value of "
                    "the State"
                )

    def _check_group(self, group: TaskGroupSpec, values: Mapping[str, str]) -> None:
        """Check one group's inputs, arguments and skipped steps."""
        for row in group.inputs:
            if row.kind == "memory" and row.name not in values:  # needs the State
                raise ValueError(
                    f"group '{group.name}' reads '{row.name}', which the State does not have. "
                    "Name a value of the State"
                )
        for name in group.arguments:
            if name not in values:  # needs the State
                raise ValueError(
                    f"group '{group.name}' takes '{name}', which the State does not have. "
                    "Name a value of the State"
                )
        for step, value in group.skip_when_confirmed.items():
            if self.confirm.get(value) != step:  # needs the confirm map, which the spec has not
                raise ValueError(
                    f"group '{group.name}' skips '{step}' when '{value}' is confirmed, but "
                    f"'{step}' does not confirm it. Name a value that step confirms"
                )

    def _check_confirms(self) -> None:
        """Check every confirmed value is saved by a task the agent runs directly."""
        for value, step in self.confirm.items():
            task = self.tasks.get(step)
            if task is None or (task.when is None and not task.via):  # needs the confirm map
                raise ValueError(
                    f"'{value}' is confirmed by '{step}', which the agent does not run directly. "
                    "Name a task with a when sentence or a via tool"
                )
            if value not in {row.variable for row in task.assign}:
                raise ValueError(
                    f"confirming task '{step}' must assign '{value}', the value it confirms"
                )

    def check_then_targets(self, known: Iterable[str]) -> None:
        """Refuse a group that transfers to an agent the adapter was not given.

        Args:
            known: The names the adapter's ``then_targets`` gives.

        Raises:
            ValueError: Naming the first group and its unknown target.
        """
        names = set(known)
        if stray := sorted(
            (group, target) for group, target in self.then_targets.items() if target not in names
        ):
            group, target = stray[0]
            raise ValueError(
                f"group '{group}' transfers the call to '{target}', which then_targets does not "
                "name. Pass the handoff to that agent in then_targets."
            )

    def check_keys(self, limit: int | None) -> None:
        """Refuse a task placeholder whose key is longer than the model call takes.

        Args:
            limit: The longest key, from ``variable_key_max``. None checks nothing.

        Raises:
            ValueError: Naming the task and the placeholder.
        """
        if limit is None:
            return
        for task in self.tasks.values():
            check_keys(task.instructions, f"task '{task.name}'", limit)

    def check_lookups(self, names: Iterable[str]) -> None:
        """Refuse a ``tool`` input that no prefetch lookup runs.

        Args:
            names: The names of the prefetch ``Lookup`` entries. Clock and call values record
                nothing, so leave them out.

        Raises:
            ValueError: Naming the first task or group and the input.
        """
        known = set(names)
        rows = [(f"task {task.name}", row) for task in self.spec.tasks for row in task.inputs] + [
            (f"group {group.name}", row) for group in self.spec.groups for row in group.inputs
        ]
        for where, row in rows:
            if row.kind == "tool" and row.name not in known:
                raise ValueError(
                    f"{where} reads the tool '{row.name}', which no prefetch Lookup runs. "
                    "Add a Lookup with that name, or fix the name."
                )

    def check_native_tools(self, names: Iterable[str], *, task: TaskSpec | None = None) -> None:
        """Refuse a task that lists a native tool the agent does not have.

        Args:
            names: The names of the agent's native tools.
            task: Check only this task. Every task when None.

        Raises:
            ValueError: Naming the first task and its missing tool.
        """
        have = set(names)
        if missing := [
            (each.name, name)
            for each in ([task] if task else self.tasks.values())
            for name in each.native_tools
            if name not in have
        ]:
            owner, name = missing[0]
            raise ValueError(
                f"task '{owner}' lists the native tool '{name}', which the agent does not "
                "have. Pass it in native_tools."
            )

    def check_handoff_tools(
        self, names: Iterable[str], *, fix: str, task: TaskSpec | None = None
    ) -> None:
        """Refuse a task that lists a handoff the agent has no tool for.

        Args:
            names: The names of the agent's handoff-capable native tools.
            fix: The end of the message, saying how to add the tool.
            task: Check only this task. Every task when None.

        Raises:
            ValueError: Naming the first task and its missing handoff.
        """
        have = set(names)
        if missing := [
            (each.name, name)
            for each in ([task] if task else self.tasks.values())
            for name in each.handoffs
            if name not in have
        ]:
            owner, name = missing[0]
            raise ValueError(
                f"task '{owner}' lists the handoff '{name}', which the agent has no tool for. {fix}"
            )

    def delegates(self) -> list[Delegate]:
        """Return the tools the main agent gains: startable tasks in order, then groups."""
        return [
            *(Delegate("task", task.name, task.when) for task in self.spec.tasks if task.when),
            *(Delegate("group", group.name, group.when) for group in self.spec.groups),
        ]

    def arguments(self, name: str) -> tuple[str, ...]:
        """Return the State values a delegate tool takes from the main agent."""
        if name in self.groups:
            return self.groups[name].arguments
        return self.tasks[name].arguments

    def delegate_for(self, name: str) -> Delegate:
        """Return the delegate of one task."""
        return Delegate("task", name, self.tasks[name].when)

    def visible_to_main(self, tool: str) -> bool:
        """Return True for a tool the main agent may call."""
        return tool not in self.spec.task_only_tools

    def skipped_steps(self, group: str, confirmed: Callable[[str], bool]) -> frozenset[int]:
        """Return the steps this group skips, decided once as it starts."""
        spec = self.groups[group]
        return frozenset(
            index
            for index, step in enumerate(spec.steps)
            if step in spec.skip_when_confirmed and confirmed(spec.skip_when_confirmed[step])
        )

    def inputs_for(self, task: TaskSpec, group: str | None = None) -> list[TaskInput]:
        """Return what a step reads: its group's inputs, then its own, each once."""
        kept: dict[tuple[str, str], TaskInput] = {}
        for row in (*(self.groups[group].inputs if group is not None else ()), *task.inputs):
            kept.setdefault((row.kind, row.name), row)
        return list(kept.values())

    def missing(self, task: TaskSpec, state: CallState) -> list[str]:
        """Return the values ``task`` requires that are not set and settled yet."""
        return [name for name in task.requires if not state.is_confirmed(name)]

    def prerequisite(self, task: TaskSpec, state: CallState) -> TaskSpec | None:
        """Return the task to run first, or None when ``task`` can run now.

        The provider requires nothing itself, so this is never more than one deep.
        """
        if not self.missing(task, state):
            return None
        return self.tasks[self.providers[task.name]]

    def steps(self, name: str) -> list[TaskSpec]:
        """Return the tasks a delegate runs, in order: one for a task, more for a group."""
        if name in self.tasks:
            return [self.tasks[name]]
        return [self.tasks[step] for step in self.groups[name].steps]

    def ends_on_tool(self, name: str) -> bool:
        """Return True when a step of this delegate can end on its own tool."""
        return any(step.finish for step in self.steps(name))

    @cached_property
    def _results(self) -> dict[str, type[StepResult]]:
        """Each task's result model, built once."""
        return {task.name: result_model(task, self.state) for task in self.spec.tasks}

    def result_model(self, task: str) -> type[StepResult]:
        """Return the finish model of one task."""
        return self._results[task]

    @cached_property
    def _values(self) -> dict[str, StepValues]:
        """Each task's value checks and saves, built once."""
        return {task.name: StepValues(task, self._results[task.name]) for task in self.spec.tasks}

    def values(self, task: str) -> StepValues:
        """Return how one task's values are checked and saved."""
        return self._values[task]

    @cached_property
    def _finish_parameters(self) -> dict[str, dict[str, Any]]:
        """Each task's finish parameters, built once."""
        return {
            task.name: finish_parameters(
                self._results[task.name], record=self.record, derived=task.derived_arguments()
            )
            for task in self.spec.tasks
        }

    def finish_parameters(self, task: str) -> dict[str, Any]:
        """Return a copy of the parameters of one task's finish tool."""
        return copy.deepcopy(self._finish_parameters[task])

    @cached_property
    def _argument_parameters(self) -> dict[str, dict[str, Any]]:
        """The parameters of each delegate tool that takes arguments, built once."""
        return {
            delegate.name: argument_parameters(names, self.state)
            for delegate in self.delegates()
            if (names := self.arguments(delegate.name))
        }

    def argument_parameters(self, name: str) -> dict[str, Any] | None:
        """Return a copy of a delegate tool's parameters, or None when it takes no arguments."""
        found = self._argument_parameters.get(name)
        return None if found is None else copy.deepcopy(found)

    def finish_rules(self, task: TaskSpec) -> dict[str, list[FinishRule]]:
        """Return each closing tool of a task with its rules."""
        rules: defaultdict[str, list[FinishRule]] = defaultdict(list)
        for rule in task.finish:
            rules[rule.tool].append(rule)
        return dict(rules)
