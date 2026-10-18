"""The tasks and task groups an agent declares: the library's own input models.

Unmute's emitted code builds them as literals. Code that defines agents as data builds them
from JSON. They serialize to JSON and back unchanged, and
``TasksSpec.version`` names the format. The rules that need the State or the tools run in
``TaskPlan``.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    model_validator,
)

from unmute_core.handoff import History
from unmute_core.model import FINISH_TOOL_NAME, SUMMARY_SITE
from unmute_core.state._paths import PLACEHOLDER
from unmute_core.tools._announce import checked_lines
from unmute_core.tools._lines import (
    RECEIPT_PLACEHOLDER,
    Assign,
    FieldName,
    MatchValues,
    SpokenLine,
    ToolName,
    ToolReply,
    VariableName,
    read_path,
    say_variables,
    spoken_lines,
)
from unmute_core.tools._lines import say_lines as _say

GO_BACK_TOOL_NAME = "go_back_to_step"
UNSERVED_FIELD = "unserved_request"


def _task_name(value: str) -> str:
    """Refuse the names the step's own tools and the summary use.

    A task or group is a tool of the main agent. A name another tool of the agent has, like
    ``end_call``, is refused where the agent's tools are put together.
    """
    if value in (FINISH_TOOL_NAME, GO_BACK_TOOL_NAME):
        raise ValueError(f"'{value}' is reserved for the generated task tool")
    if value == SUMMARY_SITE:
        raise ValueError(
            f"'{value}' is reserved: a handoff summary uses the prompt site task:{value}"
        )
    return value


def _sentence(value: str | None) -> str | None:
    """Strip a ``when`` sentence and refuse a blank one."""
    if value is None:
        return None
    if not (value := value.strip()):
        raise ValueError("a when sentence must not be blank; omit it instead")
    return value


def _announce(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Strip each announce line, and refuse a blank one or one with a placeholder."""
    return checked_lines(lines)


def _confirm(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Check lines asked before a tool runs, filled from its input."""
    return spoken_lines(lines, "input")


def input_fields(lines: Iterable[str]) -> set[str]:
    """Return the tool inputs confirm lines read: ``{{input.date}}`` reads ``date``."""
    return {
        match[1].split(".")[1]
        for line in lines
        for match in RECEIPT_PLACEHOLDER.finditer(line)
        if match[1].startswith("input.")
    }


def _first_repeat[T](values: Iterable[T]) -> T | None:
    """Return the first value that appears more than once, or None."""
    return next((value for value, count in Counter(values).items() if count > 1), None)


def _distinct[T](
    template: str, key: Callable[[T], object] | None = None, *, kind: str = "task"
) -> AfterValidator:
    """Return a validator that refuses an entry listed twice.

    Args:
        template: The error, with ``{owner}`` (like ``task 'book'``) and ``{value}``.
        key: What makes two entries the same. The entry itself when None.
        kind: ``task`` or ``group``: what the owner is called, by its ``name``.
    """

    def check(entries: Sequence[T], info: ValidationInfo) -> Sequence[T]:
        """Raise ``template`` for the first entry that repeats."""
        keys = entries if key is None else [key(entry) for entry in entries]
        if (repeated := _first_repeat(keys)) is not None:
            owner = f"{kind} '{info.data.get('name', '?')}'"
            raise ValueError(template.format(owner=owner, value=repeated))
        return entries

    return AfterValidator(check)


def _distinct_success(success: dict[str, tuple[str | int | bool, ...]]) -> dict[str, tuple]:
    """Refuse a success value listed twice for one field."""
    for field, values in success.items():
        if _first_repeat(values) is not None:
            raise ValueError(f"success field '{field}' lists a value twice")
    return success


def _own_tools(tools: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
    """Refuse a tool named like the step's own finish or go-back tool."""
    if reserved := next((tool for tool in tools if tool in STEP_TOOL_NAMES), None):
        raise ValueError(
            f"task '{info.data.get('name', '?')}' lists a tool named '{reserved}', the name of "
            "the step's own tool. Rename the tool"
        )
    return tools


def _not_unserved(rows: tuple["Assign", ...], info: ValidationInfo) -> tuple["Assign", ...]:
    """Refuse a value saved through the finish parameter a hand-back keeps."""
    if any(row.argument == UNSERVED_FIELD for row in rows):
        raise ValueError(
            f"task '{info.data.get('name', '?')}' saves through a finish parameter called "
            f"'{UNSERVED_FIELD}', which the finish tool keeps for a hand-back"
        )
    return rows


def _bad_placeholder(text: str) -> bool:
    """Return True when text has a ``{{ ... }}`` that is no placeholder, or a ``{{`` never closed.

    A lone ``}}`` is plain text, so JSON like ``{"a": {"b": 1}}`` is allowed.
    """
    spans = list(re.finditer(r"\{\{.*?\}\}", text, re.DOTALL))
    if any(not PLACEHOLDER.fullmatch(span[0]) for span in spans):
        return True
    return "{{" in re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)


def _instructions(text: str, info: ValidationInfo) -> str:
    """Refuse blank instructions and placeholders the prompt cannot fill."""
    name = info.data.get("name", "?")
    if not text.strip():
        raise ValueError(f"task '{name}': instructions must not be blank")
    if _bad_placeholder(text):
        raise ValueError(
            f"task '{name}': instructions take only {{{{name}}}} placeholders, "
            "and a path into a value like {{customer.status}} or {{customer__status}}"
        )
    return text


STEP_TOOL_NAMES = frozenset({FINISH_TOOL_NAME, GO_BACK_TOOL_NAME})
"""The names of the two tools every step has of its own."""

TaskName = Annotated[
    str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$"), AfterValidator(_task_name)
]
"""A task or group name: lower snake case, at most 64 characters. Also the prompt site."""

AgentName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
"""The name of a main agent a group hands the call to, as the adapter's ``then_targets`` keys it."""

When = Annotated[str | None, Field(max_length=1000), AfterValidator(_sentence)]
"""When the main agent starts it, said to the model. None for a step only a group runs."""

Announce = Annotated[tuple[Annotated[str, Field(max_length=500)], ...], AfterValidator(_announce)]
"""Lines spoken as it starts. Several are alternatives: one is picked each time."""


class _Spec(BaseModel):
    """Immutable, and refuses keys it does not know."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class FinishRule(_Spec):
    """A step ends by itself when this tool returns a result the rule calls a success.

    With ``confirm``, the tool does not run on the model's first call. Code asks the caller
    a confirm line filled from that call's input, and the tool runs when the model calls it
    again with the same input, once the caller heard the line. A line cut off after it
    said every input it holds counts as heard, where the adapter knows the words that
    played (LiveKit). A line cut off sooner is asked again.

    An input is said by its JSON Schema ``format``, in the call's language. A ``date`` (ISO
    ``YYYY-MM-DD``, the one date format tools take) is named from today where the business
    is: "today, October 8", "tomorrow, October 9", "Friday, October 16". So a line puts a
    date after "for" or "to", never after "on". An ``enum`` value is said as words:
    "hair-color" is "hair color".

    The rules of one tool are tried in order, and the first that matches decides: its
    ``say`` line is spoken, its ``assign`` rows are saved, and the step ends as ``ends``
    says. A rule with no ``assign`` rows of its own that ends ``completed`` saves every row
    of the task's ``assign``, as packages written before rules had rows expect.

    Attributes:
        tool: The tool, one of the step's own.
        success: Result field to the values that count as success. Every field must hold
            one of its values. Empty only with ``ends="failed"``: it matches every result,
            like an author's ``else``.
        say: Lines spoken from the result, with ``{{result.<field>}}`` and
            ``{{variable}}`` placeholders. One is picked each time.
        confirm: Lines asked before the tool runs, with ``{{input.<field>}}`` and
            ``{{variable}}`` placeholders. One is picked each time. They belong to the
            tool: every rule of the tool gives the same ones.
        assign: The values a matching result saves, each one the task's ``assign``
            declares. Read from the result, or from the input the tool ran with.
        ends: How the step ends: ``completed``, or ``failed``, which stops a group unless
            the step is ``non_blocking``.
    """

    tool: ToolName
    success: Annotated[
        dict[FieldName, MatchValues], Field(max_length=16), AfterValidator(_distinct_success)
    ]
    say: Annotated[tuple[SpokenLine, ...], AfterValidator(_say)] = ()
    confirm: Annotated[tuple[SpokenLine, ...], AfterValidator(_confirm)] = ()
    assign: tuple[Assign, ...] = ()
    ends: Literal["completed", "failed"] = "completed"

    @model_validator(mode="after")
    def _matches_some_results(self) -> Self:
        """Refuse a rule that completes the step on every result."""
        if not self.success and self.ends == "completed":
            raise ValueError(
                f"a finish rule on '{self.tool}' has no success fields, so every result would "
                "complete the step. Name the fields that count as success, or set ends: failed"
            )
        return self

    @property
    def saves(self) -> tuple[Assign, ...] | None:
        """The rows a match saves, or None when it saves every row of the task's ``assign``."""
        return None if not self.assign and self.ends == "completed" else self.assign

    @property
    def say_variables(self) -> set[str]:
        """The State values the spoken and confirm lines read."""
        return say_variables((*self.say, *self.confirm))


class OpeningTool(_Spec):
    """A tool a step runs in code as it opens, before any model request.

    When a ``say`` line fills from the result, the line is spoken, the call and its result
    join the step's history, and the step waits for the caller: no model request at all.
    Otherwise the step opens as it would without one. Use a tool that only reads, because
    the model may then call it again.

    A tool the step ends on may open it too, when it has ``confirm`` lines. Nothing runs:
    code asks its confirm line about the State values and waits for the caller's yes. When
    every value is one a step confirms and is already confirmed, the opening is skipped:
    a step that runs again on a confirmed value is a correction.

    Attributes:
        tool: One of the step's own tools. One it ends on only with ``confirm`` lines.
        arguments: Tool parameter to the State value it takes. The tool runs only when
            every value is set, except one for a parameter the tool's schema declares and
            does not require: that one is left out of the call.
        say: Lines spoken from the result, with ``{{result.<field>}}`` and
            ``{{variable}}`` placeholders. The first one that fills is spoken. None for a
            tool with ``confirm`` lines.
    """

    tool: ToolName
    arguments: Annotated[dict[FieldName, VariableName], Field(max_length=16)] = Field(
        default_factory=dict
    )
    say: Annotated[tuple[SpokenLine, ...], AfterValidator(_say)] = ()

    @property
    def say_variables(self) -> set[str]:
        """The State values the spoken lines read."""
        return say_variables(self.say)


def _repeated_argument(kind: str) -> AfterValidator:
    """Return the validator that refuses a State value a delegate takes twice."""
    return _distinct("{owner} takes the argument '{value}' twice", kind=kind)


TaskArguments = Annotated[
    tuple[VariableName, ...], Field(max_length=16), _repeated_argument("task")
]
"""State values the main agent passes when it starts a task, saved before its first step opens."""

GroupArguments = Annotated[
    tuple[VariableName, ...], Field(max_length=16), _repeated_argument("group")
]
"""State values the main agent passes when it starts a group, saved before its first step opens."""

_VALUE_NAME = TypeAdapter(VariableName)


class TaskInput(_Spec):
    """Something a step reads at the end of its prompt.

    Attributes:
        kind: ``memory`` reads a State value. ``input`` is a fixed text, in ``value``.
            ``tool`` is what a tool that runs at call start returned, recorded in
            ``CallStartResults`` under ``name``.
        name: The State value, the input's name, or the call-start tool's label.
        value: The text of an ``input``. Only that kind carries one.
    """

    kind: Literal["tool", "memory", "input"]
    name: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    value: Annotated[str | None, Field(max_length=4000)] = None

    @model_validator(mode="after")
    def _value_for_input_only(self) -> Self:
        """Refuse a value on any kind but ``input``, and an ``input`` without one."""
        if (self.kind == "input") != (self.value is not None):
            raise ValueError(
                "only an input of kind input carries a value. Give an input its value, and "
                "leave it out for memory and tool inputs"
            )
        if self.kind == "memory":
            try:
                _VALUE_NAME.validate_python(self.name)
            except ValidationError:
                raise ValueError(
                    f"memory input '{self.name}' is not a value name. Name a State value: "
                    "letters, digits and underscores, not starting with a digit"
                ) from None
        return self


def _inputs(kind: str) -> AfterValidator:
    """Return the validator that refuses an input a task or group lists twice."""
    return _distinct(
        "{owner} takes an input twice: {value}. List each input once",
        lambda row: (row.kind, row.name),
        kind=kind,
    )


TaskInputs = Annotated[tuple[TaskInput, ...], _inputs("task")]
GroupInputs = Annotated[tuple[TaskInput, ...], _inputs("group")]


class TaskSpec(_Spec):
    """One task: a prompt, its tools, and the values it saves when it finishes.

    Attributes:
        name: Lower snake case. The prompt site is ``task:<name>``.
        instructions: The step's prompt. ``{{name}}`` reads a State value, and
            ``{{customer.status}}`` or ``{{customer__status}}`` a field inside one.
        when: When the main agent starts it. None for a step only a group or a ``via``
            tool starts.
        tools: Names of the tools the step may call.
        assign: The values it saves, from its finish call or a closing tool's result.
        finish: Tools whose result ends the step by itself.
        announce: Lines spoken as it starts.
        opening: ``generate`` opens with a model reply. ``listen`` speaks the announce
            line and waits for the caller.
        history: What the step hears of the call so far: ``messages``, ``full``,
            ``last_n``, ``summary`` or ``reset`` (``unmute_core.handoff.History``).
        max_messages: How many items ``last_n`` keeps.
        include_tool_calls: ``full`` and ``last_n`` keep tool calls and their results.
        handoffs: Handoff tools of the agent the step offers too. Calling one ends the
            step, and its run, and gives the call to another agent.
        native_tools: Framework tools of the agent the step offers too, as they are, like
            an MCP toolset. A LiveKit toolset is named by its id.
        requires: Values that must be set before it runs. One other task saves them all,
            and runs first when they are not.
        via: Tools of its own that the main agent may call too. A success moves the
            caller into this task.
        inputs: What it reads at the end of its prompt.
        arguments: State values the main agent passes when it starts the task. Each
            call saves all of them, so a value it leaves out is cleared.
        opens_with: A tool it runs in code as it opens.
        replies: Lines code speaks for a tool result, in place of a model reply.
        model: The model its step runs on, by a name the adapter maps to a model, like
            ``fast``. None runs it on the agent's model. Pipecat refuses it.
    """

    name: TaskName
    instructions: Annotated[
        str,
        StringConstraints(min_length=1),
        AfterValidator(_instructions),
    ]
    when: When = None
    tools: Annotated[
        tuple[ToolName, ...],
        _distinct("{owner} lists the tool '{value}' twice"),
        AfterValidator(_own_tools),
    ] = ()
    assign: Annotated[
        tuple[Assign, ...],
        _distinct("{owner} assigns '{value}' twice", lambda row: row.variable),
        _distinct(
            "{owner} saves two values through the finish parameter '{value}'. Give one of "
            "them a longer path",
            lambda row: row.argument,
        ),
        AfterValidator(_not_unserved),
    ] = ()
    finish: tuple[FinishRule, ...] = ()
    announce: Announce = ()
    opening: Literal["generate", "listen"] = "generate"
    history: History = "messages"
    max_messages: Annotated[int | None, Field(ge=1, le=200)] = None
    include_tool_calls: bool = True
    handoffs: Annotated[
        tuple[ToolName, ...],
        Field(max_length=16),
        _distinct("{owner} lists the handoff '{value}' twice"),
    ] = ()
    native_tools: Annotated[
        tuple[ToolName, ...],
        Field(max_length=16),
        _distinct("{owner} lists the native tool '{value}' twice"),
    ] = ()
    requires: Annotated[
        tuple[VariableName, ...],
        _distinct("{owner} requires '{value}' twice"),
    ] = ()
    via: Annotated[
        tuple[ToolName, ...],
        Field(max_length=16),
        _distinct("{owner} lists the via tool '{value}' twice"),
    ] = ()
    inputs: TaskInputs = ()
    arguments: TaskArguments = ()
    opens_with: OpeningTool | None = None
    replies: Annotated[tuple[ToolReply, ...], Field(max_length=16)] = ()
    model: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Check the rules between the fields of one task."""
        name = f"task '{self.name}'"
        if stray := [rule.tool for rule in self.finish if rule.tool not in self.tools]:
            raise ValueError(f"{name} ends on '{stray[0]}', which it does not list in tools")
        if saved := set(self.requires) & {row.variable for row in self.assign}:
            raise ValueError(f"{name} cannot require '{min(saved)}', a value it saves itself")
        if clash := set(self.handoffs) & {*self.tools, *STEP_TOOL_NAMES}:
            raise ValueError(
                f"{name} lists '{min(clash)}' as a handoff and as one of its own "
                "tools. A handoff is the agent's tool. Rename one of them"
            )
        own = {*self.tools, *self.handoffs, *STEP_TOOL_NAMES}
        if clash := set(self.native_tools) & own:
            raise ValueError(
                f"{name} lists '{min(clash)}' as a native tool and as one of its "
                "own tools or handoffs. Rename one of them"
            )
        if stray := [tool for tool in self.via if tool not in self.tools]:
            raise ValueError(f"{name} is entered via '{stray[0]}', which it does not list in tools")
        if (self.history == "last_n") != (self.max_messages is not None):
            raise ValueError(f"{name}: max_messages is required with, and only valid for, last_n")
        if self.arguments and self.when is None:
            raise ValueError(
                f"{name} takes arguments but has no when sentence, so the main "
                "agent never calls it. Put the arguments on the group that runs it"
            )
        if self.opens_with is not None:
            self._check_opening(self.opens_with)
        if any(reply.tool is None for reply in self.replies):
            raise ValueError(
                f"{name} has a reply with no tool. Name it, or put the reply on the tool's own "
                "replies"
            )
        if stray := [reply.tool for reply in self.replies if reply.tool not in self.tools]:
            raise ValueError(f"{name} replies to '{stray[0]}', which it does not list in tools")
        asked: dict[str, tuple[str, ...]] = {}
        for rule in self.finish:
            if asked.setdefault(rule.tool, rule.confirm) != rule.confirm:
                raise ValueError(
                    f"{name} gives '{rule.tool}' two different confirm lines. "
                    "Give every rule of that tool the same ones"
                )
        declared = {row.variable: row for row in self.assign}
        for rule in self.finish:
            for row in rule.assign:
                if row.variable not in declared:
                    raise ValueError(
                        f"{name}: a finish rule on '{rule.tool}' saves '{row.variable}', which "
                        "the task's assign does not declare. Add it to the task's assign"
                    )
                if row.append != declared[row.variable].append:
                    raise ValueError(
                        f"{name}: a finish rule on '{rule.tool}' saves '{row.variable}' with "
                        "append set differently from the task's assign. Make them the same"
                    )
        return self

    def _check_opening(self, opening: OpeningTool) -> None:
        """Check the tool the step opens with against its tools and finish rules."""
        name = f"task '{self.name}'"
        if opening.tool not in self.tools:
            raise ValueError(f"{name} opens with '{opening.tool}', which it does not list in tools")
        ends_on = {rule.tool: rule.confirm for rule in self.finish}
        if opening.tool in ends_on and not ends_on[opening.tool]:
            raise ValueError(
                f"{name} opens with '{opening.tool}', a tool it ends on. "
                "Open with a tool that only reads, or give it confirm lines"
            )
        if ends_on.get(opening.tool) and opening.say:
            raise ValueError(
                f"{name} opens by asking to confirm '{opening.tool}', so its "
                "confirm line is what is said. Remove the opening's say lines"
            )
        if opening.tool not in ends_on and not opening.say:
            raise ValueError(
                f"{name} opens with '{opening.tool}' but says nothing. Give the opening a say line"
            )

    def providers(self, tasks: Iterable["TaskSpec"]) -> list["TaskSpec"]:
        """Return the other tasks that each save every value this one requires."""
        needed = set(self.requires)
        return [
            other
            for other in tasks
            if other.name != self.name and needed <= {row.variable for row in other.assign}
        ]

    def derived_from(self, row: Assign) -> str | None:
        """Return the finish parameter a nested value is read from, or None.

        ``caller_email: result.contact.email`` beside ``contact: result.contact`` reads the
        address out of ``contact``. The finish offers no second parameter for it: a model
        that sends ``contact`` and leaves the copy out would save no address, and one that
        sends both could make them disagree.
        """
        if len(row.path) < 2:
            return None
        parent = row.path[0]
        return parent if any(other.path == (parent,) for other in self.assign) else None

    def derived_arguments(self) -> set[str]:
        """Return the finish parameters read from a parent, so the model is not offered them."""
        return {row.argument for row in self.assign if self.derived_from(row) is not None}

    def values_from_result(
        self,
        result: Mapping[str, Any],
        given: Mapping[str, Any] | None = None,
        rows: Sequence[Assign] | None = None,
    ) -> dict[str, Any]:
        """Return the finish arguments a closing tool's call carries, by their paths.

        Args:
            result: What the tool returned.
            given: The input the model gave the tool. A row that reads an input this
                call did not carry is left out, like the day of a booking when the step
                ended on a different tool.
            rows: The rows to read, like a finish rule's own, each keyed by the finish
                parameter of the task's row for the same value. Every row of the task when
                None.
        """
        arguments = {row.variable: row.argument for row in self.assign}
        return {
            arguments[row.variable]: read_path(
                result if row.source == "result" else given, row.path
            )
            for row in (self.assign if rows is None else rows)
            if row.source == "result"
            or (given is not None and (not row.path or row.path[0] in given))
        }


class TaskGroupSpec(_Spec):
    """Tasks the main agent runs as one flow, step after step.

    Attributes:
        name: Lower snake case.
        when: When the main agent starts it.
        steps: The task names, in order. Each runs at most once in the order.
        context_scope: ``shared``: each step hears the steps before it. ``isolated``:
            each step starts fresh.
        announce: Lines spoken as it starts.
        non_blocking: Steps whose failure the group records and carries on past.
        skip_when_confirmed: Step to the value whose confirmation lets the group skip it.
        inputs: What every step reads at the end of its prompt, before its own inputs.
        arguments: State values the main agent passes when it starts the group. Each
            call saves all of them, so a value it leaves out is cleared.
        then: What follows a group that completes: ``return`` to the main agent, ``end``
            the call once the last step's words have played, or ``transfer`` the call to
            ``then_target``. A group that hands the caller back, or fails, always returns.
        then_target: The agent a ``transfer`` hands the call to, by the name the adapter's
            ``then_targets`` gives it. Only ``transfer`` takes one.
    """

    name: TaskName
    when: When = None
    steps: Annotated[
        tuple[TaskName, ...],
        Field(min_length=1),
        _distinct("{owner} runs '{value}' twice. A group runs each task once", kind="group"),
    ]
    context_scope: Literal["shared", "isolated"]
    then: Literal["return", "end", "transfer"] = "return"
    then_target: AgentName | None = None
    announce: Announce = ()
    non_blocking: Annotated[
        tuple[TaskName, ...],
        _distinct("{owner} lists the non-blocking step '{value}' twice", kind="group"),
    ] = ()
    skip_when_confirmed: dict[TaskName, VariableName] = Field(default_factory=dict)
    inputs: GroupInputs = ()
    arguments: GroupArguments = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Check the rules between the fields of one group."""
        for label, names in (
            ("non-blocking", self.non_blocking),
            ("skipped", self.skip_when_confirmed),
        ):
            if stray := [name for name in names if name not in self.steps]:
                raise ValueError(
                    f"group '{self.name}': the {label} step '{stray[0]}' is not one of its steps"
                )
        if self.then == "transfer" and self.then_target is None:
            raise ValueError(
                f"group '{self.name}': then_target is required with transfer. Name the agent "
                "that takes the call, or pick then: return or end"
            )
        if self.then != "transfer" and self.then_target is not None:
            raise ValueError(
                f"group '{self.name}': then_target is legal with transfer only. Remove it, or "
                "set then: transfer"
            )
        return self


class TasksSpec(_Spec):
    """Every task and task group of one agent, in a versioned format.

    Attributes:
        version: The format version.
        mode: ``status``: a run hands the main agent a status, and a hand-back saves
            nothing, as Unmute's emitted agents do. ``record``: a run hands back a
            record of each step, a hand-back keeps what the caller confirmed, steps can
            go back, and a step's finish is checked against what the caller said.
        tasks: The tasks.
        groups: The task groups.
        task_only_tools: Tools only tasks may call. The main agent does not get them.
    """

    version: Literal[1] = 1
    mode: Literal["status", "record"] = "status"
    tasks: Annotated[
        tuple[TaskSpec, ...],
        _distinct("two tasks are named '{value}'", lambda task: task.name),
    ] = ()
    groups: Annotated[
        tuple[TaskGroupSpec, ...],
        _distinct("two groups are named '{value}'", lambda group: group.name),
    ] = ()
    task_only_tools: Annotated[
        tuple[ToolName, ...], _distinct("task_only_tools lists '{value}' twice")
    ] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Check the rules between tasks and groups."""
        tasks = {task.name: task for task in self.tasks}
        if both := sorted(tasks.keys() & {group.name for group in self.groups}):
            raise ValueError(f"'{both[0]}' names both a task and a group. Rename one")
        for group in self.groups:
            if stray := [step for step in group.steps if step not in tasks]:
                raise ValueError(
                    f"group '{group.name}' names task '{stray[0]}', which does not exist"
                )
        grouped = {step for group in self.groups for step in group.steps}
        for task in self.tasks:
            if task.when is None and task.name not in grouped and not task.via:
                raise ValueError(
                    f"task '{task.name}' has no when sentence, no group runs it and no via "
                    "tool enters it, so nothing can start it. Give it a when sentence, list "
                    "it in a group, or add a via tool"
                )
        if (tool := _first_repeat(tool for task in self.tasks for tool in task.via)) is not None:
            raise ValueError(f"the tool '{tool}' enters two tasks. A tool enters one task only")
        for task in self.tasks:
            if task.requires:
                self._check_provider(task, tasks)
        for group in self.groups:
            for index, step in enumerate(group.steps):
                saved = {row.variable for name in group.steps[:index] for row in tasks[name].assign}
                if missing := sorted(set(tasks[step].requires) - saved):
                    raise ValueError(
                        f"group '{group.name}' runs '{step}' before any step saves "
                        f"'{missing[0]}', which it requires. Put the step that saves it first, "
                        "or drop the requirement"
                    )
        listed = {tool for task in self.tasks for tool in task.tools}
        if stray := [tool for tool in self.task_only_tools if tool not in listed]:
            raise ValueError(
                f"the task-only tool '{stray[0]}' is listed by no task. List it in a task's "
                "tools, or remove it from task_only_tools"
            )
        return self

    def owners(self) -> dict[str, str]:
        """Return each State value a task saves, to the first task that saves it.

        A save tool for the main agent leaves these to their task (``values_tool``).
        """
        return dict(
            reversed([(row.variable, task.name) for task in self.tasks for row in task.assign])
        )

    def starters(self) -> dict[str, str]:
        """Return each State value a task saves, to what the main agent starts to save it.

        A task with a ``when`` names itself. A step names the first group with a ``when``
        that runs it. A task the main agent cannot start, like one only a ``via`` tool runs,
        is left out. The first task that saves a value wins, as in ``owners``.
        """
        groups = [group for group in self.groups if group.when is not None]
        starters = (
            (
                task,
                task.name
                if task.when is not None
                else next((group.name for group in groups if task.name in group.steps), None),
            )
            for task in self.tasks
        )
        return dict(
            reversed(
                [
                    (row.variable, starter)
                    for task, starter in starters
                    if starter is not None
                    for row in task.assign
                ]
            )
        )

    @staticmethod
    def _check_provider(task: TaskSpec, tasks: dict[str, TaskSpec]) -> None:
        """Require exactly one other task to save all ``task`` requires, and need nothing."""
        providers = task.providers(tasks.values())
        if len(providers) != 1:
            found = ", ".join(other.name for other in providers) or "none"
            raise ValueError(
                f"task '{task.name}' requires {', '.join(task.requires)}: exactly one other "
                f"task must save all of them, found: {found}"
            )
        if providers[0].requires:
            raise ValueError(
                f"task '{providers[0].name}' provides what '{task.name}' requires, so it "
                "cannot require anything itself"
            )
