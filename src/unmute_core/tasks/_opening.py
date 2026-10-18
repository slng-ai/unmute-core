"""How a step opens: the line it speaks, whether it listens, the tool it runs in code.

``StepOpening`` is the part of a ``StepRun`` that decides how the step begins. The adapter
reads what it decided through the step's ``announce``, ``listens`` and ``opening_message``.
``open_agent`` runs the same opening tool for a main agent handed the call.
"""

import json
import logging
from typing import TYPE_CHECKING

from pydantic.dataclasses import dataclass
from pydantic_core import to_jsonable_python

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.state import CallState
from unmute_core.tasks._progress import GroupProgress
from unmute_core.tasks._record import new_run_id
from unmute_core.tasks._spec import OpeningTool, TaskSpec
from unmute_core.tasks._texts import BEGIN_STEP, STEP_BOUNDARY, WAIT_FOR_CALLER, opening_ran
from unmute_core.tools import SpokenLines, ToolError, ToolHooks, ToolRegistry
from unmute_core.tools._lines import state_words

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from unmute_core.tasks._run import CallTasks, StepRun

logger = logging.getLogger(__name__)


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class Opening:
    """The tool a step ran in code as it opened, and the line said from its result.

    Attributes:
        call_id: The id of the call, as it goes in the step's history and its trace.
        tool: The tool.
        arguments: Its arguments, as JSON text.
        result: What it returned.
        line: What the caller hears, or None when no line filled. Then the step opens as
            it would without an opening, and only the trace shows the call.
    """

    call_id: str
    tool: str
    arguments: str
    result: object
    line: str | None


class StepOpening:
    """Decides how one step opens, and runs its opening tool.

    Attributes:
        listens: The step opens by speaking ``announce`` and waiting, with no model request.
        announce: The line to say as it opens, or None.
        note: What its prompt says about why it opens as it does, until it has answered.
    """

    def __init__(
        self,
        step: "StepRun",
        *,
        tasks: "CallTasks",
        task: TaskSpec,
        tools: ToolRegistry,
        confirm: "Mapping[str, Sequence[str]]",
        progress: GroupProgress | None,
        record: bool,
        speak_opening: bool,
        reentry: bool,
        follows_step: bool,
        seeded: bool,
    ) -> None:
        """Build the opening of one step run. ``StepRun`` builds it.

        Args:
            step: The step, whose ``call_tool`` runs the opening tool.
            tasks: The call's tasks.
            task: The step's task.
            tools: The tools the step may call.
            confirm: The confirm lines of each closing tool that has them.
            progress: The group's progress, or None for a task run alone.
            record: Whether the plan runs in ``record`` mode.
            speak_opening: The step says its announce line as it opens.
            reentry: The step runs again after the caller went back.
            follows_step: Another step ran just before it.
            seeded: The step starts with values a task before it saved for it.
        """
        self._step = step
        self._tasks = tasks
        self._task = task
        self._tools = tools
        self._confirm = confirm
        self._record = record
        note = progress.reentry_note() if reentry and progress else None
        self.note = (note or (STEP_BOUNDARY if follows_step else "")) if record else ""
        self.listens = task.opening == "listen" and not (reentry or seeded)
        self._opens_with = None if reentry or seeded else task.opens_with
        self.announce = (
            tasks.pick_line(f"task:{task.name}", task.announce)
            if self.listens or (speak_opening and not (reentry or seeded))
            else None
        )

    @property
    def message(self) -> str | None:
        """In ``status`` mode, the developer message the step opens on. None otherwise."""
        if self._record:
            return None
        if self.listens:
            return None if self.announce else WAIT_FOR_CALLER
        return BEGIN_STEP

    def opened(self) -> None:
        """Drop the note of a step that listens: it opens with no model request."""
        if self.listens:
            self.note = ""

    async def open(self) -> Opening | None:
        """Run the step's opening tool in code. ``StepRun.open`` documents it."""
        spec, self._opens_with = self._opens_with, None
        if spec is None:
            return None
        state = self._tasks.state
        arguments = _arguments(spec, self._tools, state)
        if arguments is None:
            return None
        asks = spec.tool in self._confirm
        if (
            asks
            and spec.arguments
            and all(
                name in state.confirm and state.is_confirmed(name)
                for name in spec.arguments.values()
            )
        ):
            return None
        try:
            result = await self._step.call_tool(spec.tool, arguments)
        except ToolError as error:
            logger.info("opening %s of %s failed: %s", spec.tool, self._task.name, error.message)
            return None
        if asks:
            line = self._step.reply
            if line is None:
                return None
        elif spec.say:
            line = self._tasks.spoken_line(self._task, spec.tool, spec.say, self._step.result)
        else:
            # No lines of its own: the step's replies pick the line for this result,
            # as they would for the same call later, and as open_agent does on a handoff.
            line = self._step.reply
        text = json.dumps(arguments)
        if line is not None:
            self.listens = True
            self.announce = None
        else:
            ran = opening_ran(spec.tool, text, json.dumps(to_jsonable_python(result)))
            self.note = f"{self.note}\n\n{ran}" if self.note else ran
        return Opening(
            call_id=f"opening_{new_run_id()}",
            tool=spec.tool,
            arguments=text,
            result=result,
            line=line,
        )


def _arguments(
    spec: OpeningTool, tools: ToolRegistry, state: CallState | None
) -> dict[str, object] | None:
    """Return the opening tool's arguments from the State, or None while one it needs is unset.

    An unset value for a parameter the tool's schema declares and does not require is left
    out of the call.
    """
    given = {
        parameter: to_jsonable_python(state.plain(name) if state is not None else None)
        for parameter, name in spec.arguments.items()
    }
    schema = tools.get(spec.tool).model_parameters
    optional = schema.get("properties", {}).keys() - set(schema.get("required", ()))
    empty = {parameter for parameter, value in given.items() if value in (None, "", [])}
    if empty - optional:
        return None
    return {parameter: value for parameter, value in given.items() if parameter not in empty}


async def open_agent(
    spec: OpeningTool,
    tools: ToolRegistry,
    state: CallState | None,
    lines: SpokenLines,
    *,
    hooks: ToolHooks | None = None,
) -> Opening | None:
    """Run the tool a main agent opens with, in code, as it takes a call handed to it.

    The registry runs the tool and saves what its ``assign`` rows name. The line is the
    first of the opening's ``say`` lines that fills, else the registry's ``ToolReply`` for
    the result. When a line fills, the adapter puts the call and its result in the history,
    says the line, and asks the model nothing. When none fills, the model replies as it
    would without an opening, and the adapter only traces the call.

    Args:
        spec: The opening. Its tool is one of the agent's own.
        tools: The agent's tools.
        state: The call state, which the arguments read.
        lines: The call's spoken lines.
        hooks: The call's hooks, run around the tool.

    Returns:
        The opening, or None when the tool did not run: a value it needs is unset, or it
        failed.
    """
    arguments = _arguments(spec, tools, state)
    if arguments is None:
        return None
    try:
        result = await tools.call(spec.tool, arguments, state, hooks=hooks)
    except ToolError as error:
        logger.info("opening %s of a handed-off agent failed: %s", spec.tool, error.message)
        return None
    if not spec.say:
        line = tools.use_result(spec.tool, result, state, lines, hooks=hooks)
    elif tools.save_result(spec.tool, result, state, hooks=hooks):
        tool = tools.get(spec.tool)
        words = state_words(state, "", spec.say)
        line = lines.line(f"opening:{spec.tool}", spec.say, result, tool.result_form, words)
        if line is not None:
            tools.replied(spec.tool, hooks=hooks)
    else:
        line = None
    return Opening(
        call_id=f"opening_{new_run_id()}",
        tool=spec.tool,
        arguments=json.dumps(arguments),
        result=result,
        line=line,
    )


def check_opening(spec: OpeningTool | None, tools: ToolRegistry, *, realtime: bool) -> None:
    """Refuse a handoff's opening tool the agent taking the call cannot run.

    Args:
        spec: The opening, or None.
        tools: The tools of the agent taking the call.
        realtime: The call runs on a realtime model.

    Raises:
        ValueError: When the agent lacks the tool, or the model is a realtime one, which
            keeps the history on its server. The message names the fix.
    """
    if spec is None:
        return
    if realtime:  # the session's model: runtime state
        raise ValueError(
            "hand_off(opens_with=...) needs a text model: a realtime model keeps the "
            "history on its server. Leave opens_with out on a realtime call"
        )
    if spec.tool not in tools:  # the agent's tools: another object
        raise ValueError(
            f"hand_off opens with '{spec.tool}', which the agent taking the call does not "
            "have. Add it to that agent's tools"
        )
