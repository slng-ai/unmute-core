"""How a step opens: the line it speaks, whether it listens, the tool it runs in code.

``StepOpening`` is the part of a ``StepRun`` that decides how the step begins. The adapter
reads what it decided through the step's ``announce``, ``listens`` and ``opening_message``.
"""

import json
import logging
from typing import TYPE_CHECKING

from pydantic.dataclasses import dataclass
from pydantic_core import to_jsonable_python

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.tasks._progress import GroupProgress
from unmute_core.tasks._record import new_run_id
from unmute_core.tasks._spec import TaskSpec
from unmute_core.tasks._texts import BEGIN_STEP, STEP_BOUNDARY, WAIT_FOR_CALLER, opening_ran
from unmute_core.tools import ToolError, ToolRegistry

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
        given = {
            parameter: to_jsonable_python(state.plain(name))
            for parameter, name in spec.arguments.items()
        }
        schema = self._tools.get(spec.tool).model_parameters
        optional = schema.get("properties", {}).keys() - set(schema.get("required", ()))
        empty = {parameter for parameter, value in given.items() if value in (None, "", [])}
        if empty - optional:
            return None
        arguments = {
            parameter: value for parameter, value in given.items() if parameter not in empty
        }
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
        else:
            line = self._tasks.spoken_line(self._task, spec.tool, spec.say, self._step.result)
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
