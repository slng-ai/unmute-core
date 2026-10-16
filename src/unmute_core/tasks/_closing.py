"""How a step runs its own tools, and how a closing tool ends it.

``ClosingTools`` is the part of a ``StepRun`` that runs a tool call. A tool with a ``finish``
rule is a closing tool: it may ask the caller to confirm its input first, it runs one at a
time, and its success saves the step's values and ends the step.
"""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import SkipValidation
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.tasks._spec import FinishRule, TaskSpec, input_fields
from unmute_core.tasks._texts import (
    TERMINAL_SAVE_REFUSED,
    already_succeeded,
    confirm_asked,
    confirm_unfilled,
    refused,
    still_running,
)
from unmute_core.tasks._values import TaskValueRefused, success_matches
from unmute_core.tools import ToolRegistry
from unmute_core.tools._spec import ToolStarted, tool_succeeded

if TYPE_CHECKING:
    from collections.abc import Callable

    from unmute_core.tasks._run import CallTasks, StepRun
    from unmute_core.tasks._values import StepValues

logger = logging.getLogger(__name__)


@original_errors
@dataclass(slots=True, config=VALUE)
class _Asked:
    """A confirm line code asked for a closing tool's input, and whether the caller heard it.

    ``locked`` holds the locked values the input was asked with. A yes runs the tool only
    while they are the same, so a change the model never sends is asked again too.
    """

    tool: str
    given: SkipValidation[dict[str, Any]]
    locked: SkipValidation[dict[str, Any]]
    line: str
    said: tuple[str, ...] = ()
    heard: bool = False


class ClosingTools:
    """Runs the tools of one step, and ends the step on a closing tool's success.

    Attributes:
        reply: The line code speaks for the tool call that just ran, or None: a ``replies``
            line for its result, or a ``confirm`` line asked before it runs.
        result: The whole result of the tool call that just ran, for code.
        ends_call: The tool that just ran succeeded and its effect ends the call.
        receipt: The spoken line the step ended on, from a closing tool's result.
        tool_returned: One of the step's own tools has returned.
        tool_succeeded: One of the step's own tools has succeeded.
        running: The closing tool that runs now, or None.
        settled: Set while no closing tool runs.
    """

    def __init__(
        self,
        step: "StepRun",
        *,
        tasks: "CallTasks",
        task: TaskSpec,
        tools: ToolRegistry,
        rules: Mapping[str, Sequence[FinishRule]],
        confirm: Mapping[str, Sequence[str]],
        values: "StepValues",
        record: bool,
        event: "Callable[..., None]",
    ) -> None:
        """Build the closing tools of one step run. ``StepRun`` builds them.

        Args:
            step: The step. A closing tool that succeeds sets its outcome and
                ``carries_turn``.
            tasks: The call's tasks.
            task: The step's task.
            tools: The tools the step may call.
            rules: The ``finish`` rules of each closing tool.
            confirm: The confirm lines of each closing tool that has them.
            values: How the task's values are saved.
            record: Whether the plan runs in ``record`` mode.
            event: Sends one task event of this step.
        """
        self._step = step
        self._tasks = tasks
        self._task = task
        self._tools = tools
        self._rules = rules
        self._confirm = confirm
        self._values = values
        self._record = record
        self._event = event
        self._terminal: tuple[str, Mapping[str, Any]] | None = None
        self._asked: _Asked | None = None
        self.running: str | None = None
        self.settled = asyncio.Event()
        self.reply: str | None = None
        self.result: object = None
        self.ends_call = False
        self.receipt: str | None = None
        self.tool_returned = False
        self.tool_succeeded = False

    @property
    def closing_result(self) -> Mapping[str, Any] | None:
        """What the closing tool that ended the step returned, or None."""
        return self._terminal[1] if self._terminal else None

    @property
    def busy(self) -> bool:
        """True while a closing tool runs, or once one has ended the step."""
        return self._terminal is not None or self.running is not None

    async def call(
        self,
        name: str,
        arguments: str | Mapping[str, Any],
        *,
        started: ToolStarted | None = None,
    ) -> object:
        """Run one of the step's own tools. ``StepRun.call_tool`` documents it.

        Raises:
            ToolError: From the tool.
        """
        rules = self._rules.get(name)
        self.reply = None
        self.ends_call = False
        if not rules:
            result = await self._invoke(name, arguments, started)
            self._tools.save_result(name, result, self._tasks.state, hooks=self._tasks.hooks)
            self._returned(result)
            self.reply = self._reply_to(name, result)
            return self._tools.for_model(name, result)
        if self._terminal is not None:
            logger.info("closing tool refused: %s cannot run again in this step", name)
            return already_succeeded(name)
        if self._step.done:
            return None
        if lines := self._confirm.get(name):
            if (outcome := self._tools.precheck(name, arguments, self._tasks.state)) is not None:
                return self._prechecked(name, outcome)
            if not self._confirmed(name, arguments, lines):
                return self._ask(name, lines)
        if self._record and self.running is not None:
            logger.info("closing tool refused: %s is still running", self.running)
            return still_running(self.running, name)
        self.running = name
        self.settled.clear()
        try:
            result = await self._invoke(name, arguments, started)
            ended = self._end_on(name, rules, result)
            if ended is result:
                return self._tools.for_model(name, result)
            if (
                isinstance(ended, dict)
                and "refused" in ended
                and not self._tools.get(name).show_result
            ):
                return refused(ended["refused"])
            return ended
        finally:
            self.running = None
            self.settled.set()

    async def _invoke(
        self, name: str, arguments: str | Mapping[str, Any], started: ToolStarted | None
    ) -> object:
        """Run the tool through the registry, and keep its whole result for code."""
        result = await self._tools.call(
            name,
            arguments,
            self._tasks.state,
            started=started,
            hooks=self._tasks.hooks,
        )
        self.result = result
        self.ends_call = self._tools.ends_call(name, result)
        return result

    def _prechecked(self, name: str, outcome: dict[str, Any]) -> object:
        """Answer a closing tool's precheck outcome before its input is read back.

        Nothing is asked and the tool does not run. The replies speak the outcome.
        """
        logger.info("%s: the precheck answered before the read-back", name)
        self._asked = None
        self.result = outcome
        self._returned(outcome)
        self.reply = self._reply_to(name, outcome)
        return self._tools.for_model(name, outcome)

    def _returned(self, result: object) -> None:
        """Note that one of the step's own tools returned, and whether it succeeded."""
        self.tool_returned = True
        self.tool_succeeded = self.tool_succeeded or tool_succeeded(result)

    def _reply_to(self, name: str, result: object) -> str | None:
        """Return the line code speaks for this result of a tool, or None.

        The step's ``replies`` are tried first, then the tool's own.
        """
        rule = next(
            (
                reply
                for reply in (*self._task.replies, *self._tools.get(name).replies)
                if reply.tool == name and reply.matches(result, self._tasks.state)
            ),
            None,
        )
        if rule is None:
            return None
        line = self._tasks.spoken_line(self._task, name, rule.say, result)
        if line is not None:
            self._tools.replied(name, hooks=self._tasks.hooks)
        return line

    def reply_played(self, line: str, *, heard: bool, played: str = "") -> None:
        """Note how a ``reply`` line played. ``StepRun.reply_played`` documents it."""
        asked = self._asked
        if asked is not None and asked.line == line:
            played = " ".join(played.split())
            asked.heard = heard or (bool(asked.said) and all(v in played for v in asked.said))

    def _confirmed(
        self, name: str, arguments: str | Mapping[str, Any], lines: Sequence[str]
    ) -> bool:
        """Return True when the caller heard this exact input asked to its end.

        Otherwise note the input to ask about. A cut-off line is asked again, never handed
        to the model to retry.

        Raises:
            ToolError: When the input is refused, so nothing wrong is ever asked.
        """
        given = self._tools.check(name, arguments, self._tasks.state)
        locked = self._locked(name, given)
        asked = self._asked
        same = (
            asked is not None
            and asked.tool == name
            and asked.given == given
            and asked.locked == locked
        )
        if same and asked is not None and asked.heard:
            self._asked = None
            return True
        self._asked = _Asked(name, given, locked, "")
        if same and asked is not None:
            logger.info("%s: the caller spoke before %r ended; asked again", name, asked.line)
        return False

    def _locked(self, name: str, given: dict[str, Any]) -> dict[str, Any]:
        """Return the locked values this input would run with, read from the state now.

        A ``Transcript`` grows every turn, so it is left out: it never makes a yes stale.

        Raises:
            ToolError: When a locked value is not set.
        """
        tool = self._tools.get(name)
        names = [key for key, source in tool.locked.items() if isinstance(source, str)]
        if not names:
            return {}
        filled = tool.filled(given, self._tasks.state)
        return {key: filled[key] for key in names}

    def _ask(self, name: str, lines: Sequence[str]) -> dict[str, str]:
        """Have code ask the confirm line for the input ``_confirmed`` noted."""
        asked = self._asked
        if asked is None:
            raise RuntimeError("no confirm input was noted: call _confirmed before _ask")
        tool = self._tools.get(name)
        say = self._tasks.spoken_lines.say
        said = {
            field: say(value, tool.input_form((field,))) for field, value in asked.given.items()
        }
        line = self._tasks.spoken_line(self._task, name, lines, said, site="confirm", said=True)
        if line is None:
            self._asked = None
            return confirm_unfilled(name, sorted(input_fields(lines) - said.keys()))
        asked.line = self.reply = line
        words = " ".join(line.split())
        values = (" ".join(str(value).split()) for value in said.values())
        asked.said = tuple(value for value in values if value and value in words)
        logger.info("%s: asked the caller %r", name, line)
        return confirm_asked(name, line)

    def _end_on(self, name: str, rules: Sequence[FinishRule], result: object) -> object:
        """End the step on a closing tool's successful result, or hand the result back."""
        matched = [rule for rule in rules if success_matches(rule, result)]
        if self._step.done:
            if matched:
                logger.warning(
                    "step %s had ended when %s succeeded; nothing records it",
                    self._task.name,
                    name,
                )
            return None
        if not matched or not isinstance(result, Mapping):
            self._returned(result)
            self.reply = self._reply_to(name, result)
            return result
        self._terminal = (name, result)
        self._step.carries_turn = True
        try:
            outcome = self._values.save(
                self._task.values_from_result(result), state=self._tasks.state
            )
        except TaskValueRefused as error:
            logger.warning("closing tool %s: not recorded: %s", name, error.message)
            self._event("task.value_refused")
            self.tool_returned = True
            return {**result, "refused": "Not recorded: " + error.message + TERMINAL_SAVE_REFUSED}
        logger.info("step %s ended on %s", self._task.name, name)
        self.receipt = self._tasks.receipt(self._task, name, matched, result)
        self._event("task.ended_on_tool", saved=outcome.saved)
        self._step.outcome = outcome
        return None
