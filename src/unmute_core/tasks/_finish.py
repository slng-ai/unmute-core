"""The gates a step's finish call passes, and what follows a reply that ended.

``FinishGate`` is the part of a ``StepRun`` that answers ``finish``. It owns what the
refusals remember: the finish held back, the save refused before the caller spoke, the
result a reply left unread.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field, SkipValidation
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.state import HeldSave
from unmute_core.state._call_state import HELD_DECISION_SECONDS
from unmute_core.tasks._progress import GroupProgress
from unmute_core.tasks._spec import FINISH_TOOL_NAME, TaskSpec
from unmute_core.tasks._texts import (
    FINISH_CHECK_INSTRUCTION,
    FINISH_CHECK_SECONDS,
    HELD_SAVE_INSTRUCTION,
    SAVE_BEFORE_CALLER,
    UNSERVED_ON_REENTRY,
    finish_after_failure,
    finish_before_heard,
    finish_refusal,
    finish_while_running,
    finish_without_closing,
    not_yet_tool,
)
from unmute_core.tasks._values import (
    TaskValueRefused,
    hand_back_keeps_values,
    refusal_key,
    saves_values,
    unserved_request,
)
from unmute_core.tools import ToolRegistry

if TYPE_CHECKING:
    from unmute_core.tasks._closing import ClosingTools
    from unmute_core.tasks._run import CallTasks, StepRun
    from unmute_core.tasks._values import StepValues

logger = logging.getLogger(__name__)

TERMINAL_WAIT_SECONDS = 30.0
"""How long a finish waits for a closing tool of its step that still runs."""
NOT_HEARD_MAX = 4000
"""The most characters of an unread tool result a finish keeps."""


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class FinishCheck:
    """A narrow second request after a step reply: may the step finish now?

    The adapter sends the step's chat and the reply just said, under the step's prompt
    with ``instruction`` appended, offers only ``tools`` and requires a call. Nothing of it
    is spoken. A finish it returns goes out as that reply's own finish call, with every
    check a finish has (``decided_finish``).

    Attributes:
        instruction: Appended to the step's prompt for this request only.
        tools: The only tools offered: the step's finish and ``not_yet``, each as name,
            description and parameters.
        timeout_s: How long it may take. Past it, the step goes on.
    """

    instruction: str
    tools: SkipValidation[tuple[dict[str, Any], ...]]
    timeout_s: Annotated[float, Field(gt=0)] = FINISH_CHECK_SECONDS


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class AfterReply:
    """What follows a step reply that ended.

    Attributes:
        finish: Finish arguments to send now, as that reply's own finish call: a finish
            held back so the caller heard a result first.
        check: A finish check to run, whose finish goes out the same way.
    """

    finish: SkipValidation[dict[str, Any] | None] = None
    check: FinishCheck | None = None


def decided_finish(calls: Iterable[tuple[str, str]]) -> dict[str, Any] | None:
    """Return the finish arguments a finish check chose, or None for ``not_yet``.

    Args:
        calls: The check's tool calls, as name and JSON arguments.
    """
    arguments = next((text for name, text in calls if name == FINISH_TOOL_NAME), None)
    if arguments is None:
        return None
    try:
        parsed = json.loads(arguments or "{}")
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


class FinishGate:
    """Answers a step's ``finish``: saves and ends the step, or refuses it once per gate.

    Attributes:
        refused_unheard: The step tool and result text the first refused finish answered
            with no words, or None. Pass it as ``unheard`` for the reply to that refusal.
        not_heard: A tool's result the step finished on without telling the caller.
        answer_refusal: Whether the model replies to the refusal ``finish`` last gave.
            False when the reply that made the finish already spoke to the caller.
    """

    def __init__(
        self,
        step: "StepRun",
        closing: "ClosingTools",
        *,
        tasks: "CallTasks",
        task: TaskSpec,
        tools: ToolRegistry,
        values: "StepValues",
        confirm: Mapping[str, Sequence[str]],
        progress: GroupProgress | None,
        record: bool,
        event: Callable[..., None],
    ) -> None:
        """Build the gate of one step run. ``StepRun`` builds it.

        Args:
            step: The step. The gate reads ``done``, ``finish_tool`` and the outcome fields.
            closing: Its closing tools, which say whether one ran or runs.
            tasks: The call's tasks.
            task: The step's task.
            tools: The tools the step may call.
            values: How the task's values are saved.
            confirm: The confirm lines of each closing tool that has them.
            progress: The group's progress, or None for a task run alone.
            record: Whether the plan runs in ``record`` mode.
            event: Sends one task event of this step.
        """
        self._step = step
        self._closing = closing
        self._tasks = tasks
        self._task = task
        self._tools = tools
        self._values = values
        self._confirm = confirm
        self._progress = progress
        self._record = record
        self._event: Callable[..., None] = event
        self._entry_turns: int | None = None
        self._checked_turns: int | None = None
        self._held_finish: dict[str, Any] | None = None
        self._refused_save: dict[str, Any] | None = None
        self._last_refusal: tuple[str, str] | None = None
        self.refused_unheard: tuple[str, str] | None = None
        self.not_heard = ""
        self.answer_refusal = True

    def opened(self, *, caller_turns: int) -> None:
        """Note how many caller turns the step's chat holds as it opens."""
        self._entry_turns = caller_turns

    def reply_ended(
        self,
        *,
        spoke: bool,
        called: bool,
        completed: bool,
        answers_result: bool,
        caller_turns: int,
    ) -> AfterReply | None:
        """Learn what follows a reply of the step that ended (``record`` mode).

        A finish held back by ``finish_before_heard`` goes out once a reply to that refusal
        has spoken and called nothing: weak models say the result and stop there, and the
        step waits for a finish that never comes. It is dropped when the reply was cut off,
        answered a new caller turn, or called a tool itself. An empty reply keeps it for
        the retry.

        Otherwise a reply that spoke and called nothing, once the caller has spoken in the
        step, gets one finish check per caller turn. A step with tools of its own needs one
        of them to have succeeded first: its values come from that result, and a model that
        skipped the tool would have one made up saved.

        Args:
            spoke: The reply said something.
            called: The reply called a tool.
            completed: The reply ran to its end: not cut off, not failed.
            answers_result: The request it answered ended on a tool's result, not on a
                new caller turn.
            caller_turns: Caller turns in the step's chat for that request.

        Returns:
            A finish to send or a check to run, or None.
        """
        if not self._record or self._step.done:
            return None
        held = self._held_finish
        if held is not None and not (completed and answers_result and not (spoke or called)):
            self._held_finish = None
            if not completed or not answers_result:
                logger.info("finish %s: held finish dropped", self._task.name)
                return None
            if spoke and not called:
                logger.info("finish %s: the runtime sends the held finish", self._task.name)
                return AfterReply(finish=held)
            return None
        if (
            completed
            and spoke
            and not called
            and self._entry_turns is not None
            and caller_turns > self._entry_turns
            and caller_turns != self._checked_turns
            and (self._closing.tool_succeeded or not self._task.tools)
            and not self._closing.busy
        ):
            self._checked_turns = caller_turns
            tools = (self._step.finish_tool(), not_yet_tool())
            return AfterReply(check=FinishCheck(FINISH_CHECK_INSTRUCTION, tools))
        return None

    def unheard(
        self, results: Sequence[tuple[str, str]], *, refused: bool
    ) -> tuple[str, str] | None:
        """Return the step tool result a reply answers without a word, or None.

        Only the step's own tools count. A framework tool or a handoff tool result is not
        the step's to read back.

        Args:
            results: The ``(tool name, result text)`` pairs the reply answers, oldest first.
            refused: True when the reply also answers the step's refused finish call.

        Returns:
            The newest result of one of the step's own tools, else ``refused_unheard`` when
            ``refused``, else None.
        """
        own = [(name, text) for name, text in results if name in self._tools]
        if own:
            return own[-1]
        return self.refused_unheard if refused else None

    async def finish(
        self,
        arguments: Mapping[str, Any] | None,
        *,
        caller_turns: int,
        unheard: tuple[str, str] | None = None,
        spoke: bool = False,
    ) -> str | None:
        """Handle the step's finish call: save and end the step, or refuse.

        A save refused because the caller has said nothing in the step is answered by no
        reply when the reply that made it spoke: it read the value back and asked, so the
        caller answers next (``answer_refusal``). A follow-up reply said goodbye and saved
        again before they answered.

        Args:
            arguments: The finish arguments.
            caller_turns: Caller turns in the step's chat now.
            unheard: The step tool and result text this finish answered with no words,
                once the reply has ended. Pass ``refused_unheard`` for the reply to that
                refusal. ``record`` mode refuses such a finish once.
            spoke: The reply that made this finish said something, once it has ended.

        Returns:
            The refusal the model reads, or None when the step ended.
        """
        self.answer_refusal = True
        self._refused_save = None
        if self._step.done:
            return None
        self._tasks.state.release(self._task.name)
        arguments = dict(arguments or {})
        unserved = bool(unserved_request(arguments))
        if (refusal := self._refuse_unserved(unserved, caller_turns)) is not None:
            return refusal
        if (refusal := self._refuse_before_caller(arguments, caller_turns, spoke)) is not None:
            return refusal
        not_heard = ""
        if unheard is not None and self._reads_back(unserved):
            tool, text = unheard
            if self.refused_unheard is None:
                return self._hold_unheard(arguments, unheard)
            not_heard = f"{tool} returned: {text}"[:NOT_HEARD_MAX]
        running = self._closing.running
        if running is not None:
            if (refusal := await self._wait_for(running, unserved)) is not None:
                return refusal
            if self._step.done:
                return None
            if (refusal := self._refuse_after_failure(running, unserved)) is not None:
                return refusal
        closing_result = self._closing.closing_result
        if closing_result is not None:
            arguments = self._values.merge_retained(arguments, closing_result)
        elif (refusal := self._refuse_made_up(arguments, unserved)) is not None:
            return refusal
        caller_spoke = self._entry_turns is None or caller_turns > self._entry_turns
        return self._save(
            arguments,
            caller_spoke=caller_spoke,
            action_unrecorded=running is not None and closing_result is None,
            not_heard=not_heard,
        )

    def _refuse_unserved(self, unserved: bool, caller_turns: int) -> str | None:
        """Refuse an ``unserved`` finish on re-entry when the caller said nothing new."""
        if (
            self._record
            and self._progress
            and unserved
            and not self._closing.busy
            and self._progress.refuses_unserved(caller_turns)
        ):
            logger.info("finish %s: unserved refused; the caller said nothing new", self._task.name)
            return UNSERVED_ON_REENTRY
        return None

    def _refuse_before_caller(
        self, arguments: Mapping[str, Any], caller_turns: int, spoke: bool
    ) -> str | None:
        """Refuse a save the caller has not had a turn to answer, and remember it."""
        if (
            self._record
            and self._entry_turns is not None
            and saves_values(arguments)
            and not self._closing.busy
            and not self._closing.tool_returned
            and caller_turns <= self._entry_turns
        ):
            logger.info(
                "finish %s: refused; the caller has said nothing in this step", self._task.name
            )
            self.answer_refusal = not spoke
            self._refused_save = None if self._task.finish else dict(arguments)
            return SAVE_BEFORE_CALLER
        return None

    def _reads_back(self, unserved: bool) -> bool:
        """Return True when a finish that answered a result with no words is checked."""
        return self._record and not unserved and self._closing.closing_result is None

    def _hold_unheard(self, arguments: Mapping[str, Any], unheard: tuple[str, str]) -> str:
        """Refuse a finish that answered a tool's result with no words, and hold it."""
        self.refused_unheard = unheard
        self._held_finish = dict(arguments)
        logger.info("finish %s: refused; no words in reply to %s", self._task.name, unheard[0])
        return finish_before_heard(unheard[0])

    async def _wait_for(self, running: str, unserved: bool) -> str | None:
        """Wait for the closing tool that runs, and refuse when it outlasts the wait."""
        try:
            await asyncio.wait_for(self._closing.settled.wait(), timeout=TERMINAL_WAIT_SECONDS)
        except TimeoutError:
            if self._record and not unserved:
                return finish_while_running(running)
            logger.warning("finish %s: %s is still running", self._task.name, running)
        return None

    def _refuse_after_failure(self, running: str, unserved: bool) -> str | None:
        """Refuse a finish made while a closing tool ran and then failed to end the step."""
        if self._record and not self._closing.busy and not unserved:
            return finish_after_failure(running)
        return None

    def _refuse_made_up(self, arguments: Mapping[str, Any], unserved: bool) -> str | None:
        """Refuse values no confirmed closing tool returned, when the task confirms its input."""
        if not self._confirm or unserved:
            return None
        made_up = self._values.changed_values(arguments, self._tasks.state)
        if not made_up:
            return None
        logger.info(
            "finish %s: refused %s; no confirmed tool returned it", self._task.name, made_up
        )
        self._event("task.value_refused")
        return finish_without_closing(sorted(self._confirm), made_up)

    def _save(
        self,
        arguments: Mapping[str, Any],
        *,
        caller_spoke: bool,
        action_unrecorded: bool,
        not_heard: str,
    ) -> str | None:
        """Save the finish values and end the step, or return the refusal of the save."""
        hand_back_saves = self._record and hand_back_keeps_values(
            caller_spoke=caller_spoke, action_unrecorded=action_unrecorded
        )
        try:
            outcome = self._values.save(
                arguments, state=self._tasks.state, hand_back_saves=hand_back_saves
            )
        except TaskValueRefused as refused:
            attempt = refusal_key(refused.message, arguments)
            repeated = attempt == self._last_refusal
            self._last_refusal = attempt
            logger.warning("finish %s: %s", self._task.name, refused.message)
            self._event("task.value_refused")
            return finish_refusal(refused.message, record=self._record, repeated=repeated)
        if outcome.refused:
            logger.warning(
                "finish %s: handed back without %s", self._task.name, ", ".join(outcome.refused)
            )
            self._event("task.value_refused")
        if outcome.status == "unserved" and caller_spoke:
            self._step.carries_turn = True
        self.not_heard = not_heard
        self._step.outcome = outcome
        return None

    def hold_save(
        self,
        *,
        caller_spoke_since: Callable[[], bool],
        decide: Callable[[FinishCheck], Awaitable[Mapping[str, Any] | None]] | None = None,
    ) -> bool:
        """Keep the save ``finish`` just refused before the caller spoke, for the call's end.

        The caller gave the answer before the step began, and the agent told them it was
        taken; one who hangs up would lose it. ``CallState.settle_held`` makes it as the
        call ends. The step's next finish releases it. A step whose values come from its
        own tool holds nothing: the caller cannot have given those before it began.

        Args:
            caller_spoke_since: True once the caller has said anything in the call since
                now. The adapter counts on what the whole call heard, not the step's chat.
            decide: Runs the given finish check on what the caller said since, and returns
                the finish arguments it chose, or None.

        Returns:
            True when a save was held.
        """
        held, self._refused_save = self._refused_save, None
        if held is None:
            return False
        task, state, values = self._task, self._tasks.state, self._values

        def save(decided: Mapping[str, Any] | None) -> bool:
            """Save the held values, or the step's decision, unconfirmed. False if refused."""
            kept = held if decided is None else dict(decided)
            try:
                outcome = values.save(kept, state=state, confirming=False)
            except TaskValueRefused as refused:
                logger.info("held save of %s refused at the call's end: %s", task.name, refused)
                return False
            logger.info("held save of %s at the call's end: %s", task.name, outcome.status)
            return outcome.status == "completed"

        check = FinishCheck(
            HELD_SAVE_INSTRUCTION.format(values=json.dumps(held, default=str)),
            (self._step.finish_tool(), not_yet_tool()),
            HELD_DECISION_SECONDS,
        )
        state.hold(
            task.name,
            HeldSave(
                save=save,
                caller_spoke_since=caller_spoke_since,
                decide=(lambda: decide(check)) if decide is not None else None,
            ),
        )
        return True
