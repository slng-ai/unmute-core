"""One call's task runs: which step runs, what its finish does, what the main agent gets.

The rules began as Unmute's step and delegate code (unmute@4a738ad
templates/livekit_v1/agents.py.tmpl). Here they hold no framework object:
an adapter tells a ``StepRun`` what happened (a tool call, a finish, the caller's turn
count) and does what it answers. The adapter keeps only what needs its framework: chat
history, speech, and waiting for a reply to end.

The flow for one delegate call::

    run = call_tasks.start("book", caller_turns=n)   # a DelegateRun, or a refusal dict
    while (step := run.next_step()) is not None:
        ...  # open the step: step.instructions(), step.tools, step.finish_tool()
        ...  # route its tool calls to step.call_tool / step.finish / step.go_back
        run.ended(step, said=...)                     # once step.done
    result = run.result(caller_turns=n)               # what the main agent reads
"""

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import field
from datetime import date
from typing import Any, Self

from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.state import CallState, StateOrigin, StateRefused
from unmute_core.tasks._closing import ClosingTools
from unmute_core.tasks._finish import AfterReply, FinishCheck, FinishGate
from unmute_core.tasks._opening import Opening, StepOpening
from unmute_core.tasks._plan import RESUMED_SCOPE, Delegate, TaskPlan, entry_history
from unmute_core.tasks._progress import GroupProgress, go_back_targets
from unmute_core.tasks._record import (
    RunRecord,
    ServedRequests,
    TaskEvents,
    TaskEventSink,
    TaskStage,
    new_run_id,
)
from unmute_core.tasks._spec import (
    FINISH_TOOL_NAME,
    FinishRule,
    TaskGroupSpec,
    TaskSpec,
    TasksSpec,
)
from unmute_core.tasks._texts import (
    EMPTY_REPLY_GAVE_UP,
    EMPTY_REPLY_GAVE_UP_AFTER_TOOL,
    FINISH_DESCRIPTION,
    FINISH_DESCRIPTION_RECORD,
    GO_BACK_AFTER_ACTION,
    GO_BACK_TOO_EARLY,
    REFUSED_BUSY,
    REFUSED_HANDOFF_BESIDE,
    REFUSED_HANDOFF_ENDED,
    REFUSED_RERUN,
    REFUSED_RERUN_HANDED_BACK,
    CallStartResults,
    ToolDeclaration,
    ToolStatus,
    arguments_refused,
    delegate_description,
    empty_parameters,
    finish_contract,
    refused,
    render_instructions,
    via_description,
)
from unmute_core.tasks._values import (
    SaveOutcome,
    given_arguments,
)
from unmute_core.tools import ToolRegistry
from unmute_core.tools._announce import LinePicker
from unmute_core.tools._lines import SpokenLines, state_words
from unmute_core.tools._spec import ToolHooks, ToolSpec, ToolStarted, tool_succeeded

logger = logging.getLogger(__name__)


def _as_given(_path: tuple[str, ...]) -> None:
    """Read data already in its spoken form as it is: it has no format left to apply."""
    return None


def caller_turns(messages: Iterable[Mapping[str, Any]]) -> int:
    """Return how many caller turns a chat in OpenAI format holds."""
    return sum(1 for message in messages if message.get("role") == "user")


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class _Returned:
    """How many caller turns there were when a delegate returned, and how it ended."""

    turns: int
    status: str | None


@original_errors
@dataclass(eq=False, config=VALUE)
class ReentryGuard:
    """Refuses a delegate that already ran while the caller has said nothing since.

    A finished result with no new caller turn above it is the model reading its own
    finished work as a new request.
    """

    _returned: dict[str, _Returned] = field(init=False, default_factory=dict)

    def refuses(self, name: str, turns: int) -> bool:
        """Return True while the caller has not spoken since ``name`` returned."""
        return (returned := self._returned.get(name)) is not None and returned.turns == turns

    def record(self, name: str, turns: int, status: str | None = None) -> None:
        """Note how many caller turns there were when ``name`` returned, and how it ended."""
        self._returned[name] = _Returned(turns, status)

    def refusal(self, name: str, *, record: bool) -> dict[str, str]:
        """Return what a refused re-run reads. In ``record`` mode, it says a hand-back."""
        returned = self._returned.get(name)
        if record and returned is not None and returned.status == "unserved":
            return dict(REFUSED_RERUN_HANDED_BACK)
        return dict(REFUSED_RERUN)


class CallTasks:
    """The tasks of one call: the main agent's delegate tools, and every run they start.

    Attributes:
        plan: The agent's checked tasks.
        state: The call state.
        call_start: What the tools that run at call start gave the main agent.
        served: What earlier runs of this call did (``record`` mode).
    """

    def __init__(
        self,
        plan: TaskPlan,
        state: CallState,
        *,
        hooks: ToolHooks | None = None,
        call_start: CallStartResults | None = None,
        language: str = "en",
        timezone: str | None = None,
        gave_up_line: str = EMPTY_REPLY_GAVE_UP,
        gave_up_after_tool_line: str = EMPTY_REPLY_GAVE_UP_AFTER_TOOL,
    ) -> None:
        """Start a call with no run yet.

        Args:
            plan: The agent's checked tasks.
            state: The call state.
            hooks: The call's hooks, like its ``CallHooks``. They run around each tool a
                step calls, and hear each task event.
            call_start: The call-start results, filled by the runner as they land.
            language: The call's language, a BCP-47 tag like ``es-MX``. Confirm lines say
                dates and times in it. Set ``language`` again when the caller switches.
            timezone: Where the business is, an IANA name like ``Europe/Madrid``. Confirm
                lines then name a day from today: "today, October 8". Without it they say
                the full date with its year.
            gave_up_line: What a step says when the model's reply stays empty after every
                retry. Write it in the call's language.
            gave_up_after_tool_line: The same, when the last retry followed a step tool's
                result.

        Raises:
            ValueError: When the timezone is unknown.
        """
        self.plan = plan
        self.gave_up_line = gave_up_line
        self.gave_up_after_tool_line = gave_up_after_tool_line
        self.state = state
        self.suppliers = plan.spec.starters()
        self.activate()
        self.spoken_lines = SpokenLines(language, timezone)
        self.call_start = call_start or CallStartResults()
        self.served = ServedRequests(plan.tasks)
        self.hooks: ToolHooks | None = None
        self._events = TaskEvents()
        if hooks is not None:
            self.use(hooks)
        self._guard = ReentryGuard()
        self.lines = LinePicker()
        self.active: str | None = None

    @classmethod
    def empty(
        cls,
        state: CallState | None = None,
        *,
        tools: ToolRegistry | None = None,
    ) -> Self:
        """Return the tasks of an agent that has none, for an adapter that needs a stack.

        Args:
            state: The call state. None gives one that keeps no values of its own.
            tools: The agent's own tools. None gives none.
        """
        state = state if state is not None else CallState()
        plan = TaskPlan(TasksSpec(), state=state.model, tools=tools or ToolRegistry([]))
        return cls(plan, state)

    @property
    def language(self) -> str:
        """The call's language, a BCP-47 tag. Set it again when the caller switches."""
        return self.spoken_lines.language

    @language.setter
    def language(self, language: str) -> None:
        """Speak later lines in another language."""
        self.spoken_lines.language = language

    def today(self) -> date | None:
        """Return today's date where the business is, or None without a timezone."""
        return self.spoken_lines.today()

    def delegate_tools(self) -> list[dict[str, Any]]:
        """Return the main agent's tools that start a task or group, as name, description, parameters."""
        return [
            ToolDeclaration(
                name=delegate.name,
                description=delegate_description(
                    delegate.when,
                    record=self.plan.record,
                    carries_turn=self.plan.ends_on_tool(delegate.name),
                    ends_call=delegate.kind == "group"
                    and self.plan.groups[delegate.name].then == "end",
                    transfers_to=self.plan.then_targets.get(delegate.name),
                ),
                parameters=self.plan.argument_parameters(delegate.name) or empty_parameters(),
            ).to_dict()
            for delegate in self.plan.delegates()
        ]

    def via_description(self, tool: str, description: str) -> str:
        """Return the description the main agent reads for a tool that enters a task."""
        return via_description(
            description,
            record=self.plan.record,
            carries_turn=self.plan.ends_on_tool(self.plan.via_tasks[tool]),
        )

    def start(
        self, name: str, *, caller_turns: int, arguments: Mapping[str, Any] | None = None
    ) -> "DelegateRun | dict[str, str]":
        """Start a delegate the main agent called, or refuse it.

        The values the delegate takes are saved first, all of them: one the call leaves
        out is cleared, so a stale day from an earlier request never opens a step.

        Args:
            name: The task or group.
            caller_turns: Caller turns in the main agent's chat now.
            arguments: The model's arguments to the delegate tool.

        Returns:
            The run, or the refusal the main agent reads as the tool's result.
        """
        group = self.plan.groups.get(name)
        delegate = Delegate("group", name, group.when) if group else self.plan.delegate_for(name)
        run_id = new_run_id()
        task, group_name = (name, None) if delegate.kind == "task" else (None, name)
        if self._guard.refuses(name, caller_turns):
            self.emit("task.refused_rerun", task=task, group=group_name, run_id=run_id)
            return self._guard.refusal(name, record=self.plan.record)
        if self.plan.record and self.active is not None:
            logger.info("%s did not start: %s is still running", name, self.active)
            self.emit("task.refused_busy", task=task, group=group_name, run_id=run_id)
            return dict(REFUSED_BUSY)
        if names := self.plan.arguments(name):
            try:
                self.state.save(
                    given_arguments(names, arguments or {}), origin=StateOrigin("step", name)
                )
            except StateRefused as refused:
                logger.info("%s did not start: %s", name, refused.message)
                return arguments_refused(name, refused.message)
        return DelegateRun(self, delegate, run_id=run_id, via=False)

    def enter_via(self, tool: str, result: object, *, caller_turns: int) -> "DelegateRun | None":
        """Start the task a ``via`` tool enters, or None when its result stays a plain result.

        The tool must have succeeded, the task's required values must be set, no run may
        be active, and the re-entry guard must allow it.
        """
        name = self.plan.via_tasks.get(tool)
        if (
            name is None
            or self.active is not None
            or not tool_succeeded(result)
            or self.plan.missing(self.plan.tasks[name], self.state)
            or self._guard.refuses(name, caller_turns)
        ):
            return None
        return DelegateRun(self, self.plan.delegate_for(name), run_id=new_run_id(), via=True)

    def use(self, hooks: ToolHooks) -> None:
        """Run ``hooks`` around each tool a step calls, and tell them each task event.

        An adapter calls it with the call's ``CallHooks``. Calling it again with the same
        hooks does nothing.

        Raises:
            ValueError: If other hooks are already in use: build the tasks per call.
        """
        if self.hooks is not None and self.hooks is not hooks:  # needs the hooks held now
            raise ValueError("these tasks already run another call's hooks: build them per call")
        self.hooks = hooks
        self.listen(hooks.emit)

    def activate(self) -> None:
        """Make these tasks the ones a refused tool names, as their agent takes the call.

        Several agents can share one ``CallState``. The agent that holds the call calls this,
        so a tool that reads an unset value names a task that agent can start.
        """
        self.state.set_suppliers(self.suppliers)

    def listen(self, sink: TaskEventSink) -> None:
        """Send every later task event to ``sink`` too. Adding the same sink again does nothing.

        Args:
            sink: Receives each task event, like ``Telemetry.event``.
        """
        self._events.listen(sink)

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
        self._events.emit(stage, task=task, group=group, run_id=run_id, step=step, saved=saved)

    def pick_line(self, site: str, lines: Sequence[str]) -> str | None:
        """Pick one announce line, as ``LinePicker.pick`` does. None for no lines."""
        return self.lines.pick(site, lines)

    def receipt(
        self, task: TaskSpec, tool: str, rules: Sequence[FinishRule], result: object
    ) -> str | None:
        """Return the spoken line for a result that ended a step, or None."""
        lines = next((rule.say for rule in rules if rule.say), ())
        return self.spoken_line(task, tool, lines, result)

    def spoken_line(
        self,
        task: TaskSpec,
        tool: str,
        lines: Sequence[str],
        data: object,
        *,
        site: str = "",
        said: bool = False,
    ) -> str | None:
        """Return the next of ``lines`` that fills from a tool's data, or None.

        A ``{{result.x}}`` is said by the tool's output model, in the call's language.

        Args:
            task: The step's task.
            tool: The tool.
            lines: The lines, in turn.
            data: The tool's result, or for a confirm line its input.
            site: Keeps a separate turn of lines, like ``confirm`` beside the receipts.
            said: The data is already in its spoken form, so it is read as it is.
        """
        if not lines:
            return None
        spec = self.plan.tools.get(tool)
        key = f"{task.name}:{tool}" + (f":{site}" if site else "")
        values = state_words(self.state, f"task:{task.name}", lines)
        return self.spoken_lines.line(
            key, lines, data, _as_given if said else spec.result_form, values
        )

    def release(
        self, delegate: str, record: RunRecord, *, caller_turns: int, status: str | None
    ) -> None:
        """Take a run's end into the call: guard its re-run, keep what it served, free the call.

        Args:
            delegate: The task or group the run was of.
            record: The run's record.
            caller_turns: Caller turns in the main agent's chat once it is restored.
            status: How the run ended, or None when it did not.
        """
        self._guard.record(delegate, caller_turns, status)
        self.served.record(record)
        self.active = None


class DelegateRun:
    """One delegate call: its steps in order, its record, and what the main agent reads.

    Attributes:
        delegate: The task or group it runs.
        run_id: The id its events carry.
        via: A tool's success started it.
        announce: The line to say as it starts, while the first step opens, or None.
        context_scope: The group's ``context_scope``, or None for a task.
        record: The record of its steps.
        status: How it ended, once no step is left.
        closing: The spoken line the last step ended on. When the run completes on it,
            it is the main agent's reply, and no model request is made for another.
        ends_call: Set by ``result``: the group completed and its ``then`` is ``end``.
            The adapter ends the call with no reply from the main agent.
        transfers_to: Set by ``result``: the agent a group that completed hands the call
            to, when its ``then`` is ``transfer``. The adapter hands the call over through
            its ``then_targets``, with no reply from the main agent. None otherwise.
        handed_off: A step handed the call to another agent. The run is over, and the
            main agent never reads its result.
    """

    def __init__(self, tasks: CallTasks, delegate: Delegate, *, run_id: str, via: bool) -> None:
        """Claim the call for this run and decide what runs first.

        Args:
            tasks: The call's tasks.
            delegate: The task or group.
            run_id: The id its events carry.
            via: A tool's success started it.
        """
        self._tasks = tasks
        plan = tasks.plan
        self.delegate = delegate
        self.run_id = run_id
        self.via = via
        self.status: str | None = None
        self.closing: str | None = None
        self.ends_call = False
        self.transfers_to: str | None = None
        self.handed_off = False
        self._last: SaveOutcome | None = None
        self._phase = 0
        tasks.active = delegate.name
        if delegate.kind == "task":
            task = plan.tasks[delegate.name]
            self._group: TaskGroupSpec | None = None
            self._progress: GroupProgress | None = None
            self._prerequisite = None if via else plan.prerequisite(task, tasks.state)
            self.record = RunRecord.of_task(
                task.name,
                provider=plan.providers.get(task.name),
                provider_runs=self._prerequisite is not None,
            )
            first = self._prerequisite or task
            self.announce = (
                None
                if via or first.opening == "listen"
                else tasks.pick_line(f"task:{first.name}", first.announce)
            )
            self.context_scope = None
            if via:
                tasks.emit("task.entered_via", task=task.name, run_id=run_id)
            return
        self._group = group = plan.groups[delegate.name]
        self._progress = GroupProgress(
            group, plan.steps(group.name), plan.skipped_steps(group.name, tasks.state.is_confirmed)
        )
        self.record = RunRecord.of_group(self._progress)
        self.announce = tasks.pick_line(group.name, group.announce)
        self.context_scope = group.context_scope
        tasks.emit("group.started", group=group.name, run_id=run_id)
        for index in sorted(self._progress.skipped):
            logger.info("skipped %s: its value is already confirmed", group.steps[index])
            tasks.emit(
                "group.step_skipped",
                group=group.name,
                task=group.steps[index],
                run_id=run_id,
                step=index,
            )

    @property
    def tasks(self) -> CallTasks:
        """The call's tasks this run belongs to."""
        return self._tasks

    @property
    def progress(self) -> GroupProgress | None:
        """The order of a group's steps, or None for a task."""
        return self._progress

    @property
    def group(self) -> TaskGroupSpec | None:
        """The group this run runs, or None for a task."""
        return self._group

    def next_step(self) -> "StepRun | None":
        """Return the step to open now, or None when the run is over (see ``status``)."""
        if self.status is not None:
            return None
        if self._progress is None:
            return self._next_task_step()
        return self._next_group_step()

    def _next_task_step(self) -> "StepRun | None":
        """Return a task's prerequisite, then the task, or None."""
        plan = self._tasks.plan
        task = plan.tasks[self.delegate.name]
        if self._phase == 0:
            self._phase = 1 if self._prerequisite else 2
            if self._prerequisite is None:
                return self._open(task, seeded=self.via)
            self.emit("task.requires", task=task.name)
            return self._open(self._prerequisite, next_step=task.name)
        if self._phase == 1:
            self._phase = 2
            provided = self._last or SaveOutcome("failed")
            if provided.status != "completed":
                self.status = provided.status
                return None
            if plan.missing(task, self._tasks.state):
                self.emit("task.unserved", task=task.name)
                self.status = "unserved"
                return None
            return self._open(
                task, speak_opening=True, group_scope=RESUMED_SCOPE, follows_step=True
            )
        self.status = (self._last or SaveOutcome("failed")).status
        return None

    def _next_group_step(self) -> "StepRun | None":
        """Return the group's next step, ending steps that cannot start, or None."""
        progress = self._progress
        group = self._group
        if progress is None or group is None:
            raise RuntimeError("a task run has no group steps: call next_step to walk it")
        while (index := progress.next_step()) is not None:
            task = progress.task
            self.emit("group.step_started", task=task.name, step=index)
            if not self._tasks.plan.missing(task, self._tasks.state):
                upcoming = progress.next_task
                return self._open(
                    task,
                    index=index,
                    speak_opening=not progress.reentered,
                    group_scope=group.context_scope,
                    reentry=progress.reentered,
                    follows_step=progress.follows_step,
                    next_step=upcoming.name if upcoming else None,
                )
            self.emit("task.requires", task=task.name, step=index)
            self._ended(task, SaveOutcome("failed"), index=index)
            if self.status is not None:
                return None
        self.emit("group.completed")
        self.status = progress.status()
        return None

    def _open(
        self,
        task: TaskSpec,
        *,
        index: int | None = None,
        speak_opening: bool = False,
        group_scope: str | None = None,
        reentry: bool = False,
        follows_step: bool = False,
        next_step: str | None = None,
        seeded: bool = False,
    ) -> "StepRun":
        """Build a step and report that it started."""
        if task.name in self._tasks.plan.withdrawing:
            self._tasks.state.withdraw(task.name)
        step = StepRun(
            self,
            task,
            index=index,
            speak_opening=speak_opening,
            group_scope=group_scope,
            reentry=reentry,
            follows_step=follows_step,
            next_step=next_step,
            seeded=seeded,
        )
        self.emit("task.started", task=task.name, step=index)
        return step

    def ended(self, step: "StepRun", *, said: str = "") -> None:
        """Record a step that is done. A step left without an outcome counts as failed.

        Args:
            step: The step.
            said: What the caller heard in the reply that finished it, once it played.
        """
        if step.went_back:
            self.emit("group.went_back", task=step.task.name, step=step.index)
            return
        if step.handed_off:
            self.emit("task.handed_off", task=step.task.name, step=step.index)
            outcome = SaveOutcome("unserved")
            self.record.ended(step.task.name, outcome, index=step.index, said=said)
            self._last = outcome
            self.closing = None
            self.handed_off = True
            self.status = outcome.status
            return
        self._ended(
            step.task,
            step.outcome or SaveOutcome("failed"),
            index=step.index,
            said=said,
            not_heard=step.not_heard,
        )
        self.closing = step.receipt

    def _ended(
        self,
        task: TaskSpec,
        outcome: SaveOutcome,
        *,
        index: int | None,
        said: str = "",
        not_heard: str = "",
    ) -> None:
        """Report and record how a step ended, and stop a group it stops."""
        if outcome.status == "completed":
            self.emit("task.completed", task=task.name, step=index, saved=outcome.saved)
        else:
            stage: TaskStage = "task.unserved" if outcome.status == "unserved" else "task.failed"
            self.emit(stage, task=task.name, step=index, saved=outcome.saved or None)
        self.record.ended(task.name, outcome, index=index, said=said, not_heard=not_heard)
        self._last = outcome
        self.closing = None
        if self._progress is None and self._phase == 2:
            self.status = outcome.status
        if (
            self._progress is not None
            and index is not None
            and self._progress.record(index, outcome) == "stop"
        ):
            logger.info("group stopped: %s ended %s", task.name, outcome.status)
            self.emit("group.stopped", task=task.name, step=index)
            self.status = self._progress.status()

    def result(self, *, caller_turns: int) -> dict[str, Any]:
        """Return what the main agent reads, and release the call. Call it once, at the end.

        Args:
            caller_turns: Caller turns in the main agent's chat once it is restored,
                including a turn a step carried back.

        Returns:
            ``{"status": ...}``, or the run record in ``record`` mode.
        """
        status = self.status or "failed"
        if status != "completed":
            self.closing = None
        completed_group = self._group if status == "completed" else None
        self.ends_call = completed_group is not None and completed_group.then == "end"
        self.transfers_to = (
            completed_group.then_target
            if completed_group is not None and completed_group.then == "transfer"
            else None
        )
        tasks = self._tasks
        tasks.release(
            self.delegate.name, self.record, caller_turns=caller_turns, status=self.status
        )
        if tasks.plan.record:
            return self.record.build(status, tasks.state)
        return ToolStatus(status=status).model_dump()

    def emit(
        self,
        stage: TaskStage,
        *,
        task: str | None = None,
        step: int | None = None,
        saved: Sequence[str] | None = None,
    ) -> None:
        """Send an event of this run."""
        self._tasks.emit(
            stage,
            task=task,
            group=self._group.name if self._group else None,
            run_id=self.run_id,
            step=step,
            saved=saved,
        )


class StepRun:
    """One run of one step: its prompt, its tools, and what each of its calls does.

    Attributes:
        task: The task.
        index: Its place in the group, or None.
        history: What it hears of the call as it opens (``entry_history``).
        blocking: The group stops when it fails.
        tools: The tools it may call.
        listens: It opens by speaking ``announce`` and waiting, with no model request.
        announce: The line to say as it opens, or None.
        ends_call: The tool ``call_tool`` just ran succeeded and its effect ends the call.
        outcome: How it ended, once it has.
        went_back: It ended because the caller went back to earlier steps.
        handed_off: It ended on one of its task's handoffs. Its run ends with it.
        receipt: The spoken line it ended on, from a closing tool's result.
        reply: The line code speaks for the tool call that just ran, or None: a
            ``replies`` line for its result, or a ``confirm`` line asked before it runs.
            The adapter says it, asks the model for no reply, and calls ``reply_played``.
        not_heard: A tool's result it finished on without telling the caller.
        carries_turn: It answered the caller's newest turn, which the main agent never
            saw. The adapter puts that turn back in front of the main agent.
        answer_refusal: Whether the model replies to the refusal ``finish`` last gave.
            False when the reply that made the finish already spoke to the caller: the
            next move is theirs.
    """

    def __init__(
        self,
        run: DelegateRun,
        task: TaskSpec,
        *,
        index: int | None,
        speak_opening: bool,
        group_scope: str | None,
        reentry: bool,
        follows_step: bool,
        next_step: str | None,
        seeded: bool,
    ) -> None:
        """Build one step run. ``DelegateRun.next_step`` builds them."""
        self._run = run
        self._tasks = tasks = run.tasks
        plan = tasks.plan
        self._record = plan.record
        progress = run.progress
        self._progress = progress if index is not None else None
        self.task = task
        self.index = index
        self.history = entry_history(task, group_scope=group_scope, reentry=reentry)
        self.blocking = (
            self._progress.is_blocking(index) if self._progress and index is not None else True
        )
        self.tools: ToolRegistry = plan.tools.only(task.tools)
        self._rules = plan.finish_rules(task)
        self._derived = task.derived_arguments()
        self._group = run.group.name if run.group and index is not None else None
        self._go_back = self._progress.go_back_tool() if self._record and self._progress else None
        self._next_step = next_step if self._record else None
        self._answered = False
        self._contract: str | None = None
        self._finish_declaration: ToolDeclaration | None = None
        self.outcome: SaveOutcome | None = None
        self.went_back = False
        self.handed_off = False
        self.carries_turn = False
        confirm = {rule.tool: rule.confirm for rule in task.finish if rule.confirm}
        values = plan.values(task.name)
        self._closing = ClosingTools(
            self,
            tasks=tasks,
            task=task,
            tools=self.tools,
            rules=self._rules,
            confirm=confirm,
            values=values,
            record=self._record,
            event=self._event,
        )
        self._gate = FinishGate(
            self,
            self._closing,
            tasks=tasks,
            task=task,
            tools=self.tools,
            values=values,
            confirm=confirm,
            progress=self._progress,
            record=self._record,
            event=self._event,
        )
        self._opening = StepOpening(
            self,
            tasks=tasks,
            task=task,
            tools=self.tools,
            confirm=confirm,
            progress=self._progress,
            record=self._record,
            speak_opening=speak_opening,
            reentry=reentry,
            follows_step=follows_step,
            seeded=seeded,
        )

    @property
    def done(self) -> bool:
        """True once the step has ended."""
        return self.outcome is not None or self.went_back

    @property
    def status(self) -> str:
        """How the step ended: the outcome's status, ``went_back``, or empty while it runs."""
        if self.outcome is not None:
            return self.outcome.status
        return "went_back" if self.went_back else ""

    @property
    def listens(self) -> bool:
        """The step opens by speaking ``announce`` and waiting, with no model request."""
        return self._opening.listens

    @property
    def announce(self) -> str | None:
        """The line to say as it opens, or None."""
        return self._opening.announce

    @property
    def ends_call(self) -> bool:
        """The tool ``call_tool`` just ran succeeded and its effect ends the call."""
        return self._closing.ends_call

    @property
    def receipt(self) -> str | None:
        """The spoken line it ended on, from a closing tool's result."""
        return self._closing.receipt

    @property
    def reply(self) -> str | None:
        """The line code speaks for the tool call that just ran, or None.

        A ``replies`` line for its result, or a ``confirm`` line asked before it runs. The
        adapter says it, asks the model for no reply, and calls ``reply_played``.
        """
        return self._closing.reply

    @property
    def result(self) -> object:
        """The whole result of the tool call that just ran, for code."""
        return self._closing.result

    @property
    def refused_unheard(self) -> tuple[str, str] | None:
        """The step tool and result the first refused finish answered with no words."""
        return self._gate.refused_unheard

    @property
    def not_heard(self) -> str:
        """A tool's result it finished on without telling the caller."""
        return self._gate.not_heard

    @property
    def answer_refusal(self) -> bool:
        """Whether the model replies to the refusal ``finish`` last gave.

        False when the reply that made the finish already spoke to the caller: the next
        move is theirs.
        """
        return self._gate.answer_refusal

    @property
    def closing_result(self) -> Mapping[str, Any] | None:
        """What the closing tool that ended the step returned, or None.

        ``call_tool`` gives the adapter None for that call, so the model reads only how
        the step ended. The trace shows this result as the tool's output.
        """
        return self._closing.closing_result

    @property
    def site(self) -> str:
        """The prompt site its model requests carry, ``task:<name>``."""
        return f"task:{self.task.name}"

    @property
    def opening_message(self) -> str | None:
        """In ``status`` mode, the developer message the step opens on. None otherwise."""
        return self._opening.message

    def instructions(self) -> str:
        """Return the step's prompt now, its ``{{name}}`` placeholders still in it.

        Render it as the step opens, so values saved since read current. Until the step
        has answered once, it ends on why it opens as it does.
        """
        plan = self._tasks.plan
        result_names = [
            row.argument for row in self.task.assign if row.argument not in self._derived
        ]
        if self._contract is None:
            self._contract = finish_contract(
                FINISH_TOOL_NAME,
                result_names,
                record=self._record,
                ends_on=self._rules,
                go_back=self._go_back["name"] if self._go_back else None,
                next_step=self._next_step,
            )
        return render_instructions(
            self.task,
            record=self._record,
            finish_name=FINISH_TOOL_NAME,
            result_names=result_names,
            ends_on=self._rules,
            go_back=self._go_back["name"] if self._go_back else None,
            next_step=self._next_step,
            inputs=plan.inputs_for(self.task, self._group),
            call_start=self._tasks.call_start,
            served=self._tasks.served.section(self.task.name) if self._record else "",
            opening="" if self._answered else self._opening.note,
            contract=self._contract,
        )

    def finish_tool(self) -> dict[str, Any]:
        """Return the finish tool, as name, description and parameters."""
        if self._finish_declaration is None:
            self._finish_declaration = ToolDeclaration(
                name=FINISH_TOOL_NAME,
                description=FINISH_DESCRIPTION_RECORD if self._record else FINISH_DESCRIPTION,
                parameters=self._tasks.plan.finish_parameters(self.task.name),
            )
        return self._finish_declaration.to_dict()

    def go_back_tool(self) -> dict[str, Any] | None:
        """Return the go-back tool, or None. Offer it only while ``offers_go_back``."""
        return self._go_back

    def offers_go_back(self, *, caller_turns: int) -> bool:
        """Return True when a model request may offer the go-back tool.

        Only once the caller has spoken in the step, so the step's opening reply cannot
        call it and be refused. Both adapters leave it out of a request until then.

        Args:
            caller_turns: Caller turns in the step's chat now.
        """
        return (
            self._go_back is not None
            and self._progress is not None
            and self._progress.may_go_back(caller_turns)
        )

    def opened(self, *, caller_turns: int) -> None:
        """Tell the step how many caller turns its chat holds as it opens."""
        self._gate.opened(caller_turns=caller_turns)
        if self._progress is not None:
            self._progress.mark_opened(caller_turns=caller_turns)
        self._opening.opened()

    def answered(self) -> None:
        """Tell the step it has answered once. Its prompt drops the opening from now on."""
        self._answered = True

    async def open(self) -> "Opening | None":
        """Run the step's opening tool in code, and pick the line its result fills.

        Call it once, before ``opened``. When a line fills, the step listens and says no
        announce: the adapter puts the call and its result in the history, says the line,
        and asks the model nothing. When no line fills, the step opens as it would without
        an opening, and the adapter only traces the call. None means the tool did not run:
        the step has no opening tool, a value it takes is unset, or the tool failed. An
        unset value for a parameter the schema declares as optional is left out instead.

        The call goes in the history only when a line is said, so the model next reads it
        after the caller's turn. Gemini refuses a request whose own turn holds a call no
        model made, and accepts one in an earlier turn.

        A tool with ``confirm`` lines does not run: its line is the confirm question, and the
        caller's yes runs it. It is skipped when every value it takes is one a step confirms
        and that confirmation stands.
        """
        return await self._opening.open()

    async def call_tool(
        self,
        name: str,
        arguments: str | Mapping[str, Any],
        *,
        started: ToolStarted | None = None,
    ) -> object:
        """Run one of the step's own tools.

        A closing tool that succeeds ends the step: the result is saved and None comes
        back, so the adapter asks for no reply.

        Args:
            name: The tool.
            arguments: The model's arguments, as JSON text or parsed.
            started: Called once the call is checked, just before the tool runs: the
                adapter says the tool's announce line there.

        Returns:
            What the model reads, or None when the step ended. ``ends_call`` then says
            whether the tool ran, succeeded, and ends the call. A tool that hides its
            result gives ``{"ok": true}``. ``result`` keeps the whole result for code.

        Raises:
            ToolError: From the tool.
        """
        return await self._closing.call(name, arguments, started=started)

    def reply_ended(
        self,
        *,
        spoke: bool,
        called: bool,
        completed: bool,
        answers_result: bool,
        caller_turns: int,
    ) -> AfterReply | None:
        """Tell the step a reply of it ended, and learn what follows (``record`` mode).

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
        return self._gate.reply_ended(
            spoke=spoke,
            called=called,
            completed=completed,
            answers_result=answers_result,
            caller_turns=caller_turns,
        )

    def reply_played(self, line: str, *, heard: bool, played: str = "") -> None:
        """Note how a ``reply`` line played. The adapter calls it once the line is over.

        A confirm line cut off after it said every value it holds counts as heard: a
        caller who answers "yes" over the last words has heard what they agreed to. A line
        that holds no value counts only when it played to its end.

        Args:
            line: The line, as ``reply`` gave it.
            heard: False when it was cut off before its end.
            played: The words that played before the cut, when the adapter knows them.
        """
        self._closing.reply_played(line, heard=heard, played=played)

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
        return self._gate.unheard(results, refused=refused)

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
        return await self._gate.finish(
            arguments, caller_turns=caller_turns, unheard=unheard, spoke=spoke
        )

    def announcement(self, tool: ToolSpec) -> str | None:
        """Pick the line one of the step's tools says as it starts, or None."""
        return self._tasks.pick_line(f"tool:{tool.name}", tool.announce)

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
        return self._gate.hold_save(caller_spoke_since=caller_spoke_since, decide=decide)

    def go_back(self, arguments: object, *, caller_turns: int) -> dict[str, str] | None:
        """Handle the go-back call: end the step so earlier ones run again, or refuse.

        Returns:
            The refusal the model reads, or None when the step ended.
        """
        if self.done:
            return None
        if self._progress is None or self._go_back is None:
            return refused("There is no earlier step to go back to.")
        if self._closing.busy:
            return dict(GO_BACK_AFTER_ACTION)
        if not self._progress.may_go_back(caller_turns):
            return dict(GO_BACK_TOO_EARLY)
        try:
            targets = self._progress.go_back(go_back_targets(arguments))
        except ValueError as error:
            return refused(str(error))
        logger.info(
            "step %s: going back to %s",
            self.task.name,
            [self._progress.steps[i].name for i in targets],
        )
        self.went_back = True
        return None

    def may_hand_off(self, *, alone: bool) -> dict[str, str] | None:
        """Return the refusal of one of its task's handoffs now, or None when it may go.

        A handoff made beside another call of the step is refused, and the other call
        runs: the step's own work comes first. The model hands the caller over in a reply
        of its own, from wherever the call is then.

        Args:
            alone: The reply that made the handoff made no other call.
        """
        if self.done:
            return dict(REFUSED_HANDOFF_ENDED)
        if not alone:
            return dict(REFUSED_HANDOFF_BESIDE)
        return None

    def hand_off(self) -> None:
        """End the step on a handoff ``may_hand_off`` allowed, once the next agent is ready.

        The adapter calls it only when the handoff can no longer fail, so a broken
        handoff leaves the step running.
        """
        if not self.done:
            self.handed_off = True
            self.outcome = SaveOutcome("unserved")

    def fail(self) -> None:
        """End the step as failed, with nothing saved: it broke, or cannot answer."""
        if not self.done:
            self.outcome = SaveOutcome("failed")

    def gave_up(self, *, finish_only: bool) -> str | None:
        """Handle a reply that stayed empty after every retry.

        A step its group can do without fails, so the group carries on. Any other step
        stays open and says a short apology.

        Args:
            finish_only: The last retry offered only finish, after a step tool's result.

        Returns:
            The line to say, or None when the step failed.
        """
        if not self.blocking:
            self.fail()
            return None
        return self._tasks.gave_up_after_tool_line if finish_only else self._tasks.gave_up_line

    def _event(self, stage: TaskStage, *, saved: Sequence[str] | None = None) -> None:
        """Send an event of this step."""
        self._run.emit(stage, task=self.task.name, step=self.index, saved=saved)
