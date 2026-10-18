"""Running a delegate tool's steps on LiveKit, inside the tool call."""

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from livekit.agents import NotGivenOr, llm, tts
from livekit.agents.voice import Agent, AgentSession, RunContext

from unmute_core.livekit._calls import end_call_now, say, tool_text
from unmute_core.livekit._dev import dev_observer
from unmute_core.livekit._history import CallerHistory, step_history, summary_history
from unmute_core.livekit._model import ModelNode, metrics_llm, model_parts
from unmute_core.livekit._realtime import session_closing, speaks_itself
from unmute_core.livekit._step import StepTask
from unmute_core.model import ModelCall
from unmute_core.tasks import DelegateRun, StepRun, TaskSpec

if TYPE_CHECKING:
    from unmute_core.livekit._agent import UnmuteAgent

logger = logging.getLogger(__name__)

type StepTools = Callable[
    [TaskSpec], tuple[list[llm.FunctionTool | llm.RawFunctionTool], list[llm.Tool]]
]
"""Returns the handoff tools and the framework tools of the agent that a task lists, with
each toolset opened into its tools, so a step never owns the agent's toolset."""


@dataclass
class _Progress:
    """What a run has done to the chat so far, and who holds the call."""

    snapshot: llm.ChatContext
    running: llm.ChatContext | None
    latest: llm.ChatContext
    carried: llm.ChatMessage | None = None
    handed_to: Agent | None = None


class DelegateRunner:
    """Awaits each step of a delegate run as a ``StepTask``, and gives the agent its chat back.

    Owns the step that runs now. The agent's ``hand_off`` reads it through ``step``.

    Attributes:
        step: The step task that runs now, or None.
    """

    def __init__(
        self,
        owner: "UnmuteAgent",
        *,
        models: ModelNode,
        model_llm: NotGivenOr[llm.LLM],
        summarizer: ModelCall | None,
        voice: tts.TTS | None,
        task_models: Mapping[str, ModelCall],
        then_targets: Mapping[str, Callable[["UnmuteAgent"], Awaitable[object]]],
        step_tools: StepTools,
        handed_off: Callable[[], bool],
        ends_call: Callable[[], None],
    ) -> None:
        """Keep what a run asks the agent for.

        Args:
            owner: The agent that runs the delegates.
            models: Sends the agent's requests.
            model_llm: The LLM LiveKit holds for the agent.
            summarizer: Writes a ``summary`` history, or None to ask the agent's model.
            voice: The agent's voice, which its steps speak with too.
            task_models: The models a task's ``model`` names.
            then_targets: The agents a group that transfers hands the call to.
            step_tools: Returns the agent's tools a task lists.
            handed_off: Tells whether the agent already handed the call on.
            ends_call: Tells the call's hooks the agent ends the call, before it shuts down.
        """
        self._owner = owner
        self._models = models
        self._model_llm = model_llm
        self._summarizer = summarizer
        self._voice = voice
        self._task_models = task_models
        self._then_targets = then_targets
        self._step_tools = step_tools
        self._handed_off = handed_off
        self._ends_call = ends_call
        self.step: StepTask | None = None

    async def run(
        self, run: DelegateRun, context: RunContext, *, seed: Sequence[llm.ChatItem] = ()
    ) -> object:
        """Await each step of a run, then give the agent its history back with the result.

        A shared group, and a task's steps, hear the steps before them. An isolated group's
        step starts fresh. A spoken receipt line is the agent's last word: the tool then asks
        for no reply. So does a run the call closed on: a reply would only play before the
        line drops. A group whose ``then`` is ``end`` ends the call once it completes, and
        one whose ``then`` is ``transfer`` hands it to its target. A step that handed off
        ends the run, and the tool returns the agent it handed to.

        Args:
            run: The run the core started.
            context: The delegate or via tool's run context.
            seed: The via call and its output, which the first step opens on.
        """
        session = context.session
        snapshot = self._owner.chat_ctx.copy()
        if run.announce:
            say(session, run.announce)
        isolated = run.context_scope == "isolated"
        progress = _Progress(snapshot, None if isolated else snapshot.copy(), llm.ChatContext())
        closed = False
        try:
            await self._steps(run, session, progress, seed)
        except Exception:
            if not session_closing(session):
                raise
            closed = True
        finally:
            result = await self._give_back(run, progress)
        return await self._outcome(run, session, progress, result, closed)

    async def _steps(
        self,
        run: DelegateRun,
        session: AgentSession,
        progress: _Progress,
        seed: Sequence[llm.ChatItem],
    ) -> None:
        """Run the steps one after another, each opening on what the earlier ones left."""
        while (step := run.next_step()) is not None:
            source = progress.running if progress.running is not None else progress.latest
            history = await self._step_history(source, step)
            history.insert(list(seed))
            seed = ()
            task = self._task(step, history)
            self.step = task
            try:
                await self._await_step(run, task, session)
            finally:
                self.step = None
            run.ended(step, said=task.said)
            progress.handed_to = task.handed_to
            progress.latest = task.chat_ctx.copy()
            progress.carried = CallerHistory(progress.latest).newest if step.carries_turn else None
            if progress.running is not None:
                progress.running.merge(
                    progress.latest.copy(),
                    exclude_instructions=True,
                    exclude_function_call=step.went_back,
                )
            if step.receipt is not None:
                self._say_receipt(session, step.receipt, progress.running)

    def _task(self, step: StepRun, history: llm.ChatContext) -> StepTask:
        """Return the step's task, on its own model when its task names one."""
        models, model_llm = self._step_models(step.task.model)
        handoffs, natives = self._step_tools(step.task)
        return StepTask(
            step,
            history=history,
            models=models,
            model_llm=model_llm,
            voice=self._voice,
            transcriber=self._owner.stt,
            handoffs=handoffs,
            native_tools=natives,
        )

    def _say_receipt(
        self, session: AgentSession, receipt: str, running: llm.ChatContext | None
    ) -> None:
        """Say a step's receipt, show it to the trace and the dev page, and keep it in the chat."""
        say(session, receipt, add_to_chat_ctx=False)
        if (trace := self._models.trace) is not None:
            trace.said(receipt)
        if not speaks_itself(session) and (dev := dev_observer(session)):
            dev.said(receipt)
        if running is not None:
            running.insert(llm.ChatMessage(role="assistant", content=[receipt]))

    async def _give_back(self, run: DelegateRun, progress: _Progress) -> dict[str, object]:
        """Give the agent its chat back, with the turn the last step answered, and the result."""
        snapshot = progress.snapshot
        if progress.carried is not None and snapshot.get_by_id(progress.carried.id) is None:
            snapshot.insert(progress.carried)
        result = run.result(caller_turns=CallerHistory(snapshot).turns)
        if run.closing is not None:
            snapshot.insert(llm.ChatMessage(role="assistant", content=[run.closing]))
        await self._owner.update_chat_ctx(snapshot, exclude_invalid_function_calls=False)
        return result

    async def _outcome(
        self,
        run: DelegateRun,
        session: AgentSession,
        progress: _Progress,
        result: dict[str, object],
        closed: bool,
    ) -> object:
        """Return what the delegate tool returns once the run is over."""
        if progress.handed_to is not None:
            return progress.handed_to
        if run.transfers_to is not None:
            target = await self._transfer(run.transfers_to)
            return target if target is not None else tool_text(result)
        if run.ends_call:
            self._ends_call()
            end_call_now(session)
            return llm.ToolResult(tool_text(result), reply_required=False)
        if run.closing is not None or closed:
            return llm.ToolResult(tool_text(result), reply_required=False)
        return tool_text(result)

    async def _transfer(self, name: str) -> Agent | None:
        """Hand the call to the agent a completed group transfers to.

        It hears this agent's history as the run left it, shaped by its handoff. A transfer
        that raises, or hands nothing over, is logged, and the main agent reads the group's
        result and replies instead: the caller is never left in silence.

        Returns:
            The agent LiveKit switches to, or None when the call stays here.
        """
        try:
            target = await self._then_targets[name](self._owner)
        except Exception:
            logger.warning("the transfer to %s failed", name, exc_info=True)
            return None
        if isinstance(target, Agent) and self._handed_off():
            return target
        logger.warning("the transfer to %s handed nothing over: %s", name, target)
        return None

    def _step_models(self, name: str | None) -> tuple[ModelNode, NotGivenOr[llm.LLM]]:
        """Return what sends a step's requests: the agent's, or its task's own model."""
        if name is None:
            return self._models, self._model_llm
        chosen, model_llm = model_parts(self._task_models[name])
        models = self._models
        node = ModelNode(
            state=models.state,
            model=chosen,
            trace=models.trace,
            call=models.call,
            metrics_llm=metrics_llm(model_llm),
        )
        return node, model_llm

    async def _step_history(self, source: llm.ChatContext, step: StepRun) -> llm.ChatContext:
        """Return the chat a step opens on. A ``summary`` step asks for its summary first."""
        if step.history == "summary" and (
            text := await self._models.summarize(self._owner, source, model=self._summarizer)
        ):
            return summary_history(text)
        return step_history(source, step)

    async def _await_step(self, run: DelegateRun, task: StepTask, session: AgentSession) -> None:
        """Run one step, with its span. A step that breaks fails, unless the call is closing.

        A closing session completes the running task with an error
        (voice/agent_session.py:1375). That is the call ending, not the step failing, and
        ``run`` ends the run with no reply.
        """
        step: StepRun = task.step
        span = None
        trace = self._models.trace
        telemetry = trace.telemetry if trace else None
        if telemetry is not None:
            span = telemetry.start_task_step(task=run.delegate.name, step=step.task.name)
        error = None
        try:
            await task
        except Exception as exc:
            if session_closing(session):
                raise
            error = type(exc).__name__
            logger.warning("step %s broke", step.task.name, exc_info=True)
            step.fail()
        finally:
            if telemetry is not None and span is not None:
                outcome = "handed_off" if step.handed_off else step.status or "went_back"
                saved = step.outcome.saved if step.outcome else ()
                state = self._models.state
                telemetry.end_task_step(
                    span,
                    outcome=outcome,
                    saved={name: state.plain(name) for name in saved} if state else None,
                    error=error,
                )
