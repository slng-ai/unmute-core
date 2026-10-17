"""Delegate runs on Pipecat: a Flows node for each step, and the run's way back.

``RunLifecycle`` owns the one run a call has at a time. It claims the call, copies the
main agent's context, opens each step as a node, and gives the main agent its context
back with the run's result. ``UnmuteFlows`` supplies the handlers a node's tools run.
"""

import asyncio
import copy
import json
import logging
import re
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from opentelemetry.trace import Span
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.flows import (
    NO_RESPONSE,
    ContextStrategy,
    ContextStrategyConfig,
    FlowManager,
    NodeConfig,
)
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMRunFrame,
    LLMUpdateSettingsFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import (
    NOT_GIVEN,
    LLMContext,
    LLMContextMessage,
    NotGiven,
)
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings

from unmute_core.call import AGENT_ENDED, CallHooks
from unmute_core.handoff import summary_message
from unmute_core.model import STEP_START_CUE
from unmute_core.pipecat._history import (
    STEP_START_MESSAGE,
    below_replies,
    caller_turns,
    made_other_calls,
    newest_caller_message,
    set_tool_reply,
    shape_history,
    tool_pair,
    without_call,
    without_calls_to,
)
from unmute_core.pipecat._model import ModelCallLLMService
from unmute_core.pipecat._replies import ReplyTracker
from unmute_core.pipecat._side import SideRequests
from unmute_core.pipecat._tools import FunctionHandler, joins_context
from unmute_core.tasks import (
    FINISH_TOOL_NAME,
    CallTasks,
    DelegateRun,
    Opening,
    StepRun,
    TaskSpec,
    ToolStatus,
)
from unmute_core.telemetry import Telemetry

logger = logging.getLogger(__name__)

type NodeResult = tuple[object, NodeConfig | object | None]
type NodeFunctions = Callable[["ActiveRun", "ActiveStep"], list[Any]]
type HandoffTool = Callable[["ActiveRun", "ActiveStep", FunctionSchema], FunctionSchema]

FRESH_HISTORY = frozenset({"reset", "caller_turn"})
"""The histories that start a step from nothing the main agent said."""
SHAPE_HISTORY = "unmute_shape_history"
"""The Flows action that cuts a step's history as its node opens."""
FLOWS_PLACEHOLDER = re.compile(
    r"\\?\{\{\s*[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\s*\}\}"
)
"""The ``{{ key }}`` Flows fills from its state (pipecat.flows.manager ``_PLACEHOLDER``)."""


def for_flows(text: str) -> str:
    """Escape every ``{{ key }}`` so Flows leaves the text as it is."""
    return FLOWS_PLACEHOLDER.sub(lambda match: "\\" + match[0], text)


def result_text(result: object) -> str:
    """Return a tool result as Pipecat's assistant aggregator writes it."""
    return json.dumps(result, ensure_ascii=False) if result else "COMPLETED"


def opening_messages(opening: Opening, line: str) -> list[dict[str, Any]]:
    """Return a step's opening call, its result and the line said, as the context writes them."""
    call = {"name": opening.tool, "arguments": opening.arguments}
    return [
        {
            "role": "assistant",
            "tool_calls": [{"id": opening.call_id, "type": "function", "function": call}],
        },
        {"role": "tool", "tool_call_id": opening.call_id, "content": result_text(opening.result)},
        {"role": "assistant", "content": line},
    ]


async def shielded[T](work: Coroutine[Any, Any, T]) -> T:
    """Run ``work`` to the end even if an interruption cancels the tool call.

    Entering and leaving a step rewrite the context in several awaited steps. Stopping
    half way would lose the main agent's copy.
    """
    task = asyncio.ensure_future(work)
    return await asyncio.shield(task)


def context_messages(context: LLMContext) -> list[Any]:
    """Return a context's messages as plain dicts and provider objects."""
    return cast(list[Any], context.get_messages())


def replace_messages(context: LLMContext, messages: list[Any]) -> None:
    """Replace a context's messages."""
    context.set_messages(cast(Any, messages))


@dataclass(eq=False)
class ActiveStep:
    """One step's node.

    Attributes:
        step: The core step.
        baseline: Caller turns in its context that the main agent already saw.
        span: Its ``task_step`` span, or None without telemetry.
        prompt: The prompt it was last sent.
        answered: A reply that started in this step has ended.
        advanced: Its ending has been handled.
        tools: Every tool its node set, go-back included, kept while a request leaves
            go-back out.
        opening_turn: Caller turns the call held when the step opened. Its handoffs stay
            out of its opening reply and come back once the caller speaks.
        handoffs: The agent's handoff tools its task offers too.
        native: The agent's native tools its task offers too, as they are.
        handing_off: One of those runs now. ``hand_off`` reads it.
    """

    step: StepRun
    baseline: int
    opening_turn: int = 0
    span: Span | None = None
    prompt: str = ""
    answered: bool = False
    advanced: bool = False
    tools: ToolsSchema | None = None
    handoffs: list[FunctionSchema] = field(default_factory=list)
    native: list[FunctionSchema] = field(default_factory=list)
    handing_off: bool = False


@dataclass(eq=False)
class ActiveRun:
    """One delegate run on Pipecat.

    Attributes:
        run: The core run.
        call_id: The delegate call's id. Its reply carries the run's result.
        worker: The pipeline worker the call came from.
        snapshot: The main agent's messages, copied as the run took the context.
        snapshot_tools: The main agent's tools.
        beside: Results of main-agent calls that landed while the run held the context.
        baseline: Caller turns in the run's context that the main agent saw.
        step: The step running now.
        shape_next: The task whose history applies once a move's result is recorded.
        carried: The caller's turn a step answered, for the main agent to read.
        running: Everything the run's steps said and did so far, after the main agent's
            messages. Each step opens on its own history of it, as on LiveKit, so a
            ``messages`` step does not take the tool calls away from a ``full`` step after it.
        opened_at: Where the live context's messages that ``running`` lacks begin.
    """

    run: DelegateRun
    call_id: str
    worker: PipelineWorker
    snapshot: list[LLMContextMessage] | None = None
    snapshot_tools: ToolsSchema | NotGiven = NOT_GIVEN
    beside: dict[str, object] = field(default_factory=dict)
    baseline: int = 0
    step: ActiveStep | None = None
    shape_next: TaskSpec | None = None
    carried: dict[str, Any] | None = None
    running: list[Any] = field(default_factory=list)
    opened_at: int = 0


@dataclass
class AgentSeat:
    """The agent that holds the call. A handoff gives the seat to another.

    Attributes:
        tasks: The agent's tasks.
        native: The agent's Pipecat tools, which a step's task may offer too.
    """

    tasks: CallTasks
    native: list[FunctionSchema]


class RunLifecycle:
    """The one delegate run a call has, from its start to its result.

    Attributes:
        task_results: For ``TelemetryObserver``: the result the main agent reads for each
            task call, by call ID. None while the task runs.
        openings: For ``TelemetryObserver``: the tools steps ran in code as they opened.
    """

    def __init__(
        self,
        *,
        llm: LLMService[Any],
        context: LLMContext,
        aggregators: LLMContextAggregatorPair,
        telemetry: Telemetry | None,
        side: SideRequests,
        replies: ReplyTracker,
        seat: AgentSeat,
        then_targets: Mapping[str, FunctionHandler],
        main_instruction: Callable[[], str],
        functions: NodeFunctions,
        handoff_tool: HandoffTool,
        call: CallHooks | None = None,
    ) -> None:
        """Hold what a run reads and writes: the call's objects and the node builders.

        Args:
            llm: The LLM service, shared by the main agent and every step.
            context: The shared context.
            aggregators: The context's aggregator pair.
            telemetry: Records a ``task_step`` span per step.
            side: Renders a step's prompt and asks for a summary.
            replies: Counts the caller's turns and plays a step's lines.
            seat: The agent that holds the call.
            then_targets: The handoffs a group's ``then_target`` names.
            main_instruction: Returns the main agent's prompt as it reads now.
            functions: Builds the Flows functions of a step's node.
            handoff_tool: Wraps one of the agent's handoff tools for a step.
            call: The call's hooks, when known. A step's end is noted as the agent's own.
        """
        self._call = call
        self._llm = llm
        self._context = context
        self._aggregators = aggregators
        self._telemetry = telemetry
        self._side = side
        self._replies = replies
        self._seat = seat
        self._then_targets = then_targets
        self._main_instruction = main_instruction
        self._functions = functions
        self._handoff_tool = handoff_tool
        self._run: ActiveRun | None = None
        self._flow: FlowManager | None = None
        self._held: list[dict[str, Any]] = []
        self._ending: set[asyncio.Task[None]] = set()
        self.task_results: dict[str, str | None] = {}
        self.openings: list[Opening] = []

    @property
    def current(self) -> ActiveRun | None:
        """The run that holds the context now, or None."""
        return self._run

    @property
    def step(self) -> ActiveStep | None:
        """The step running now, or None."""
        return self._run.step if self._run is not None else None

    @property
    def site(self) -> str:
        """The prompt site of the step running now, or ``""``."""
        step = self.step
        return step.step.site if step is not None else ""

    def is_current(self, run: ActiveRun, step: ActiveStep) -> bool:
        """Return True while ``step`` is the live step of the live run."""
        return self._run is run and run.step is step and not step.advanced

    def hold(self, message: dict[str, Any]) -> bool:
        """Keep a message for the main agent while a run holds its context.

        Returns:
            True when the message was kept, False when the main agent's context is live.
        """
        if self._run is None or self._run.snapshot is None:
            return False
        self._held.append(message)
        return True

    def end_soon(self, worker: PipelineWorker) -> None:
        """End the call once the tool's result is in: a Flows handler cannot drain it.

        The call's hooks, when known, note the end as the agent's own.
        """
        if self._call is not None:
            self._call.end_requested(AGENT_ENDED)
        task = asyncio.create_task(worker.end())
        self._ending.add(task)
        task.add_done_callback(self._ending.discard)

    def carry(self, run: ActiveRun, step: ActiveStep) -> None:
        """Keep the caller turn a step answered, when the main agent never saw it."""
        messages = context_messages(self._context)
        if step.step.carries_turn and caller_turns(messages) > step.baseline:
            run.carried = newest_caller_message(messages)

    def set_site(self, site: str) -> None:
        """Tell a ``ModelCallLLMService`` which prompt its next requests come from."""
        if isinstance(self._llm, ModelCallLLMService):
            self._llm.site = site

    def reserve(self, run: DelegateRun, params: FunctionCallParams) -> ActiveRun:
        """Claim the call for one run, before anything is awaited.

        Pipecat runs the calls of one reply in parallel. A second delegate must see this
        run, or it would start its own on the same context.
        """
        self._run = ActiveRun(run=run, call_id=params.tool_call_id, worker=params.pipeline_worker)
        self.task_results[params.tool_call_id] = None
        return self._run

    async def start(
        self, run: ActiveRun, params: FunctionCallParams, *, via: object = None
    ) -> None:
        """Resolve the delegate call, copy the main context, and open the first step."""
        delegate = run.run.delegate
        try:
            noun = "flow" if delegate.kind == "group" else "task"
            await params.result_callback(
                via
                if via is not None
                else ToolStatus(status=f"running the {delegate.name} {noun}").model_dump(),
                properties=FunctionCallResultProperties(run_llm=False),
            )
            await run.worker.flush_pipeline()
            if run.run.announce:
                joins_context(self._llm)
                await run.worker.queue_frame(TTSSpeakFrame(run.run.announce))
            live = list(context_messages(self._context))
            run.snapshot = copy.deepcopy(live)
            seed = tool_pair(live, run.call_id) if via is not None else []
            run.running = [*without_call(live, run.call_id), *seed]
            run.snapshot_tools = self._context.tools
            step = run.run.next_step()
            if step is None:
                await self._finish_empty(run)
                return
            shaped = (
                []
                if step.history in FRESH_HISTORY
                else await self._opening_history(step.task, run.running)
            )
            shaped.extend(m for m in seed if m not in shaped)
            replace_messages(self._context, shaped)
            run.opened_at = len(shaped)
            self._replies.count_caller(context_messages(self._context))
            run.baseline = caller_turns(shaped)
            flow = await self._flow_for(run.worker)
            node = await self._node(run, step)
            run.shape_next = None
            await flow.set_node_from_config(node)
        except BaseException as error:
            if self._run is run:
                logger.exception("task %s could not start", delegate.name)
                await self._abandon(run, answer=isinstance(error, Exception))
            raise

    async def _finish_empty(self, run: ActiveRun) -> None:
        """End a run that has no step: return it, then end, transfer or let the main agent reply."""
        await self._return(run)
        if run.run.ends_call:
            self.end_soon(run.worker)
            return
        if (target := run.run.transfers_to) is not None and await self._transfer(
            run, target, None, said=False
        ):
            return
        await run.worker.queue_frame(LLMRunFrame())

    async def _flow_for(self, worker: PipelineWorker) -> FlowManager:
        """Return the call's one FlowManager, built for its first run.

        One per run would leak: each adds a frame handler to the worker that nothing
        removes.
        """
        if self._flow is None:
            flow = FlowManager(llm=self._llm, context_aggregator=self._aggregators, worker=worker)
            flow.register_action(SHAPE_HISTORY, self._shape_action)
            await flow.initialize()
            self._flow = flow
        return self._flow

    async def _shape_action(self, _action: dict[str, Any], _flow: FlowManager) -> None:
        """Flows' pre-action on every node: cut the history of the step being entered.

        Flows runs it as the node is set, once the call that moved the caller has its
        result recorded and before the node's own messages land.
        """
        if self._run is not None:
            await self._shape_entry(self._run)

    def _leave_flow(self) -> None:
        """Drop a transition the run left pending, so it cannot fire into the next context."""
        if self._flow is not None:
            self._flow._pending_transition = None

    async def _opening_history(self, task: TaskSpec, messages: list[Any]) -> list[Any]:
        """Return what a step opens on of ``messages``. A ``summary`` step asks for it first."""
        if task.history == "summary" and (text := await self._side.summary(messages, None)):
            return [summary_message(text)]
        return shape_history(task, messages)

    async def _shape_entry(self, run: ActiveRun) -> None:
        """Give the step being entered its own history of everything the run did so far.

        The messages the step being left added join ``running`` first. A fresh step's node
        replaces the live context with its own messages, so its additions start at 0.
        It never keeps an earlier finish, which it would read as its own work, nor an
        earlier step's cue, which it would read as the caller. The caller turns the step
        opens on are counted again on what it now reads, as LiveKit counts them once the
        history is set: a ``last_n`` window can keep fewer than the step being left held.
        """
        task, run.shape_next = run.shape_next, None
        messages = context_messages(self._context)
        run.running.extend(messages[run.opened_at :])
        fresh = run.step is not None and run.step.step.history in FRESH_HISTORY
        run.opened_at = 0 if fresh else len(messages)
        if task is None:
            return
        shaped = [
            message
            for message in without_calls_to(
                await self._opening_history(task, run.running), {FINISH_TOOL_NAME}
            )
            if message != STEP_START_MESSAGE
        ]
        replace_messages(self._context, shaped)
        run.opened_at = len(shaped)
        if run.step is not None:
            run.step.step.opened(caller_turns=caller_turns(shaped))

    async def _node(self, run: ActiveRun, step: StepRun) -> NodeConfig:
        """Build one step's node, as it opens, so its prompt reads current values."""
        reset = step.history in FRESH_HISTORY
        current = ActiveStep(
            step, baseline=0 if reset else run.baseline, opening_turn=self._replies.caller_count
        )
        native = self._seat.native
        current.handoffs = [
            self._handoff_tool(run, current, tool)
            for tool in native
            if tool.name in step.task.handoffs
        ]
        current.native = [tool for tool in native if tool.name in step.task.native_tools]
        run.step = current
        run.carried = None
        messages = await self._opening_messages(run, current, reset=reset)
        step.opened(
            caller_turns=caller_turns(messages)
            if reset
            else caller_turns(context_messages(self._context))
        )
        run.shape_next = None if reset else step.task
        self.set_site(step.site)
        current.prompt = self._side.render(step.instructions(), step.site, "step")
        if self._telemetry is not None:
            current.span = self._telemetry.start_task_step(
                task=run.run.delegate.name, step=step.task.name
            )
        return self._config(step, current, messages, self._functions(run, current), reset=reset)

    async def _opening_messages(
        self, run: ActiveRun, current: ActiveStep, *, reset: bool
    ) -> list[dict[str, Any]]:
        """Return the messages a step's node opens with, and queue the lines it says.

        A request must not end on the agent's own line, the step's announce: a provider
        like Gemini answers it with nothing. The cue after it is a developer message, so
        it is no caller turn.
        """
        step = current.step
        messages: list[dict[str, Any]] = []
        seed = newest_caller_message(context_messages(self._context))
        if step.history == "caller_turn" and seed is not None:
            messages.append(seed)
        if (opening := await step.open()) is not None:
            self.openings.append(opening)
        if opening is not None and opening.line is not None:
            messages.extend(opening_messages(opening, opening.line))
            self._replies.play(step, opening.line)
            await run.worker.queue_frame(TTSSpeakFrame(opening.line, append_to_context=False))
        if step.announce:
            await run.worker.queue_frame(TTSSpeakFrame(step.announce, append_to_context=False))
            messages.append({"role": "assistant", "content": step.announce})
        if opening := step.opening_message:
            messages.append({"role": "developer", "content": opening})
        if messages and messages[-1].get("role") == "assistant":
            messages.append({"role": "developer", "content": STEP_START_CUE})
        if reset and not messages and not step.listens:
            messages.append(dict(STEP_START_MESSAGE))
            current.baseline = 1
        if reset:
            run.baseline = 0
        return messages

    @staticmethod
    def _config(
        step: StepRun,
        current: ActiveStep,
        messages: list[dict[str, Any]],
        functions: list[Any],
        *,
        reset: bool,
    ) -> NodeConfig:
        """Return the Flows node of a step: its prompt, its opening messages and its tools."""
        node = NodeConfig(
            name=step.task.name,
            role_message=for_flows(current.prompt),
            task_messages=[
                {**message, "content": for_flows(message["content"])}
                if isinstance(message.get("content"), str)
                else message
                for message in messages
            ],
            functions=functions,
            pre_actions=[{"type": SHAPE_HISTORY}],
            respond_immediately=not step.listens,
        )
        if reset:
            node["context_strategy"] = ContextStrategyConfig(strategy=ContextStrategy.RESET)
        return node

    async def advance(self, run: ActiveRun, step: ActiveStep) -> NodeResult:
        """Move on from an ended step. If that breaks, hand the main agent back."""
        step.advanced = True
        try:
            return await self._move_on(run, step)
        except Exception:
            logger.exception("step %s could not move on", step.step.task.name)
            if self._run is not run:
                return ToolStatus(status="ok").model_dump(), None
            await self._abandon(run)
            return ToolStatus(status="failed").model_dump(), NO_RESPONSE

    async def _move_on(self, run: ActiveRun, step: ActiveStep) -> NodeResult:
        """Open the next step, or return the run to the main agent.

        A receipt line is spoken as a queued frame. When the run completes on it, it is the
        main agent's reply, and no model request follows.
        """
        core = step.step
        run.run.ended(core, said=self._replies.text)
        outcome = core.status
        if self._telemetry is not None and step.span is not None:
            saved = core.outcome.saved if core.outcome else ()
            self._telemetry.end_task_step(
                step.span, outcome=outcome, saved={name: self._plain(name) for name in saved}
            )
        receipt = core.receipt
        ended = ToolStatus(status=outcome).model_dump()
        status = None if core.went_back else ended
        upcoming = run.run.next_step()
        if upcoming is not None:
            if receipt:
                joins_context(self._llm)
                await run.worker.queue_frame(TTSSpeakFrame(receipt))
            return status, await self._node(run, upcoming)
        closing = run.run.closing if run.run.status == "completed" else None
        answers = closing is not None and (
            bool(run.beside) or made_other_calls(run.snapshot or [], run.call_id)
        )
        await shielded(self._return(run, said=closing if answers else None))
        if (target := run.run.transfers_to) is not None:
            handed = await shielded(self._transfer(run, target, closing or receipt, said=answers))
            return ended, NO_RESPONSE if handed else None
        if run.run.ends_call:
            if line := closing or receipt:
                joins_context(self._llm)
                await run.worker.queue_frame(TTSSpeakFrame(line))
            self.end_soon(run.worker)
            return ended, NO_RESPONSE
        if closing is not None and not answers:
            joins_context(self._llm)
            await run.worker.queue_frame(TTSSpeakFrame(closing))
            return ended, NO_RESPONSE
        if receipt:
            await run.worker.queue_frame(TTSSpeakFrame(receipt, append_to_context=False))
        return ended, None

    def _plain(self, name: str) -> Any:
        """Return a saved value's plain form, for the step's span."""
        return self._seat.tasks.state.plain(name)

    async def _transfer(self, run: ActiveRun, target: str, line: str | None, *, said: bool) -> bool:
        """Hand the call to the agent a completed group transfers to, through ``then_targets``.

        The step's last line plays first and joins the main agent's history, which the
        target hears shaped by its handoff. The handler gets params for the group's own call,
        already answered as the run started, so ``hand_off`` leaves that call out. A transfer
        that raises, or hands nothing over, is logged, and the main agent replies instead.

        Args:
            run: The run that completed, already returned to the main agent.
            target: The group's ``then_target``.
            line: The last step's closing line, or None.
            said: The line is already in the main agent's history.

        Returns:
            True once the call is handed over. False leaves the main agent to reply, with
            the line already said.
        """
        tasks = self._seat.tasks
        if line:
            if not said:
                self._context.add_message({"role": "assistant", "content": line})
            await run.worker.queue_frame(TTSSpeakFrame(line, append_to_context=False))
        await run.worker.flush_pipeline()

        async def answered(
            result: object, *, properties: FunctionCallResultProperties | None = None
        ) -> None:
            """Drop the handoff's own answer: the group's call has its result."""

        params = FunctionCallParams(
            function_name=run.run.delegate.name,
            tool_call_id=run.call_id,
            arguments={},
            llm=self._llm,
            pipeline_worker=run.worker,
            context=self._context,
            result_callback=answered,
            app_resources=run.worker.app_resources,
        )
        try:
            await self._then_targets[target](params)
        except Exception:
            logger.warning("the transfer to %s failed", target, exc_info=True)
        if self._seat.tasks is tasks:
            logger.warning("the transfer to %s handed nothing over", target)
            return False
        return True

    async def _return(self, run: ActiveRun, *, said: str | None = None) -> None:
        """Give the main agent back its prompt, messages and tools, with the run's result."""
        await run.worker.queue_frame(
            LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=self._main_instruction()))
        )
        await run.worker.flush_pipeline()
        self.set_site("")
        tail = [{"role": "assistant", "content": said}] if said else []
        self._restore(run, tail)
        self._run = None
        self._leave_flow()

    def _restore(self, run: ActiveRun, tail: list[dict[str, Any]]) -> None:
        """Make the main agent's copy the context again, with the run's result in it.

        A carried caller turn goes right after the run's result, as on LiveKit.
        """
        assert run.snapshot is not None
        carried = [run.carried] if run.carried else []
        result = run.run.result(caller_turns=caller_turns(run.snapshot) + len(carried))
        text = self.task_results[run.call_id] = json.dumps(result, ensure_ascii=False)
        set_tool_reply(run.snapshot, run.call_id, text)
        for call_id, beside in run.beside.items():
            set_tool_reply(run.snapshot, call_id, result_text(beside))
        held, self._held = self._held, []
        replace_messages(
            self._context, [*below_replies(run.snapshot, run.call_id, carried), *held, *tail]
        )
        self._context.set_tools(run.snapshot_tools)

    async def _abandon(self, run: ActiveRun, *, answer: bool = True) -> None:
        """Hand the main agent back a run that broke, as failed.

        The delegate call asked for no reply, so the main agent is asked for one here.
        """
        if self._run is not run:
            return
        step = run.step
        if step is not None and not step.advanced:
            step.advanced = True
            step.step.fail()
            run.run.ended(step.step, said=self._replies.text)
            if self._telemetry is not None and step.span is not None:
                self._telemetry.end_task_step(step.span, outcome="failed", error="abandoned")
        if run.snapshot is None:
            messages = context_messages(self._context)
            result = run.run.result(caller_turns=caller_turns(messages))
            text = self.task_results[run.call_id] = json.dumps(result, ensure_ascii=False)
            set_tool_reply(messages, run.call_id, text)
        else:
            self._restore(run, [])
        self._run = None
        self._leave_flow()
        self.set_site("")
        if answer:
            await run.worker.queue_frame(
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(system_instruction=self._main_instruction())
                )
            )
            await run.worker.queue_frame(LLMRunFrame())

    def end_handed_off(self, run: ActiveRun, step: ActiveStep) -> None:
        """End a run whose step handed the call off. Nothing goes back to the main agent.

        Messages the main agent was given while the run held the context join it now, so
        the new agent's history can keep them.
        """
        step.advanced = True
        held, self._held = self._held, []
        for message in held:
            self._context.add_message(cast(Any, message))
        run.run.ended(step.step, said=self._replies.text)
        if self._telemetry is not None and step.span is not None:
            self._telemetry.end_task_step(step.span, outcome="handed_off")
        run.run.result(caller_turns=caller_turns(context_messages(self._context)))
        self.task_results[run.call_id] = None
        self._run = None
        self._leave_flow()
        self.set_site("")
