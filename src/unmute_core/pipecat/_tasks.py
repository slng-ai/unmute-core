"""Tasks and task groups on Pipecat, with a Flows node for each step.

It began as unmute@4a738ad templates/pipecat_v1/agents.py.tmpl. The step rules
now live in ``unmute_core.tasks``. What is left here is Pipecat glue:

1. The main agent is not a Flows node. It keeps its own context and tools.
2. A delegate call starts a ``DelegateRun``. Its context is copied, and each ``StepRun``
   becomes a node on the call's one ``FlowManager``.
3. When the run ends, the copy comes back with the run's result as the delegate's reply.
4. ``hand_off`` gives the call to another main agent: its prompt, its tools and its tasks,
   on the same context, with what its ``HandoffContext`` keeps. A step whose task lists
   the handoff tool offers it too, and calling it there ends the run first. A group whose
   ``then`` is ``transfer`` runs the handoff ``then_targets`` names once it completes.

Pipecat 1.12 behaviours this module works around:

* Flows defaults ``cancel_on_interruption`` to False, an async tool. Its result can arrive
  as a developer message, and its guidance is added to the system prompt. Every node
  function sets it to True, so every tool is an ordinary one.
* The flush probe waits behind queued audio (``base_output.py`` sync frames), so a
  delegate flushes before it queues its announce line.
* Flows fills ``{{ key }}`` in a node's prompts from its own state and fails on a missing
  key. Everything handed to Flows is escaped.
* An abandoned run's pending transition would still fire, so leaving a run clears
  ``FlowManager._pending_transition``.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Any, cast

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.flows import (
    NO_RESPONSE,
    FlowManager,
)
from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    FunctionCallResultProperties,
    LLMContextFrame,
    LLMRunFrame,
    LLMSetToolsFrame,
    LLMUpdateSettingsFrame,
    ManuallySwitchServiceFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
)
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings

from unmute_core.agent import AgentSpec
from unmute_core.agent._setup import agent_name, agent_state, agent_tasks, check_agent
from unmute_core.call import CallHooks
from unmute_core.handoff import (
    HandoffContext,
    summary_message,
)
from unmute_core.model import (
    ModelCall,
)
from unmute_core.pipecat._call import before_request
from unmute_core.pipecat._history import (
    caller_turns,
    shaped,
    without_calls_to,
)
from unmute_core.pipecat._model import ModelCallLLMService
from unmute_core.pipecat._realtime import (
    ask,
    fixed_instructions,
    follow_replies,
    is_realtime,
    keep_resend,
    speak,
)
from unmute_core.pipecat._replies import ReplyTracker
from unmute_core.pipecat._requests import RequestHooks, service_model
from unmute_core.pipecat._runs import (
    ActiveRun,
    ActiveStep,
    AgentSeat,
    NodeResult,
    RunLifecycle,
    shielded,
)
from unmute_core.pipecat._runs import for_flows as for_flows
from unmute_core.pipecat._runs import opening_messages as opening_messages
from unmute_core.pipecat._runs import result_text as result_text
from unmute_core.pipecat._services import hide_async_guidance, llm_settings
from unmute_core.pipecat._side import RequestSpans, SideRequests
from unmute_core.pipecat._tools import (
    FlowsHandler,
    FunctionHandler,
    announcer,
    flows_function,
    function_schema,
    joins_context,
    mark_ender,
    report,
    run_tool,
    say_line,
    tool_dicts,
)
from unmute_core.pipecat._voices import VoiceSwitcher
from unmute_core.realtime import check_realtime, check_realtime_handoff, realtime_instructions
from unmute_core.state import CallState
from unmute_core.tasks import (
    FINISH_TOOL_NAME,
    GO_BACK_TOOL_NAME,
    REFUSED_BUSY,
    AfterReply,
    CallTasks,
    FinishCheck,
    Opening,
    OpeningTool,
    TaskEventSink,
    TaskPlan,
    ToolStatus,
)
from unmute_core.tasks._opening import check_opening, open_agent
from unmute_core.telemetry import Telemetry
from unmute_core.tools import (
    SAVE_VALUES_TOOL,
    ToolError,
    ToolRegistry,
    ToolSpec,
    check_tool_names,
    free_values,
    is_values,
    replies_to,
    values_block,
    values_tool,
)

logger = logging.getLogger(__name__)

type Handler = FunctionHandler

ALREADY_HANDLED = MappingProxyType(ToolStatus(status="already handled").model_dump())
"""What a call answers when its step has already ended."""
HANDOFF_LINE_SECONDS = 30.0
"""The longest a handoff waits for its line to play (Unmute's ``HANDOFF_CUE_TIMEOUT_SECS``)."""


def _warn_unasked(tasks: CallTasks, side: SideRequests, summarizer: ModelCall | None) -> None:
    """Log once each way a task needs a model request the call's LLM cannot answer."""
    if side.can_ask:
        return
    if summarizer is None and tasks.plan.record and tasks.plan.tasks:
        logger.warning(
            "the plan records values, so each step runs a finish check, and this call's LLM "
            "cannot be asked outside the pipeline: the steps go on as the model leaves them. "
            "Give the AgentSpec a summarizer model, or use an OpenAI-compatible or Google "
            "service, or a ModelCallLLMService."
        )
    if summarizer is None and any(t.history == "summary" for t in tasks.plan.tasks.values()):
        logger.warning(
            "a task has history 'summary', and this call has no summarizer and an LLM that "
            "cannot be asked outside the pipeline: the next step hears the chat. Give the "
            "AgentSpec a summarizer model."
        )


def _check_handoffs(
    handoffs: Sequence[str], native_tools: Sequence[FunctionSchema], *, new_agent: bool = False
) -> None:
    """Refuse a handoff tool name that is not one of the native tools.

    Raises:
        ValueError: The message names the handoff and the fix.
    """
    names = {tool.name for tool in native_tools}
    if stray := [name for name in handoffs if name not in names]:
        call = "hand_off's AgentSpec(handoffs=...)" if new_agent else "AgentSpec(handoffs=...)"
        owner = "the new agent's" if new_agent else "the agent's"
        raise ValueError(
            f"{call} names '{stray[0]}', which is not one of {owner} native_tools. "
            "Pass its handoff tool in native_tools."
        )


def _free_values(tasks: CallTasks) -> list[str]:
    """Return the State values the main agent can save: no task saves them, no start sets them."""
    return free_values(tasks.state, tasks.plan.spec.owners())


def _check_tool_names(
    tasks: CallTasks, native_tools: Sequence[FunctionSchema], save_values: bool
) -> None:
    """Refuse an agent that offers one tool name twice.

    The names come from the registry tools the main agent sees, the task and group
    delegates, ``save_values`` when a value is free for the main agent, and the native tools.

    Raises:
        ValueError: The message names the tool and the sources that clash.
    """
    own = [
        tool["name"]
        for tool in tool_dicts(tasks.plan.tools)
        if tasks.plan.visible_to_main(tool["name"])
    ]
    check_tool_names(
        [
            ("a registry tool", own),
            ("a task or group", [tool["name"] for tool in tasks.delegate_tools()]),
            ("save_values", [SAVE_VALUES_TOOL] if save_values and _free_values(tasks) else []),
            ("a native tool", [tool.name for tool in native_tools]),
        ]
    )


def _check_tasks(
    tasks: CallTasks, native_tools: Sequence[FunctionSchema], save_values: bool
) -> None:
    """Refuse what only Pipecat cannot run: a task model, or a tool the agent lacks.

    Also refuse a tool name the agent offers twice. ``check_agent`` holds the rules every
    framework shares.

    Args:
        tasks: The agent's tasks.
        native_tools: The agent's own tools.
        save_values: Whether the agent has the ``save_values`` tool when a value is free.

    Raises:
        ValueError: The message names the task and the handoff, or the model, or the tool.
    """
    _check_tool_names(tasks, native_tools, save_values)
    names = {tool.name for tool in native_tools}
    for task in tasks.plan.tasks.values():
        if task.model is not None:
            raise ValueError(
                f"task '{task.name}' runs on the model '{task.model}', and Pipecat runs every "
                "task on the agent's LLM service. Leave model out, or run the agent on LiveKit."
            )
        tasks.plan.check_handoff_tools(names, fix="Pass it in native_tools.", task=task)
        tasks.plan.check_native_tools(names, task=task)


def _is_values(message: object) -> bool:
    """Return True for the system message that lists the call's values."""
    return isinstance(message, dict) and is_values(message.get("role"), message.get("content"))


class UnmuteFlows:
    """The tasks of one call on Pipecat.

    Build one per call, then ``install`` it before the pipeline runs. Give the LLM
    ``main_instruction()`` as its system instruction.

    A native Pipecat LLM service gets its prompts filled here. A ``ModelCallLLMService``
    gets them unfilled, with the prompt site of each request. A handoff can give the next
    agent its own model call, on a ``ModelCallLLMService``, and its own voice, with ``tts=``.

    Attributes:
        task_results: For ``TelemetryObserver``: the result the main agent reads for each
            task call, by call ID. None while the task runs. The call's first result only
            says the task is running.
        closing_results: For ``TelemetryObserver``: what a step's closing tool returned, by
            tool name. The model reads only how the step ended.
        openings: For ``TelemetryObserver``: the tools steps ran in code as they opened,
            which no frame shows.
        request_spans: Set by ``TelemetryObserver``: traces a finish check, a model request
            no frame shows, in the turn it belongs to.
    """

    def __init__(
        self,
        spec: AgentSpec,
        *,
        llm: LLMService[Any],
        context: LLMContext,
        aggregators: LLMContextAggregatorPair,
        state: CallState | None = None,
        call: CallHooks | None = None,
        telemetry: Telemetry | None = None,
        native_tools: Sequence[FunctionSchema] = (),
        then_targets: Mapping[str, Handler] | None = None,
        tts: VoiceSwitcher | None = None,
    ) -> None:
        """Hold the first agent of the call and the Pipecat objects it runs on.

        Args:
            spec: The first agent: its prompt, tools, tasks, models and handoff tools. Its
                ``model`` needs ``llm`` to be a ``ModelCallLLMService``. Its ``summarizer``
                also answers each finish check when the call's LLM cannot be asked. Without
                one, a ``ModelCallLLMService`` or an OpenAI-compatible service is asked, and
                any other LLM gets what was said instead, and no finish check.
            llm: The LLM service, shared by every agent and every step.
            context: The shared context.
            aggregators: The context's aggregator pair.
            state: The call state. None takes the state of ``call``, else one that keeps no
                values of its own.
            call: The call's hooks. Each model request waits for the tools on events, a
                main-agent request ends with their notes, and steps read the call-start
                results.
            telemetry: Records a ``task_step`` span per step, and each task event.
            native_tools: Pipecat tools the first agent gets as they are, each with its
                handler, like a handoff or an end-call tool. A step gets the ones its task
                lists in ``handoffs`` or ``native_tools``.
            then_targets: The agents a group whose ``then`` is ``transfer`` hands the call
                to, by its ``then_target``: the handler of a handoff to that agent, which
                calls ``hand_off``. It covers every agent of the call, since the agents
                the call is handed to transfer through it too.
            tts: The call's voices, a ``VoiceSwitcher`` of TTS services in the pipeline.
                Its first service speaks first. ``hand_off(tts=...)`` switches to another.

        Raises:
            ValueError: If the spec has tasks or ``save_values`` and there is no state, a
                task lists a handoff that is not one of ``native_tools``, or names a model,
                a group transfers to an agent ``then_targets`` lacks, the spec's
                ``handoffs`` names a tool ``native_tools`` lacks, the spec has a model and
                ``llm`` is no ``ModelCallLLMService``, or ``tts`` is a plain
                ``ServiceSwitcher``.
        """
        if tts is not None and not isinstance(tts, VoiceSwitcher):
            raise ValueError(
                "UnmuteFlows(tts=...) takes a VoiceSwitcher. A plain ServiceSwitcher holds "
                "every pipeline flush until it times out, so each handoff would hang."
            )
        state = agent_state(state, call)
        self._llm = llm
        self._call = call
        self._then_targets = dict(then_targets or {})
        self._first_model = llm.model_call if isinstance(llm, ModelCallLLMService) else None
        self._check_model(spec.model, "UnmuteFlows(spec) with a model")
        check_agent(
            spec, state, call=call, then_targets=self._then_targets, models=self._models(spec)
        )
        tasks = cast("CallTasks", agent_tasks(spec, state, call, always=True))
        self.spec = spec
        self._seat = AgentSeat(tasks, list(native_tools))
        tasks.activate()
        self._save_values = spec.save_values
        self._main = self._main_registry()
        self._instructions = spec.instructions
        hide_async_guidance(llm)
        self._context = context
        self._aggregators = aggregators
        self._telemetry = telemetry
        self._name = agent_name(spec, tasks.state) or "main"
        self._voices = tts
        self._model: ModelCall | None = None
        self._use_model(spec.model)
        self._own(spec.summarizer)
        _check_tasks(tasks, self._native, spec.save_values)
        _check_handoffs(spec.handoffs, self._native)
        self._realtime = is_realtime(llm)
        self._withheld = frozenset() if self._realtime else frozenset(spec.handoffs)
        self._opening_turn: int | None = 0 if self._withheld else None
        self._check_realtime(tasks)
        self._sinks: list[TaskEventSink] = [telemetry.event] if telemetry is not None else []
        for sink in self._sinks:
            tasks.listen(sink)
        if call is not None and isinstance(llm, ModelCallLLMService):
            llm.failure_is_final = True
        self._requests = (
            RequestHooks(llm, call, site=lambda: self.site)
            if call is not None and not self._realtime
            else None
        )
        self._replies = ReplyTracker()
        self._side = SideRequests(
            llm, tasks.state, call=call, requests=self._requests, summarizer=spec.summarizer
        )
        _warn_unasked(tasks, self._side, spec.summarizer)
        self._runs = RunLifecycle(
            llm=llm,
            context=context,
            aggregators=aggregators,
            telemetry=telemetry,
            side=self._side,
            replies=self._replies,
            seat=self._seat,
            then_targets=self._then_targets,
            main_instruction=self.main_instruction,
            functions=self._node_functions,
            handoff_tool=self._step_handoff,
            call=call,
        )
        self._instructing: set[asyncio.Task[None]] = set()
        self._decisions: set[asyncio.Task[None]] = set()
        # ponytail: the observer pops this; without one it keeps one entry per call
        self.closing_results: dict[str, Any] = {}

    @property
    def tasks(self) -> CallTasks:
        """The task runs of the agent that holds the call."""
        return self._seat.tasks

    def listen(self, sink: TaskEventSink) -> None:
        """Send every later task event of the call to ``sink``, of each agent handed it too.

        Args:
            sink: Receives each task event, like ``DevRecorder.task_event``.
        """
        self._sinks.append(sink)
        self._tasks.listen(sink)

    @property
    def state(self) -> CallState:
        """The call state."""
        return self._tasks.state

    def _models(self, spec: AgentSpec) -> tuple[ModelCall | None]:
        """Return the model call an agent's requests go through: its own, else the service's."""
        return (spec.model if spec.model is not None else self._first_model,)

    def _check_model(self, model: ModelCall | None, where: str) -> None:
        """Refuse an agent's own model on an LLM service that runs only one.

        Raises:
            ValueError: If ``model`` comes without a ``ModelCallLLMService``.
        """
        if model is not None and not isinstance(self._llm, ModelCallLLMService):
            raise ValueError(
                f"{where} needs the call's LLM to be a ModelCallLLMService. A native Pipecat "
                "service runs one model: wrap the models in ModelCalls."
            )

    def _own(self, model: ModelCall | None) -> None:
        """Have the call's LLM service close ``model`` with its own."""
        if model is not None and isinstance(self._llm, ModelCallLLMService):
            self._llm.owned.append(model)

    def _use_model(self, model: ModelCall | None) -> None:
        """Send the next requests through an agent's model: its own, else the service's."""
        if not isinstance(self._llm, ModelCallLLMService) or self._first_model is None:
            return
        chosen = model or self._first_model
        self._llm.model_call = chosen
        self._model = None if chosen is self._first_model else chosen
        if self._model is not None and self._model not in self._llm.switched:
            self._llm.switched.append(self._model)

    @property
    def llm(self) -> LLMService[Any]:
        """The call's LLM service, which the flows run against."""
        return self._llm

    @property
    def model(self) -> ModelCall | None:
        """The own model call of the agent that holds the call, or None on the service's."""
        return self._model

    @property
    def site(self) -> str:
        """The prompt site of the next model request: ``task:<name>`` in a step, else ``""``."""
        return self._runs.site

    @property
    def request_spans(self) -> RequestSpans | None:
        """Set by ``TelemetryObserver``: how a model request no frame shows is traced."""
        return self._side.request_spans

    @request_spans.setter
    def request_spans(self, spans: RequestSpans | None) -> None:
        """Set how a model request no frame shows is traced."""
        self._side.request_spans = spans

    @property
    def task_results(self) -> dict[str, str | None]:
        """For ``TelemetryObserver``: the result the main agent reads for each task call."""
        return self._runs.task_results

    @property
    def openings(self) -> list[Opening]:
        """For ``TelemetryObserver``: the tools steps ran in code as they opened."""
        return self._runs.openings

    @property
    def _tasks(self) -> CallTasks:
        """The tasks of the agent that holds the call."""
        return self._seat.tasks

    @property
    def _plan(self) -> TaskPlan:
        """The plan of the agent that holds the call."""
        return self._seat.tasks.plan

    @property
    def _native(self) -> list[FunctionSchema]:
        """The Pipecat tools of the agent that holds the call."""
        return self._seat.native

    @property
    def _run(self) -> ActiveRun | None:
        """The run that holds the context now, or None."""
        return self._runs.current

    def _check_realtime(self, tasks: CallTasks) -> None:
        """Refuse a realtime model this agent cannot run on, and have it follow replies."""
        if not self._realtime:
            return
        call = self._call
        check_realtime(
            fixed_instructions=fixed_instructions(self._llm),
            tasks=bool(tasks.delegate_tools()),
            save_values=self._save_values,
            hooks=call.hooks if call is not None else None,
        )
        follow_replies(self._llm)

    def _main_registry(self) -> ToolRegistry:
        """Return the main agent's tools: the plan's, and the save tool with ``save_values``.

        The save tool is left out when every value belongs to a task or is set at call start.
        """
        registry = self._plan.tools
        if self._save_values and free_values(self.state, self._plan.spec.owners()):
            saves = values_tool(self.state, owners=self._plan.spec.owners())
            registry = registry.with_tools(saves)
        if self._call is not None:
            self._call.watch(registry)
        return registry

    def _show_values(self, context: LLMContext) -> None:
        """End a main-agent request with the current values. A step's request gets none.

        A State with no values has no block.
        """
        kept = [message for message in context.get_messages() if not _is_values(message)]
        if self._runs.current is None and self.state.names:
            block = values_block(self.state, owners=self._plan.spec.owners())
            kept.append({"role": "system", "content": block})
        context.set_messages(cast(Any, kept))

    async def _refresh_main(self) -> None:
        """Give a native LLM the main prompt as it reads now, so a value saved since shows.

        It runs inside the LLM's own frame handler, before the request goes out, so it
        sets the prompt at once instead of queueing a settings frame behind the request.
        """
        try:
            prompt = self.main_instruction()
            if prompt != llm_settings(self._llm).system_instruction:
                await self._llm._update_settings(LLMSettings(system_instruction=prompt))
        except Exception:
            logger.exception("the main prompt could not be refreshed; the request keeps the last")

    def main_instruction(self) -> str:
        """Return the main agent's prompt as the LLM reads it now.

        A realtime model reads one text: the prompt, then the notes of the tools on events
        and, with ``save_values``, the values.
        """
        if not self._realtime:
            return self._side.render(self._instructions, "")
        call = self._call
        values = self._save_values
        return realtime_instructions(
            self._instructions,
            self.state,
            hooks=call.hooks if call is not None else None,
            model=service_model(self._llm),
            blocks=(
                call.notes_block() if call is not None else None,
                values_block(self.state, owners=self._plan.spec.owners())
                if values and self.state.names
                else None,
            ),
        )

    def main_tools(self) -> list[dict[str, Any]]:
        """Return the main agent's tools: its own, then one per task or group."""
        tools = [
            {**tool, "description": self._tasks.via_description(tool["name"], tool["description"])}
            if tool["name"] in self._plan.via_tasks
            else tool
            for tool in tool_dicts(self._main)
            if self._plan.visible_to_main(tool["name"])
        ]
        return [*tools, *self._tasks.delegate_tools()]

    def install(self) -> None:
        """Set the main agent's tools, each with its handler, and watch the LLM's frames.

        Pipecat registers a handler from the tools the context advertises, and rebinds it
        when a node advertises another one for the same name. So the main agent's tools
        and each node's carry their own handlers, and a step never reaches a main tool.

        A realtime model gets the main instructions here, and again after each save.
        """
        self._context.set_tools(self._main_schemas())
        if self._realtime:
            self._instruct()
        if self._realtime and not fixed_instructions(self._llm):
            self.state.listen(lambda _event: self._instruct())
        if self._realtime and self._call is not None:
            self._llm.add_event_handler("on_after_push_frame", self._heard)
        if not self._realtime:
            self._llm.add_event_handler("on_before_process_frame", self._on_process)
        self._llm.add_event_handler("on_before_push_frame", self._on_push)

    async def _heard(self, _llm: object, frame: Frame) -> None:
        """Run the tools on events for a caller's words, beside the realtime model's reply."""
        if isinstance(frame, TranscriptionFrame):
            task = asyncio.create_task(self._caller_spoke(frame.text))
            self._instructing.add(task)
            task.add_done_callback(self._instructing.discard)

    async def _caller_spoke(self, text: str) -> None:
        """Wait for the tools on events, then send the instructions with their notes."""
        assert self._call is not None  # ponytail: _heard is added only with a call
        await self._call.before_request(text)
        if not fixed_instructions(self._llm):
            self._instruct()

    def _instruct(self) -> None:
        """Send a realtime model its instructions as they read now, after the frames queued."""
        frame = LLMUpdateSettingsFrame(
            delta=LLMSettings(system_instruction=self.main_instruction())
        )
        task = asyncio.create_task(self._llm.queue_frame(frame))
        self._instructing.add(task)
        task.add_done_callback(self._instructing.discard)
        keep_resend(self._llm, task)

    async def hand_off(
        self,
        params: FunctionCallParams,
        spec: AgentSpec,
        *,
        native_tools: Sequence[FunctionSchema] = (),
        tts: FrameProcessor | None = None,
        announce: str | None = None,
        context: HandoffContext | None = None,
        opens_with: OpeningTool | None = None,
    ) -> None:
        """Give the call to another main agent. Call it from the handoff tool's handler.

        The agent hears what ``context`` keeps, and replies first. Its tools and tasks
        replace this agent's, on this call's state. A task this agent started beside the
        handoff is refused, and a handoff beside a running task is refused, unless that
        task's step made it. The line is said in this agent's voice, and a summary is written
        by this agent's summarizer or model. The new agent's model, voice and summarizer take
        over after both.

        Args:
            params: The handoff call's params. ``hand_off`` reports its result.
            spec: The agent that takes the call. Its ``model`` needs the call's LLM to be a
                ``ModelCallLLMService``. None there: the model call the service was built
                with. Its ``handoffs`` are left out of its first reply, so it cannot hand the
                call straight back. They come back once the caller speaks. A realtime model
                keeps every tool.
            native_tools: The new agent's Pipecat tools, like its own handoffs.
            tts: The new agent's voice, one of the services of ``UnmuteFlows(tts=...)``.
                None: that switcher's first service.
            announce: A line said first, while the handoff call still runs. The handoff
                waits for it to play.
            context: What the new agent hears. A summary is asked for while the line
                plays. None keeps what was said, without tool calls.
            opens_with: A tool of the new agent's own, run in code before its first reply.
                When a line fills from the result, the line is that reply, with no model
                request. Not on a realtime model.

        Raises:
            ValueError: If the spec lists a handoff that is not one of ``native_tools``,
                a task names a model, or transfers to an agent ``then_targets`` lacks, or
                if the spec has a model without a ``ModelCallLLMService``, or ``tts`` is not
                one of the call's voices, or ``opens_with`` names a tool the spec lacks or
                the model is a realtime one.
        """
        self._check_switch(spec, tts, native_tools)
        holding = self._tasks
        tasks = cast("CallTasks", agent_tasks(spec, self.state, self._call, always=True))
        try:
            if self._realtime:
                fixed = fixed_instructions(self._llm)
                check_realtime_handoff(context or HandoffContext(), fixed_instructions=fixed)
                check_realtime(fixed_instructions=fixed, tasks=bool(tasks.delegate_tools()))
            check_agent(
                spec,
                self.state,
                call=self._call,
                then_targets=self._then_targets,
                models=self._models(spec),
            )
            _check_tasks(tasks, native_tools, spec.save_values)
            check_opening(opens_with, tasks.plan.tools, realtime=self._realtime)
        except ValueError:
            holding.activate()
            raise
        run = self._runs.current
        step = run.step if run is not None else None
        if run is not None and (step is None or not step.handing_off):
            holding.activate()
            await params.result_callback(
                dict(REFUSED_BUSY), properties=FunctionCallResultProperties(run_llm=False)
            )
            return
        previous, name = self._name, agent_name(spec, self.state) or "main"
        if self._telemetry is not None:
            self._telemetry.event("handoff", {"from": previous, "to": name})
        if run is not None and step is not None:
            step.step.hand_off()
            self._runs.end_handed_off(run, step)
        self._seat.tasks, self._seat.native = tasks, list(native_tools)
        tasks.activate()
        self.spec = spec
        self._save_values = spec.save_values
        self._main = self._main_registry()
        self._instructions, self._name = spec.instructions, name
        self._withheld = frozenset() if self._realtime else frozenset(spec.handoffs)
        self._opening_turn = None
        self._own(spec.summarizer)
        for sink in self._sinks:
            tasks.listen(sink)
        await shielded(
            self._take_over(
                params,
                announce,
                context or HandoffContext(),
                model=spec.model,
                tts=tts,
                opens_with=opens_with,
            )
        )
        self._side.summarizer = spec.summarizer
        _warn_unasked(tasks, self._side, spec.summarizer)

    def _check_switch(
        self, spec: AgentSpec, tts: FrameProcessor | None, native_tools: Sequence[FunctionSchema]
    ) -> None:
        """Refuse a model, a voice or a handoff tool the call cannot give the new agent.

        Raises:
            ValueError: If one of them cannot be given. The message names the fix.
        """
        self._check_model(spec.model, "hand_off(spec) with a model")
        if tts is not None and (self._voices is None or tts not in self._voices.services):
            raise ValueError(
                "hand_off(tts=...) needs that TTS service in the VoiceSwitcher given as "
                "UnmuteFlows(tts=...), placed in the pipeline after the LLM."
            )
        _check_handoffs(spec.handoffs, native_tools, new_agent=True)

    async def _switch(
        self, worker: PipelineWorker, model: ModelCall | None, tts: FrameProcessor | None
    ) -> None:
        """Give the agent taking the call its model and its voice, before its first reply."""
        self._use_model(model)
        if self._voices is not None:
            voice = tts or self._voices.services[0]
            await worker.queue_frame(ManuallySwitchServiceFrame(service=voice))

    async def _take_over(
        self,
        params: FunctionCallParams,
        announce: str | None,
        context: HandoffContext,
        *,
        model: ModelCall | None,
        tts: FrameProcessor | None,
        opens_with: OpeningTool | None,
    ) -> None:
        """Say the line, answer the handoff, then set the new agent's prompt, tools and history.

        With an opening tool, it runs once the new agent's tools are in place. When its line
        fills, the call, its result and the line join the history, and no model request is
        made.

        The line plays before the call answers, so a strategy that mutes the caller during
        tool calls protects it. It never joins the live context, which only text from a TTS
        service does: it goes into the new agent's history here, the same on every pipeline.
        """
        if self._realtime:
            await self._take_over_realtime(params, announce, tts)
            return
        worker = params.pipeline_worker
        source = self._messages()
        summary = (
            asyncio.create_task(self._side.summary(source, context.summarizer))
            if context.history == "summary"
            else None
        )
        try:
            if announce:
                await self._say_line(worker, announce)
            text = await summary if summary is not None else None
        finally:
            if summary is not None and not summary.done():
                summary.cancel()
        await params.result_callback(
            ToolStatus(status="handed off").model_dump(),
            properties=FunctionCallResultProperties(run_llm=False),
        )
        await worker.flush_pipeline()
        said = [{"role": "assistant", "content": announce}] if announce else []
        self._set_messages(
            [summary_message(text)]
            if text
            else shaped(
                [*without_calls_to(self._messages(), {params.function_name}), *said],
                context.history,
                max_messages=context.max_messages,
                include_tool_calls=context.include_tool_calls,
                tools={tool["name"] for tool in self.main_tools()}
                | {tool.name for tool in self._native},
            )
        )
        self._replies.count_caller(self._messages())
        self._opening_turn = self._replies.caller_count
        self._context.set_tools(self._main_schemas())
        await self._switch(worker, model, tts)
        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=self.main_instruction()))
        )
        if not await self._open(worker, opens_with):
            await worker.queue_frame(LLMRunFrame())

    async def _open(self, worker: PipelineWorker, spec: OpeningTool | None) -> bool:
        """Run the new agent's opening tool, and say its line. Return True when said."""
        if spec is None:
            return False
        tasks = self._tasks
        opening = await open_agent(
            spec, self._main, tasks.state, tasks.spoken_lines, hooks=tasks.hooks
        )
        if opening is None:
            return False
        self._runs.openings.append(opening)
        if opening.line is None:
            return False
        self._set_messages([*self._messages(), *opening_messages(opening, opening.line)])
        await worker.queue_frame(TTSSpeakFrame(opening.line, append_to_context=False))
        return True

    async def _take_over_realtime(
        self, params: FunctionCallParams, announce: str | None, tts: FrameProcessor | None
    ) -> None:
        """Hand a realtime model's call over: the line, then new instructions and tools.

        The model keeps the whole conversation on its server. The settings and the tools
        go to the model in order, and the new agent replies once the tools have reached it.
        """
        if announce:
            await speak(self._llm, announce, wait=True)
        await params.result_callback(
            ToolStatus(status="handed off").model_dump(),
            properties=FunctionCallResultProperties(run_llm=False),
        )
        await self._switch(params.pipeline_worker, None, tts)
        schemas = self._main_schemas()
        self._context.set_tools(schemas)
        reached = asyncio.Event()

        async def processed(_llm: object, _frame: Frame, _direction: object) -> None:
            """Note that the new tools reached the model."""
            reached.set()

        prompt = self.main_instruction()
        await self._llm.queue_frame(
            LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=prompt))
        )
        await self._llm.queue_frame(LLMSetToolsFrame(tools=schemas), callback=processed)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(reached.wait(), timeout=HANDOFF_LINE_SECONDS)
        await ask(self._llm, None)

    async def _say_line(self, worker: PipelineWorker, line: str) -> None:
        """Speak a line, kept out of the context, and wait until the bot stops speaking it.

        It waits at most ``HANDOFF_LINE_SECONDS``. A line cut off counts as played.
        """
        if self._realtime:
            await speak(self._llm, line, wait=True)
            return
        with self._replies.awaiting_line() as played:
            await worker.queue_frame(TTSSpeakFrame(line, append_to_context=False))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(played.wait(), timeout=HANDOFF_LINE_SECONDS)

    def _main_schemas(self) -> ToolsSchema:
        """Return the main agent's tools, each with its handler, then its native tools.

        The first reply after a handoff leaves out the agent's handoff tools.
        """
        delegates = {delegate.name for delegate in self._plan.delegates()}
        schemas = [
            function_schema(
                tool,
                self._main_handler(tool["name"], delegates),
                cancel_on_interruption=not self._main.keeps_running(tool["name"]),
            )
            for tool in self.main_tools()
        ]
        native = [tool for tool in self._native if tool.name not in self._withheld]
        return ToolsSchema(standard_tools=[*schemas, *native])

    def _main_handler(self, name: str, delegates: set[str]) -> FunctionHandler:
        """Return the handler of one of the main agent's tools."""
        if name in delegates:
            return self._delegate_handler(name)
        if name in self._plan.via_tasks:
            return self._via_handler(name)
        return self._main_tool_handler(name)

    def add_main_message(self, message: dict[str, Any]) -> None:
        """Add a message to the main agent's context, held back while a run owns it."""
        if not self._runs.hold(message):
            self._context.add_message(cast(Any, message))

    def _beside(self, params: FunctionCallParams, tasks: CallTasks) -> FunctionCallParams:
        """Return a main call's params whose result is kept if a run holds the context.

        A call made beside a delegate in the same reply can resolve after the run copied
        the main context. Its result is kept for the restore, and asks for no reply: the
        main agent answers after the run. A call of an agent that has handed off asks for
        no reply either: the new agent speaks.
        """
        report = params.result_callback

        async def result_callback(
            result: object, *, properties: FunctionCallResultProperties | None = None
        ) -> None:
            """Keep the result for the main agent while a run holds the context."""
            run = self._runs.current
            final = properties is None or properties.is_final
            if run is not None and final and params.tool_call_id != run.call_id:
                run.beside[params.tool_call_id] = result
                properties = replace(properties or FunctionCallResultProperties(), run_llm=False)
            elif tasks is not self._tasks:
                properties = replace(properties or FunctionCallResultProperties(), run_llm=False)
            await report(result, properties=properties)

        return replace(params, result_callback=result_callback)

    def _messages(self) -> list[Any]:
        """Return the live context's messages as plain dicts and provider objects."""
        return cast(list[Any], self._context.get_messages())

    def _set_messages(self, messages: list[Any]) -> None:
        """Replace the live context's messages."""
        self._context.set_messages(cast(Any, messages))

    def _turns(self) -> int:
        """Return the caller turns in the live context."""
        return caller_turns(self._messages())

    async def _on_process(self, _llm: object, frame: Frame) -> None:
        """Prepare each model request, and note how its frames play out."""
        if isinstance(frame, LLMContextFrame):
            await self._prepare(frame)
        self._replies.observe(frame)

    async def _prepare(self, frame: LLMContextFrame) -> None:
        """Ready a model request: hooks, prompt, values and the tools its agent offers now."""
        runs = self._runs
        if self._requests is not None:
            await self._requests.give_back()
        if self._call is not None:
            await before_request(self._call, frame.context, main=runs.current is None)
        if runs.current is None and isinstance(self._llm, LLMService):
            await self._refresh_main()
        if self._save_values:
            self._show_values(frame.context)
        self._replies.note_request(frame.context.get_messages())
        if self._withheld and runs.current is None:
            self._end_opening(frame.context)
        self._offer_step_tools(frame.context)
        if self._requests is not None:
            await self._requests.before(frame)

    def _end_opening(self, context: LLMContext) -> None:
        """Offer the handoff tools again on the first request after the caller speaks.

        The handoff took the caller turn the opening answers. The opening's tool follow-ups
        answer that same turn, so they still leave the handoffs out, as on LiveKit. An
        opening cut off before its request leaves nothing behind: the next request answers
        the caller's new turn, with every tool.
        """
        if self._opening_turn is None or self._replies.caller_count > self._opening_turn:
            self._withheld, self._opening_turn = frozenset(), None
            context.set_tools(self._main_schemas())

    async def settle(self) -> list[str]:
        """Make what the call's end decides: the saves steps held. Call it once, as the call ends.

        Call it when the caller has hung up, before the pipeline ends (``EndFrame``) and
        the telemetry closes, so a decision's model request is traced in the last turn.

        Returns:
            The tasks whose held saves were made.
        """
        return await self.state.settle_held()

    def _hold_save(self, step: ActiveStep) -> None:
        """Keep the save a step refused before the caller spoke, for the call's end."""
        heard = self._replies.caller_count

        async def decide(check: FinishCheck) -> dict[str, Any] | None:
            """Ask the step about its held save, on the chat the call has now."""
            decided = await self._side.check(step.step, check, "", self._messages())
            return None if decided is None else decided[1]

        step.step.hold_save(
            caller_spoke_since=lambda: self._replies.caller_count > heard, decide=decide
        )

    def _offer_step_tools(self, context: LLMContext) -> None:
        """Give a step's request the tools it offers now, beyond its node's own.

        Go-back stays out until the step offers it: LiveKit's step does the same, and a
        model that calls it on the step's opening reply would only be refused. The
        handoffs its task lists come last, once the caller has spoken since the step opened:
        its opening reply leaves them out. Pipecat registers their handlers from the tools
        the request carries.
        """
        step = self._runs.step
        if step is None or (
            step.step.go_back_tool() is None and not step.handoffs and not step.native
        ):
            return
        if step.tools is None:
            step.tools = context.tools if isinstance(context.tools, ToolsSchema) else None
        if step.tools is None:
            return
        offered = step.step.go_back_tool() is not None and step.step.offers_go_back(
            caller_turns=caller_turns(context.get_messages())
        )
        context.set_tools(
            ToolsSchema(
                standard_tools=[
                    *(
                        tool
                        for tool in step.tools.standard_tools
                        if offered or tool.name != GO_BACK_TOOL_NAME
                    ),
                    *(step.handoffs if self._replies.caller_count > step.opening_turn else ()),
                    *step.native,
                ],
                custom_tools=step.tools.custom_tools,
            )
        )

    def _step_handoff(
        self, run: ActiveRun, step: ActiveStep, tool: FunctionSchema
    ) -> FunctionSchema:
        """Return one of the agent's handoff tools, as a step offers it.

        The core decides whether the step may hand off now. When it may, the agent's own
        handler runs, and its ``hand_off`` ends the step and the run once its checks pass. A
        handler that raises first leaves the step running. A refusal asks for no reply: the
        other call of the same reply decides what comes next.
        """
        handler = tool.handler
        assert handler is not None

        async def hand_off(params: FunctionCallParams) -> None:
            """End the step on the handoff, or give the model the refusal."""
            refusal = (
                step.step.may_hand_off(alone=self._replies.alone(params.tool_call_id))
                if self._runs.is_current(run, step)
                else dict(ALREADY_HANDLED)
            )
            if refusal is None:
                step.handing_off = True
                try:
                    await cast(Any, handler)(params)
                finally:
                    step.handing_off = False
                return
            await params.result_callback(
                refusal, properties=FunctionCallResultProperties(run_llm=False)
            )

        return FunctionSchema(
            name=tool.name,
            description=tool.description,
            properties=tool.properties,
            required=tool.required,
            handler=hand_off,
        )

    async def _on_push(self, _llm: object, frame: Frame) -> None:
        """Follow each reply: its text, and the end of a step's first answer."""
        step = self._replies.pushed(frame, self._runs.step)
        run = self._runs.current
        if run is not None and step is not None and self._runs.is_current(run, step):
            await self._answered(run, step)
            self._reply_ended(run, step)

    def _reply_ended(self, run: ActiveRun, step: ActiveStep) -> None:
        """Tell the step its reply ended, and send the finish that follows, if any.

        A held finish, or a finish check's, goes out as that reply's own call, after it:
        Pipecat has run the reply's calls by its end frame. It runs as a task, so the
        reply's speech is not held up.
        """
        after = self._replies.finished(step.step)
        if after is None:
            return
        replies = self._replies
        task = asyncio.create_task(self._after(run, step, after, replies.text, replies.request))
        self._decisions.add(task)
        task.add_done_callback(self._decisions.discard)

    async def _after(
        self, run: ActiveRun, step: ActiveStep, after: AfterReply, said: str, request: list[Any]
    ) -> None:
        """Send a held finish, or run a finish check and send its finish."""
        arguments = after.finish
        call_id = f"unmute_finish_{uuid.uuid4().hex[:12]}"
        if after.check is not None:
            decided = await self._side.check(step.step, after.check, said, request)
            if decided is None:
                return
            call_id, arguments = decided
        if arguments is None or not self._runs.is_current(run, step):
            return
        call = FunctionCallFromLLM(
            function_name=FINISH_TOOL_NAME,
            tool_call_id=call_id,
            arguments=arguments,
            context=self._context,
        )
        await cast(Any, self._llm).run_function_calls([call])

    async def _answered(self, run: ActiveRun, step: ActiveStep) -> None:
        """Drop why the step opens as it does from its prompt, once it has answered."""
        if step.answered:
            return
        step.answered = True
        step.step.answered()
        prompt = self._side.render(step.step.instructions(), step.step.site, "step")
        if prompt != step.prompt:
            step.prompt = prompt
            await run.worker.queue_frame(
                LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=prompt))
            )

    @property
    def reply_text(self) -> str:
        """The text of the newest reply: the one streaming now, or the last one."""
        return self._replies.text

    def _main_tool_handler(self, name: str) -> FunctionHandler:
        """Return the handler of a main agent tool."""
        tasks = self._tasks
        registry = self._main

        async def handler(params: FunctionCallParams) -> None:
            """Run the tool."""
            params = self._beside(params, tasks)
            result = await run_tool(
                registry,
                name,
                params.arguments,
                tasks.state,
                started=announcer(params, tasks.lines),
                hooks=tasks.hooks,
            )
            await report(
                params,
                registry.for_model(name, result),
                ends_call=registry.ends_call(name, result),
                hooks=tasks.hooks,
                line=registry.use_result(
                    name, result, tasks.state, tasks.spoken_lines, hooks=tasks.hooks
                )
                if self._runs.current is None and tasks is self._tasks
                else None,
                reply=replies_to(result, spoke=bool(self._replies.text)),
            )

        return mark_ender(handler) if registry.get(name).effect == "ends_conversation" else handler

    def _delegate_handler(self, name: str) -> FunctionHandler:
        """Return the handler of the tool that starts a task or group."""
        tasks = self._tasks

        async def handler(params: FunctionCallParams) -> None:
            """Start the run, or refuse it."""
            params = self._beside(params, tasks)
            if tasks is not self._tasks:
                logger.info("%s did not start: the call was handed off", name)
                await params.result_callback(
                    dict(REFUSED_BUSY), properties=FunctionCallResultProperties(run_llm=False)
                )
                return
            if (running := self._runs.current) is not None:
                logger.info("%s did not start: %s is running", name, running.run.delegate.name)
                await params.result_callback(
                    dict(REFUSED_BUSY), properties=FunctionCallResultProperties(run_llm=False)
                )
                return
            started = self._tasks.start(
                name, caller_turns=self._turns(), arguments=params.arguments
            )
            if isinstance(started, dict):
                await params.result_callback(started)
                return
            await shielded(self._runs.start(self._runs.reserve(started, params), params))

        return handler

    def _via_handler(self, name: str) -> FunctionHandler:
        """Return the handler of a main tool whose success enters a task."""
        tasks = self._tasks

        async def handler(params: FunctionCallParams) -> None:
            """Run the tool. On success, enter its task with the result.

            A result that enters no task is saved and answered as any main tool's is.
            """
            params = self._beside(params, tasks)
            result = await run_tool(
                tasks.plan.tools,
                name,
                params.arguments,
                tasks.state,
                started=announcer(params, tasks.lines),
                hooks=tasks.hooks,
            )
            shown = tasks.plan.tools.for_model(name, result)
            if self._runs.current is not None or tasks is not self._tasks:
                await params.result_callback(
                    shown, properties=FunctionCallResultProperties(run_llm=False)
                )
                return
            started = self._tasks.enter_via(name, result, caller_turns=self._turns())
            if started is None:
                line = tasks.plan.tools.use_result(
                    name, result, tasks.state, tasks.spoken_lines, hooks=tasks.hooks
                )
                if line is None:
                    await params.result_callback(shown)
                else:
                    await report(params, shown, ends_call=False, hooks=tasks.hooks, line=line)
                return
            await shielded(self._runs.start(self._runs.reserve(started, params), params, via=shown))

        return handler

    def _node_functions(self, run: ActiveRun, current: ActiveStep) -> list[Any]:
        """Return the Flows functions of a step's node: its tools, its finish and go-back."""
        step = current.step
        functions: list[Any] = [
            flows_function(
                tool,
                self._step_tool(run, current, tool["name"]),
                cancel_on_interruption=not step.tools.keeps_running(tool["name"]),
            )
            for tool in tool_dicts(step.tools)
        ]
        functions.append(flows_function(step.finish_tool(), self._finish(run, current)))
        if (go_back := step.go_back_tool()) is not None:
            functions.append(flows_function(go_back, self._go_back(run, current)))
        return functions

    def _step_tool(self, run: ActiveRun, step: ActiveStep, name: str) -> FlowsHandler:
        """Return the Flows handler of one of a step's own tools."""

        async def handler(args: dict[str, Any], _flow: FlowManager) -> NodeResult:
            """Run the tool. A closing tool that succeeds ends the step."""
            if not self._runs.is_current(run, step):
                return dict(ALREADY_HANDLED), NO_RESPONSE

            async def started(tool: ToolSpec) -> None:
                """Say the tool's announce line as it starts."""
                if (line := step.step.announcement(tool)) is not None:
                    await say_line(self._llm, tool, line)

            try:
                result = await step.step.call_tool(name, args, started=started)
            except ToolError as error:
                result = {"error": error.message}
            else:
                if step.step.ends_call:
                    self._runs.end_soon(run.worker)
                    return result, NO_RESPONSE
            if not self._runs.is_current(run, step):
                return result, NO_RESPONSE
            if result is None and step.step.closing_result is not None:
                self.closing_results[name] = step.step.closing_result
            if result is None and step.step.done:
                self._runs.carry(run, step)
                return await self._runs.advance(run, step)
            if not replies_to(result, spoke=bool(self._replies.text)):
                return result, NO_RESPONSE
            if result is not None and (line := step.step.reply) is not None:
                self._replies.play(step.step, line)
                joins_context(self._llm)
                await run.worker.queue_frame(TTSSpeakFrame(line))
                return result, NO_RESPONSE
            return result, None

        return handler

    def _finish(self, run: ActiveRun, step: ActiveStep) -> FlowsHandler:
        """Return the Flows handler of the step's finish."""

        async def handler(args: dict[str, Any], _flow: FlowManager) -> NodeResult:
            """Save and end the step, or refuse."""
            if not self._runs.is_current(run, step):
                return dict(ALREADY_HANDLED), NO_RESPONSE
            refusal = await step.step.finish(
                args,
                caller_turns=self._turns(),
                unheard=self._replies.unheard(step.step),
                spoke=bool(self._replies.text),
            )
            if not self._runs.is_current(run, step):
                return dict(ALREADY_HANDLED), NO_RESPONSE
            if refusal is not None:
                self._hold_save(step)
                return {"refused": refusal}, None if step.step.answer_refusal else NO_RESPONSE
            self._runs.carry(run, step)
            return await self._runs.advance(run, step)

        return handler

    def _go_back(self, run: ActiveRun, step: ActiveStep) -> FlowsHandler:
        """Return the Flows handler of the go-back tool."""

        async def handler(args: dict[str, Any], _flow: FlowManager) -> NodeResult:
            """End the step so earlier ones run again, or refuse."""
            if not self._runs.is_current(run, step):
                return dict(ALREADY_HANDLED), NO_RESPONSE
            refusal = step.step.go_back(args, caller_turns=self._turns())
            if refusal is not None:
                return {"refused": refusal}, None
            return await self._runs.advance(run, step)

        return handler
