"""The main agent: its tools, the delegate tools that run tasks, and its model requests.

A delegate tool asks the core for a ``DelegateRun`` and awaits one ``StepTask`` per step,
inside the tool call. When the run ends, the agent gets its own history back, plus the
caller turn the last step answered for it, and reads the run's result as the tool's output.
A later step that runs has answered an earlier step's turn, so that turn is not carried.

A handoff tool returns ``await agent.hand_off(other)``. LiveKit then makes ``other`` the
session's agent. It hears what its ``HandoffContext`` keeps, keeps the call's spans, and
speaks first. A step whose task lists the handoff offers the same tool. Calling it there
ends the step, and the delegate tool returns ``other``. A group whose ``then`` is
``transfer`` hands the call on the same way once it completes, through ``then_targets``.
"""

import asyncio
import logging
import weakref
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from functools import cached_property
from typing import Any

from livekit import rtc
from livekit.agents import NOT_GIVEN, llm
from livekit.agents import tts as livekit_tts
from livekit.agents.llm.chat_context import Instructions
from livekit.agents.voice import (
    Agent,
    AgentSession,
    CloseEvent,
    ModelSettings,
    RunContext,
)

from unmute_core.agent import AgentSpec
from unmute_core.agent._setup import agent_name, agent_state, agent_tasks, check_agent
from unmute_core.call import AGENT_ENDED, CallHooks
from unmute_core.handoff import HandoffContext
from unmute_core.livekit._call import attach_call
from unmute_core.livekit._calls import end_call_now, say
from unmute_core.livekit._close import CloseGuard
from unmute_core.livekit._delegate import DelegateRunner
from unmute_core.livekit._dev import dev_observer
from unmute_core.livekit._gate import HandoffGate
from unmute_core.livekit._history import (
    CallerHistory,
    shaped_history,
    summary_history,
)
from unmute_core.livekit._model import ModelNode, metrics_llm, model_parts
from unmute_core.livekit._realtime import realtime_model, speak, speaks_itself
from unmute_core.livekit._step import announce, raw_tool, tool_text
from unmute_core.livekit._sync import RealtimeSync
from unmute_core.livekit._trace import SessionTrace, traced_tts
from unmute_core.model import close_model
from unmute_core.realtime import check_realtime_handoff
from unmute_core.state import CallState
from unmute_core.tasks import REFUSED_BUSY, OpeningTool, TaskPlan, TaskSpec
from unmute_core.tasks._opening import check_opening, open_agent
from unmute_core.telemetry import Telemetry
from unmute_core.tools import (
    LinePicker,
    SpokenLines,
    ToolRegistry,
    ToolSpec,
    check_tool_names,
    free_values,
    values_block,
    values_tool,
)

logger = logging.getLogger(__name__)

type ThenTarget = Callable[[UnmuteAgent], Awaitable[object]]
"""Hands the call to the agent a group's ``then_target`` names: it gets the agent that ran
the group and returns its ``hand_off``, as a handoff tool does."""


class UnmuteAgent(Agent):
    """A LiveKit agent that runs Unmute's tools, tasks, model call and telemetry.

    Start it like any agent: ``await session.start(agent=UnmuteAgent(spec), room=...)``.
    Build one per call. A subclass that overrides ``on_enter`` calls ``super().on_enter()``
    so the spans start.

    Attributes:
        spec: The agent it runs.
        state: The call state, or None.
        tasks: The call's task runs, or None for an agent without tasks.
    """

    def __init__(
        self,
        spec: AgentSpec,
        *,
        state: CallState | None = None,
        call: CallHooks | None = None,
        telemetry: Telemetry | None = None,
        native_tools: Sequence[llm.Tool | llm.Toolset] = (),
        then_targets: Mapping[str, ThenTarget] | None = None,
        tts: livekit_tts.TTS | None = None,
        **agent_options: Any,
    ) -> None:
        """Build the agent and its tools.

        Args:
            spec: The agent: its prompt, tools, tasks, models and handoff tools. Its name
                is the LiveKit id a ``handoff`` event shows. With no name and no state,
                LiveKit derives the id from the class name.
            state: The call state the prompts and tools read. None takes the state of
                ``call``, so the call keeps one state and one ``session_id``.
            call: The call's hooks: tools on events, the failure line and the call's end.
                Each model request waits for the tools on events. Give every agent of the
                call the same one.
            telemetry: The call's telemetry, new for this call. Its spans start as the
                agent enters, and it closes when the session closes.
            native_tools: LiveKit tools the main agent gets as they are, like
                ``EndCallTool``. A step gets the ones its task lists in ``handoffs`` or
                ``native_tools``. A ``@function_tool`` method of a
                subclass is one too.
            then_targets: The agents a group whose ``then`` is ``transfer`` hands the call
                to, by its ``then_target``. Each one returns ``agent.hand_off(...)``, like
                ``lambda agent: agent.hand_off(UnmuteAgent(BILLING, state=agent.state))``.
            tts: The agent's voice, which its steps speak with too. None uses the session's.
            **agent_options: LiveKit's own ``Agent`` options, passed on as they are, like
                ``chat_ctx``, ``stt``, ``vad``, ``turn_handling`` or ``allow_interruptions``.

        Raises:
            TypeError: If the spec's ``model``, ``summarizer`` or a task model is a LiveKit
                plugin, not a ``ModelCall``: wrap it in ``LiveKitModel``.
            ValueError: If the spec has tasks or ``save_values`` and there is no state, or
                the timezone is unknown, or a task lists a handoff the agent has no tool for,
                a group transfers to an agent ``then_targets`` lacks, or the spec's
                ``handoffs`` names a tool the agent does not have, or a task's tool input
                names a prefetch lookup the call's ``prefetch`` lacks, or a placeholder makes
                a key longer than the model call takes (the router takes 64 characters).
        """
        _refuse_plugins(spec)
        state = agent_state(state, call)
        self._then_targets = dict(then_targets or {})
        check_agent(
            spec,
            state,
            call=call,
            then_targets=self._then_targets,
            models=(spec.model, *spec.task_models.values()),
        )
        self.spec = spec
        self.state = state
        self.tasks = agent_tasks(spec, state, call, always=False)
        self._lines = self.tasks.lines if self.tasks else LinePicker()
        self._spoken = (
            self.tasks.spoken_lines if self.tasks else SpokenLines(spec.language, spec.timezone)
        )
        self._registry, self._owners = _registry_for(
            spec.tasks, spec.tools, state, spec.save_values
        )
        self._save_values = spec.save_values
        self._prompt = spec.instructions
        self._call = call
        if call is not None:
            call.watch(self._registry)
        self._trace = SessionTrace(telemetry, state, call) if telemetry else None
        if self.tasks is not None and telemetry is not None:
            self.tasks.listen(telemetry.event)
        model_call, self._model_llm = model_parts(spec.model)
        self._models = ModelNode(
            state=state,
            model=model_call,
            trace=self._trace,
            call=call,
            metrics_llm=metrics_llm(self._model_llm),
        )
        self._summarizer = spec.summarizer
        self._owned = [spec.model, spec.summarizer, *spec.task_models.values()]
        self._voice = tts
        self._handoffs = spec.handoffs
        self._task_models = dict(spec.task_models)
        self._gate = HandoffGate(self._handoffs, armed=True)
        self._opens_with: OpeningTool | None = None
        self._handed_off = False
        self._settles = False
        self._settling: asyncio.Task[list[str]] | None = None
        self._releasing: asyncio.Task[None] | None = None
        self._held_session: AgentSession | None = None
        self._guard = CloseGuard(
            lambda: [self._settling, self._trace.closing if self._trace else None, self._releasing]
        )
        self._realtime = RealtimeSync(
            self,
            prompt=spec.instructions,
            state=state,
            call=call,
            save_values=spec.save_values,
            owners=self._owners,
            has_delegates=lambda: self.tasks is not None and bool(self.tasks.delegate_tools()),
        )
        self._runner = DelegateRunner(
            self,
            models=self._models,
            model_llm=self._model_llm,
            summarizer=self._summarizer,
            voice=tts,
            task_models=self._task_models,
            then_targets=self._then_targets,
            step_tools=self._step_tools,
            handed_off=lambda: self._handed_off,
            ends_call=self._end_requested,
        )
        main = self._main_tools()
        super().__init__(
            instructions=spec.instructions,
            tools=[*main, *native_tools],
            llm=self._model_llm,
            tts=tts if tts is not None else NOT_GIVEN,
            id=agent_name(spec, state),
            **agent_options,
        )
        check_tool_names(
            [
                ("a registry tool, task or group", [tool.id for tool in main]),
                ("a native tool", [tool.id for tool in self.tools if tool not in main]),
            ]
        )
        offered = self._check_handoffs()
        if spec.tasks is not None:
            self._check_plan(spec.tasks, offered)

    def _check_handoffs(self) -> set[str]:
        """Return the ids of the agent's own function tools, once ``handoffs`` names only them.

        Raises:
            ValueError: If ``handoffs`` names a tool the agent does not have.
        """
        offered = {tool.id for tool in self._native_functions()}
        if unknown := [name for name in self._handoffs if name not in offered]:
            raise ValueError(
                f"handoffs names '{unknown[0]}', which is not one of the agent's tools. "
                "Pass it in native_tools, or write it as a @function_tool method."
            )
        return offered

    def _check_plan(self, tasks: TaskPlan, offered: set[str]) -> None:
        """Check the plan's native and handoff tools against the agent's own.

        Raises:
            ValueError: If a task lists a tool the agent does not have.
        """
        tasks.check_native_tools({tool.id for tool in self._natives()})
        tasks.check_handoff_tools(
            offered, fix="Pass it in native_tools, or write it as a @function_tool method."
        )

    def _step_tools(
        self, task: TaskSpec
    ) -> tuple[list[llm.FunctionTool | llm.RawFunctionTool], list[llm.Tool]]:
        """Return the handoff tools and the framework tools a task's steps offer too."""
        return (
            [tool for tool in self._native_functions() if tool.id in task.handoffs],
            _flat(tool for tool in self._natives() if tool.id in task.native_tools),
        )

    @cached_property
    def _library_names(self) -> frozenset[str]:
        """The names of the tools the library gives the agent: its registry and its delegates."""
        delegates = self.tasks.delegate_tools() if self.tasks else []
        return frozenset(self._registry.names) | {schema["name"] for schema in delegates}

    def _natives(self) -> list[llm.Tool | llm.Toolset]:
        """Return the agent's own framework tools: native ones and ``@function_tool`` methods."""
        return [tool for tool in self.tools if tool.id not in self._library_names]

    def _native_functions(self) -> list[llm.FunctionTool | llm.RawFunctionTool]:
        """Return the agent's own function tools: native ones and ``@function_tool`` methods."""
        return [
            tool
            for tool in self._natives()
            if isinstance(tool, llm.FunctionTool | llm.RawFunctionTool)
        ]

    def _main_tools(self) -> list[llm.Tool | llm.Toolset]:
        """Return the tools the main agent sees: its own, via tools, and delegates."""
        tasks = self.tasks
        tools: list[llm.Tool | llm.Toolset] = []
        for schema in self._registry.schemas():
            function = schema["function"]
            name = function["name"]
            if tasks is not None and not tasks.plan.visible_to_main(name):
                continue
            if tasks is not None and name in tasks.plan.via_tasks:
                function = {
                    **function,
                    "description": tasks.via_description(name, function["description"]),
                }
            tools.append(
                raw_tool(
                    function,
                    self._call_tool,
                    keep_running=self._registry.keeps_running(name),
                )
            )
        if tasks is not None:
            tools.extend(raw_tool(schema, self._delegate) for schema in tasks.delegate_tools())
        return tools

    @property
    def call(self) -> CallHooks | None:
        """The call's hooks, or None for an agent without them."""
        return self._call

    @property
    def handoffs(self) -> tuple[str, ...]:
        """The names of this agent's handoff tools, which give the call to another agent."""
        return self._handoffs

    @property
    def instructions(self) -> str | Instructions:
        """The prompt. A realtime model reads it filled, with the call's notes and values."""
        try:
            session = self.session
        except RuntimeError:
            return super().instructions
        if realtime_model(session) is None:
            return super().instructions
        return self._realtime.text(session)

    async def instructions_settled(self) -> None:
        """Wait until a realtime model has the instructions the call's saves asked for."""
        await self._realtime.settled()

    async def on_enter(self) -> None:
        """Start the call's spans on the first entry. An agent handed the call speaks first.

        With an opening tool, it runs the tool first. When its line fills, the call and its
        result join the chat, the line is said, and no model request is made.

        Raises:
            ValueError: On a realtime model, if the agent has something it cannot run. The
                session closes first: the agent would run without it.
        """
        if (model := realtime_model(self.session)) is not None:
            try:
                self._realtime.enter(model)
            except ValueError:
                self.session.shutdown(drain=False)
                raise
        if self.tasks is not None:
            self.tasks.activate()
        elif self.state is not None:
            self.state.clear_suppliers()
        if self._trace is not None:
            self._trace.attach(self.session)
        if (dev := dev_observer(self.session)) is not None:
            dev.entered(self)
        if self._call is not None:
            attach_call(self.session, self._call)
        if not self._settles:
            self._settles = True
            self._held_session = self.session
            self.session.once("close", self._closed)
            self._guard.hold_job()
        if self._gate.opens:
            self._gate.opens = False
            said = await self._open()
            self._gate.start_opening(self.chat_ctx)
            if not said:
                self.session.generate_reply()

    def adopt_trace(self, trace: SessionTrace) -> None:
        """Record into another agent's trace from now on, and its task events too.

        ``hand_off`` calls it on the agent that takes the call, so one call has one trace.
        """
        self._trace = trace
        self._models.adopt_trace(trace)
        if self.tasks is not None:
            self.tasks.listen(trace.telemetry.event)

    def open_on_enter(self, opens_with: OpeningTool | None = None) -> None:
        """Speak first when the agent enters: it takes a call handed to it.

        Args:
            opens_with: A tool of the agent's own, run in code before its first reply.
        """
        self._gate.opens = True
        self._opens_with = opens_with

    async def _open(self) -> bool:
        """Run the opening tool the handoff gave, and say its line. Return True when said."""
        spec, self._opens_with = self._opens_with, None
        if spec is None:
            return False
        opening = await open_agent(spec, self._registry, self.state, self._spoken, hooks=self._call)
        if opening is None:
            return False
        result = tool_text(opening.result)
        if self._trace is not None:
            self._trace.tool_ran(
                name=opening.tool,
                call_id=opening.call_id,
                arguments=opening.arguments,
                result=result,
            )
        if opening.line is None:
            return False
        history = self.chat_ctx.copy()
        history.insert(
            [
                llm.FunctionCall(
                    call_id=opening.call_id, name=opening.tool, arguments=opening.arguments
                ),
                llm.FunctionCallOutput(
                    call_id=opening.call_id, name=opening.tool, output=result, is_error=False
                ),
            ]
        )
        await self.update_chat_ctx(history, exclude_invalid_function_calls=False)
        say(self.session, opening.line)
        return True

    def _closed(self, _event: CloseEvent) -> None:
        """Settle the call's held saves. Its trace or its call's end settles them instead."""
        self._guard.closed()
        self._realtime.close()
        if self.state is not None and self._trace is None and self._call is None:
            self._settling = asyncio.create_task(self.state.settle_held())
        if self._held_session is not None:
            self._releasing = asyncio.create_task(self._release_models(self._held_session))

    async def _release_models(self, session: AgentSession) -> None:
        """Close the model calls this agent was given, once the session is closed.

        The session is the call, so what the agent was given ends with it: a router's HTTP
        client, a fallback chain's calls. A model call two agents share, or one the agent
        was given twice, closes once. A LiveKit LLM plugin stays open: LiveKit and the host
        own its lifecycle. A close that fails is logged. With a ``CallHooks``, the models close
        after the call's end has finished, since a held save decided there still asks the model.
        """
        if self._call is not None and (ending := self._call.ending) is not None:
            await asyncio.wait([ending])
        closed = _CLOSED.setdefault(session, [])
        for model in self._owned:
            if model is None or isinstance(model, llm.LLM) or any(model is done for done in closed):
                continue
            closed.append(model)
            try:
                await close_model(model)
            except Exception:
                logger.warning("closing the model %r failed", model, exc_info=True)

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        """Speak through LiveKit's own TTS node, with a ``tts`` span per piece of speech."""
        return traced_tts(self, self._trace, text, model_settings)

    async def hand_off(
        self,
        agent: "UnmuteAgent",
        *,
        announce: str | None = None,
        context: HandoffContext | None = None,
        opens_with: OpeningTool | None = None,
    ) -> "UnmuteAgent | str":
        """Ready another main agent to take the call. Return the result from a tool.

        The agent hears what ``context`` keeps of the call. It records into this agent's
        spans, so build it without ``telemetry``. Once handed off, this agent starts no
        task. A handoff is refused while a task runs, unless that task's step made it.

        Args:
            agent: The agent that takes the call, built on this agent's ``CallState``.
            announce: A line said first, which the caller cannot cut off. The handoff
                waits for it to play.
            context: What the agent hears. A summary is asked for while the line plays.
                None keeps what was said, without tool calls.
            opens_with: A tool of the agent's own it runs in code as it enters. When a line
                fills from the result, the line is its first reply, with no model request.

        Returns:
            ``agent``, which LiveKit switches to, or a refusal while a task runs.

        Raises:
            TypeError: If ``agent`` is not an ``UnmuteAgent``.
            ValueError: If ``agent`` holds another call state, or cannot run ``opens_with``.
        """
        if not isinstance(agent, UnmuteAgent):  # own text: names the plain-agent way
            raise TypeError(
                f"hand_off needs an UnmuteAgent, got {type(agent).__name__}. For a plain "
                "Agent, return it from the tool, built with chat_ctx=shape_chat_ctx(...)"
            )
        if agent.state is not self.state:
            raise ValueError("hand_off needs an agent built on the same CallState.")
        check_opening(
            opens_with, agent._registry, realtime=realtime_model(self.session) is not None
        )
        running = self._runner.step
        step = running if running is not None and running.handing_off else None
        if self.tasks is not None and self.tasks.active is not None and step is None:
            return tool_text(dict(REFUSED_BUSY))
        context = context or HandoffContext()
        if (model := realtime_model(self.session)) is not None:
            check_realtime_handoff(
                context, fixed_instructions=not model.capabilities.mutable_instructions
            )
        source = step if step is not None else self
        summary = None
        if context.history == "summary":
            model = context.summarizer or self._summarizer
            summary = asyncio.create_task(
                self._models.summarize(self, source.chat_ctx, model=model)
            )
        if announce:
            await self._announce(announce)
        text = await summary if summary is not None else None
        history = source.chat_ctx.copy()
        if announce and not speaks_itself(self.session):
            history.add_message(role="assistant", content=announce)
        await agent.update_chat_ctx(
            summary_history(text)
            if text
            else shaped_history(
                history,
                context.history,
                max_messages=context.max_messages,
                include_tool_calls=context.include_tool_calls,
            )
        )
        if self._trace is not None:
            agent.adopt_trace(self._trace)
            self._trace.telemetry.event("handoff", {"from": self.id, "to": agent.id})
            self._trace.start_handoff()
        if step is not None:
            step.step.hand_off()
        agent.open_on_enter(opens_with)
        self._handed_off = True
        return agent

    async def _announce(self, line: str) -> None:
        """Say a handoff's line, which the caller cannot cut off, and wait for it to play.

        It stays out of this agent's chat. ``hand_off`` puts it in the next agent's
        history, as on Pipecat, where only a TTS service writes spoken text to the context.
        A realtime model that speaks itself says it in a reply, which joins its chat and
        the trace as any reply does.
        """
        try:
            await speak(self.session, line, allow_interruptions=False, add_to_chat_ctx=False)
        except Exception:
            logger.warning("the handoff line could not be spoken", exc_info=True)
            return
        if not speaks_itself(self.session):
            if self._trace is not None:
                self._trace.said(line)
            if (dev := dev_observer(self.session)) is not None:
                dev.said(line)

    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterator[llm.ChatChunk | str]:
        """Send the main agent's request through the call's model.

        With ``save_values``, the request ends with the current values. The first reply after
        a handoff offers no handoff tool.
        """
        tools = self._gate.offered(chat_ctx, tools)
        if self._save_values and self.state is not None and self.state.names:
            chat_ctx = chat_ctx.copy()
            chat_ctx.add_message(
                role="system", content=values_block(self.state, owners=self._owners)
            )
        async for chunk in self._models.stream(
            self, chat_ctx, tools, model_settings, prompt=self._prompt, site=""
        ):
            yield chunk

    async def _call_tool(self, arguments: dict[str, Any], context: RunContext) -> object:
        """Run one of the agent's tools. A via tool that succeeds enters its task.

        Otherwise the registry saves what its ``assign`` names, and a registry reply line
        that fills is spoken with no model reply. Once this agent has handed off, a result
        asks for no reply: the new agent speaks.
        """
        call = context.function_call
        result = await self._registry.call(
            call.name,
            arguments,
            self.state,
            started=self._started,
            hooks=self._call,
        )
        shown = tool_text(self._registry.for_model(call.name, result))
        if self._registry.ends_call(call.name, result):
            self._end_requested()
            end_call_now(self.session)
            return llm.ToolResult(shown, reply_required=False)
        if self._handed_off:
            return llm.ToolResult(shown, reply_required=False)
        run = (
            self.tasks.enter_via(call.name, result, caller_turns=CallerHistory(self.chat_ctx).turns)
            if self.tasks is not None
            else None
        )
        if run is None:
            line = self._registry.use_result(
                call.name, result, self.state, self._spoken, hooks=self._call
            )
            if line is None:
                return shown
            say(self.session, line)
            return llm.ToolResult(shown, reply_required=False)
        output = llm.FunctionCallOutput(
            call_id=call.call_id, name=call.name, output=shown, is_error=False
        )
        return await self._runner.run(run, context, seed=[call.model_copy(), output])

    def _end_requested(self) -> None:
        """Tell the call's hooks that the agent ends the call, before the session shuts down."""
        if self._call is not None:
            self._call.end_requested(AGENT_ENDED)

    def _started(self, tool: ToolSpec) -> Awaitable[None] | None:
        """Say the tool's announce line as it starts."""
        if (line := self._lines.pick(f"tool:{tool.name}", tool.announce)) is not None:
            return announce(self.session, tool, line)
        return None

    async def _delegate(self, arguments: dict[str, Any], context: RunContext) -> object:
        """Run the task or group a delegate tool names, or give the model the refusal."""
        assert self.tasks is not None
        name = context.function_call.name
        if self._handed_off:
            return llm.ToolResult(tool_text(dict(REFUSED_BUSY)), reply_required=False)
        run = self.tasks.start(
            name, caller_turns=CallerHistory(self.chat_ctx).turns, arguments=arguments
        )
        if isinstance(run, dict):
            return tool_text(run)
        return await self._runner.run(run, context)


_CLOSED: weakref.WeakKeyDictionary[AgentSession, list[object]] = weakref.WeakKeyDictionary()
"""The model calls each session's agents already closed."""


def _flat(tools: Iterable[llm.Tool | llm.Toolset]) -> list[llm.Tool]:
    """Return the tools with every toolset replaced by the tools inside it, at any depth.

    A step gets the owner's tools one by one. LiveKit sets up a toolset when an agent starts
    and closes it when the agent ends, so a step that owned the owner's toolset would close
    it under the owner.
    """
    return [
        leaf
        for tool in tools
        for leaf in (_flat(tool.tools) if isinstance(tool, llm.Toolset) else [tool])
    ]


def _refuse_plugins(spec: AgentSpec) -> None:
    """Refuse a LiveKit plugin where the spec takes a ``ModelCall``, naming the wrapper.

    Raises:
        TypeError: Naming the field that holds a plugin.
    """
    given = {
        "model": spec.model,
        "summarizer": spec.summarizer,
        **{f"task_models[{name!r}]": model for name, model in spec.task_models.items()},
    }
    # a plugin type only this adapter knows: the core cannot check it
    if field := next((name for name, model in given.items() if isinstance(model, llm.LLM)), None):
        raise TypeError(
            f"AgentSpec {field} takes a ModelCall, got a LiveKit LLM plugin. "
            "Wrap it: LiveKitModel(plugin)."
        )


def _registry_for(
    tasks: TaskPlan | None, tools: ToolRegistry | None, state: CallState | None, save_values: bool
) -> tuple[ToolRegistry, Mapping[str, str]]:
    """Return the agent's tool registry, and who owns each value.

    With ``save_values`` the registry gets the save tool when some value is free for the main
    agent: no task saves it and the call does not start with it.
    """
    registry = tasks.tools if tasks else tools or ToolRegistry([])
    owners = tasks.spec.owners() if tasks else {}
    if save_values and state is not None and free_values(state, owners):
        registry = registry.with_tools(values_tool(state, owners=owners))
    return registry, owners
