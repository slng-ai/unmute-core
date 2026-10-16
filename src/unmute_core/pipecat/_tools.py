"""Library tools as Pipecat function schemas and handlers.

A tool here is the ``{"name", "description", "parameters"}`` shape the core hands out
(``CallTasks.delegate_tools``, ``StepRun.finish_tool``, ``ToolRegistry.schemas``).
"""

import asyncio
import contextlib
import weakref
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.flows import FlowManager, FlowsFunctionSchema
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallResultProperties,
    InterruptionFrame,
    LLMAssistantPushAggregationFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams, LLMService

from unmute_core.call import AGENT_ENDED, CallHooks
from unmute_core.pipecat._realtime import (
    can_hold,
    hold_result,
    is_realtime,
    reply_ended,
    send_quietly,
    speaks_itself,
    withhold_result,
)
from unmute_core.pipecat._services import mark_keeps_running
from unmute_core.state import CallState
from unmute_core.tools import (
    LinePicker,
    SpokenLines,
    ToolError,
    ToolHooks,
    ToolRegistry,
    ToolSpec,
    ToolStarted,
)

_REF = "#/$defs/"
LINE_WAIT_SECONDS = 30.0
"""The longest a tool waits for its announce line to play."""


def inline_refs(node: object, defs: Mapping[str, Any] | None = None) -> Any:
    """Return a schema with every local ``$ref`` replaced by its definition.

    Pipecat's ``FunctionSchema`` keeps only ``properties`` and ``required``, so a
    ``$defs`` at the root would be lost.
    """
    if defs is None:
        defs = node.get("$defs", {}) if isinstance(node, dict) else {}
    if isinstance(node, list):
        return [inline_refs(item, defs) for item in node]
    if not isinstance(node, dict):
        return node
    target = node.get("$ref")
    if isinstance(target, str) and target.startswith(_REF):
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        return inline_refs({**defs.get(target.removeprefix(_REF), {}), **siblings}, defs)
    return {key: inline_refs(value, defs) for key, value in node.items() if key != "$defs"}


def tool_dicts(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Return the registry's tools in the core's name, description, parameters shape."""
    return [schema["function"] for schema in registry.schemas()]


type FunctionHandler = Callable[[FunctionCallParams], Awaitable[None]]
type FlowsHandler = Callable[[dict[str, Any], FlowManager], Awaitable[tuple[object, object | None]]]


def _parts(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Return the arguments both schema classes take, refs inlined."""
    parameters = inline_refs(tool.get("parameters") or {})
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "properties": dict(parameters.get("properties") or {}),
        "required": list(parameters.get("required") or []),
    }


def function_schema(
    tool: Mapping[str, Any],
    handler: FunctionHandler | None = None,
    *,
    cancel_on_interruption: bool = True,
) -> FunctionSchema:
    """Return a tool as a Pipecat ``FunctionSchema``. Pipecat registers its handler."""
    if handler is not None and not cancel_on_interruption:
        handler = mark_keeps_running(tool_options(handler, cancel_on_interruption=False))
    return FunctionSchema(**_parts(tool), handler=handler)


def flows_function(
    tool: Mapping[str, Any], handler: FlowsHandler, *, cancel_on_interruption: bool = True
) -> FlowsFunctionSchema:
    """Return a tool as a Flows node function.

    ``cancel_on_interruption`` is set even when True, because Flows defaults to False, an
    async tool, whose guidance Pipecat adds to the system prompt.
    """
    if not cancel_on_interruption:
        mark_keeps_running(handler)
    return FlowsFunctionSchema(
        **_parts(tool), handler=handler, cancel_on_interruption=cancel_on_interruption
    )


async def run_tool(
    registry: ToolRegistry,
    name: str,
    arguments: Mapping[str, Any],
    state: CallState | None,
    *,
    started: ToolStarted | None = None,
    hooks: ToolHooks | None = None,
) -> object:
    """Run one tool call. A ``ToolError`` becomes ``{"error": ...}``, Pipecat's failed result.

    Args:
        registry: The tools.
        name: The tool the model called.
        arguments: Its arguments.
        state: The call state, or None.
        started: Called once the call is checked, just before the tool runs.
        hooks: The tool hooks to run around the call, like the call's ``CallHooks``.
    """
    try:
        return await registry.call(name, dict(arguments), state, started=started, hooks=hooks)
    except ToolError as error:
        return {"error": error.message}


_ENDING: weakref.WeakSet[object] = weakref.WeakSet()
"""The pipeline workers whose call a tool has ended."""
_ENDERS: weakref.WeakSet[Callable[..., object]] = weakref.WeakSet()
"""The handlers of tools that can end the call."""
_HELD: weakref.WeakKeyDictionary[object, set[str]] = weakref.WeakKeyDictionary()
"""The tool calls whose result a realtime service waits for, per service."""


def ending(worker: object) -> bool:
    """Return True once a tool of the call has ended it. Later results ask for no reply."""
    return worker in _ENDING


def mark_ender[H: Callable[..., object]](handler: H) -> H:
    """Mark a tool handler as one that can end the call, and return it."""
    _ENDERS.add(handler)
    return handler


def _ender_running(params: FunctionCallParams) -> bool:
    """Return True when another call of the service, one that can end the call, still runs."""
    return any(
        call.tool_call_id != params.tool_call_id and call.registry_item.handler in _ENDERS
        for call in params.llm._function_call_tasks.values()
    )


async def answer_ending(params: FunctionCallParams, result: object) -> None:
    """Answer the tool call that ends the call, with no reply to follow.

    On a realtime service the result stays local too, so it does not ask for a reply.
    Pipecat still records the result: the tool span, the transcript and the dev record
    show the tool.
    """
    _ENDING.add(params.pipeline_worker)
    withhold_result(params.llm, params.tool_call_id)
    await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))


async def end_with(params: FunctionCallParams, result: object, hooks: object | None = None) -> None:
    """Answer the tool call with no reply, note the agent's own end, and end the pipeline.

    Args:
        params: The tool call's parameters.
        result: What Pipecat records as the tool's result.
        hooks: The call's hooks. When they are ``CallHooks``, the end is noted as
            ``AGENT_ENDED`` first.
    """
    await answer_ending(params, result)
    if isinstance(hooks, CallHooks):
        hooks.end_requested(AGENT_ENDED)
    await params.pipeline_worker.end()


async def report(
    params: FunctionCallParams,
    result: object,
    *,
    ends_call: bool,
    hooks: object | None = None,
    line: str | None = None,
    reply: bool = True,
) -> None:
    """Give Pipecat a tool's result. A tool that ends the call ends it, with no reply first.

    A tool that finishes after another one ended the call asks for no reply either: the
    call is over. On a realtime service, a tool that finishes while its reply still runs
    waits for that reply to end, so a later call of the same reply can still end the call.
    A tool that finishes while another one that can end the call still runs waits for it:
    its result is held back, and goes out with the next result that is not held, unless
    the call ends first.

    Args:
        params: The tool call's parameters.
        result: What the model reads.
        ends_call: True when the tool ends the call.
        hooks: The tool hooks of the call. When they are the call's ``CallHooks``, the end
            is noted as the agent's own (``AGENT_ENDED``) before the pipeline ends.
        line: A reply line code speaks in place of the model's reply. A realtime model
            replies in its own words instead: its result starts a reply of its own, which
            a second spoken line would talk over.
        reply: False asks for no model reply, as ``replies_to`` decides. A realtime
            service that cannot read a result without replying still replies.
    """
    if can_hold(params.llm) and not ends_call and not _ender_running(params):
        await reply_ended(params.llm)
    if ends_call:
        await end_with(params, result, hooks)
    elif ending(params.pipeline_worker):
        withhold_result(params.llm, params.tool_call_id)
        await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
    elif not reply and (
        not is_realtime(params.llm) or await send_quietly(params.llm, params.tool_call_id, result)
    ):
        await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
    elif line is not None and not is_realtime(params.llm):
        joins_context(params.llm)
        await params.llm.push_frame(TTSSpeakFrame(line))
        await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
    elif can_hold(params.llm) and _ender_running(params):
        hold_result(params.llm, params.tool_call_id, held=True)
        _HELD.setdefault(params.llm, set()).add(params.tool_call_id)
        await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
    else:
        for call_id in _HELD.pop(params.llm, ()):
            hold_result(params.llm, call_id, held=False)
        await params.result_callback(result)


class _Playout:
    """Waits for lines an LLM service pushed to stop playing. One per service.

    The output transport reports the bot stopping upstream, and the LLM service sees it.
    """

    def __init__(self, llm: FrameProcessor) -> None:
        """Watch the frames that reach ``llm``."""
        self._waiting: set[asyncio.Event] = set()
        llm.add_event_handler("on_before_process_frame", self._seen)

    async def _seen(self, _llm: object, frame: Frame) -> None:
        """Release every waiting line once the bot stops speaking or is cut off."""
        if isinstance(frame, BotStoppedSpeakingFrame | InterruptionFrame):
            for played in self._waiting:
                played.set()

    async def played_after(self, start: Awaitable[None]) -> None:
        """Wait until the bot stops speaking after ``start``, at most ``LINE_WAIT_SECONDS``.

        ponytail: the first stop after the start counts, so speech queued behind the model's
        own words is released when those words stop. Per-line playout needs TTS word times.
        """
        played = asyncio.Event()
        self._waiting.add(played)
        try:
            await start
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(played.wait(), timeout=LINE_WAIT_SECONDS)
        finally:
            self._waiting.discard(played)


_PLAYOUTS: weakref.WeakKeyDictionary[FrameProcessor, _Playout] = weakref.WeakKeyDictionary()


def played_after(llm: FrameProcessor, start: Awaitable[None]) -> Awaitable[None]:
    """Run ``start``, which makes the bot speak, and wait until the bot stops speaking.

    Waits at most ``LINE_WAIT_SECONDS``. The service ``llm`` must be in the pipeline: it
    sees the output transport report the bot stopping.
    """
    playout = _PLAYOUTS.get(llm)
    if playout is None:
        playout = _PLAYOUTS[llm] = _Playout(llm)
    return playout.played_after(start)


class _Commits:
    """Commits a line code says to the context once it has played. One per service.

    Pipecat's TTS commits a ``TTSSpeakFrame`` to the context only when no model reply is
    under way, and a reply that only called a tool stays under way: its line would wait
    for the caller's next words and reach the model after them. The output transport
    reports the bot stopping upstream, and the LLM service sees it: the line has played,
    so its text is in the assistant aggregator, and the commit goes out then. An
    interruption commits what played by itself.
    """

    def __init__(self, llm: FrameProcessor) -> None:
        """Watch the frames that reach ``llm``."""
        self._llm = llm
        self._pending: bool | None = None
        llm.add_event_handler("on_before_process_frame", self._seen)

    def expect(self) -> None:
        """Note that a line code queued must join the context once it has played."""
        self._pending = False

    async def _seen(self, _llm: object, frame: Frame) -> None:
        """Follow the line: started, then played to its end, then committed."""
        if self._pending is None:
            return
        if isinstance(frame, InterruptionFrame):
            self._pending = None
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._pending = True
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._pending:
            self._pending = None
            await self._llm.push_frame(LLMAssistantPushAggregationFrame())


_COMMITS: weakref.WeakKeyDictionary[FrameProcessor, _Commits] = weakref.WeakKeyDictionary()


def joins_context(llm: FrameProcessor) -> None:
    """Have the line code is about to queue join the context as soon as it has played.

    Call it before a ``TTSSpeakFrame`` that follows a tool call, like a tool's reply line
    or a step's receipt. A realtime model keeps its own context, so it is left alone.

    Args:
        llm: The call's LLM service. It must be in the pipeline: it sees the bot stop.
    """
    if is_realtime(llm):
        return
    commits = _COMMITS.get(llm)
    if commits is None:
        commits = _COMMITS[llm] = _Commits(llm)
    commits.expect()


def say_line(llm: FrameProcessor, tool: ToolSpec, line: str) -> Awaitable[None]:
    """Push a tool's announce line. With ``announce_wait``, the result waits for it to play.

    A realtime model that speaks itself says nothing: it words its own line as it calls
    the tool, and a second reply with the line would come while the tool's result is
    still out.
    """
    if isinstance(llm, LLMService) and speaks_itself(llm):
        return asyncio.sleep(0)
    if not tool.announce_wait:
        return llm.push_frame(TTSSpeakFrame(line))
    return played_after(llm, llm.push_frame(TTSSpeakFrame(line)))


def announcer(params: FunctionCallParams, pick: LinePicker) -> ToolStarted:
    """Return what says a tool's announce line as it starts, before the result."""

    async def started(tool: ToolSpec) -> None:
        """Push the line, unless the tool has none."""
        if (line := pick.pick(f"tool:{tool.name}", tool.announce)) is not None:
            await say_line(params.llm, tool, line)

    return started


def tools_schema(
    registry: ToolRegistry,
    state: CallState | None = None,
    *,
    hooks: ToolHooks | None = None,
    language: str = "en",
    timezone: str | None = None,
) -> ToolsSchema:
    """Return the tools of an agent without tasks, each with its handler. Build it per call.

    Put them on the ``LLMContext``. Pipecat registers the handlers from there.

    Args:
        registry: The agent's tools.
        state: The call state. None takes the state of ``hooks`` when it is a ``CallHooks``.
        hooks: The tool hooks to run around each call, like the call's ``CallHooks``. A
            ``CallHooks`` also drains the registry's webhooks at the call's end.
        language: The call's language, a BCP-47 tag like ``es-MX``. A reply line says the
            result's dates and times in it.
        timezone: Where the business is, an IANA name like ``Europe/Madrid``. A reply line
            then names a day from today: "tomorrow, October 9".

    Raises:
        ValueError: When the timezone is unknown.
    """
    if isinstance(hooks, CallHooks):
        hooks.watch(registry)
        state = state if state is not None else hooks.state
    pick = LinePicker()
    lines = SpokenLines(language, timezone)

    def handler_for(name: str) -> FunctionHandler:
        """Return the Pipecat handler of one tool."""

        async def handler(params: FunctionCallParams) -> None:
            """Run the call and report its result."""
            started = announcer(params, pick)
            result = await run_tool(
                registry,
                name,
                params.arguments,
                state,
                started=started,
                hooks=hooks,
            )
            await report(
                params,
                registry.for_model(name, result),
                ends_call=registry.ends_call(name, result),
                hooks=hooks,
                line=registry.use_result(name, result, state, lines, hooks=hooks),
            )

        return mark_ender(handler) if registry.get(name).effect == "ends_conversation" else handler

    return ToolsSchema(
        standard_tools=[
            function_schema(
                t,
                handler_for(t["name"]),
                cancel_on_interruption=not registry.keeps_running(t["name"]),
            )
            for t in tool_dicts(registry)
        ]
    )
