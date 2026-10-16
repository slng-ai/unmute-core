"""What happens around a call: its hooks, tools on events, the first reply's wait, failure, end.

An adapter binds the actions it can take (say a line, hang up, read the transcript) and
reports what happens: the call opens, a model request starts, a tool ends, a part fails
for good, the call ends. ``CallHooks`` runs the call's ``Hooks`` and decides the rest, the
same way on every framework.

Work at call start is a ``Prefetch``: before the greeting, or beside it. A tool on events
runs on ``first_user_message``, ``call_end``, or the outcome of one named tool. It saves
fields of its result into the state. With ``note=True`` the result also reaches the model
as a note in the library's own words: every main-agent request ends with all the notes so
far, in one system message, just before the values block.
"""

import asyncio
import dataclasses
import inspect
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import field
from typing import Any

from pydantic import BaseModel, ConfigDict, SkipValidation
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.call._end import CallEnd, EndRun
from unmute_core.call._events import NOTE_MAX_CHARS as NOTE_MAX_CHARS
from unmute_core.call._events import NOTES_HEADER as NOTES_HEADER
from unmute_core.call._events import EventToolRunner
from unmute_core.hooks import EventTool, Hooks
from unmute_core.hooks._event_tool import OUTCOMES
from unmute_core.prefetch import Prefetch
from unmute_core.state import CallState, StateEvent
from unmute_core.tasks._texts import CallStartResults
from unmute_core.tools import ToolRegistry, ToolRun

logger = logging.getLogger(__name__)

MAX_CHAIN = 8
"""How many tools deep one event may lead, each following the one before."""
FAILURE_LINE = "I'm sorry, something went wrong on our side. Please call back later. Goodbye."
"""What the agent says before it hangs up when a part of the call fails for good."""
MODEL_HOOK_POINTS = ("prompt", "before_model", "wrap_model", "after_model")
"""The hook points that run around a model request the library sends."""
FAILURE_SECONDS = 15.0
"""The longest the failure line or audio may take before the call hangs up anyway."""

CALLER_HUNG_UP = "caller_hung_up"
"""The caller left the call."""
AGENT_ENDED = "agent_ended"
"""The agent ended the call itself, for example by a tool."""
MAX_DURATION = "max_duration"
"""The call ran as long as it may."""
IDLE = "idle"
"""Nobody spoke for too long."""
ERROR = "error"
"""A part of the call failed for good."""
ENDED = "ended"
"""The call ended and the adapter knows no more than that."""
TRANSFERRED = "transferred"
"""The call was transferred to another number and this agent is done with it."""


class _FirstUserMessage(BaseModel):
    """What the ``call.first_user_message`` observers are told."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str


class _CallFailed(BaseModel):
    """What the ``call.failed`` observers are told."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    component: str
    error: SkipValidation[BaseException]


@original_errors
@dataclass(config=VALUE)
class _Actions:
    """What the adapter lets ``CallHooks`` do."""

    say: Callable[[str], Awaitable[object]]
    hang_up: Callable[[], Awaitable[object] | object]
    transcript: Callable[[], list[dict[str, Any]]]


@dataclasses.dataclass
class _Run:
    """What happened in one call so far. A plain dataclass: it holds asyncio objects."""

    background: set[asyncio.Task[None]] = field(default_factory=set)
    failed: str | None = None
    requested: str | None = None
    unsaid: tuple[str, str | None] | None = None
    hung_up: bool = False
    warned_model_hooks: bool = False
    model_requests: bool = False


class CallHooks:
    """Run one call's hooks, tools on events, failure line and end, for any adapter.

    The adapter calls ``bind`` once it can speak, then ``open`` before the greeting,
    ``before_request`` at the start of each model request, ``fail`` when a part fails for
    good, and ``end`` once the call is over. It runs the ``hooks`` at each point, and passes
    ``CallHooks`` itself to each tool call, so tool events reach the call's observers.

    Attributes:
        hooks: The call's hooks.
    """

    def __init__(
        self,
        state: CallState | None = None,
        *,
        hooks: Hooks | None = None,
        prefetch: Prefetch | None = None,
        ready_timeout_s: float = 10.0,
        first_request_timeout_s: float = 10.0,
        failure_line: str | None = FAILURE_LINE,
        end_timeout_s: float = 5.0,
    ) -> None:
        """Check the tools on events and keep the hooks.

        Args:
            state: The call's state. Tools read and the end reports its values. Each save
                is told to the ``state.changed`` observers. None gives one that keeps no
                values of its own, with a web call's ``call``. Give the same state to the
                agent, or let the adapter take it from here, so the call has one
                ``session_id``.
            hooks: The call's hooks, its tools on events among them. None runs none.
            prefetch: What the call reads as it starts. Its entries run before the greeting,
                its background lookups beside it.
            ready_timeout_s: The longest the ``before_call`` hooks may take together.
            first_request_timeout_s: The longest a model request waits for tools on events.
            failure_line: Said before hanging up on a failure. None says nothing. A
                ``wrap_failure`` hook may say something else.
            end_timeout_s: The longest the held saves and the call-end tools may take, and
                again the longest the ``call.ended`` observers and the webhook drain may
                take. A slow tool never keeps the observers from running.

        Raises:
            ValueError: If two tools share a name, a tool follows itself, a chain of tools is
                deeper than ``MAX_CHAIN``, a tool assigns a value the state lacks, or a
                prefetch or an assign comes without a state.
        """
        if prefetch is not None and state is None:
            raise ValueError("CallHooks with a prefetch needs state=CallState(...)")
        self._own_state = state is None
        self.state: CallState = state if state is not None else CallState()
        self.prefetch = prefetch
        self.results = CallStartResults()
        """What each prefetch lookup returned. Pass it to ``CallTasks(call_start=...)``."""
        self.hooks = hooks if hooks is not None else Hooks()
        events = list(self.hooks.event_tools)
        names = set(self.state.names)
        if stray := [
            (spec.tool.name, value)
            for spec in events
            for value in spec.assign
            if value not in names
        ]:
            tool, value = stray[0]
            raise ValueError(
                f"event tool {tool} assigns {value!r}, which the state does not have. "
                f"The state has: {', '.join(sorted(names))}"
            )
        tools = ToolRegistry(event.tool for event in events)
        self._ready_timeout_s = ready_timeout_s
        self._first_request_timeout_s = first_request_timeout_s
        self._failure_line = failure_line
        self._actions: _Actions | None = None
        self._watched: list[ToolRegistry] = [tools]
        self._run = _Run()
        self._began = time.monotonic()
        _check_chains(events)
        self._runner = EventToolRunner(
            events,
            tools=tools,
            state=self.state,
            transcript=self.transcript,
            tool_hooks=self,
        )
        self._end = EndRun(
            state=self.state,
            hooks=self.hooks,
            prefetch=prefetch,
            registries=self._watched,
            runner=self._runner,
            timeout_s=end_timeout_s,
            build_end=self._ended,
        )
        self.state.listen(self._state_changed)

    def bind(
        self,
        *,
        say: Callable[[str], Awaitable[object]],
        hang_up: Callable[[], Awaitable[object] | object],
        transcript: Callable[[], list[dict[str, Any]]],
    ) -> None:
        """Give the adapter's actions.

        Args:
            say: Say a line, and wait until it has played.
            hang_up: End the call.
            transcript: Return the chat so far, as ``{"role", "content"}`` messages.

        A failure before ``bind`` says its line and hangs up now.

        Raises:
            ValueError: If an adapter bound already: each call needs its own ``CallHooks``.
        """
        if self._actions is not None:
            raise ValueError("this CallHooks already serves a call: build one CallHooks per call")
        self._actions = _Actions(say, hang_up, transcript)
        if (unsaid := self._run.unsaid) is not None:
            self._run.unsaid = None
            task = asyncio.ensure_future(self._say_and_hang_up(*unsaid))
            self._run.background.add(task)
            task.add_done_callback(self._run.background.discard)

    @property
    def model_hooks(self) -> list[str]:
        """The model hook points that have hooks: ``prompt``, ``before_model`` and the rest.

        They run only where the library sends the model request. An adapter warns when they
        would be skipped.
        """
        return [point for point in MODEL_HOOK_POINTS if self.hooks.has(point)]

    def note_model_requests(self) -> None:
        """Note that an adapter runs this call's model hooks around each model request."""
        self._run.model_requests = True

    def warn_skipped_model_hooks(self, fix: str) -> None:
        """Log once that this call's model hooks will not run, and how to make them run.

        Nothing is logged once an adapter noted that it runs them (``note_model_requests``).

        Args:
            fix: What the developer changes, like ``pass llm= to attach_call``.
        """
        if self._run.model_requests or self._run.warned_model_hooks:
            return
        if points := self.model_hooks:
            self._run.warned_model_hooks = True
            logger.warning(
                "the call has %s hooks, and the model request does not go through the "
                "library, so they do not run: %s",
                ", ".join(points),
                fix,
            )

    @property
    def bound(self) -> bool:
        """True once an adapter gave its actions."""
        return self._actions is not None

    def watch(self, registry: ToolRegistry) -> None:
        """Drain the webhooks of ``registry`` at the end of the call.

        The tools that follow a tool start from its events, which reach ``emit`` because
        the adapter passes ``hooks=call`` to each tool call.
        """
        if all(registry is not seen for seen in self._watched):
            self._watched.append(registry)

    async def run_before_tool(self, run: ToolRun) -> ToolRun:
        """Run the ``before_tool`` hooks. See ``Hooks.run_before_tool``."""
        return await self.hooks.run_before_tool(run)

    async def run_tool(self, run: ToolRun, call: Callable[[ToolRun], Awaitable[object]]) -> object:
        """Run a tool call through the ``wrap_tool`` hooks. See ``Hooks.run_tool``."""
        return await self.hooks.run_tool(run, call)

    async def run_after_tool(self, run: ToolRun, result: object) -> object:
        """Run the ``after_tool`` hooks. See ``Hooks.run_after_tool``."""
        return await self.hooks.run_after_tool(run, result)

    def emit(self, event: str, data: object) -> None:
        """Tell the call's observers of an event, and start the tools that follow a tool."""
        self._runner.on_tool_event(event, data)
        self.hooks.emit(event, data)

    def _state_changed(self, event: StateEvent) -> None:
        """Tell the ``state.changed`` observers of a save."""
        self.hooks.emit("state.changed", event)

    @property
    def failed(self) -> bool:
        """True once a part of the call failed for good."""
        return self._run.failed is not None

    async def open(self) -> bool:
        """Pass the checks before the greeting, run the prefetch, then start the rest.

        The checks come first, so a failed one sets off no lookup. The background lookups
        run beside the greeting. The first model request waits for them, so a fixed greeting
        plays at once. The call's end or a failure stops them.

        Returns:
            True when the call may greet. False after a failed check: the failure line
            played and the call hung up.
        """
        try:
            async with asyncio.timeout(self._ready_timeout_s):
                await self.hooks.run_before_call()
        except Exception as error:  # noqa: BLE001 - any failed check fails the call, which logs it
            await self.fail("ready", error)
            return False
        if self.prefetch is not None:
            await self.prefetch.run(self.state, results=self.results)
            background = self.prefetch.run_background(self.state, results=self.results)
            self._end.lookups = self._runner.spawn(background)
        return True

    async def before_request(self, caller: str | None) -> None:
        """Wait for the tools on events before a model request.

        Args:
            caller: The newest caller message in the request. The first one starts the
                ``first_user_message`` tools.
        """
        if caller and caller.strip() and self._runner.heard_caller(caller):
            self.hooks.emit("call.first_user_message", _FirstUserMessage(text=caller).model_dump())
            self._runner.spawn(self._runner.fire("first_user_message"))
        if pending := list(self._runner.pending):
            _, late = await asyncio.wait(pending, timeout=self._first_request_timeout_s)
            if late:
                logger.warning("%d tools on events still run as the model is asked", len(late))

    @property
    def notes(self) -> tuple[str, ...]:
        """The notes the tools on events left so far, oldest first."""
        return self._runner.notes

    def notes_block(self) -> str | None:
        """The system message a main-agent request ends with, or None without notes."""
        return self._runner.notes_block()

    @property
    def ending(self) -> asyncio.Task[CallEnd] | None:
        """The end's task once it started, so a shutdown can wait for it."""
        return self._end.task

    def fail_soon(self, component: str, error: BaseException) -> None:
        """Start ``fail`` in the background, from a framework's event handler."""
        task = asyncio.ensure_future(self.fail(component, error))
        self._run.background.add(task)
        task.add_done_callback(self._run.background.discard)

    async def fail(self, component: str, error: BaseException) -> None:
        """Say the failure line, then hang up. Only the first failure counts.

        A ``tts`` failure logs a warning and tells the ``call.tts_failed`` observers, so one can
        switch voices. The failure line would need the voice that failed, and the framework's
        own fallback or a retry may still speak, so the call stays open.

        The ``call.failed`` observers hear it first. The ``wrap_failure`` hooks run around
        the line: one may play audio, say other words, or hang up itself. A hook that returns
        without calling its handler keeps the call going: the call is no longer failed, the
        ``call.recovered`` observers hear it, and a later failure counts again. A failure
        once the call's end has started, or was requested, changes nothing: the call is over.

        Args:
            component: The part that failed: ``stt``, ``llm``, ``tts`` or ``ready``.
            error: What went wrong.
        """
        if component == "tts":
            logger.warning("the speech output failed and the call stays open: %s", error)
            self.hooks.emit(
                "call.tts_failed", _CallFailed(component=component, error=error).model_dump()
            )
            return
        if (
            self._run.failed is not None
            or self._end.task is not None
            or self._run.requested is not None
        ):
            return
        self._run.failed = component
        logger.error("the call fails: %s failed: %s", component, error)
        self.hooks.emit("call.failed", _CallFailed(component=component, error=error).model_dump())
        handled = False

        async def say_and_hang_up(line: str | None = self._failure_line) -> None:
            """Say the line, then hang up. Before ``bind``, both wait for it."""
            nonlocal handled
            handled = True
            self._end.stop_lookups()
            if self._actions is None:
                logger.warning("the call failed before bind: the line plays once it binds")
                self._run.unsaid = (component, line)
                return
            await self._say_and_hang_up(component, line)

        try:
            await self.hooks.run_failure(component, error, say_and_hang_up)
        except Exception:
            logger.exception("a wrap_failure hook failed; the call hangs up")
            self._end.stop_lookups()
            if self._actions is not None:
                await self._hang_up()
            return
        if not handled:
            self._run.failed = None
            logger.warning("a wrap_failure hook kept the call going after %s failed", component)
            self.hooks.emit("call.recovered", {"component": component})

    def end_requested(self, reason: str) -> None:
        """Note that the call is ending for ``reason``. The first request wins.

        Later failures change nothing, and ``end`` reports this reason unless a failure came
        first. Background lookups stop.

        Args:
            reason: Why, like ``AGENT_ENDED`` or ``MAX_DURATION``.
        """
        if self._run.requested is None:
            self._run.requested = reason
            self._end.stop_lookups()

    def caller_left(self) -> None:
        """Note that the caller hung up."""
        self.end_requested(CALLER_HUNG_UP)

    async def _say_and_hang_up(self, component: str, line: str | None) -> None:
        """Say the line, then hang up, whatever happens."""
        assert self._actions is not None
        try:
            async with asyncio.timeout(FAILURE_SECONDS):
                if line:
                    await self._actions.say(line)
        except Exception:
            logger.exception("the failure line could not play")
        finally:
            await self._hang_up()

    async def _hang_up(self) -> None:
        """Hang up through the adapter, once."""
        assert self._actions is not None
        if self._run.hung_up:
            return
        self._run.hung_up = True
        ended = self._actions.hang_up()
        if inspect.isawaitable(ended):
            await ended

    def resolved_reason(self, reason: str) -> str:
        """Return the reason the call ends with: a failure, else the requested one, else ``reason``.

        Args:
            reason: What the adapter knows, like ``caller_hung_up``.
        """
        if self._run.failed is not None:
            return ERROR
        return self._run.requested if self._run.requested is not None else reason

    async def end(self, reason: str) -> CallEnd:
        """Settle the held saves, run the call-end tools, tell the hook, drain the webhooks.

        Every caller waits for the same run. Each step that fails is logged, and the whole
        end stops waiting after ``end_timeout_s``.

        Args:
            reason: Why the call ended, like ``participant_disconnected``.

        Returns:
            How the call ended.
        """
        task = self._end.start(self.resolved_reason(reason), failed=self._run.failed is not None)
        return await asyncio.shield(task)

    def _ended(self, reason: str) -> CallEnd:
        """Read how the call ended now."""
        values = None if self._own_state else self.state.dump()
        confirmed = [name for name in values or {} if self.state.is_confirmed(name)]
        return CallEnd(reason, values, confirmed, self.transcript(), time.monotonic() - self._began)

    def transcript(self) -> list[dict[str, Any]]:
        """Return the chat so far, as ``{"role", "content"}`` messages, or nothing before ``bind``.

        A ``Transcript`` locked argument reads it, on an event tool and on a tool the model
        calls.
        """
        if self._actions is None:
            return []
        try:
            return self._actions.transcript()
        except Exception:
            logger.exception("the transcript could not be read")
            return []


def _check_chains(events: Sequence[EventTool]) -> None:
    """Refuse a tool that follows itself, or a chain of tools deeper than ``MAX_CHAIN``."""
    followers: defaultdict[str, list[str]] = defaultdict(list)
    for spec in events:
        for event in spec.on:
            kind, _, source = event.partition(":")
            if kind in OUTCOMES:
                followers[source].append(spec.tool.name)

    def depth(name: str, path: tuple[str, ...]) -> int:
        """The longest chain that starts at ``name``."""
        if name in path:
            raise ValueError(
                f"event tools follow each other in a loop: {' -> '.join((*path, name))}"
            )
        return 1 + max(
            (depth(next_, (*path, name)) for next_ in followers.get(name, ())), default=0
        )

    for spec in events:
        if depth(spec.tool.name, ()) > MAX_CHAIN:
            raise ValueError(f"event tool {spec.tool.name} starts a chain deeper than {MAX_CHAIN}")
