"""Hooks: your code before, around and after each prompt, model request, tool call and event.

Write a hook as a decorated function or as a method of a class, and give the ``Hooks`` to
``CallHooks``. Every adapter then runs it at the same points:

- ``before_*`` and ``prompt`` hooks run first to last. They may change what goes on.
- ``wrap_*`` hooks nest, the first one outermost. Each decides whether and how to call on.
- ``after_*`` hooks run last to first. ``after_tool`` may replace the result.
- ``on`` observers only watch.
"""

import asyncio
import fnmatch
import functools
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Annotated, Any, Literal, get_args

from pydantic import AfterValidator, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors, own_error
from unmute_core.hooks._event_tool import EventTool
from unmute_core.model import ModelEvent, ModelRequest, ToolCall
from unmute_core.tasks._record import TaskStage
from unmute_core.tools._spec import ToolRefused, ToolRun

logger = logging.getLogger(__name__)

type PromptKind = Literal["agent", "step", "finish_check", "summary"]
"""Which prompt a ``prompt`` hook reads: an agent's, a task step's, a finish check's or a
summary's."""

POINTS = (
    "prompt",
    "before_model",
    "wrap_model",
    "after_model",
    "before_tool",
    "wrap_tool",
    "after_tool",
    "before_call",
    "wrap_failure",
)
"""Every point a hook can run at. A class hook defines methods with these names."""
EVENTS = (
    "call.first_user_message",
    "call.failed",
    "call.recovered",
    "call.tts_failed",
    "call.ended",
    "tool.started",
    "tool.succeeded",
    "tool.failed",
    "tool.refused",
    "state.changed",
    *get_args(TaskStage),
)
"""Every event an ``on`` observer can watch."""


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class Refuse:
    """What a ``before_tool`` hook returns to stop a call. The model reads the message.

    Attributes:
        message: Why the tool did not run, in words the model can pass on.
    """

    message: str


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class PromptInfo:
    """Which prompt a ``prompt`` hook is filling.

    Attributes:
        kind: See ``PromptKind``.
        site: ``""`` for the main prompt, ``task:<name>`` in a step, ``summary`` for one.
        model: The model that will read it, like ``gpt-4.1-mini``, or ``unknown``.
    """

    kind: PromptKind
    site: str
    model: str


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class ModelResponse:
    """What a model request gave back, as an ``after_model`` hook sees it.

    Attributes:
        text: The text the agent said.
        tool_calls: The tools the model called.
        error: The error's type name when the request failed, else None. A Pipecat
            pipeline request reports its failure through ``call.failed`` instead.
        interrupted: True when the caller cut the reply off.
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    error: str | None = None
    interrupted: bool = False


type Observer = Callable[[str, Any], Awaitable[None] | None]
"""Told an event's name and its data. The data is a dict for a tool, call or task event and a
``StateEvent`` for ``state.changed``, so it is typed ``Any`` for the observer to narrow."""
ModelHandler = Callable[[ModelRequest], AsyncIterator[ModelEvent]]
"""Sends a request on, to the next ``wrap_model`` hook or the model."""
ToolHandler = Callable[[ToolRun], Awaitable[object]]
"""Runs a call on, to the next ``wrap_tool`` hook or the tool."""
FailureHandler = Callable[..., Awaitable[None]]
"""Says the failure line and hangs up. ``line=None`` says nothing first."""
_Hook = Callable[..., Any]
"""Any hook, as ``Hooks`` keeps it: each point takes and gives different types."""


def _known_pattern(pattern: str) -> str:
    """Refuse a pattern that matches no event.

    Raises:
        ValueError: Naming the pattern and the events.
    """
    if not fnmatch.filter(EVENTS, pattern):
        raise ValueError(f"{pattern!r} matches no event. The events are: {', '.join(EVENTS)}")
    return pattern


def _some_patterns(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """Refuse an empty list of patterns.

    Raises:
        ValueError: With an example to follow.
    """
    if not patterns:
        raise ValueError("on() needs at least one event, like on('call.ended')")
    return patterns


EventPattern = Annotated[str, AfterValidator(_known_pattern)]
"""An event name, or a pattern with ``*`` that matches at least one name in ``EVENTS``."""
_PATTERNS = TypeAdapter(Annotated[tuple[EventPattern, ...], AfterValidator(_some_patterns)])


async def _done[T](value: T | Awaitable[T]) -> T:
    """Return a hook's answer, awaited when the hook is async."""
    return await value if inspect.isawaitable(value) else value


class Hooks:
    """Your hooks for a call. Define them once, and give them to every call's ``CallHooks``.

    Register a function with a decorator, like ``@hooks.before_model``, or an object with
    ``add``. An object hooks in with each method it has whose name is a point in ``POINTS``,
    and watches events with an ``on_event(event, data)`` method and an ``events`` tuple of
    patterns. ``add`` also takes an ``EventTool``.
    """

    def __init__(self, *hooks: object) -> None:
        """Start with no hooks, then add each one given.

        Args:
            *hooks: Objects for ``add``.
        """
        self._points: dict[str, list[_Hook]] = {point: [] for point in POINTS}
        self._observers: dict[str, list[Observer]] = {event: [] for event in EVENTS}
        self._patterns: list[tuple[tuple[str, ...], Observer]] = []
        self._running: set[asyncio.Task[None]] = set()
        self.event_tools: list[EventTool] = []
        """The ``EventTool`` objects added, in order."""
        for hook in hooks:
            self.add(hook)

    def _point[F: _Hook](self, point: str, hook: F) -> F:
        """Register a hook at one point and give it back, so it works as a decorator."""
        self._points[point].append(hook)
        return hook

    def prompt[F: _Hook](self, hook: F) -> F:
        """Change each system prompt: ``(text, PromptInfo) -> str``. The text is unfilled.

        A prompt hook is a plain function, never async: adapters build prompts where
        nothing can wait. Do async work in ``before_model``.
        """
        return self._point("prompt", hook)

    def before_model[F: _Hook](self, hook: F) -> F:
        """Change a model request: ``(ModelRequest) -> ModelRequest | None``. None keeps it."""
        return self._point("before_model", hook)

    def wrap_model[F: _Hook](self, hook: F) -> F:
        """Wrap a model request: an async generator ``(request, handler)`` yielding events.

        It reaches only a request a ``ModelCall`` answers. A framework's own LLM plugin
        keeps its own retries and fallback.
        """
        return self._point("wrap_model", hook)

    def after_model[F: _Hook](self, hook: F) -> F:
        """Watch a request's answer: ``(ModelRequest, ModelResponse) -> None``."""
        return self._point("after_model", hook)

    def before_tool[F: _Hook](self, hook: F) -> F:
        """Change or stop a tool call: ``(ToolRun) -> ToolRun | Refuse | None``."""
        return self._point("before_tool", hook)

    def wrap_tool[F: _Hook](self, hook: F) -> F:
        """Wrap a tool call: ``async (ToolRun, handler) -> result``."""
        return self._point("wrap_tool", hook)

    def after_tool[F: _Hook](self, hook: F) -> F:
        """Replace a tool's result: ``(ToolRun, result) -> result | None``. None keeps it."""
        return self._point("after_tool", hook)

    def before_call[F: _Hook](self, hook: F) -> F:
        """Check something before the greeting: ``() -> None``. Raising fails the call."""
        return self._point("before_call", hook)

    def wrap_failure[F: _Hook](self, hook: F) -> F:
        """Wrap a failure: ``async (component, error, handler)``.

        ``handler()`` says the failure line and hangs up, and ``handler(line=None)`` only
        hangs up. A hook that never calls it says its own words and hangs up itself. Before
        the adapter binds, ``handler`` returns at once and the line plays once it binds.
        """
        return self._point("wrap_failure", hook)

    def on(self, *patterns: str) -> Callable[[Observer], Observer]:
        """Watch events: ``(event, data) -> None``. A pattern may hold ``*``, like ``task.*``.

        Raises:
            ValueError: If a pattern matches no event in ``EVENTS``.
        """
        try:
            known = _PATTERNS.validate_python(patterns)
        except ValidationError as error:
            raise own_error(error) from None
        events = [event for event in EVENTS if any(fnmatch.fnmatchcase(event, p) for p in known)]

        def register(observer: Observer) -> Observer:
            """Keep the observer under each event its patterns match."""
            for event in events:
                self._observers[event].append(observer)
            self._patterns.append((known, observer))
            return observer

        return register

    def add(self, hook: object) -> None:
        """Add an ``EventTool``, another ``Hooks``, or an object with hook methods.

        Another ``Hooks`` adds its hooks, observers and event tools as they are now. So
        ``Hooks(SHARED, Audit(room))`` builds one call's hooks from hooks shared by all.

        Raises:
            ValueError: If the object has no method named after a point, and no
                ``on_event``.
        """
        if isinstance(hook, EventTool):
            self.event_tools.append(hook)
            return
        if isinstance(hook, Hooks):
            for point, hooks in hook._points.items():
                self._points[point].extend(hooks)
            for event, observers in hook._observers.items():
                self._observers[event].extend(observers)
            self._patterns.extend(hook._patterns)
            self.event_tools.extend(hook.event_tools)
            return
        found = [point for point in POINTS if callable(getattr(hook, point, None))]
        for point in found:
            self._points[point].append(getattr(hook, point))
        if callable(observer := getattr(hook, "on_event", None)):
            self.on(*getattr(hook, "events", ("*",)))(observer)
        elif not found:
            raise ValueError(
                f"{type(hook).__name__} hooks in nowhere: give it a method named after one "
                f"of {', '.join(POINTS)}, or on_event"
            )

    def has(self, point: str) -> bool:
        """Return True when any hook runs at ``point``."""
        return bool(self._points[point])

    def run_prompt(self, text: str, info: PromptInfo) -> str:
        """Run the ``prompt`` hooks, first to last, and return the prompt they leave.

        Raises:
            TypeError: If a hook is async.
        """
        for hook in tuple(self._points["prompt"]):
            text = hook(text, info)
            if inspect.isawaitable(text):
                if inspect.iscoroutine(text):
                    text.close()
                raise TypeError(
                    f"prompt hook {getattr(hook, '__name__', hook)!r} is async: a prompt hook "
                    "must be a plain function. Do async work in before_model"
                )
        return text

    async def run_before_model(self, request: ModelRequest) -> ModelRequest:
        """Run the ``before_model`` hooks, first to last, and return the request they leave."""
        for hook in tuple(self._points["before_model"]):
            request = await _done(hook(request)) or request
        return request

    def run_model(self, request: ModelRequest, send: ModelHandler) -> AsyncIterator[ModelEvent]:
        """Send a request through the ``wrap_model`` hooks, the first one outermost."""
        return _nested(self._points["wrap_model"], send, _wrapped)(request)

    async def run_after_model(self, request: ModelRequest, response: ModelResponse) -> None:
        """Tell the ``after_model`` hooks, last to first. A hook that fails is only logged."""
        for hook in reversed(self._points["after_model"]):
            try:
                await _done(hook(request, response))
            except Exception:
                logger.exception("an after_model hook failed")

    async def run_before_tool(self, run: ToolRun) -> ToolRun:
        """Run the ``before_tool`` hooks, first to last.

        Raises:
            ToolRefused: If a hook returned ``Refuse``. The model reads its message.
        """
        for hook in tuple(self._points["before_tool"]):
            answer = await _done(hook(run))
            if isinstance(answer, Refuse):
                raise ToolRefused(answer.message)
            run = answer or run
        return run

    async def run_tool(self, run: ToolRun, call: ToolHandler) -> object:
        """Run a tool call through the ``wrap_tool`` hooks, the first one outermost."""
        return await _nested(self._points["wrap_tool"], call, _wrapped)(run)

    async def run_after_tool(self, run: ToolRun, result: object) -> object:
        """Run the ``after_tool`` hooks, last to first, and return the result they leave."""
        for hook in reversed(self._points["after_tool"]):
            replaced = await _done(hook(run, result))
            result = result if replaced is None else replaced
        return result

    async def run_before_call(self) -> None:
        """Run the ``before_call`` hooks in order. The first that raises stops the rest."""
        for hook in tuple(self._points["before_call"]):
            await _done(hook())

    async def run_failure(
        self, component: str, error: BaseException, handler: FailureHandler
    ) -> None:
        """Run a failure through the ``wrap_failure`` hooks, the first one outermost."""
        wrapped = _nested(
            self._points["wrap_failure"],
            handler,
            lambda hook, inner: _failure(hook, component, error, inner),
        )
        await wrapped()

    def emit(self, event: str, data: object) -> None:
        """Tell the observers of ``event``. An async one runs as a task of its own."""
        for observer in self._watching(event):
            try:
                answer = observer(event, data)
            except Exception:
                logger.exception("an observer of %s failed", event)
                continue
            if inspect.isawaitable(answer):
                self._keep(answer, event)

    async def notify(self, event: str, data: object) -> None:
        """Tell the observers of ``event`` and wait for each. One that fails is only logged."""
        for observer in self._watching(event):
            try:
                await _done(observer(event, data))
            except Exception:
                logger.exception("an observer of %s failed", event)

    def _watching(self, event: str) -> tuple[Observer, ...]:
        """Return the observers of ``event``, in the order they came.

        A listed event reads the index built by ``on``. An event outside ``EVENTS`` is
        matched against the patterns, so a ``*`` observer still hears it.
        """
        if (indexed := self._observers.get(event)) is not None:
            return tuple(indexed)
        return tuple(
            observer
            for patterns, observer in self._patterns
            if any(fnmatch.fnmatchcase(event, pattern) for pattern in patterns)
        )

    def _keep(self, work: Awaitable[None], event: str) -> None:
        """Run an async observer as a task, kept until it ends, its failure logged."""
        try:
            task = asyncio.ensure_future(work)
        except RuntimeError:
            logger.warning("an observer of %s needs a running event loop; it was skipped", event)
            return
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        task.add_done_callback(lambda done: _log_failure(done, event))


def _nested[H](hooks: Iterable[_Hook], handler: H, wrap: Callable[[_Hook, H], H]) -> H:
    """Wrap a handler in the hooks, the first hook outermost."""
    return functools.reduce(lambda inner, hook: wrap(hook, inner), reversed(list(hooks)), handler)


def _wrapped[H: Callable[..., Any]](hook: _Hook, handler: H) -> H:
    """Return a handler that calls ``hook`` with the next handler."""
    return lambda value: hook(value, handler)  # ty: ignore[invalid-return-type]


def _failure(
    hook: _Hook, component: str, error: BaseException, handler: FailureHandler
) -> FailureHandler:
    """Return a failure handler that calls ``hook`` with the next handler."""

    async def run(**kwargs: object) -> None:
        """Run the hook. Arguments given here reach the inner handler too."""

        async def inner(**more: object) -> None:
            """Call on, with the outer hook's arguments and this one's."""
            await handler(**(kwargs | more))

        await hook(component, error, inner)

    return run


def _log_failure(task: asyncio.Task[None], event: str) -> None:
    """Log an async observer that failed."""
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error("an observer of %s failed", event, exc_info=error)
