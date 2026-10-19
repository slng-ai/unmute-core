"""Hooks around each attempt of a ``FallbackModelCall``, for code that cannot reach the chain.

An adapter traces each call a chain tries, as LiveKit's ``FallbackAdapter`` does. It also
times the call that answered. The chain is often out of its reach: an app builds one per
call, or one per request inside its own model call. So the adapter sets a hook for the time a
request runs, and every chain in that request reads it.
"""

from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextvars import ContextVar
from typing import Any

from unmute_core.model._call import ModelCall, ModelEvent, ModelRequest

AttemptStream = Callable[[ModelCall, ModelRequest], AsyncIterator[ModelEvent]]
"""Streams one attempt: given the call a chain tries and its request, yield the call's events."""

_ATTEMPT_STREAM: ContextVar[AttemptStream | None] = ContextVar(
    "unmute_attempt_stream", default=None
)


class AnswerClock:
    """When the attempt that answered a request began, as a chain reports it.

    Attributes:
        started: The ``time.perf_counter()`` reading at the start of the attempt that
            answered, or None while no call has answered.
    """

    def __init__(self) -> None:
        """Start with no answer."""
        self.started: float | None = None


_ANSWER_CLOCK: ContextVar[AnswerClock | None] = ContextVar("unmute_answer_clock", default=None)


def note_answer(started: float) -> None:
    """Tell the clock of the running request that a call answered, and when its attempt began.

    A chain calls it when a call gives its first event. The first call wins: a chain inside
    a chain notes its own attempt before the outer chain notes the whole inner request. Does
    nothing when no ``with_answer_clock`` runs.

    Args:
        started: The ``time.perf_counter()`` reading when the answering attempt began.
    """
    clock = _ANSWER_CLOCK.get()
    if clock is not None and clock.started is None:
        clock.started = started


def attempt_stream(call: ModelCall, request: ModelRequest) -> AsyncIterator[ModelEvent]:
    """Stream one attempt of a chain: through the hook ``with_attempts`` set, else directly.

    Args:
        call: The model call the chain tries.
        request: The request it sends.
    """
    stream = _ATTEMPT_STREAM.get()
    return stream(call, request) if stream is not None else call.stream(request)


async def _within[T](
    events: AsyncIterable[T], hook: ContextVar[Any], value: object
) -> AsyncIterator[T]:
    """Read ``events`` with ``hook`` set to ``value`` only while each item is computed."""
    source = aiter(events)
    try:
        while True:
            token = hook.set(value)
            try:
                item = await anext(source)
            except StopAsyncIteration:
                return
            finally:
                hook.reset(token)
            yield item
    finally:
        if (close := getattr(source, "aclose", None)) is not None:
            await close()


def with_attempts[T](events: AsyncIterable[T], stream: AttemptStream) -> AsyncIterator[T]:
    """Read ``events`` with ``stream`` run around each attempt of a chain inside it.

    The hook is set only while ``events`` computes its next item, so it never leaks to the
    reader, and a reader in another task can close the stream safely.

    Args:
        events: A request's stream, like the chunks an adapter reads from a model call.
        stream: Runs each attempt. A chain inside an attempt runs its own attempts through it
            too.
    """
    return _within(events, _ATTEMPT_STREAM, stream)


def with_answer_clock[T](events: AsyncIterable[T], clock: AnswerClock) -> AsyncIterator[T]:
    """Read ``events`` and let ``clock`` learn when the attempt that answered began.

    Every ``FallbackModelCall`` inside ``events`` reports to ``clock`` when one of its calls
    answers. A request that no call answered leaves ``clock.started`` None. The hook is set
    only while ``events`` computes its next item, as ``with_attempts`` does.

    Args:
        events: A request's stream, like the chunks an adapter reads from a model call.
        clock: Gets the start of the answering attempt.
    """
    return _within(events, _ANSWER_CLOCK, clock)
