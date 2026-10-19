"""A hook around each attempt of a ``FallbackModelCall``, for code that cannot reach the chain.

An adapter traces each call a chain tries, as LiveKit's ``FallbackAdapter`` does. The chain is
often out of its reach: an app builds one per call, or one per request inside its own model
call. So the adapter sets the hook for the time a request runs, and every chain in that
request reads it.
"""

from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextvars import ContextVar

from unmute_core.model._call import ModelCall, ModelEvent, ModelRequest

AttemptStream = Callable[[ModelCall, ModelRequest], AsyncIterator[ModelEvent]]
"""Streams one attempt: given the call a chain tries and its request, yield the call's events."""

_ATTEMPT_STREAM: ContextVar[AttemptStream | None] = ContextVar(
    "unmute_attempt_stream", default=None
)


def attempt_stream(call: ModelCall, request: ModelRequest) -> AsyncIterator[ModelEvent]:
    """Stream one attempt of a chain: through the hook ``with_attempts`` set, else directly.

    Args:
        call: The model call the chain tries.
        request: The request it sends.
    """
    stream = _ATTEMPT_STREAM.get()
    return stream(call, request) if stream is not None else call.stream(request)


async def with_attempts[T](events: AsyncIterable[T], stream: AttemptStream) -> AsyncIterator[T]:
    """Read ``events`` with ``stream`` run around each attempt of a chain inside it.

    The hook is set only while ``events`` computes its next item, so it never leaks to the
    reader, and a reader in another task can close the stream safely.

    Args:
        events: A request's stream, like the chunks an adapter reads from a model call.
        stream: Runs each attempt. A chain inside an attempt runs its own attempts through it
            too.
    """
    source = aiter(events)
    try:
        while True:
            token = _ATTEMPT_STREAM.set(stream)
            try:
                item = await anext(source)
            except StopAsyncIteration:
                return
            finally:
                _ATTEMPT_STREAM.reset(token)
            yield item
    finally:
        if (close := getattr(source, "aclose", None)) is not None:
            await close()
