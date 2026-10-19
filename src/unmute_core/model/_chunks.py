"""Turn the chunks of an OpenAI-compatible streaming reply into model events."""

import logging
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import replace
from typing import Any

from unmute_core.model._call import AnswerStarted, ModelEvent, TextDelta, ToolCall, Usage
from unmute_core.model._signatures import EXTRA_CONTENT

logger = logging.getLogger(__name__)


class _ToolCalls:
    """The tool calls of one reply, joined from their pieces in the order they began."""

    def __init__(self, request_id: str | None) -> None:
        """Start with no calls.

        Args:
            request_id: The provider's ``x-request-id`` for the request, put on each call.
        """
        self._request_id = request_id
        self._calls: dict[int | str | None, dict[str, str]] = {}
        self._extras: dict[int | str | None, dict[str, Any]] = {}
        self._key: int | str | None = None
        self._sent: set[int | str | None] = set()

    def add(self, part: Any, response_id: str | None) -> list[ToolCall]:
        """Join one piece into its call, and return the calls that piece shows to be whole.

        Pieces are joined by ``index``. A provider that sends no ``index`` is joined by the
        call's ``id``. A piece with neither continues the last call. A piece that starts a
        new call ends every call before it. A piece for a call that already went out is
        dropped with a warning, as the call cannot be taken back.

        Args:
            part: One ``ChoiceDeltaToolCall`` of a chunk.
            response_id: The answer's id, as far as the chunks have named it.
        """
        if part.index is not None:
            self._key = part.index
        elif part.id:
            self._key = part.id
        key = self._key
        if key in self._sent:
            logger.warning("tool call piece %r came after its call was sent, dropped", key)
            return []
        whole = [] if key in self._calls else self.drain(response_id)
        call = self._calls.setdefault(key, {"id": "", "name": "", "arguments": ""})
        call["id"] = part.id or call["id"]
        if part.function:
            call["name"] += part.function.name or ""
            call["arguments"] += part.function.arguments or ""
        if isinstance(extra := (part.model_extra or {}).get(EXTRA_CONTENT), dict):
            self._extras[key] = self._extras.get(key, {}) | extra
        return whole

    def drain(self, response_id: str | None) -> list[ToolCall]:
        """Return every call so far, in the order they began, and forget them.

        Args:
            response_id: The answer's id, as far as the chunks have named it.
        """
        whole = [
            ToolCall(
                **call,
                extra=self._extras.get(key, {}),
                response_id=response_id,
                request_id=self._request_id,
            )
            for key, call in self._calls.items()
        ]
        self._sent |= self._calls.keys()
        self._calls.clear()
        self._extras.clear()
        return whole


async def chunk_events(
    chunks: AsyncIterable[Any], *, request_id: str | None = None
) -> AsyncIterator[ModelEvent]:
    """Yield ``AnswerStarted``, the text, each tool call once it is whole, then the ``Usage``.

    ``AnswerStarted`` comes at the first chunk, whatever it holds.

    Tool call pieces are joined by their ``index``. A provider that sends no ``index``, as
    some OpenAI-compatible endpoints do, is joined by the call's ``id``, so several whole
    calls in one chunk stay separate calls. A piece with neither continues the last call.
    A call is whole when a later piece starts another call, or at a chunk whose choice has a
    ``finish_reason``, and it goes out then, without waiting for the rest of the stream. Calls
    still open when the stream ends, as when a provider sends no ``finish_reason``, go out at the
    end. Calls come out in the order they first appeared. A provider's ``extra_content`` on a
    piece, like Gemini's thought signature, stays on its call. A provider's ``extra_content``
    on the text goes out on the text: data that comes before any text waits for the first
    piece, and data with no text at all is dropped, so it never starts an answer. The ``Usage``
    carries the cached input tokens, the tokens written to the cache, the reasoning tokens and
    the provider's own total. Every event carries the answer's id from its chunks, as far as
    they have named it, and ``request_id``, so a reply cut off before its ``Usage`` still
    names both.

    Args:
        chunks: ``ChatCompletionChunk`` objects, as the ``openai`` SDK streams them.
        request_id: The provider's ``x-request-id`` for the request, when known.
    """
    calls = _ToolCalls(request_id)
    usage: Usage | None = None
    response_id: str | None = None
    started = False
    spoken = False
    held: dict[str, Any] = {}
    async for chunk in chunks:
        response_id = chunk.id or response_id
        if not started:
            started = True
            yield AnswerStarted(response_id=response_id, request_id=request_id)
        if chunk.usage is not None:
            details = getattr(chunk.usage, "prompt_tokens_details", None)
            output_details = getattr(chunk.usage, "completion_tokens_details", None)
            usage = Usage(
                chunk.usage.prompt_tokens,
                chunk.usage.completion_tokens,
                cached_input_tokens=getattr(details, "cached_tokens", None),
                reasoning_tokens=getattr(output_details, "reasoning_tokens", None),
                total_tokens=getattr(chunk.usage, "total_tokens", None),
                cache_write_tokens=getattr(details, "cache_write_tokens", None),
            )
        for choice in chunk.choices:
            if isinstance(extra := (choice.delta.model_extra or {}).get(EXTRA_CONTENT), dict):
                held |= extra
            spoken = spoken or bool(choice.delta.content)
            if spoken and (choice.delta.content or held):
                yield TextDelta(
                    choice.delta.content or "",
                    response_id=response_id,
                    request_id=request_id,
                    extra=held,
                )
                held = {}
            for part in choice.delta.tool_calls or []:
                for call in calls.add(part, response_id):
                    yield call
            if choice.finish_reason:
                for call in calls.drain(response_id):
                    yield call
    for call in calls.drain(response_id):
        yield call
    if usage is not None:
        yield replace(usage, response_id=response_id, request_id=request_id)
