"""Turn the chunks of an OpenAI-compatible streaming reply into model events."""

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import replace
from typing import Any

from unmute_core.model._call import AnswerStarted, ModelEvent, TextDelta, ToolCall, Usage
from unmute_core.model._signatures import EXTRA_CONTENT


async def chunk_events(
    chunks: AsyncIterable[Any], *, request_id: str | None = None
) -> AsyncIterator[ModelEvent]:
    """Yield ``AnswerStarted``, the text, then each whole tool call, then the ``Usage``.

    ``AnswerStarted`` comes at the first chunk, whatever it holds.

    Tool call pieces are joined by their ``index``. A provider that sends no ``index``, as
    some OpenAI-compatible endpoints do, is joined by the call's ``id``, so several whole
    calls in one chunk stay separate calls. A piece with neither continues the last call.
    Calls come out in the order they first appeared. A provider's ``extra_content`` on a piece,
    like Gemini's thought signature, stays on the finished call. The ``Usage`` carries the
    cached input tokens, the reasoning tokens and the provider's own total. Every event
    carries the answer's id from its chunks, as far as they have named it, and
    ``request_id``, so a reply cut off before its ``Usage`` still names both.

    Args:
        chunks: ``ChatCompletionChunk`` objects, as the ``openai`` SDK streams them.
        request_id: The provider's ``x-request-id`` for the request, when known.
    """
    calls: dict[int | str | None, dict[str, str]] = {}
    extras: dict[int | str | None, dict[str, Any]] = {}
    key: int | str | None = None
    usage: Usage | None = None
    response_id: str | None = None
    started = False
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
            )
        for choice in chunk.choices:
            if choice.delta.content:
                yield TextDelta(
                    choice.delta.content, response_id=response_id, request_id=request_id
                )
            for part in choice.delta.tool_calls or []:
                if part.index is not None:
                    key = part.index
                elif part.id:
                    key = part.id
                call = calls.setdefault(key, {"id": "", "name": "", "arguments": ""})
                call["id"] = part.id or call["id"]
                if part.function:
                    call["name"] += part.function.name or ""
                    call["arguments"] += part.function.arguments or ""
                if isinstance(extra := (part.model_extra or {}).get(EXTRA_CONTENT), dict):
                    extras[key] = extras.get(key, {}) | extra
    for key, call in calls.items():
        yield ToolCall(
            **call,
            extra=extras.get(key, {}),
            response_id=response_id,
            request_id=request_id,
        )
    if usage is not None:
        yield replace(usage, response_id=response_id, request_id=request_id)
