"""Turn the chunks of an OpenAI-compatible streaming reply into model events."""

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import replace
from typing import Any

from unmute_core.model._call import ModelEvent, TextDelta, ToolCall, Usage
from unmute_core.model._signatures import EXTRA_CONTENT


async def chunk_events(
    chunks: AsyncIterable[Any], *, request_id: str | None = None
) -> AsyncIterator[ModelEvent]:
    """Yield the text, then each whole tool call, then the ``Usage`` of a chunk stream.

    Tool call pieces are joined by their index. A provider's ``extra_content`` on a piece,
    like Gemini's thought signature, stays on the finished call. The ``Usage`` carries the
    cached input tokens. Every event carries the answer's id from its chunks, as far as
    they have named it, and ``request_id``, so a reply cut off before its ``Usage`` still
    names both.

    Args:
        chunks: ``ChatCompletionChunk`` objects, as the ``openai`` SDK streams them.
        request_id: The provider's ``x-request-id`` for the request, when known.
    """
    calls: dict[int, dict[str, str]] = {}
    extras: dict[int, dict[str, Any]] = {}
    usage: Usage | None = None
    response_id: str | None = None
    async for chunk in chunks:
        response_id = chunk.id or response_id
        if chunk.usage is not None:
            details = getattr(chunk.usage, "prompt_tokens_details", None)
            usage = Usage(
                chunk.usage.prompt_tokens,
                chunk.usage.completion_tokens,
                cached_input_tokens=getattr(details, "cached_tokens", None),
            )
        for choice in chunk.choices:
            if choice.delta.content:
                yield TextDelta(
                    choice.delta.content, response_id=response_id, request_id=request_id
                )
            for part in choice.delta.tool_calls or []:
                call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                call["id"] = part.id or call["id"]
                if part.function:
                    call["name"] += part.function.name or ""
                    call["arguments"] += part.function.arguments or ""
                if isinstance(extra := (part.model_extra or {}).get(EXTRA_CONTENT), dict):
                    extras[part.index] = extras.get(part.index, {}) | extra
    for index, call in calls.items():
        yield ToolCall(
            **call,
            extra=extras.get(index, {}),
            response_id=response_id,
            request_id=request_id,
        )
    if usage is not None:
        yield replace(usage, response_id=response_id, request_id=request_id)
