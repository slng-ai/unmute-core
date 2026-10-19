"""``chunk_events`` turns OpenAI-compatible stream chunks into model events."""

from collections.abc import AsyncIterator
from typing import Any

from openai.types.chat import ChatCompletionChunk

from unmute_core.model import AnswerStarted, ModelEvent, Usage
from unmute_core.model._chunks import chunk_events


def usage_chunk(usage: dict[str, Any]) -> ChatCompletionChunk:
    """A final chunk with no choice, as a provider sends the token counts."""
    return ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [],
            "usage": usage,
        }
    )


async def stream(*chunks: ChatCompletionChunk) -> AsyncIterator[ChatCompletionChunk]:
    """Yield the chunks as a provider's stream would."""
    for item in chunks:
        yield item


async def events_of(*chunks: ChatCompletionChunk) -> list[ModelEvent]:
    """Run the chunks through ``chunk_events`` and collect what comes out."""
    return [event async for event in chunk_events(stream(*chunks))]


async def test_usage_carries_the_tokens_written_to_the_cache() -> None:
    events = await events_of(
        usage_chunk(
            {
                "prompt_tokens": 120,
                "completion_tokens": 7,
                "total_tokens": 127,
                "prompt_tokens_details": {"cached_tokens": 96, "cache_write_tokens": 24},
            }
        )
    )
    assert events == [
        AnswerStarted(response_id="c1"),
        Usage(
            120,
            7,
            cached_input_tokens=96,
            total_tokens=127,
            response_id="c1",
            cache_write_tokens=24,
        ),
    ]


async def test_usage_has_no_cache_write_tokens_when_the_provider_sends_none() -> None:
    plain = usage_chunk({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})
    cached = usage_chunk(
        {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 2},
        }
    )
    [_, plain_usage] = await events_of(plain)
    [_, cached_usage] = await events_of(cached)
    assert isinstance(plain_usage, Usage)
    assert isinstance(cached_usage, Usage)
    assert (plain_usage.cache_write_tokens, cached_usage.cache_write_tokens) == (None, None)
