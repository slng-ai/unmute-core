"""``chunk_events`` turns OpenAI-compatible stream chunks into model events."""

from collections.abc import AsyncIterator
from typing import Any, cast

from openai._models import construct_type
from openai.types.chat import ChatCompletionChunk

from unmute_core.model import AnswerStarted, ModelEvent, ToolCall, ToolCallStarted, Usage
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


SIGNED = {"google": {"thought_signature": "sig-1"}}


def piece(
    index: int | None, call_id: str | None, name: str | None, arguments: str, **extra: Any
) -> dict[str, Any]:
    """One tool call piece, as a provider streams it."""
    function = {"name": name, "arguments": arguments} if name else {"arguments": arguments}
    return {"index": index, "id": call_id, "type": "function", "function": function, **extra}


def calls_chunk(*pieces: dict[str, Any], finish_reason: str | None = None) -> ChatCompletionChunk:
    """A chunk whose one choice carries these tool call pieces.

    The chunk is built the way the SDK builds a streamed one, without validation, so a piece
    may send a null ``index`` as some providers do.
    """
    return cast(
        ChatCompletionChunk,
        construct_type(
            type_=ChatCompletionChunk,
            value={
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": list(pieces)},
                        "finish_reason": finish_reason,
                    }
                ],
            },
        ),
    )


USAGE = usage_chunk({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})


async def reads_at_each_event(
    *chunks: ChatCompletionChunk, tool_call_starts: bool = False
) -> list[tuple[ModelEvent, int]]:
    """Run the chunks through ``chunk_events``, noting how many chunks were read at each event."""
    read = 0

    async def counted() -> AsyncIterator[ChatCompletionChunk]:
        nonlocal read
        for item in chunks:
            read += 1
            yield item

    return [
        (event, read) async for event in chunk_events(counted(), tool_call_starts=tool_call_starts)
    ]


def tool_calls(events: list[ModelEvent]) -> list[ToolCall]:
    """The tool calls among the events, in order."""
    return [event for event in events if isinstance(event, ToolCall)]


async def test_a_call_is_yielded_before_the_next_call_has_streamed() -> None:
    seen = await reads_at_each_event(
        calls_chunk(piece(0, "t1", "book", '{"a"')),
        calls_chunk(piece(0, None, None, ": 1}")),
        calls_chunk(piece(1, "t2", "find", '{"b"')),
        calls_chunk(piece(1, None, None, ": 2}")),
        calls_chunk(piece(1, None, None, ""), finish_reason="tool_calls"),
        USAGE,
    )
    [(first, first_read), (second, second_read)] = [
        (event, read) for event, read in seen if isinstance(event, ToolCall)
    ]
    assert (first.id, first.arguments, first_read) == ("t1", '{"a": 1}', 3)
    assert (second.id, second.arguments, second_read) == ("t2", '{"b": 2}', 5)


async def test_a_call_is_yielded_at_its_finish_reason_before_the_usage_chunk() -> None:
    seen = await reads_at_each_event(
        calls_chunk(piece(0, "t1", "book", "{}")),
        calls_chunk(finish_reason="tool_calls"),
        USAGE,
    )
    assert [(type(event).__name__, read) for event, read in seen] == [
        ("AnswerStarted", 1),
        ("ToolCall", 2),
        ("Usage", 3),
    ]


async def test_calls_with_no_finish_reason_are_yielded_at_the_end_before_the_usage() -> None:
    seen = await reads_at_each_event(
        calls_chunk(piece(0, "t1", "book", "{}")),
        calls_chunk(piece(1, "t2", "find", "{}")),
        USAGE,
    )
    assert [(type(event).__name__, read) for event, read in seen] == [
        ("AnswerStarted", 1),
        ("ToolCall", 2),
        ("ToolCall", 3),
        ("Usage", 3),
    ]


async def test_calls_without_an_index_stay_apart_in_one_chunk_and_across_chunks() -> None:
    events = await events_of(
        calls_chunk(piece(None, "t1", "book", "{}"), piece(None, "t2", "find", "{}")),
        calls_chunk(piece(None, "t3", "list", "{"), piece(None, None, None, "}")),
        calls_chunk(piece(None, "t4", "end", "{}")),
    )
    assert [(call.id, call.name, call.arguments) for call in tool_calls(events)] == [
        ("t1", "book", "{}"),
        ("t2", "find", "{}"),
        ("t3", "list", "{}"),
        ("t4", "end", "{}"),
    ]


async def test_parallel_calls_keep_their_order_and_their_provider_data() -> None:
    late = {"extra_content": SIGNED}
    events = await events_of(
        calls_chunk(piece(0, "t1", "book", "{")),
        calls_chunk(piece(0, None, None, "}", **late)),
        calls_chunk(piece(1, "t2", "find", "{}")),
        calls_chunk(piece(2, "t3", "list", "{}", extra_content={"other": 1})),
        calls_chunk(finish_reason="tool_calls"),
    )
    assert [(call.id, call.arguments, dict(call.extra)) for call in tool_calls(events)] == [
        ("t1", "{}", SIGNED),
        ("t2", "{}", {}),
        ("t3", "{}", {"other": 1}),
    ]


async def test_a_piece_for_a_call_that_was_already_sent_is_dropped() -> None:
    events = await events_of(
        calls_chunk(piece(0, "t1", "book", '{"a": 1}')),
        calls_chunk(piece(1, "t2", "find", "{}")),
        calls_chunk(piece(0, None, None, "junk")),
    )
    assert [(call.id, call.arguments) for call in tool_calls(events)] == [
        ("t1", '{"a": 1}'),
        ("t2", "{}"),
    ]


TWO_CALLS = (
    calls_chunk(piece(0, "t1", "book", '{"a"')),
    calls_chunk(piece(0, None, None, ": 1}")),
    calls_chunk(piece(1, "t2", "find", "{}"), finish_reason="tool_calls"),
    USAGE,
)


async def test_no_tool_call_start_is_yielded_unless_asked() -> None:
    seen = await reads_at_each_event(*TWO_CALLS)
    assert not [event for event, _ in seen if isinstance(event, ToolCallStarted)]


async def test_each_tool_call_start_comes_at_its_first_piece() -> None:
    seen = await reads_at_each_event(*TWO_CALLS, tool_call_starts=True)
    assert [(type(event).__name__, read) for event, read in seen] == [
        ("AnswerStarted", 1),
        ("ToolCallStarted", 1),
        ("ToolCall", 3),
        ("ToolCallStarted", 3),
        ("ToolCall", 3),
        ("Usage", 4),
    ]
    starts = [event for event, _ in seen if isinstance(event, ToolCallStarted)]
    assert starts == [
        ToolCallStarted(id="t1", name="book", response_id="c1"),
        ToolCallStarted(id="t2", name="find", response_id="c1"),
    ]
