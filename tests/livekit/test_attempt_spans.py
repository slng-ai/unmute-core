"""``ModelCallLLM(trace_attempts=True)``: one LiveKit span pair per call a chain tries."""

import asyncio
import gc
from collections.abc import AsyncIterator

import pytest
from livekit.agents import llm
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from unmute_core.livekit import ModelCallLLM
from unmute_core.model import FallbackModelCall, ModelEvent, ModelRequest, TextDelta, Usage


class ModelNotFoundError(Exception):
    """A model the endpoint does not serve: the chain moves to the next one."""

    status_code = 404


class Model:
    """Answers with one word, or raises, under its own name."""

    def __init__(self, model: str, answer: str | Exception) -> None:
        self.model = model
        self.provider = "test"
        self.answer = answer

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if isinstance(self.answer, Exception):
            raise self.answer
        yield TextDelta(self.answer)
        yield Usage(12, 1)


async def _say(model: ModelCallLLM) -> str:
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="Is my order on its way?")
    async with model.chat(chat_ctx=ctx) as stream:
        return "".join(
            [chunk.delta.content async for chunk in stream if chunk.delta and chunk.delta.content]
        )


def _tree(spans: InMemorySpanExporter) -> list[tuple[object, ...]]:
    finished = spans.get_finished_spans()
    names = {span.context.span_id: span.name for span in finished}
    return sorted(
        (
            span.name,
            names.get(span.parent.span_id, "") if span.parent else "",
            (span.attributes or {}).get("gen_ai.request.model"),
            span.status.status_code.name,
        )
        for span in finished
    )


@pytest.mark.parametrize("trace_attempts", [True, False], ids=["each attempt", "the request"])
async def test_each_call_a_chain_tries_gets_its_own_span_pair(
    spans: InMemorySpanExporter, trace_attempts: bool
) -> None:
    chain = FallbackModelCall(
        [Model("primary", ModelNotFoundError("no such model")), Model("backup", "Yes.")]
    )
    assert await _say(ModelCallLLM(chain, trace_attempts=trace_attempts)) == "Yes."
    if trace_attempts:
        assert _tree(spans) == [
            ("llm_fallback_adapter", "", "primary", "UNSET"),
            ("llm_request", "llm_fallback_adapter", "backup", "UNSET"),
            ("llm_request", "llm_fallback_adapter", "primary", "ERROR"),
            ("llm_request_run", "llm_request", "backup", "UNSET"),
            ("llm_request_run", "llm_request", "primary", "ERROR"),
        ]
    else:
        assert _tree(spans) == [
            ("llm_request", "", "primary", "UNSET"),
            ("llm_request_run", "llm_request", "primary", "UNSET"),
        ]


async def test_a_traced_attempt_gives_the_chain_the_calls_own_error(
    spans: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    error = ModelNotFoundError("no such model")
    seen: list[BaseException | None] = []
    chain = FallbackModelCall(
        [Model("primary", error), Model("backup", "Yes.")],
        on_fail_over=lambda move: seen.append(move.error),
    )
    assert await _say(ModelCallLLM(chain, trace_attempts=True)) == "Yes."
    assert seen == [error]
    gc.collect()
    assert "never retrieved" not in caplog.text


async def test_a_reply_the_caller_talks_over_closes_its_attempt(
    spans: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    closed = asyncio.Event()

    class Endless:
        model = "primary"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            try:
                yield TextDelta("Your order")
                await asyncio.Event().wait()
            finally:
                closed.set()

    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="Is my order on its way?")
    model = ModelCallLLM(FallbackModelCall([Endless()]), trace_attempts=True)
    async with model.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                break
    await asyncio.wait_for(closed.wait(), 2)
    assert "never retrieved" not in caplog.text
