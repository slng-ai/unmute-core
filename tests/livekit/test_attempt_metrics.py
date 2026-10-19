"""After a fallback, a request's ``LLMMetrics`` are timed from the call that answered."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from livekit.agents import AgentSession, llm, metrics
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from unmute_core.agent import AgentSpec
from unmute_core.livekit import ModelCallLLM, UnmuteAgent
from unmute_core.model import FallbackModelCall, ModelEvent, ModelRequest, TextDelta, Usage

FAILED_AFTER = 0.25
"""How long the first call takes to fail. Metrics that count it exceed this."""
TRACE = pytest.mark.parametrize(
    "trace_attempts", [True, False], ids=["each attempt", "the request"]
)


class SlowFailureError(Exception):
    """A model that timed out: the chain moves to the next one."""

    status_code = 504


class Fails:
    """A model call that fails after a delay."""

    model = "primary"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        await asyncio.sleep(FAILED_AFTER)
        raise SlowFailureError("timed out")
        yield TextDelta("")


class Answers:
    """A model call that answers at once, or with an empty reply."""

    model = "backup"

    def __init__(self, text: str = "Yes.") -> None:
        self.text = text

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if self.text:
            yield TextDelta(self.text, response_id="chatcmpl-1")
            await asyncio.sleep(0)
        yield Usage(12, 4, response_id="chatcmpl-1")


async def _reported(model: ModelCallLLM) -> list[metrics.LLMMetrics]:
    reported: list[metrics.LLMMetrics] = []
    model.on("metrics_collected", reported.append)
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="Is my order on its way?")
    async with model.chat(chat_ctx=ctx) as stream:
        async for _ in stream:
            pass
    return reported


@TRACE
async def test_metrics_after_a_fallback_start_at_the_call_that_answered(
    trace_attempts: bool,
) -> None:
    chain = FallbackModelCall([Fails(), Answers()])
    [reported] = await _reported(ModelCallLLM(chain, trace_attempts=trace_attempts))
    assert 0 <= reported.ttft < FAILED_AFTER
    assert 0 <= reported.duration < FAILED_AFTER
    assert reported.ttft <= reported.duration
    assert (reported.request_id, reported.completion_tokens, reported.prompt_tokens) == (
        "chatcmpl-1",
        4,
        12,
    )
    assert reported.tokens_per_second == pytest.approx(4 / reported.duration)
    assert reported.cancelled is False


@TRACE
async def test_metrics_after_a_pause_between_retries_start_at_the_retry(
    trace_attempts: bool,
) -> None:
    chain = FallbackModelCall([Fails(), Answers()], same_model_retries=(0.2,))
    [reported] = await _reported(ModelCallLLM(chain, trace_attempts=trace_attempts))
    assert reported.duration < FAILED_AFTER


@TRACE
async def test_metrics_of_a_request_no_call_answered_keep_the_whole_time(
    trace_attempts: bool,
) -> None:
    chain = FallbackModelCall([Fails(), Answers("")])
    [reported] = await _reported(ModelCallLLM(chain, trace_attempts=trace_attempts))
    assert reported.ttft == -1.0
    assert reported.duration >= FAILED_AFTER


@TRACE
async def test_metrics_of_a_call_that_needed_no_fallback_are_unchanged(
    trace_attempts: bool,
) -> None:
    [reported] = await _reported(
        ModelCallLLM(FallbackModelCall([Answers()]), trace_attempts=trace_attempts)
    )
    assert 0 <= reported.ttft <= reported.duration < FAILED_AFTER


@TRACE
async def test_the_request_span_holds_the_same_metrics(
    spans: InMemorySpanExporter, trace_attempts: bool
) -> None:
    model = ModelCallLLM(FallbackModelCall([Fails(), Answers()]), trace_attempts=trace_attempts)
    [reported] = await _reported(model)
    [request] = [
        span
        for span in spans.get_finished_spans()
        if span.name in ("llm_request", "llm_fallback_adapter") and span.parent is None
    ]
    held = metrics.LLMMetrics.model_validate_json(str((request.attributes or {})["lk.llm_metrics"]))
    assert (held.ttft, held.duration) == (reported.ttft, reported.duration)
    assert held.duration < FAILED_AFTER
    assert (request.attributes or {})["gen_ai.response.time_to_first_chunk"] == reported.ttft


async def test_an_agent_turn_after_a_fallback_reports_the_answering_calls_time() -> None:
    collected: list[metrics.AgentMetrics] = []
    agent = UnmuteAgent(AgentSpec("Orders.", model=FallbackModelCall([Fails(), Answers()])))
    async with AgentSession() as session:
        session.on("metrics_collected", lambda event: collected.append(event.metrics))
        await session.start(agent)
        await session.run(user_input="Hi")

    [reported] = [item for item in collected if isinstance(item, metrics.LLMMetrics)]
    assert 0 <= reported.ttft <= reported.duration < FAILED_AFTER
    assert (reported.request_id, reported.completion_tokens) == ("chatcmpl-1", 4)
