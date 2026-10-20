"""Every chunk of a ``ModelCallLLM`` request carries the id the provider gave the answer."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import pytest
from livekit.agents import AgentSession, APIConnectOptions, NotGivenOr, llm, metrics
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from unmute_core.agent import AgentSpec
from unmute_core.livekit import LiveKitModel, ModelCallLLM, UnmuteAgent
from unmute_core.model import (
    FallbackModelCall,
    ModelEvent,
    ModelRequest,
    TextDelta,
    ToolCall,
    ToolCallStarted,
    Usage,
)

ID = "chatcmpl-7"
TRACE = pytest.mark.parametrize(
    "trace_attempts", [True, False], ids=["each attempt", "the request"]
)


class Events:
    """A model call that streams the events it was given."""

    def __init__(self, *events: ModelEvent) -> None:
        self.model = "primary"
        self.events = events

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        for event in self.events:
            yield event
            await asyncio.sleep(0)


def _llm(
    trace_attempts: bool, *events: ModelEvent
) -> tuple[ModelCallLLM, list[metrics.LLMMetrics]]:
    model = ModelCallLLM(FallbackModelCall([Events(*events)]), trace_attempts=trace_attempts)
    reported: list[metrics.LLMMetrics] = []
    model.on("metrics_collected", reported.append)
    return model, reported


async def _chunks(model: ModelCallLLM, *, stop_at: int | None = None) -> list[llm.ChatChunk]:
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="Is my order on its way?")
    chunks: list[llm.ChatChunk] = []
    async with model.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            chunks.append(chunk)
            if stop_at is not None and len(chunks) == stop_at:
                break
    return chunks


@TRACE
async def test_metrics_name_the_completion_id_when_the_reply_has_no_usage_chunk(
    trace_attempts: bool,
) -> None:
    model, reported = _llm(
        trace_attempts,
        TextDelta("Your order", response_id=ID),
        TextDelta(" is on its way.", response_id=ID),
    )
    chunks = await _chunks(model)
    assert [chunk.id for chunk in chunks] == [ID, ID]
    [only] = reported
    assert only.request_id == ID


@TRACE
async def test_metrics_of_a_reply_the_caller_talked_over_name_the_completion_id(
    trace_attempts: bool,
) -> None:
    model, reported = _llm(
        trace_attempts,
        TextDelta("Your order", response_id=ID),
        TextDelta(" is on its way.", response_id=ID),
    )
    await _chunks(model, stop_at=1)
    [only] = reported
    assert only.request_id == ID


@TRACE
async def test_text_tool_calls_and_usage_all_carry_the_completion_id(trace_attempts: bool) -> None:
    model, reported = _llm(
        trace_attempts,
        TextDelta("One moment.", response_id=ID),
        ToolCallStarted("call_1", "lookup_order", response_id=ID),
        ToolCall("call_1", "lookup_order", "{}", response_id=ID),
        Usage(11, 4, response_id=ID),
    )
    chunks = await _chunks(model)
    assert [chunk.id for chunk in chunks] == [ID, ID, ID]
    assert [r.request_id for r in reported] == [ID]


@TRACE
async def test_a_reply_with_no_provider_id_gets_a_random_one(trace_attempts: bool) -> None:
    model, reported = _llm(trace_attempts, TextDelta("Hi."), Usage(11, 4))
    chunks = await _chunks(model)
    assert len({chunk.id for chunk in chunks}) >= 1
    assert all(chunk.id.startswith("unmute_") for chunk in chunks)
    assert reported[0].request_id.startswith("unmute_")


@TRACE
async def test_words_a_reasoning_filter_held_back_keep_the_completion_id(
    trace_attempts: bool,
) -> None:
    model, _ = _llm(trace_attempts, TextDelta("Done. <thi", response_id=ID))
    chunks = await _chunks(model)
    assert "".join(chunk.delta.content or "" for chunk in chunks if chunk.delta) == "Done. <thi"
    assert {chunk.id for chunk in chunks} == {ID}


@TRACE
async def test_every_span_names_only_the_completion_id(
    spans: InMemorySpanExporter, trace_attempts: bool
) -> None:
    model, _ = _llm(
        trace_attempts,
        TextDelta("One moment.", response_id=ID),
        ToolCall("call_1", "lookup_order", "{}", response_id=ID),
        Usage(11, 4, response_id=ID),
    )
    await _chunks(model)
    named = [
        tuple(cast("Sequence[str]", (span.attributes or {})["lk.provider_request_ids"]))
        for span in spans.get_finished_spans()
        if "lk.provider_request_ids" in (span.attributes or {})
    ]
    assert named
    assert set(named) == {(ID,)}


class _Answers(llm.LLMStream):
    """Sends a scripted list of chunks."""

    def __init__(self, owner: llm.LLM, chunks: Sequence[llm.ChatChunk], **kwargs: Any) -> None:
        super().__init__(owner, **kwargs)
        self._chunks = chunks

    async def _run(self) -> None:
        for chunk in self._chunks:
            self._event_ch.send_nowait(chunk)


class Plugin(llm.LLM):
    """A LiveKit LLM whose chunks carry a completion id."""

    @property
    def model(self) -> str:
        return "gpt-4o"

    @property
    def provider(self) -> str:
        return "scripted"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        call = llm.FunctionToolCall(name="lookup_order", arguments="{}", call_id="call_1")
        chunks = [
            llm.ChatChunk(id=ID, delta=llm.ChoiceDelta(role="assistant", content="Hi.")),
            llm.ChatChunk(id=ID, delta=llm.ChoiceDelta(role="assistant", tool_calls=[call])),
        ]
        return _Answers(
            self, chunks, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


async def test_a_livekit_plugin_in_a_chain_passes_its_completion_id_on() -> None:
    events = [event async for event in LiveKitModel(Plugin()).stream(ModelRequest(messages=[]))]
    assert [event.response_id for event in events if not isinstance(event, Usage)] == [ID, ID]


async def test_an_agent_turn_reports_the_completion_id_without_a_usage_chunk() -> None:
    collected: list[metrics.AgentMetrics] = []
    script = deque([[TextDelta("Hello.", response_id=ID)]])

    class Once:
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            for event in script.popleft():
                yield event

    agent = UnmuteAgent(AgentSpec("Orders.", model=Once()))
    async with AgentSession() as session:
        session.on("metrics_collected", lambda event: collected.append(event.metrics))
        await session.start(agent)
        await session.run(user_input="Hi")

    [reported] = [item for item in collected if isinstance(item, metrics.LLMMetrics)]
    assert reported.request_id == ID
