"""A model that fails on LiveKit: in the middle of a reply, before any text, and after a
failover. LiveKit's own retry decides part of it, so these cases run on LiveKit only.
"""

from collections.abc import AsyncIterator
from typing import cast

import pytest
from livekit.agents import AgentSession, APIConnectOptions, llm
from livekit.agents.voice.agent_session import SessionConnectOptions

from tests.livekit.test_standalone import FailsOnce, StatusError
from tests.livekit.test_trace import Memory
from tests.tasks.job import Canned, plan, registry, state
from unmute_core.agent import AgentSpec
from unmute_core.livekit import (
    ModelCallLLM,
    UnmuteAgent,
)
from unmute_core.model import FallbackModelCall, ModelEvent, ModelRequest, TextDelta, ToolCall
from unmute_core.telemetry import Telemetry

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}


class Scripted:
    """A ``ModelCall`` that plays one script per request. An ``Exception`` in it is raised."""

    def __init__(self, *scripts: list[object], model: str = "scripted") -> None:
        self.scripts = list(scripts)
        self.requests: list[ModelRequest] = []
        self.model = model
        self.closed = 0

    async def aclose(self) -> None:
        """Count the close."""
        self.closed += 1

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Play the next script."""
        self.requests.append(request)
        for item in self.scripts.pop(0) if self.scripts else []:
            if isinstance(item, Exception):
                raise item
            yield cast(ModelEvent, item)


def call_event(name: str) -> ToolCall:
    """A model event that calls a tool with no arguments."""
    return ToolCall(f"call_{name}", name, "{}", {})


class Down:
    """A ``ModelCall`` that always fails with an HTTP status, as an SDK error does."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Fail at once."""
        self.requests += 1
        raise StatusError("down")
        yield TextDelta("never")


async def test_a_reply_that_fails_after_its_text_is_not_sent_again() -> None:
    model = Scripted([TextDelta("Hello"), StatusError("dropped")], [TextDelta("again")])
    stream = ModelCallLLM(model).chat(
        chat_ctx=llm.ChatContext.empty(),
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
    )
    text = [chunk.delta.content async for chunk in stream if chunk.delta]
    assert text == ["Hello"]
    assert len(model.requests) == 1


async def test_a_request_that_fails_before_any_text_is_still_retried() -> None:
    model = FailsOnce(StatusError("boom"))
    stream = ModelCallLLM(model).chat(
        chat_ctx=llm.ChatContext.empty(),
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
    )
    assert [chunk.delta.content async for chunk in stream if chunk.delta] == ["Hello."]
    assert model.requests == 2


async def test_a_step_reply_that_fails_after_its_text_is_not_asked_again_or_checked() -> None:
    model = Scripted(
        [call_event("find_order")], [TextDelta("Let me look"), RuntimeError("dropped")]
    )
    tasks = plan(tools=registry(lookup_order=Canned(FOUND)))
    agent = UnmuteAgent(AgentSpec("Orders.", tasks=tasks, model=model), state=state(tasks))
    async with AgentSession() as session:
        await session.start(agent)
        await session.run(user_input="Where is my order?")
    assert [request.site for request in model.requests] == ["", "task:find_order"]


async def test_an_agent_retries_its_model_call_as_the_session_does_before_any_event() -> None:
    model = FailsOnce(StatusError("boom"))
    agent = UnmuteAgent(AgentSpec("You help.", model=model))
    async with AgentSession(conn_options=_quick()) as session:
        await session.start(agent)
        await session.run(user_input="Hi.")
        said = [m.text_content for m in session.history.messages() if m.role == "assistant"]
    assert model.requests == 2
    assert said == ["Hello."]


async def test_an_agent_does_not_retry_a_model_call_that_retries_itself() -> None:
    first, second = Down("a"), Down("b")
    agent = UnmuteAgent(AgentSpec("You help.", model=FallbackModelCall([first, second])))
    async with AgentSession(conn_options=_quick()) as session:
        await session.start(agent)
        with pytest.raises(Exception, match="down"):
            await session.run(user_input="Hi.")
    assert (first.requests, second.requests) == (1, 1)


def _quick() -> SessionConnectOptions:
    """Session connection options that retry with no pause."""
    return SessionConnectOptions(llm_conn_options=APIConnectOptions(max_retry=2, retry_interval=0))


async def test_the_span_names_the_model_that_answered_after_a_failover() -> None:
    sink = Memory()
    chain = FallbackModelCall([Down("first"), Scripted([TextDelta("Hi.")], model="second")])
    agent = UnmuteAgent(
        AgentSpec("You help.", model=chain), telemetry=Telemetry(sink, agent="desk", call_id="RM_1")
    )
    async with AgentSession() as session:
        await session.start(agent)
        await session.run(user_input="Hi.")
    assert agent._trace is not None and agent._trace.closing is not None
    await agent._trace.closing
    [span] = [s for s in sink.spans.get_finished_spans() if s.name == "llm"]
    attributes = dict(span.attributes or {})
    assert attributes["gen_ai.request.model"] == "first"
    assert attributes["gen_ai.response.model"] == "second"


async def test_a_chain_that_is_not_sticky_starts_the_next_request_on_its_first_call() -> None:
    sink = Memory()
    chain = FallbackModelCall(
        [Down("first"), Scripted([TextDelta("Hi."), TextDelta("Hi.")], model="second")]
    )
    agent = UnmuteAgent(
        AgentSpec("You help.", model=chain), telemetry=Telemetry(sink, agent="desk", call_id="RM_1")
    )
    async with AgentSession() as session:
        await session.start(agent)
        await session.run(user_input="Hi.")
        await session.run(user_input="Again.")
    assert agent._trace is not None and agent._trace.closing is not None
    await agent._trace.closing
    spans = [s for s in sink.spans.get_finished_spans() if s.name == "llm"]
    assert len(spans) == 2
    for span in spans:
        attributes = dict(span.attributes or {})
        assert attributes["gen_ai.request.model"] == "first"
        assert attributes["gen_ai.response.model"] == "second"
