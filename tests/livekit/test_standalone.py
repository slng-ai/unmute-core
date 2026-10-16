"""The LiveKit pieces work on a plain ``Agent``, without ``UnmuteAgent``.

What they do on a call is in tests/conformance (the ``livekit-plain`` driver). These cases
cover the rest: span names, the prompt site and values, a refused tool, the ``llm`` span of
a native LLM, and a tool that waits for its line.
"""

from collections.abc import AsyncIterator
from typing import Any

import httpx
import httpx2
import openai
import pytest
from livekit.agents import (
    Agent,
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    llm,
)
from livekit.agents.voice import ModelSettings
from openai import AsyncOpenAI

from tests.livekit.fake import ScriptedLLM, call
from tests.livekit.test_model import ScriptedCall
from tests.livekit.test_trace import Memory
from tests.outputs import AnyResult
from tests.tasks.job import Canned, Sending, job, plan, registry, state
from unmute_core.call import CallHooks
from unmute_core.hooks import Hooks, PromptSuffix
from unmute_core.livekit import (
    ModelCallLLM,
    attach_call,
    filled_llm,
    function_tools,
    trace_session,
    traced_llm,
)
from unmute_core.livekit._model import livekit_error
from unmute_core.model import (
    FallbackModelCall,
    MissingVariablesError,
    ModelEvent,
    ModelRequest,
    TextDelta,
)
from unmute_core.slng import RouterModelCall
from unmute_core.state import CallInfo
from unmute_core.telemetry import Telemetry
from unmute_core.tools import ToolError, ToolRegistry, ToolSpec


class Refuses:
    """A tool executor that always refuses."""

    async def run(self, arguments: dict[str, Any], state: object) -> object:
        raise ToolError("The order system is down.")


def _router() -> RouterModelCall:
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://router.test/v1",
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(lambda r: httpx2.Response(500))
        ),
    )
    return RouterModelCall(CallInfo(agent_id="a", session_id="s"), client=client)


def test_the_llm_names_the_model_and_who_serves_it() -> None:
    router = _router()
    assert (ModelCallLLM(router).model, ModelCallLLM(router).provider) == ("slng/auto", "slng")
    fallback = ModelCallLLM(FallbackModelCall([router, ScriptedCall()]))
    assert (fallback.model, fallback.provider) == ("slng/auto", "slng")
    assert (ModelCallLLM(ScriptedCall()).model, ModelCallLLM(ScriptedCall()).provider) == (
        "unknown",
        "unknown",
    )


async def test_a_plain_agent_sends_the_prompt_unfilled_with_its_values_at_its_site() -> None:
    task_plan = plan(job(mode="status"))
    call_state = state(task_plan)
    call_state.save({"order_id": "B-7"}, step="find_order")
    called = ScriptedCall("Order B-7 is open.")
    agent = Agent(
        instructions="Order {{order_id}}.",
        llm=ModelCallLLM(called, state=call_state, site="task:find_order"),
    )
    async with AgentSession() as session:
        await session.start(agent)
        await session.run(user_input="Where is my order?")

    [request] = called.requests
    assert request.site == "task:find_order"
    assert request.variables == {"order_id": "B-7"}
    assert any(m["content"] == "Order {{order_id}}." for m in request.messages)


async def test_a_refused_tool_reaches_the_model_as_an_error() -> None:
    called = ScriptedCall(call("search_web", q="hours"), "Sorry, try later.")
    tools = registry(search_web=Refuses(), normalize=Canned())
    agent = Agent(instructions="Help.", tools=function_tools(tools), llm=ModelCallLLM(called))
    async with AgentSession() as session:
        await session.start(agent)
        await session.run(user_input="When are you open?")

    outputs = [m["content"] for m in called.requests[1].messages if m["role"] == "tool"]
    assert outputs == ["The order system is down."]


class StatusError(Exception):
    """An SDK error that carries an HTTP status, like openai's ``APIStatusError``."""

    status_code = 500


class FailsOnce:
    """A ``ModelCall`` that fails its first request with ``error``, then answers."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Fail once, then answer."""
        self.requests += 1
        if self.requests == 1:
            raise self.error
        yield TextDelta("Hello.")


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (StatusError("boom"), APIStatusError),
        (httpx.ReadTimeout("slow"), APITimeoutError),
        (httpx.ConnectError("down"), APIConnectionError),
        (httpx2.ReadTimeout("slow"), APITimeoutError),
        (httpx2.ConnectError("down"), APIConnectionError),
        (openai.APITimeoutError(httpx2.Request("POST", "https://m.test")), APITimeoutError),
        (
            openai.APIConnectionError(request=httpx2.Request("POST", "https://m.test")),
            APIConnectionError,
        ),
        (RuntimeError("wrapped"), APIConnectionError),
    ],
)
def test_a_failed_request_becomes_a_livekit_error(error: Exception, kind: type) -> None:
    if isinstance(error, RuntimeError):
        error.__cause__ = httpx.ConnectError("down")
    assert type(livekit_error(error)) is kind


def test_a_client_error_is_not_retried_and_other_errors_stay_as_they_are() -> None:
    bad_request = StatusError("bad")
    bad_request.status_code = 400
    converted = livekit_error(bad_request)
    assert isinstance(converted, APIStatusError)
    assert converted.retryable is False
    assert livekit_error(MissingVariablesError(["x"])) is None


async def test_livekit_retries_a_request_that_failed_with_a_5xx() -> None:
    model = FailsOnce(StatusError("boom"))
    stream = ModelCallLLM(model).chat(
        chat_ctx=llm.ChatContext.empty(),
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
    )
    text = [chunk.delta.content async for chunk in stream if chunk.delta]
    assert model.requests == 2
    assert text == ["Hello."]


async def test_traced_llm_gives_a_native_llm_on_a_plain_agent_an_llm_span() -> None:
    sink = Memory()
    task_plan = plan(job(mode="status"))
    call_state = state(task_plan)
    call_state.save({"order_id": "B-7"}, step="find_order")
    native = ScriptedLLM("Order B-7 is open.")

    class Orders(Agent):
        async def llm_node(
            self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
        ) -> AsyncIterator[llm.ChatChunk | str]:
            async for chunk in traced_llm(self, trace, chat_ctx, tools, model_settings):
                yield chunk

    async with AgentSession(llm=native) as session:
        trace = trace_session(session, Telemetry(sink, agent="orders", call_id="RM_1"), call_state)
        await session.start(Orders(instructions="Order {{order_id}}."))
        await session.run(user_input="Where is my order?")
    assert trace.closing is not None
    await trace.closing

    [request] = native.requests
    assert request.prompt == "Order B-7."
    [span] = [span for span in sink.spans.get_finished_spans() if span.name == "llm"]
    assert "Order B-7 is open." in str((span.attributes or {}).get("output.value"))


async def test_a_tool_that_waits_runs_once_its_line_has_played() -> None:
    order: list[str] = []

    class Spy:
        async def run(self, arguments: dict[str, Any], state: object) -> object:
            order.append("tool ran")
            return {"ok": True}

    tool = ToolSpec(
        "check",
        "Check.",
        {"type": "object", "properties": {}},
        AnyResult,
        Spy(),
        announce="One moment.",
        announce_wait=True,
    )
    native = ScriptedLLM(call("check"), "Done.")
    async with AgentSession(llm=native) as session:
        session.on(
            "conversation_item_added",
            lambda event: (
                order.append(f"said {event.item.text_content}")
                if isinstance(event.item, llm.ChatMessage) and event.item.role == "assistant"
                else None
            ),
        )
        await session.start(Agent(instructions="x", tools=function_tools(ToolRegistry([tool]))))
        await session.run(user_input="Check it.")
    assert order[:2] == ["said One moment.", "tool ran"]


async def test_plain_tools_have_their_sends_drained_at_the_calls_end() -> None:
    sending = Sending()
    tools = ToolRegistry(
        [ToolSpec("note", "Note.", {"type": "object", "properties": {}}, AnyResult, sending)]
    )
    hooks = CallHooks()
    function_tools(tools, hooks=hooks)
    await hooks.end("ended")
    assert sending.drained


async def test_filled_llm_gives_a_plain_agent_its_prompt_filled_from_the_values_saved_since() -> (
    None
):
    task_plan = plan(job(mode="status"))
    call_state = state(task_plan)
    call_state.save({"order_id": "B-7"}, step="find_order")
    native = ScriptedLLM("Order B-7.", "Order C-9.")

    class Orders(Agent):
        async def llm_node(
            self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
        ) -> AsyncIterator[llm.ChatChunk | str]:
            async for chunk in filled_llm(self, chat_ctx, tools, model_settings, state=call_state):
                yield chunk

    async with AgentSession(llm=native) as session:
        await session.start(Orders(instructions="Order {{order_id}}."))
        await session.run(user_input="Where is my order?")
        call_state.save({"order_id": "C-9"}, step="find_order")
        await session.run(user_input="And now?")

    assert [request.prompt for request in native.requests] == ["Order B-7.", "Order C-9."]


async def test_model_hooks_on_a_plain_agent_with_a_native_llm_warn_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hooks = Hooks()
    hooks.add(PromptSuffix("Be brief."))
    call = CallHooks(hooks=hooks)
    async with AgentSession(llm=ScriptedLLM("Hi.")) as session:
        attach_call(session, call)
        await session.start(Agent(instructions="Orders."))
        with caplog.at_level("WARNING"):
            await session.run(user_input="Hello")
    assert "prompt hooks" in caplog.text
    assert "filled_llm" in caplog.text
