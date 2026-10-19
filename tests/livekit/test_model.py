"""Every model request goes through ``llm_node``: native, through a ``ModelCall``, or the router.

The router case checks that task requests carry their prompt site and the main agent's
requests do not.
"""

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import httpx2
from livekit.agents import AgentSession, APIConnectOptions, APIError, llm, metrics
from pydantic import BaseModel

from tests.livekit.fake import Reply, ScriptedLLM, call
from tests.slng.router_fake import FakeRouter, chunk
from tests.tasks.job import Canned, job, plan, registry, state
from unmute_core.agent import AgentSpec
from unmute_core.call import CallHooks
from unmute_core.hooks import Hooks, PromptSuffix
from unmute_core.livekit import UnmuteAgent
from unmute_core.livekit._model import (
    RUNNING_PLACEHOLDER_KEY,
    ModelCallLLM,
    request_context,
)
from unmute_core.model import (
    FallbackModelCall,
    ModelEvent,
    ModelRequest,
    TextDelta,
    ToolCall,
    Usage,
    fill_messages,
    retries_itself,
)
from unmute_core.slng import RouterModelCall
from unmute_core.state import CallInfo, CallState
from unmute_core.tools import SAVE_VALUES_TOOL, is_values

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}


class ScriptedCall:
    """A ``ModelCall`` that answers each request with the next scripted reply."""

    def __init__(self, *replies: Reply | str) -> None:
        self.replies = deque(Reply(r) if isinstance(r, str) else r for r in replies)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        reply = self.replies.popleft() if self.replies else Reply()
        if reply.text:
            yield TextDelta(reply.text)
        for index, (name, arguments) in enumerate(reply.calls):
            yield ToolCall(f"call_{len(self.requests)}_{index}", name, json.dumps(arguments))


async def _run(agent: UnmuteAgent, model: ScriptedLLM | None, *turns: str) -> None:
    async with AgentSession(llm=model) if model else AgentSession() as session:
        await session.start(agent)
        for turn in turns:
            await session.run(user_input=turn)


def _system(messages: list[dict[str, Any]]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "system")


async def test_both_paths_give_the_model_the_same_prompt() -> None:
    def agent(model: ScriptedCall | None) -> UnmuteAgent:
        task_plan = plan(job(mode="status"), tools=registry(lookup_order=Canned(FOUND)))
        call_state = state(task_plan)
        call_state.save({"order_id": "B-7"}, step="find_order")
        return UnmuteAgent(
            AgentSpec("Order {{order_id}}.", tasks=task_plan, model=model), state=call_state
        )

    native = ScriptedLLM(call("find_order"), "Checking.")
    await _run(agent(None), native, "Where is my order?")
    called = ScriptedCall(call("find_order"), "Checking.")
    await _run(agent(called), None, "Where is my order?")

    assert [request.site for request in called.requests] == ["", "task:find_order"]
    assert "{{order_id}}" in _system(called.requests[1].messages)
    assert called.requests[1].variables == {"order_id": "B-7"}
    for sent, request in zip(native.requests, called.requests, strict=True):
        assert sent.prompt == _system(fill_messages(request.messages, request.variables))


async def test_both_paths_give_the_model_the_same_hooked_prompt_and_params() -> None:
    def agent(model: ScriptedCall | None) -> UnmuteAgent:
        task_plan = plan(job(mode="status"), tools=registry(lookup_order=Canned(FOUND)))
        call_state = state(task_plan)
        call_state.save({"order_id": "B-7"}, step="find_order")
        hooks = Hooks(PromptSuffix("/no_think"))
        hooks.before_model(lambda request: replace(request, params={"temperature": 0}))
        return UnmuteAgent(
            AgentSpec("Order {{order_id}}.", tasks=task_plan, model=model),
            state=call_state,
            call=CallHooks(call_state, hooks=hooks),
        )

    native = ScriptedLLM(call("find_order"), "Checking.")
    await _run(agent(None), native, "Where is my order?")
    called = ScriptedCall(call("find_order"), "Checking.")
    await _run(agent(called), None, "Where is my order?")

    for sent, request in zip(native.requests, called.requests, strict=True):
        assert sent.prompt == _system(fill_messages(request.messages, request.variables))
        assert request.messages[0]["content"].endswith("\n\n/no_think")
        assert sent.extra == {"extra_body": {"temperature": 0}}
        assert request.params == {"temperature": 0}


async def test_a_model_call_answers_without_a_session_llm() -> None:
    task_plan = plan(job(mode="status"), tools=registry(lookup_order=Canned(FOUND)))
    call_state = state(task_plan)
    called = ScriptedCall(call("find_order"), call("lookup_order"), "It is open.")
    agent = UnmuteAgent(AgentSpec("Orders.", tasks=task_plan, model=called), state=call_state)
    await _run(agent, None, "Where is order A-12?")

    assert call_state.plain("status") == "open"
    step = called.requests[1]
    assert [tool["function"]["name"] for tool in step.tools] == ["lookup_order", "finish"]


class ScriptedRouter(FakeRouter):
    """A fake router that answers each request with the next scripted stream."""

    def __init__(self, *streams: list[dict[str, Any]]) -> None:
        super().__init__()
        self.streams = deque(streams)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path != "/livez":
            self.chunks = self.streams.popleft() if self.streams else [chunk({"content": "ok"})]
        return super().handle(request)


def _router_call(name: str) -> list[dict[str, Any]]:
    tool = {"index": 0, "id": f"c_{name}", "function": {"name": name, "arguments": "{}"}}
    return [chunk({"tool_calls": [tool]})]


async def test_step_requests_carry_their_prompt_site_and_the_main_agents_do_not() -> None:
    router = ScriptedRouter(_router_call("find_order"), _router_call("lookup_order"))
    model = RouterModelCall(
        CallInfo(agent_id="orders", session_id="call-1"),
        model="m",
        base_url="unused",
        client=router.client(),
    )
    task_plan = plan(job(mode="status"), tools=registry(lookup_order=Canned(FOUND)))
    agent = UnmuteAgent(AgentSpec("Orders.", tasks=task_plan, model=model), state=state(task_plan))
    await _run(agent, None, "Where is order A-12?")

    sites = [request.headers.get("x-slng-prompt-site") for request in router.requests]
    assert sites == [None, "task:find_order", None]
    assert {request.headers["x-slng-agent-id"] for request in router.requests} == {"orders"}


SIGNED = {"google": {"thought_signature": "sig-find"}}


async def test_a_routed_calls_signature_goes_back_with_the_call() -> None:
    signed = _router_call("find_order")
    signed[0]["choices"][0]["delta"]["tool_calls"][0]["extra_content"] = SIGNED
    router = ScriptedRouter(signed, _router_call("lookup_order"))
    model = RouterModelCall(
        CallInfo(agent_id="orders", session_id="call-1"),
        model="m",
        base_url="unused",
        client=router.client(),
    )
    task_plan = plan(job(mode="status"), tools=registry(lookup_order=Canned(FOUND)))
    agent = UnmuteAgent(AgentSpec("Orders.", tasks=task_plan, model=model), state=state(task_plan))
    await _run(agent, None, "Where is order A-12?")

    replayed = [
        call
        for message in json.loads(router.requests[-1].content)["messages"]
        for call in message.get("tool_calls") or []
    ]
    find = next(call for call in replayed if call["id"] == "c_find_order")
    assert find["extra_content"] == SIGNED


async def _hooked_native_request(added: list[dict[str, Any]]) -> tuple[Any, llm.ChatContext]:
    """Run one turn on a native LLM whose ``before_model`` hook adds ``added`` to the chat."""
    hooks = Hooks()
    hooks.before_model(lambda request: replace(request, messages=[*request.messages, *added]))
    native = ScriptedLLM("Hi.")
    agent = UnmuteAgent(AgentSpec("Orders."), call=CallHooks(hooks=hooks))
    async with AgentSession(llm=native) as session:
        await session.start(agent)
        await session.run(user_input="hi")
        asked = next(item for item in session.history.items if item.type == "message")
    return asked, native.requests[0].chat_ctx


async def test_a_hook_written_call_keeps_its_provider_data() -> None:
    call = {
        "id": "c9",
        "type": "function",
        "function": {"name": "find", "arguments": "{}"},
        "extra_content": SIGNED,
    }
    _, sent = await _hooked_native_request(
        [
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c9", "content": "{}"},
        ]
    )
    [written] = [item for item in sent.items if isinstance(item, llm.FunctionCall)]
    assert written.extra == SIGNED


def test_a_request_drops_the_running_call_livekit_adds_and_reads_the_prompt_now() -> None:
    ctx = llm.ChatContext()
    ctx.add_message(role="system", content="old prompt", id="lk.agent_task.instructions")
    ctx.add_message(role="user", content="hi")
    running = llm.FunctionCall(
        call_id="c1", name="order_flow", arguments="{}", extra={RUNNING_PLACEHOLDER_KEY: True}
    )
    ctx.insert(
        [
            running,
            llm.FunctionCallOutput(call_id="c1", name="order_flow", output="...", is_error=False),
        ]
    )

    request = request_context(ctx, "new prompt")

    assert [item.type for item in request.items] == ["message", "message"]
    prompt = request.items[0]
    assert isinstance(prompt, llm.ChatMessage) and prompt.text_content == "new prompt"
    assert len(ctx.items) == 4


async def test_reasoning_tags_never_reach_what_the_agent_says() -> None:
    called = ScriptedCall("</think>Hello <think>weighing it</think>there.")
    agent = UnmuteAgent(AgentSpec("Orders.", model=called))
    async with AgentSession() as session:
        await session.start(agent)
        await session.run(user_input="Hi")
        said = [item.text_content for item in session.history.items if item.type == "message"]
    assert said[-1] == "Hello there."


async def test_a_before_model_hook_that_raises_fails_the_call() -> None:
    hooks = Hooks()

    @hooks.before_model
    def broken(request: ModelRequest) -> None:
        raise RuntimeError("bug")

    failed: list[str] = []
    hooks.on("call.failed")(lambda event, data: failed.append(data["component"]))
    agent = UnmuteAgent(
        AgentSpec("Orders.", model=ScriptedCall("Hi.")), call=CallHooks(hooks=hooks)
    )
    await _run(agent, None, "Hello?")
    await asyncio.sleep(0)
    assert failed == ["llm"]


async def test_a_hook_that_adds_a_message_keeps_every_other_item_as_it_was() -> None:
    asked, sent = await _hooked_native_request([{"role": "system", "content": "Note."}])
    assert asked.id in [item.id for item in sent.items]
    note = sent.items[-1]
    assert isinstance(note, llm.ChatMessage)
    assert note.text_content == "Note."


class UnavailableError(RuntimeError):
    status_code = 503


class Failing:
    """A ``ModelCall`` that raises a 503 on every request."""

    def __init__(self, *, retries_itself: bool) -> None:
        self.retries_itself = retries_itself
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.calls += 1
        raise UnavailableError("down")
        yield TextDelta("")


async def _attempts(model: Any) -> None:
    stream = ModelCallLLM(model).chat(
        chat_ctx=llm.ChatContext.empty(),
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
    )
    try:
        async for _ in stream:
            pass
    except APIError:
        pass
    await stream.aclose()


async def test_a_self_retrying_model_call_is_not_retried_by_livekit() -> None:
    own, plain = Failing(retries_itself=True), Failing(retries_itself=False)
    await _attempts(own)
    await _attempts(plain)
    assert (own.calls, plain.calls) == (1, 3)


def test_the_chain_retries_itself_and_a_plain_router_does_not() -> None:
    assert retries_itself(FallbackModelCall([Failing(retries_itself=False)]))
    assert not retries_itself(
        RouterModelCall(CallInfo(agent_id="a", session_id="s"), client=FakeRouter().client())
    )
    assert not retries_itself(ScriptedCall())


class UsageCall(ScriptedCall):
    """A scripted model call that ends each reply with its token usage."""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        async for event in super().stream(request):
            yield event
        yield Usage(11, 4)


async def test_a_model_call_turn_reports_one_llm_metrics_with_its_tokens() -> None:
    collected: list[metrics.AgentMetrics] = []
    agent = UnmuteAgent(AgentSpec("Orders.", model=UsageCall("Hello.")))
    async with AgentSession() as session:
        session.on("metrics_collected", lambda event: collected.append(event.metrics))
        await session.start(agent)
        await session.run(user_input="Hi")

    reported = [item for item in collected if isinstance(item, metrics.LLMMetrics)]
    assert len(reported) == 1
    assert (reported[0].prompt_tokens, reported[0].completion_tokens, reported[0].total_tokens) == (
        11,
        4,
        15,
    )
    assert reported[0].cancelled is False
    assert reported[0].label.endswith("ModelCallLLM")
    assert reported[0].metadata is not None
    assert reported[0].metadata.model_name == "unknown"
    rows = [row for row in session.usage.model_usage if isinstance(row, metrics.LLMModelUsage)]
    assert [(row.input_tokens, row.output_tokens) for row in rows] == [(11, 4)]


class CachedUsageCall(ScriptedCall):
    """A scripted model call whose usage says how much of the prompt was cached."""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        async for event in super().stream(request):
            yield event
        yield Usage(11, 4, cached_input_tokens=8, response_id="chatcmpl-1")


async def test_llm_metrics_carry_the_cached_tokens_and_the_answers_id() -> None:
    collected: list[metrics.AgentMetrics] = []
    agent = UnmuteAgent(AgentSpec("Orders.", model=CachedUsageCall("Hello.")))
    async with AgentSession() as session:
        session.on("metrics_collected", lambda event: collected.append(event.metrics))
        await session.start(agent)
        await session.run(user_input="Hi")

    [reported] = [item for item in collected if isinstance(item, metrics.LLMMetrics)]
    assert (reported.prompt_cached_tokens, reported.request_id) == (8, "chatcmpl-1")


class CountedUsageCall(ScriptedCall):
    """A scripted model call whose provider counts its thinking apart, in its own total."""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        async for event in super().stream(request):
            yield event
        yield Usage(11, 4, reasoning_tokens=3, total_tokens=20)


async def test_llm_metrics_carry_the_reasoning_tokens_and_the_providers_total() -> None:
    collected: list[metrics.AgentMetrics] = []
    agent = UnmuteAgent(AgentSpec("Orders.", model=CountedUsageCall("Hello.")))
    async with AgentSession() as session:
        session.on("metrics_collected", lambda event: collected.append(event.metrics))
        await session.start(agent)
        await session.run(user_input="Hi")

    [reported] = [item for item in collected if isinstance(item, metrics.LLMMetrics)]
    assert (reported.reasoning_tokens, reported.total_tokens) == (3, 20)


async def test_each_model_call_request_of_a_turn_reports_once_steps_included() -> None:
    collected: list[metrics.AgentMetrics] = []
    task_plan = plan(job(mode="status"), tools=registry(lookup_order=Canned(FOUND)))
    called = UsageCall(call("find_order"), call("lookup_order"), "It is open.")
    agent = UnmuteAgent(AgentSpec("Orders.", tasks=task_plan, model=called), state=state(task_plan))
    async with AgentSession() as session:
        session.on("metrics_collected", lambda event: collected.append(event.metrics))
        await session.start(agent)
        await session.run(user_input="Where is order A-12?")

    reported = [item for item in collected if isinstance(item, metrics.LLMMetrics)]
    assert len(reported) == len(called.requests)


class Caller(BaseModel):
    """Two values: the brand the call starts with, and a note nobody saves."""

    brand: str | None = None
    note: str | None = None


async def _values_turn(call_state: CallState) -> ScriptedCall:
    called = ScriptedCall("Hello.")
    agent = UnmuteAgent(AgentSpec("Help.", model=called, save_values=True), state=call_state)
    await _run(agent, None, "Hi")
    return called


async def test_save_values_with_nothing_free_shows_the_values_and_offers_no_tool() -> None:
    called = await _values_turn(
        CallState(Caller, call_start={"brand": "optional", "note": "optional"})
    )

    [request] = called.requests
    assert any(is_values(m["role"], m["content"]) for m in request.messages)
    assert [tool["function"]["name"] for tool in request.tools] == []


async def test_save_values_with_a_free_value_shows_the_values_and_offers_the_tool() -> None:
    called = await _values_turn(CallState(Caller, call_start={"brand": "optional"}))

    [request] = called.requests
    assert any(is_values(m["role"], m["content"]) for m in request.messages)
    assert [tool["function"]["name"] for tool in request.tools] == [SAVE_VALUES_TOOL]


async def test_a_prompt_sent_as_written_keeps_its_braces() -> None:
    model = ScriptedCall("Sure.")
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content="Reply with {{name}} in braces.")
    ctx.add_message(role="user", content="Hi")
    async with ModelCallLLM(model, prompts_as_written=True).chat(chat_ctx=ctx) as stream:
        [chunk async for chunk in stream]

    [request] = model.requests
    assert _system(request.messages) == "Reply with {{name}} in braces."
    assert request.variables == {}
