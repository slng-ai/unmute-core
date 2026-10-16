"""``ModelCallLLMService``: Pipecat requests through a ``ModelCall``, and the same prompt."""

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, cast

from pipecat.frames.frames import ErrorFrame, LLMContextFrame, LLMFullResponseEndFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.tests.utils import run_test
from pydantic import BaseModel

from tests.pipecat.harness import Harness
from tests.tasks.job import filled
from unmute_core.call import CallHooks
from unmute_core.hooks import Hooks, PromptSuffix
from unmute_core.model import ModelEvent, ModelRequest, TextDelta, ToolCall, Usage
from unmute_core.pipecat import ModelCallLLMService
from unmute_core.state import CallState


class Scripted:
    """A ``ModelCall`` that answers each request with the next scripted events."""

    def __init__(self, *replies: list[ModelEvent]) -> None:
        self.replies = list(replies)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        for event in self.replies.pop(0) if self.replies else []:
            yield event


def service(model: Scripted, prompt: str = "Hello.", **kwargs: Any) -> ModelCallLLMService:
    return ModelCallLLMService(
        model, settings=OpenAILLMService.Settings(model="m", system_instruction=prompt), **kwargs
    )


async def test_text_and_tool_calls_come_back_as_openai_chunks() -> None:
    model = Scripted([TextDelta("Hi "), ToolCall("c1", "find", "{}"), ToolCall("c2", "x", "{}")])
    stream = await service(model).get_chat_completions(LLMContext())
    chunks = [chunk async for chunk in stream]
    deltas = [chunk.choices[0].delta for chunk in chunks]
    assert deltas[0].content == "Hi "
    calls = [delta.tool_calls[0] for delta in deltas[1:]]
    assert [(c.index, c.id, c.function.name) for c in calls] == [(0, "c1", "find"), (1, "c2", "x")]


async def test_reasoning_tags_are_taken_out_of_the_text() -> None:
    model = Scripted([TextDelta("</think>Hi"), TextDelta(" <thi"), TextDelta("nk>x</think>there")])
    stream = await service(model).get_chat_completions(LLMContext())
    text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream])
    assert text == "Hi there"


async def test_a_request_carries_its_prompt_unfilled_its_site_and_live_variables() -> None:
    llm = service(Scripted())
    h = Harness(llm=llm)
    llm.state = h.state
    h.user("Where is my order?")
    await h.call("find_order")
    assert llm.site == "task:find_order"
    assert h.system_instruction.startswith("Find order {{order_id}}.")
    h.state.save({"order_id": "A-12"})
    llm._settings.system_instruction = h.system_instruction
    _ = [chunk async for chunk in await llm.get_chat_completions(h.context)]
    [request] = cast(Scripted, llm.model_call).requests
    assert request.site == "task:find_order"
    assert request.variables == {"order_id": "A-12"}
    assert request.messages[0] == {"role": "system", "content": h.system_instruction}


async def test_a_routed_and_a_native_step_give_the_model_the_same_final_prompt() -> None:
    routed = Harness(llm=service(Scripted()))
    native = Harness()
    for h in (routed, native):
        h.user("Where is my order?")
        await h.call("order_flow")
    assert routed.system_instruction != native.system_instruction
    assert filled(routed.system_instruction, routed.state, "task:find_order") == (
        native.system_instruction
    )


async def test_a_routed_and_a_native_call_give_the_same_hooked_prompts() -> None:
    def hooked() -> CallHooks:
        return CallHooks(hooks=Hooks(PromptSuffix("/no_think")))

    routed = Harness(llm=service(Scripted()), call=hooked())
    native = Harness(call=hooked())
    for h in (routed, native):
        assert h.flows.main_instruction().endswith("\n\n/no_think")
        h.user("Where is my order?")
        await h.call("order_flow")
    assert routed.system_instruction.endswith("\n\n/no_think")
    assert filled(routed.system_instruction, routed.state, "task:find_order") == (
        native.system_instruction
    )


async def test_a_hooked_request_goes_out_through_the_hooks_and_is_kept() -> None:
    hooks = Hooks()
    hooks.before_model(lambda request: replace(request, params={"temperature": 0}))
    llm = service(Scripted([TextDelta("Hi")]), call=CallHooks(hooks=hooks))
    _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext())]
    [request] = cast(Scripted, llm.model_call).requests
    assert request.params == {"temperature": 0}
    assert request.model == "unknown"
    assert llm.last_request is request


SIGNED = {"google": {"thought_signature": "sig-1"}}


def _recorded(call_id: str) -> list[Any]:
    """A tool call and its result, as Pipecat's aggregator writes them into the context."""
    call = {"id": call_id, "function": {"name": "find", "arguments": "{}"}, "type": "function"}
    return [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "content": "{}", "tool_call_id": call_id},
    ]


async def test_a_calls_provider_data_goes_back_when_the_call_is_replayed() -> None:
    hooks = Hooks()
    seen: list[ModelRequest] = []
    hooks.before_model(lambda request: seen.append(request) or request)
    model = Scripted([ToolCall("c1", "find", "{}", SIGNED), ToolCall("c2", "find", "{}")], [])
    llm = service(model, call=CallHooks(hooks=hooks))
    _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext())]
    assert llm.tool_extras == {"c1": SIGNED}

    history: list[Any] = [{"role": "user", "content": "hi"}, *_recorded("c1"), *_recorded("c2")]
    context = LLMContext(history)
    _ = [chunk async for chunk in await llm.get_chat_completions(context)]

    sent = model.requests[-1].messages
    calls = [message["tool_calls"][0] for message in sent if message.get("tool_calls")]
    assert calls[0]["extra_content"] == SIGNED
    assert "extra_content" not in calls[1]
    assert seen[-1].messages == sent
    assert "extra_content" not in cast(Any, context.get_messages()[1])["tool_calls"][0]


async def test_a_replayed_call_sends_back_only_the_keys_livekit_sends() -> None:
    other = {"openai": {"a": 1}}
    model = Scripted(
        [ToolCall("c1", "find", "{}", SIGNED | other), ToolCall("c2", "x", "{}", other)]
    )
    llm = service(model)
    _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext())]
    assert llm.tool_extras == {"c1": SIGNED | other, "c2": other}

    _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext(_recorded("c1")))]
    assert model.requests[-1].messages[1]["tool_calls"][0]["extra_content"] == SIGNED
    _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext(_recorded("c2")))]
    assert "extra_content" not in model.requests[-1].messages[1]["tool_calls"][0]


async def test_a_reused_call_id_never_replays_an_older_signature() -> None:
    model = Scripted([ToolCall("c1", "find", "{}", SIGNED)], [ToolCall("c1", "find", "{}")], [])
    llm = service(model)
    for _ in range(2):
        _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext())]
    assert llm.tool_extras == {}

    _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext(_recorded("c1")))]
    assert "extra_content" not in model.requests[-1].messages[1]["tool_calls"][0]


async def test_the_site_goes_back_to_the_main_prompt_when_the_run_returns() -> None:
    h = Harness(llm=service(Scripted()))
    h.user("Where is my order?")
    await h.call("find_order")
    h.user("A-12.")
    await h.call("finish", {"order_id": "A-12"})
    assert h.llm.site == ""


class Failing(Scripted):
    """A ``ModelCall`` whose every request fails."""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        raise RuntimeError("every model failed")
        yield TextDelta("")


async def test_a_failed_model_call_leaves_the_service_unusable_only_when_asked() -> None:
    usual = service(Failing())
    await run_test(usual, frames_to_send=[LLMContextFrame(LLMContext())])
    assert usual.is_usable

    llm = service(Failing())
    llm.failure_is_final = True
    _, up = await run_test(llm, frames_to_send=[LLMContextFrame(LLMContext())])
    [error] = [frame for frame in up if isinstance(frame, ErrorFrame)]
    assert "every model failed" in error.error
    assert not llm.is_usable


class FailsAfterText(Scripted):
    """A ``ModelCall`` that says a few words and then fails."""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield TextDelta("Let me check")
        raise RuntimeError("the connection dropped")


async def test_a_failure_after_the_reply_began_ends_the_reply_and_the_call_goes_on() -> None:
    llm = service(FailsAfterText())
    llm.failure_is_final = True
    down, up = await run_test(llm, frames_to_send=[LLMContextFrame(LLMContext())])

    [error] = [frame for frame in up if isinstance(frame, ErrorFrame)]
    assert "the connection dropped" in error.error
    assert llm.is_usable
    assert [frame.text for frame in down if isinstance(frame, LLMTextFrame)] == ["Let me check"]
    assert any(isinstance(frame, LLMFullResponseEndFrame) for frame in down)


async def test_a_failure_before_any_chunk_is_still_final() -> None:
    class Quiet(Scripted):
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            yield Usage(1, 1)
            raise RuntimeError("down before a word")

    llm = service(Quiet())
    llm.failure_is_final = True
    await run_test(llm, frames_to_send=[LLMContextFrame(LLMContext())])
    assert not llm.is_usable


class Closable(Scripted):
    """A ``ModelCall`` that counts how often it was closed."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


async def test_cleanup_closes_the_model_call_and_each_model_switched_in_once() -> None:
    first, other = Closable(), Closable()
    llm = service(first)
    llm.switched.extend([other, first])
    await llm.cleanup()
    assert (first.closed, other.closed) == (1, 1)


async def test_trace_headers_join_the_request_and_win_over_its_own() -> None:
    model = Scripted([TextDelta("Hi")])
    llm = service(model)
    llm.trace_headers = lambda: {"traceparent": "00-1-2-01"}
    _ = [chunk async for chunk in await llm.get_chat_completions(LLMContext())]
    assert model.requests[0].headers == {"traceparent": "00-1-2-01"}


async def test_no_trace_headers_are_sent_by_default() -> None:
    model = Scripted([TextDelta("Hi")])
    _ = [chunk async for chunk in await service(model).get_chat_completions(LLMContext())]
    assert model.requests[0].headers == {}


async def test_cleanup_closes_the_model_the_service_was_built_with_after_a_handoff() -> None:
    first, other = Closable(), Closable()
    llm = service(first)
    llm.model_call = other
    llm.switched.append(other)
    await llm.cleanup()
    assert (first.closed, other.closed) == (1, 1)


async def test_cleanup_closes_the_summarizer_the_flows_were_given() -> None:
    first, summarizer = Closable(), Closable()
    llm = service(first)
    h = Harness(llm=llm, summarizer=summarizer)
    await h.llm.cleanup()
    assert (first.closed, summarizer.closed) == (1, 1)


class Memory(BaseModel):
    name: str | None = None


async def _two_requests(llm: ModelCallLLMService, second_prompt: str) -> list[ErrorFrame]:
    pushed: list[ErrorFrame] = []

    async def push_error(error_msg: str, exception: Exception | None = None, **kw: Any) -> None:
        pushed.append(ErrorFrame(error_msg, exception=exception))

    async def noop(*args: Any, **kwargs: Any) -> None:
        return None

    llm.push_error = cast(Any, push_error)
    llm.push_frame = cast(Any, noop)
    llm.start_ttfb_metrics = cast(Any, noop)
    llm.stop_ttfb_metrics = cast(Any, noop)
    await llm._process_context(LLMContext())
    llm._settings.system_instruction = second_prompt
    await llm._process_context(LLMContext())
    return pushed


async def test_a_failure_before_the_reply_is_final_even_after_an_earlier_reply() -> None:
    llm = service(Scripted([TextDelta("Hi.")]), prompt="Hello {{name}}.")
    llm.state = CallState(Memory)
    llm.state.save({"name": "Ana"})
    llm.failure_is_final = True
    [error] = await _two_requests(llm, "Hello {{missing}}.")
    assert error.error.startswith("The model call failed:")


async def test_a_failure_after_the_reply_began_is_still_not_final() -> None:
    class Cuts(Scripted):
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            yield TextDelta("Let me")
            raise RuntimeError("dropped")

    llm = service(Cuts(), prompt="Hello.")
    llm.failure_is_final = True
    [first, second] = await _two_requests(llm, "Hello.")
    assert "after its reply began" in first.error
    assert "after its reply began" in second.error
