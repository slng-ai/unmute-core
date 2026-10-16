"""``RequestHooks``: a native Pipecat request borrows the hooks' changes and gives them back."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pipecat.frames.frames import (
    FunctionCallFromLLM,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService

from tests.pipecat.harness import Harness
from unmute_core.call import CallHooks
from unmute_core.hooks import Hooks, ModelResponse, PromptSuffix
from unmute_core.model import ModelRequest, ToolCall
from unmute_core.pipecat import ModelCallLLMService
from unmute_core.pipecat._requests import RequestHooks, _Lent


def _llm() -> OpenAILLMService:
    settings = OpenAILLMService.Settings(model="gpt-4.1-mini", system_instruction="Help.")
    return OpenAILLMService(api_key="test", settings=settings)


async def test_a_native_request_borrows_its_changes_and_gives_them_back() -> None:
    hooks = Hooks()

    @hooks.before_model
    def change(request: ModelRequest) -> ModelRequest:
        system = {"role": "system", "content": "Help. Be brief."}
        note = {"role": "system", "content": "Note."}
        messages = [system, *request.messages[1:], note]
        return replace(request, messages=messages, params={"temperature": 0})

    llm = _llm()
    requests = RequestHooks(llm, CallHooks(hooks=hooks))
    context = LLMContext(cast(Any, [{"role": "user", "content": "hi"}]))
    frame = LLMContextFrame(context)
    await requests.before(frame)

    assert frame.context is not context
    assert context.get_messages() == [{"role": "user", "content": "hi"}]
    assert frame.context.get_messages()[-1] == {"role": "system", "content": "Note."}
    assert llm._settings.system_instruction == "Help. Be brief."
    assert llm._settings.extra["extra_body"] == {"temperature": 0}

    await requests.give_back()
    assert llm._settings.system_instruction == "Help."
    assert "extra_body" not in llm._settings.extra


async def test_a_plain_pipeline_hooks_the_services_own_prompt() -> None:
    llm = _llm()
    requests = RequestHooks(llm, CallHooks(hooks=Hooks(PromptSuffix("/no_think"))), prompts=True)
    await requests.before(LLMContextFrame(LLMContext()))
    assert llm._settings.system_instruction == "Help.\n\n/no_think"
    await requests.give_back()
    await requests.before(LLMContextFrame(LLMContext()))
    assert llm._settings.system_instruction == "Help.\n\n/no_think"


async def test_after_model_hears_the_reply_or_that_it_was_cut_off() -> None:
    heard: list[ModelResponse] = []
    hooks = Hooks()
    hooks.after_model(lambda request, response: heard.append(response))
    llm = _llm()
    requests = RequestHooks(llm, CallHooks(hooks=hooks))

    await requests.before(LLMContextFrame(LLMContext()))
    for frame in (LLMFullResponseStartFrame(), LLMTextFrame("Hi."), LLMFullResponseEndFrame()):
        await requests._pushed(llm, frame)
    await requests.before(LLMContextFrame(LLMContext()))
    await requests._processed(llm, InterruptionFrame())
    await asyncio.sleep(0)

    assert [(r.text, r.interrupted) for r in heard] == [("Hi.", False), ("", True)]


async def test_after_model_hears_the_provider_data_a_model_call_sent_with_a_call() -> None:
    signed = {"google": {"thought_signature": "sig-1"}}
    heard: list[ModelResponse] = []
    hooks = Hooks()
    hooks.after_model(lambda request, response: heard.append(response))
    llm = ModelCallLLMService(cast(Any, SimpleNamespace(model="m")))
    llm.tool_extras["c1"] = signed
    requests = RequestHooks(llm, CallHooks(hooks=hooks))

    await requests.before(LLMContextFrame(LLMContext()))
    calls = [FunctionCallFromLLM("find", call_id, {}, LLMContext()) for call_id in ("c1", "c2")]
    for frame in (FunctionCallsStartedFrame(calls), LLMFullResponseEndFrame()):
        await requests._pushed(llm, frame)
    await asyncio.sleep(0)

    [response] = heard
    assert response.tool_calls == (
        ToolCall("c1", "find", "{}", signed),
        ToolCall("c2", "find", "{}"),
    )


async def test_after_model_hears_a_calls_arguments_without_unicode_escapes() -> None:
    heard: list[ModelResponse] = []
    hooks = Hooks()
    hooks.after_model(lambda request, response: heard.append(response))
    llm = _llm()
    requests = RequestHooks(llm, CallHooks(hooks=hooks))

    await requests.before(LLMContextFrame(LLMContext()))
    calls = [FunctionCallFromLLM("find", "c1", {"name": "José"}, LLMContext())]
    for frame in (FunctionCallsStartedFrame(calls), LLMFullResponseEndFrame()):
        await requests._pushed(llm, frame)
    await asyncio.sleep(0)

    [response] = heard
    assert response.tool_calls[0].arguments == '{"name": "José"}'


async def test_a_hooked_prompt_never_doubles_the_text_pipecat_adds() -> None:
    llm = _llm()
    llm.append_system_instruction("EXTRA")
    requests = RequestHooks(llm, CallHooks(hooks=Hooks(PromptSuffix("/no_think"))), prompts=True)
    for _ in range(3):
        await requests.before(LLMContextFrame(LLMContext()))
        assert llm._settings.system_instruction == "Help.\n\n/no_think\n\nEXTRA"
        await requests.give_back()
    assert llm._settings.system_instruction == "Help.\n\nEXTRA"


async def test_an_interruption_gives_back_the_borrowed_params() -> None:
    hooks = Hooks()
    hooks.before_model(lambda request: replace(request, params={"temperature": 0}))
    llm = _llm()
    requests = RequestHooks(llm, CallHooks(hooks=hooks))
    await requests.before(LLMContextFrame(LLMContext()))
    assert "extra_body" in llm._settings.extra
    await requests._processed(llm, InterruptionFrame())
    assert "extra_body" not in llm._settings.extra


async def test_a_before_model_hook_that_raises_fails_the_call() -> None:
    hooks = Hooks()

    @hooks.before_model
    def broken(request: ModelRequest) -> None:
        raise RuntimeError("bug")

    failed: list[str] = []
    hooks.on("call.failed")(lambda event, data: failed.append(data["component"]))
    requests = RequestHooks(_llm(), CallHooks(hooks=hooks))
    with pytest.raises(RuntimeError, match="bug"):
        await requests.before(LLMContextFrame(LLMContext()))
    await asyncio.sleep(0)
    assert failed == ["llm"]


class _Completions:
    """An OpenAI client's ``chat.completions`` that keeps what it was sent."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.sent.append(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def test_a_side_request_sends_only_its_own_params_and_keeps_a_chat_with_no_prompt() -> None:
    hooks = Hooks()
    hooks.before_model(lambda request: replace(request, params={"temperature": 0}))
    llm = _llm()
    completions = _Completions()
    llm._client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    h = Harness(llm=llm, call=CallHooks(hooks=Hooks()))
    lent = llm._settings.extra | {"extra_body": {"top_p": 0.1}}
    llm._settings.extra = lent
    assert h.flows._requests is not None
    h.flows._requests._extra = _Lent({}, lent)

    request = ModelRequest(messages=[{"role": "user", "content": "hi"}], params={"seed": 1})
    text = (await h.flows._side.native_ask(llm, request)).text

    assert text == "ok"
    [sent] = completions.sent
    assert sent["extra_body"] == {"seed": 1}
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


async def test_one_llm_service_serves_one_call() -> None:
    llm = _llm()
    RequestHooks(llm, CallHooks(hooks=Hooks()))
    with pytest.raises(ValueError, match="one LLM service per call"):
        RequestHooks(llm, CallHooks(hooks=Hooks()))
