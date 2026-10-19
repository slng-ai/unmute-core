"""OpenAIModelCall fills the prompt itself and talks plain OpenAI-compatible chat completions."""

import builtins
from typing import Any

import openai
import pytest

from tests.slng.router_fake import FakeRouter, chunk
from unmute_core.model import (
    AnswerStarted,
    ModelRequest,
    OpenAIModelCall,
    TextDelta,
    ToolCall,
    Usage,
    close_model,
)

REQUEST = ModelRequest(
    messages=[
        {"role": "system", "content": "Caller: {{caller_name}}."},
        {"role": "user", "content": "hi"},
    ],
    tools=[{"type": "function", "function": {"name": "book", "parameters": {"type": "object"}}}],
    site="task:book",
    variables={"caller_name": "Ana"},
    headers={"traceparent": "00-1-2-01"},
    params={"temperature": 0},
)


async def test_the_prompt_is_filled_and_nothing_of_slng_is_sent() -> None:
    router = FakeRouter()
    call = OpenAIModelCall("gpt-test", client=router.client(), params={"top_p": 1})
    [event async for event in call.stream(REQUEST)]

    sent = router.requests[-1]
    assert router.body["messages"][0]["content"] == "Caller: Ana."
    assert router.body["model"] == "gpt-test"
    assert router.body["stream_options"] == {"include_usage": True}
    assert router.body["temperature"] == 0
    assert router.body["top_p"] == 1
    assert router.body["tools"] == REQUEST.tools
    assert sent.headers["traceparent"] == "00-1-2-01"
    assert not [name for name in sent.headers if name.lower().startswith("x-slng")]
    assert not {"template_variables", "slng_config", "slng_pure_proxy"} & set(router.body)


async def test_it_streams_text_a_tool_call_with_its_extra_content_and_usage_last() -> None:
    signature = {"google": {"thought_signature": "sig"}}
    tool = {
        "tool_calls": [
            {
                "index": 0,
                "id": "t1",
                "function": {"name": "book", "arguments": '{"a":'},
                "extra_content": signature,
            }
        ]
    }
    more = {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}
    usage: dict[str, Any] = {**chunk({}), "choices": [], "usage": _usage()}
    router = FakeRouter([chunk({"content": "ok"}), chunk(tool), chunk(more), usage])
    call = OpenAIModelCall("gpt-test", client=router.client())

    events = [event async for event in call.stream(REQUEST)]

    assert events == [
        AnswerStarted(response_id="c1"),
        TextDelta("ok", response_id="c1"),
        ToolCall("t1", "book", '{"a":1}', signature, response_id="c1"),
        Usage(3, 4, total_tokens=7, response_id="c1"),
    ]


def _usage() -> dict[str, int]:
    return {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


async def test_a_required_tool_choice_is_sent() -> None:
    router = FakeRouter()
    call = OpenAIModelCall("gpt-test", client=router.client())
    request = ModelRequest(
        messages=[{"role": "user", "content": "hi"}], tools=REQUEST.tools, tool_choice="required"
    )
    [event async for event in call.stream(request)]
    assert router.body["tool_choice"] == "required"


def test_the_provider_is_openai_or_the_host_of_the_base_url() -> None:
    assert OpenAIModelCall("m", client=FakeRouter().client()).provider == "openai"
    assert OpenAIModelCall("m", base_url="https://llm.example.test/v1", api_key="k").provider == (
        "llm.example.test"
    )


async def test_it_closes_only_a_client_it_built() -> None:
    built = OpenAIModelCall("m", api_key="k")
    await close_model(built)
    await built.aclose()
    assert built.client.is_closed()

    router_client = FakeRouter().client()
    injected = OpenAIModelCall("m", client=router_client)
    await injected.aclose()
    assert not router_client.is_closed()


def test_a_missing_openai_package_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ImportError, match=r"unmute-core\[slng\], or add openai"):
        OpenAIModelCall("m", api_key="k")


TWO_SYSTEMS = ModelRequest(
    messages=[
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
        {"role": "developer", "content": [{"type": "text", "text": "Speak French."}]},
    ]
)
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


@pytest.mark.parametrize(
    ("model", "base_url"),
    [("gemini-2.5-flash", None), ("google/gemini-3-flash", GOOGLE_URL)],
)
async def test_two_system_messages_go_out_as_one_for_gemini(
    model: str, base_url: str | None
) -> None:
    router = FakeRouter()
    call = OpenAIModelCall(model, base_url=base_url, client=router.client())
    [event async for event in call.stream(TWO_SYSTEMS)]

    assert router.body["messages"] == [
        {"role": "system", "content": "Be brief.\n\nSpeak French."},
        {"role": "user", "content": "hi"},
    ]


@pytest.mark.parametrize(
    ("model", "base_url"),
    [("gpt-test", None), ("meta/llama-4-maverick", GOOGLE_URL), ("acme/llama-gemini-tuned", None)],
)
async def test_two_system_messages_stay_two_for_other_models(
    model: str, base_url: str | None
) -> None:
    router = FakeRouter()
    call = OpenAIModelCall(model, base_url=base_url, client=router.client())
    [event async for event in call.stream(TWO_SYSTEMS)]

    assert router.body["messages"] == TWO_SYSTEMS.messages


@pytest.mark.parametrize("timeout", [0, -5, "a"])
def test_a_timeout_that_is_not_seconds_above_zero_is_refused(timeout: Any) -> None:
    with pytest.raises(ValueError, match=r"OpenAIModelCall\ntimeout\n.*seconds above zero"):
        OpenAIModelCall("gpt-test", api_key="k", timeout=timeout)


@pytest.mark.parametrize(
    ("arguments", "expiry"),
    [({}, 120.0), ({"keepalive_s": 30}, 30), ({"keepalive_s": None}, None)],
)
def test_the_client_it_builds_keeps_an_idle_connection_for_keepalive_s(
    monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any], expiry: float | None
) -> None:
    built: list[dict[str, Any]] = []
    real = openai.DefaultAsyncHttpxClient
    monkeypatch.setattr(
        openai, "DefaultAsyncHttpxClient", lambda **kwargs: built.append(kwargs) or real(**kwargs)
    )
    OpenAIModelCall("m", api_key="k", **arguments)
    [limits] = [kwargs["limits"] for kwargs in built]
    assert limits.keepalive_expiry == expiry
    assert limits.max_connections == openai.DEFAULT_CONNECTION_LIMITS.max_connections
    assert (
        limits.max_keepalive_connections
        == openai.DEFAULT_CONNECTION_LIMITS.max_keepalive_connections
    )


@pytest.mark.parametrize("keepalive_s", [0, -5, "a"])
def test_a_keepalive_that_is_not_seconds_above_zero_is_refused(keepalive_s: Any) -> None:
    with pytest.raises(ValueError, match=r"OpenAIModelCall\nkeepalive_s\n"):
        OpenAIModelCall("m", api_key="k", keepalive_s=keepalive_s)


def test_a_ready_client_refuses_keepalive_s() -> None:
    with pytest.raises(ValueError, match="keepalive_s only applies to a client OpenAIModelCall"):
        OpenAIModelCall("m", client=FakeRouter().client(), keepalive_s=30)
