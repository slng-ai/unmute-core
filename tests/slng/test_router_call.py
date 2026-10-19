"""RouterModelCall sends the prompt unfilled with its variables, and streams the answer back."""

import json
import logging
from typing import Any

import openai
import pytest
from openai import AsyncOpenAI

from tests.slng.router_fake import FakeRouter, chunk
from unmute_core.model import ModelRequest, TextDelta, ToolCall, Usage, error_code, http_status
from unmute_core.slng import (
    OpenAIEndpoint,
    RouterConfig,
    RouterHeaders,
    RouterModelCall,
    SlngConfigError,
)
from unmute_core.slng._router import TIMEOUT, answer_line
from unmute_core.state import CallInfo

CALL = CallInfo(agent_id="salon", session_id="call-1")
REQUEST = ModelRequest(
    messages=[
        {"role": "system", "content": "Caller: {{caller_name}}."},
        {"role": "user", "content": "hi"},
    ],
    tools=[{"type": "function", "function": {"name": "book", "parameters": {"type": "object"}}}],
    site="task:book",
    variables={"caller_name": "Ana"},
    headers={"traceparent": "00-1-2-01"},
)


def router_call(router: FakeRouter) -> RouterModelCall:
    return RouterModelCall(
        CALL,
        model="gpt-test",
        base_url="unused",
        body={"reasoning_effort": "none"},
        client=router.client(),
    )


async def test_the_request_keeps_the_prompt_unfilled() -> None:
    router = FakeRouter()
    [event async for event in router_call(router).stream(REQUEST)]

    sent = router.requests[-1]
    assert sent.url.path == "/v1/chat/completions"
    assert sent.headers["x-slng-agent-id"] == "salon"
    assert sent.headers["x-slng-session-id"] == "call-1"
    assert sent.headers["x-slng-prompt-site"] == "task:book"
    assert sent.headers["traceparent"] == "00-1-2-01"
    assert router.body["messages"] == REQUEST.messages
    assert router.body["template_variables"] == {"caller_name": "Ana"}
    assert router.body["reasoning_effort"] == "none"
    assert router.body["tools"] == REQUEST.tools
    assert router.body["stream"] is True
    assert "tool_choice" not in router.body


async def test_body_template_variables_join_the_requests_own() -> None:
    router = FakeRouter()
    call = RouterModelCall(
        CALL,
        model="gpt-test",
        body={"template_variables": {"brand": "Salon Sol", "caller_name": "old"}},
        client=router.client(),
    )
    [event async for event in call.stream(REQUEST)]

    assert router.body["template_variables"] == {"brand": "Salon Sol", "caller_name": "Ana"}


async def test_a_required_tool_choice_reaches_the_router() -> None:
    router = FakeRouter()
    request = ModelRequest(
        messages=REQUEST.messages,
        tools=REQUEST.tools,
        variables={"caller_name": "x"},
        tool_choice="required",
    )
    [event async for event in router_call(router).stream(request)]
    assert router.body["tool_choice"] == "required"


async def test_no_tools_sends_no_tools_field() -> None:
    router = FakeRouter()
    [
        event
        async for event in router_call(router).stream(
            ModelRequest(messages=REQUEST.messages, variables={"caller_name": "x"})
        )
    ]
    assert "tools" not in router.body
    assert "x-slng-prompt-site" not in router.requests[-1].headers


async def test_text_and_tool_calls_stream_back() -> None:
    router = FakeRouter(
        [
            chunk({"role": "assistant", "content": "One "}),
            chunk({"content": "moment."}),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "book", "arguments": '{"da'},
                        }
                    ]
                }
            ),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "t2",
                            "type": "function",
                            "function": {"name": "find", "arguments": "{}"},
                        }
                    ]
                }
            ),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'y": 1}'}}]}),
        ]
    )
    events = [event async for event in router_call(router).stream(REQUEST)]
    assert events == [
        TextDelta("One ", response_id="c1"),
        TextDelta("moment.", response_id="c1"),
        ToolCall(id="t1", name="book", arguments='{"day": 1}', response_id="c1"),
        ToolCall(id="t2", name="find", arguments="{}", response_id="c1"),
    ]


SIGNED = {"google": {"thought_signature": "sig-1"}}
SKIPPED = {"google": {"thought_signature": "skip_thought_signature_validator"}}


def _piece(index: int | None, call_id: str | None, name: str | None, arguments: str) -> dict:
    """One tool call piece, as a provider streams it. A null index or id is sent as null."""
    function = {"name": name, "arguments": arguments} if name else {"arguments": arguments}
    return {"index": index, "id": call_id, "type": "function", "function": function}


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        pytest.param(
            [
                [_piece(0, "t1", "book", '{"a"')],
                [_piece(1, "t2", "find", "{}")],
                [_piece(0, None, None, ": 1}")],
            ],
            [("t1", "book", '{"a": 1}'), ("t2", "find", "{}")],
            id="numbered calls join by index",
        ),
        pytest.param(
            [[_piece(None, "t1", "book", '{"a": 1}')], [_piece(None, "t2", "find", "{}")]],
            [("t1", "book", '{"a": 1}'), ("t2", "find", "{}")],
            id="calls without index in separate chunks stay apart",
        ),
        pytest.param(
            [[_piece(None, "t1", "book", '{"a": 1}'), _piece(None, "t2", "find", "{}")]],
            [("t1", "book", '{"a": 1}'), ("t2", "find", "{}")],
            id="calls without index in one chunk stay apart",
        ),
        pytest.param(
            [
                [_piece(None, "t1", "book", '{"a"')],
                [_piece(None, "t2", "find", "{")],
                [_piece(None, None, None, "}")],
                [_piece(None, "t1", None, ": 1}")],
            ],
            [("t1", "book", '{"a": 1}'), ("t2", "find", "{}")],
            id="a piece with an id seen before, or with neither, continues a call",
        ),
    ],
)
async def test_tool_call_pieces_join_by_index_else_by_id(
    deltas: list[list[dict]], expected: list[tuple[str, str, str]]
) -> None:
    router = FakeRouter([chunk({"tool_calls": delta}) for delta in deltas])
    events = [event async for event in router_call(router).stream(REQUEST)]
    assert [(e.id, e.name, e.arguments) for e in events if isinstance(e, ToolCall)] == expected


async def test_a_call_without_index_keeps_its_provider_data() -> None:
    first = {**_piece(None, "t1", "book", "{}"), "extra_content": SIGNED}
    router = FakeRouter(
        [chunk({"tool_calls": [first, _piece(None, "t2", "find", "{}")]})],
    )
    events = [event async for event in router_call(router).stream(REQUEST)]
    assert [e.extra for e in events if isinstance(e, ToolCall)] == [SIGNED, {}]


def _replayed(*calls: dict[str, Any]) -> ModelRequest:
    """A request that replays one assistant message with these tool calls."""
    return ModelRequest(
        messages=[
            {"role": "system", "content": "Book."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": list(calls)},
            *({"role": "tool", "tool_call_id": c["id"], "content": "{}"} for c in calls),
        ]
    )


def _call(call_id: str, **extra: object) -> dict[str, Any]:
    """One replayed tool call in OpenAI format, with its provider data when given."""
    call = {"id": call_id, "type": "function", "function": {"name": "book", "arguments": "{}"}}
    return call | ({"extra_content": extra} if extra else {})


async def test_a_tool_calls_provider_data_streams_back_as_its_extra() -> None:
    router = FakeRouter(
        [
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "book", "arguments": "{"},
                            "extra_content": SIGNED,
                        }
                    ]
                }
            ),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "t2",
                            "type": "function",
                            "function": {"name": "find", "arguments": "{}"},
                        }
                    ]
                }
            ),
        ]
    )
    events = [event async for event in router_call(router).stream(REQUEST)]
    assert events == [
        ToolCall(id="t1", name="book", arguments="{}", extra=SIGNED, response_id="c1"),
        ToolCall(id="t2", name="find", arguments="{}", response_id="c1"),
    ]


async def test_a_replayed_call_carries_its_provider_data_to_the_router() -> None:
    router = FakeRouter()
    request = _replayed(_call("t1", google={"thought_signature": "sig-1"}), _call("t2"))
    [event async for event in router_call(router).stream(request)]

    assert router.body["messages"] == request.messages


async def test_a_gemini_model_gets_the_placeholder_on_an_unsigned_call() -> None:
    router = FakeRouter()
    call = RouterModelCall(
        CALL, model="google/gemini-3-flash-preview", base_url="unused", client=router.client()
    )
    [event async for event in call.stream(_replayed(_call("t1"), _call("t2")))]

    first, second = router.body["messages"][2]["tool_calls"]
    assert first["extra_content"] == SKIPPED
    assert "extra_content" not in second


async def test_a_config_naming_gemini_gets_the_placeholder_under_auto() -> None:
    router = FakeRouter()
    call = RouterModelCall(
        CALL,
        config=RouterConfig.single("gemini-3-pro-preview"),
        base_url="unused",
        client=router.client(),
    )
    [event async for event in call.stream(_replayed(_call("t1")))]

    assert router.body["messages"][2]["tool_calls"][0]["extra_content"] == SKIPPED


async def test_another_model_gets_no_placeholder() -> None:
    router = FakeRouter()
    request = _replayed(_call("t1"))
    [event async for event in router_call(router).stream(request)]

    assert router.body["messages"] == request.messages


async def test_only_the_slng_response_headers_are_kept() -> None:
    router = FakeRouter(
        headers={
            "x-slng-request-id": "r1",
            "x-slng-response-source": "cache",
            "x-slng-cache-layer": "exact",
            "x-slng-model": "gpt-test",
            "server": "fake",
        }
    )
    call = router_call(router)
    [event async for event in call.stream(REQUEST)]
    assert call.last_response == {
        "x-slng-request-id": "r1",
        "x-slng-response-source": "cache",
        "x-slng-cache-layer": "exact",
        "x-slng-model": "gpt-test",
    }


async def test_an_error_status_carries_its_status_and_retry_after() -> None:
    router = FakeRouter(status=429, headers={"retry-after": "1"})
    with pytest.raises(openai.APIStatusError) as raised:
        [event async for event in router_call(router).stream(REQUEST)]
    assert http_status(raised.value) == 429
    assert raised.value.response.headers["retry-after"] == "1"


async def test_variables_over_the_router_limit_fail_before_sending() -> None:
    router = FakeRouter()
    request = ModelRequest(messages=REQUEST.messages, variables={"a.b": "x"})
    with pytest.raises(ValueError, match=r"'a\.b'"):
        [event async for event in router_call(router).stream(request)]
    assert router.requests == []


async def test_a_body_variable_that_is_not_text_fails_before_sending() -> None:
    router = FakeRouter()
    call = RouterModelCall(
        CALL, base_url="unused", body={"template_variables": {"a": 5}}, client=router.client()
    )
    with pytest.raises(ValueError):
        [event async for event in call.stream(REQUEST)]
    assert router.requests == []


async def test_a_value_over_the_router_limit_is_cut_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = FakeRouter()
    request = ModelRequest(messages=REQUEST.messages, variables={"note": "x" * 5000})
    with caplog.at_level(logging.WARNING):
        [event async for event in router_call(router).stream(request)]
    assert router.body["template_variables"] == {"note": "x" * 4000}
    assert "note is 5000 characters and is shortened to 4000" in caplog.text


def test_missing_arguments_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLNG_API_KEY", "from-env")
    monkeypatch.setenv("SLNG_ROUTER_URL", "https://router.test/v1")
    monkeypatch.delenv("SLNG_MODEL", raising=False)
    call = RouterModelCall(CALL)
    assert call.client.api_key == "from-env"
    assert str(call.client.base_url) == "https://router.test/v1/"
    assert call.model == "slng/auto"
    assert call.client.max_retries == 0


def test_arguments_win_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLNG_API_KEY", "from-env")
    monkeypatch.setenv("SLNG_ROUTER_URL", "https://env.test/v1")
    monkeypatch.setenv("SLNG_MODEL", "env-model")
    call = RouterModelCall(CALL, model="m", base_url="https://arg.test/v1", api_key="arg")
    assert (call.client.api_key, str(call.client.base_url), call.model) == (
        "arg",
        "https://arg.test/v1/",
        "m",
    )


def test_a_missing_key_and_url_are_named_together(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SLNG_API_KEY", "SLNG_ROUTER_URL", "SLNG_WORLD_PART"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SlngConfigError) as raised:
        RouterModelCall(CALL)
    assert str(raised.value) == (
        "Set SLNG_API_KEY in the environment. Set SLNG_ROUTER_URL or SLNG_WORLD_PART in the environment."
    )


def test_a_ready_client_reads_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLNG_API_KEY", raising=False)
    client = AsyncOpenAI(api_key="test", base_url="https://router.test/v1")
    assert RouterModelCall(CALL, client=client).model == "slng/auto"


CONFIG = RouterConfig.single(
    "gpt-test", OpenAIEndpoint(url="https://api.openai.com/v1", api_key="sk-upstream")
)


async def test_the_config_proxy_flag_usage_and_a_request_id_go_with_every_request() -> None:
    router = FakeRouter()
    call = RouterModelCall(
        CALL, model="gpt-test", config=CONFIG, pure_proxy=True, client=router.client()
    )
    [event async for event in call.stream(REQUEST)]
    [event async for event in call.stream(REQUEST)]
    first = json.loads(router.requests[0].content)
    assert first["slng_config"] == CONFIG.wire()
    assert first["slng_config"]["tiers"]["1"][0]["endpoint"]["api_key"] == "sk-upstream"
    assert first["slng_pure_proxy"] is True
    assert first["stream_options"] == {"include_usage": True}
    ids = [request.headers["x-slng-request-id"] for request in router.requests]
    assert len(set(ids)) == 2
    assert all(len(value) == 32 for value in ids)


async def test_no_config_and_no_proxy_send_neither() -> None:
    router = FakeRouter()
    [event async for event in router_call(router).stream(REQUEST)]
    assert "slng_config" not in router.body
    assert "slng_pure_proxy" not in router.body


async def test_extra_headers_never_replace_the_identity_headers() -> None:
    router = FakeRouter()
    call = RouterModelCall(
        CALL,
        extra_headers={"X-Team": "a", "X-Slng-Agent-Id": "other", "X-Slng-Request-Id": "mine"},
        client=router.client(),
    )
    [event async for event in call.stream(REQUEST)]
    sent = router.requests[-1].headers
    assert (sent["x-team"], sent["x-slng-agent-id"], sent["x-slng-request-id"]) == (
        "a",
        "salon",
        "mine",
    )


async def test_usage_comes_back_last() -> None:
    usage_chunk = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127},
    }
    router = FakeRouter([chunk({"content": "Hi."}), usage_chunk])
    events = [event async for event in router_call(router).stream(REQUEST)]
    assert events == [
        TextDelta("Hi.", response_id="c1"),
        Usage(120, 7, total_tokens=127, response_id="c1"),
    ]


async def test_usage_carries_the_reasoning_tokens_and_the_providers_own_total() -> None:
    usage_chunk = {
        **chunk({}),
        "choices": [],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 7,
            "total_tokens": 140,
            "completion_tokens_details": {"reasoning_tokens": 13},
        },
    }
    router = FakeRouter([chunk({"content": "Hi."}), usage_chunk])
    events = [event async for event in router_call(router).stream(REQUEST)]
    assert events[-1] == Usage(120, 7, reasoning_tokens=13, total_tokens=140, response_id="c1")


async def test_every_event_carries_the_answers_ids_and_usage_the_cached_tokens() -> None:
    usage_chunk = {
        **chunk({}),
        "id": "chatcmpl-9",
        "choices": [],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 7,
            "total_tokens": 127,
            "prompt_tokens_details": {"cached_tokens": 96},
        },
    }
    router = FakeRouter(
        [{**chunk({"content": "Hi."}), "id": "chatcmpl-9"}, usage_chunk],
        headers={"x-request-id": "req-42"},
    )
    events = [event async for event in router_call(router).stream(REQUEST)]
    assert events[0] == TextDelta("Hi.", response_id="chatcmpl-9", request_id="req-42")
    assert events[-1] == Usage(
        120,
        7,
        cached_input_tokens=96,
        response_id="chatcmpl-9",
        request_id="req-42",
        total_tokens=127,
    )


async def test_a_refusal_carries_its_code_and_logs_its_request_id(caplog) -> None:
    router = FakeRouter(status=403, error={"message": "no", "code": "project_mode_forbidden"})
    call = RouterModelCall(CALL, extra_headers={"X-Slng-Request-Id": "r-9"}, client=router.client())
    with pytest.raises(openai.APIStatusError) as raised:
        [event async for event in call.stream(REQUEST)]
    assert error_code(raised.value) == "project_mode_forbidden"
    assert "router request r-9 failed" in caplog.text


def test_a_pinned_model_must_be_in_the_config() -> None:
    with pytest.raises(ValueError, match="'other' is not in the router config, which has gpt-test"):
        RouterModelCall(CALL, model="other", config=CONFIG, client=FakeRouter().client())
    RouterModelCall(CALL, config=CONFIG, client=FakeRouter().client())


def test_a_pinned_model_needs_no_tiers() -> None:
    RouterModelCall(
        CALL,
        model="openai/gpt-4o",
        config=RouterConfig(cache_enabled=False),
        client=FakeRouter().client(),
    )


def test_the_answer_line_keeps_a_fixed_field_order() -> None:
    full = {
        "x-slng-response-source": "cache",
        "x-slng-cache-layer": "semantic",
        "x-slng-model": "gpt-test",
        "x-slng-request-id": "r-1",
    }
    assert answer_line("salon", full) == (
        "scope=salon source=cache layer=semantic model=gpt-test request_id=r-1"
    )
    assert answer_line("salon", {}) == "scope=salon source=unknown request_id=unknown"


async def test_each_answer_logs_one_line(caplog: pytest.LogCaptureFixture) -> None:
    router = FakeRouter(
        headers={"x-slng-response-source": "llm", "x-slng-request-id": "r-2", "x-slng-model": "m"}
    )
    with caplog.at_level(logging.INFO, logger="unmute_core.slng._router"):
        [event async for event in router_call(router).stream(REQUEST)]
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("slng router:")]
    assert lines == ["slng router: scope=salon source=llm model=m request_id=r-2"]


async def test_the_fields_the_call_owns_are_refused_elsewhere() -> None:
    with pytest.raises(ValueError, match="set slng_config with RouterModelCall's own argument"):
        RouterModelCall(CALL, body={"slng_config": {}}, client=FakeRouter().client())
    router = FakeRouter()
    request = ModelRequest(messages=REQUEST.messages, params={"stream_options": {}})
    with pytest.raises(ValueError, match="must not set stream_options"):
        [event async for event in router_call(router).stream(request)]
    assert router.requests == []


def test_the_world_part_the_config_id_and_the_timeout_come_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLNG_API_KEY", "key")
    monkeypatch.delenv("SLNG_ROUTER_URL", raising=False)
    monkeypatch.setenv("SLNG_WORLD_PART", "eu-north")
    monkeypatch.setenv("SLNG_LLM_CONFIG_ID", "cfg-1")
    call = RouterModelCall(CALL)
    assert str(call.client.base_url) == "https://eu-north.context-router.slng.ai/v1/"
    assert call.headers.llm_config_id == "cfg-1"
    assert call.client.timeout == TIMEOUT
    pinned = RouterModelCall(CALL, llm_config_id="mine", world_part="us")
    assert pinned.headers.llm_config_id == "mine"
    assert str(RouterModelCall(CALL, world_part="us").client.base_url).startswith("https://us.")


async def test_a_request_id_in_any_case_replaces_the_new_one() -> None:
    router = FakeRouter()
    call = RouterModelCall(
        CALL, extra_headers={"x-slng-request-id": "mine"}, client=router.client()
    )
    [event async for event in call.stream(REQUEST)]
    assert router.requests[-1].headers.get_list("x-slng-request-id") == ["mine"]


def test_a_world_part_given_wins_over_the_environments_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLNG_API_KEY", "key")
    monkeypatch.setenv("SLNG_ROUTER_URL", "https://env.test/v1")
    call = RouterModelCall(CALL, world_part="eu")
    assert str(call.client.base_url) == "https://eu.context-router.slng.ai/v1/"


async def test_it_closes_only_a_client_it_built(monkeypatch: pytest.MonkeyPatch) -> None:
    injected = FakeRouter().client()
    call = RouterModelCall(CALL, model="gpt-test", client=injected)
    await call.aclose()
    assert not injected.is_closed()

    monkeypatch.setenv("SLNG_API_KEY", "k")
    built = RouterModelCall(CALL, base_url="https://router.test/v1")
    await built.aclose()
    await built.aclose()
    assert built.client.is_closed()


@pytest.fixture
def warned(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    from unmute_core.slng import _blocked

    fresh: set[str] = set()
    monkeypatch.setattr(_blocked, "_WARNED", fresh)
    return fresh


@pytest.mark.parametrize("key", ["api_base", "metadata", "azure_deployment"])
async def test_a_body_field_the_router_drops_is_sent_with_one_warning(
    key: str, warned: set[str], caplog: pytest.LogCaptureFixture
) -> None:
    router = FakeRouter()
    with caplog.at_level(logging.WARNING):
        call = RouterModelCall(CALL, base_url="unused", client=router.client(), body={key: 1})
        RouterModelCall(CALL, base_url="unused", client=router.client(), body={key: 1})
    [event async for event in call.stream(REQUEST)]
    assert router.body[key] == 1
    assert [r.getMessage() for r in caplog.records if repr(key) in r.getMessage()] == [
        f"RouterModelCall's body sets {key!r}, which the SLNG Context Router drops: it never "
        "reaches the model. Leave it out, or set it with the router config's endpoint"
    ]


@pytest.mark.parametrize("key", ["api_key", "extra_headers", "vertex_project"])
async def test_a_request_param_the_router_drops_is_sent_with_one_warning(
    key: str, warned: set[str], caplog: pytest.LogCaptureFixture
) -> None:
    router = FakeRouter()
    request = ModelRequest(messages=REQUEST.messages, params={key: "x"})
    with caplog.at_level(logging.WARNING):
        [event async for event in router_call(router).stream(request)]
        [event async for event in router_call(router).stream(request)]
    assert router.body[key] == "x"
    assert sum(repr(key) in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize("key", ["extra_body", "extra_query"])
async def test_extra_body_and_extra_query_pass_without_a_warning(
    key: str, warned: set[str], caplog: pytest.LogCaptureFixture
) -> None:
    router = FakeRouter()
    with caplog.at_level(logging.WARNING):
        RouterModelCall(CALL, base_url="unused", client=router.client(), body={key: {"a": 1}})
    assert not [r for r in caplog.records if repr(key) in r.getMessage()]


async def test_reasoning_effort_and_include_pass_in_params() -> None:
    router = FakeRouter()
    request = ModelRequest(
        messages=REQUEST.messages, params={"reasoning_effort": "low", "include": ["x"]}
    )
    [event async for event in router_call(router).stream(request)]
    assert router.body["reasoning_effort"] == "low"


def test_the_router_refuses_headers_in_place_of_the_call() -> None:
    with pytest.raises(TypeError, match=r"takes the call's CallInfo, like state.call"):
        RouterModelCall(RouterHeaders("salon", "call-1"), client=FakeRouter().client())  # ty: ignore[invalid-argument-type]
