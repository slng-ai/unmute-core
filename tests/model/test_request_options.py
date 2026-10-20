"""The openai request options in params act as the SDK does, on both OpenAI-compatible calls."""

from collections.abc import Callable
from typing import Any

import pytest

from tests.openai_http import http
from tests.slng.router_fake import FakeRouter
from unmute_core.model import ModelCall, ModelRequest, OpenAIModelCall, RequestRefusedError
from unmute_core.slng import RouterModelCall
from unmute_core.state import CallInfo

MESSAGES = [{"role": "user", "content": "hi"}]
Build = Callable[[FakeRouter, dict[str, Any]], ModelCall]


def build_openai(router: FakeRouter, params: dict[str, Any]) -> ModelCall:
    return OpenAIModelCall("m", client=router.client(), params=params)


def build_router(router: FakeRouter, params: dict[str, Any]) -> ModelCall:
    return RouterModelCall(
        CallInfo(agent_id="salon", session_id="call-1"),
        model="m",
        client=router.client(),
        body=params,
    )


@pytest.fixture(params=[build_openai, build_router], ids=["openai", "router"])
def build(request: pytest.FixtureRequest) -> Build:
    return request.param


class Flaky(FakeRouter):
    """Answers 429 with a short retry delay until it has been asked ``failures`` times."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def handle(self, request: http.Request) -> http.Response:
        answer = super().handle(request)
        if len(self.requests) <= self.failures:
            return http.Response(
                429, json={"error": {"message": "slow"}}, headers={"retry-after-ms": "1"}
            )
        return answer


async def send(
    call: ModelCall, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> None:
    request = ModelRequest(messages=MESSAGES, params=params or {}, headers=headers or {})
    [event async for event in call.stream(request)]


@pytest.mark.parametrize("where", ["call", "request"])
async def test_timeout_sets_the_requests_timeout_and_is_not_a_body_field(
    build: Build, where: str
) -> None:
    router = FakeRouter()
    options = {"timeout": 2.5}
    await send(
        build(router, options if where == "call" else {}), options if where == "request" else None
    )

    assert router.requests[-1].headers["x-stainless-read-timeout"] == "2.5"
    assert router.requests[-1].extensions["timeout"]["read"] == 2.5
    assert "timeout" not in router.body


async def test_a_requests_timeout_wins_over_the_calls() -> None:
    router = FakeRouter()
    await send(build_openai(router, {"timeout": 9.0}), {"timeout": 2.5})

    assert router.requests[-1].headers["x-stainless-read-timeout"] == "2.5"


async def test_max_retries_sets_the_retry_count_and_is_not_a_body_field(build: Build) -> None:
    router = Flaky(failures=2)
    await send(build(router, {}), {"max_retries": 2})

    assert len(router.requests) == 3
    assert router.requests[-1].headers["x-stainless-retry-count"] == "2"
    assert "max_retries" not in router.body


async def test_the_calls_max_retries_applies_and_a_request_can_turn_it_off(build: Build) -> None:
    router = Flaky(failures=1)
    await send(build(router, {"max_retries": 1}))
    assert len(router.requests) == 2

    refused = Flaky(failures=1)
    with pytest.raises(Exception, match="429"):
        await send(build(refused, {"max_retries": 1}), {"max_retries": 0})
    assert len(refused.requests) == 1


async def test_extra_query_becomes_url_query_parameters(build: Build) -> None:
    router = FakeRouter()
    await send(
        build(router, {"extra_query": {"api-version": "1", "a": "old"}}),
        {"extra_query": {"a": "b"}},
    )

    assert dict(router.requests[-1].url.params) == {"api-version": "1", "a": "b"}
    assert "extra_query" not in router.body


async def test_extra_headers_join_the_headers_and_are_not_a_body_field(build: Build) -> None:
    router = FakeRouter()
    call = build(router, {"extra_headers": {"x-call": "1", "x-both": "call"}})
    await send(
        call, {"extra_headers": {"x-request": "2", "X-Both": "request"}}, {"traceparent": "t"}
    )

    sent = router.requests[-1].headers
    assert (sent["x-call"], sent["x-request"], sent["x-both"], sent["traceparent"]) == (
        "1",
        "2",
        "request",
        "t",
    )
    assert "extra_headers" not in router.body


async def test_extra_headers_never_replace_the_routers_identity_headers() -> None:
    router = FakeRouter()
    await send(build_router(router, {"extra_headers": {"X-Slng-Agent-Id": "other"}}))

    assert router.requests[-1].headers["x-slng-agent-id"] == "salon"


async def test_extra_body_fields_join_the_body(build: Build) -> None:
    router = FakeRouter()
    await send(
        build(router, {"top_p": 1, "extra_body": {"seed": 1, "top_p": 0.5}}),
        {"extra_body": {"seed": 2}, "temperature": 0},
    )

    assert (router.body["top_p"], router.body["seed"], router.body["temperature"]) == (0.5, 2, 0)
    assert "extra_body" not in router.body


@pytest.mark.parametrize("where", ["params", "extra_body"])
async def test_a_reserved_field_is_refused_wherever_it_is_set(build: Build, where: str) -> None:
    router = FakeRouter()
    field = {"stream_options": {}}
    params = field if where == "params" else {"extra_body": field}

    with pytest.raises(ValueError, match="stream_options"):
        build(router, params)
    with pytest.raises(RequestRefusedError, match="stream_options"):
        await send(build(router, {}), params)
    assert not router.requests


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"timeout": 0}, "timeout must be seconds"),
        ({"timeout": "5"}, "timeout must be seconds"),
        ({"max_retries": -1}, "max_retries must be a whole number"),
        ({"max_retries": 1.5}, "max_retries must be a whole number"),
        ({"max_retries": True}, "max_retries must be a whole number"),
        ({"extra_query": "a=1"}, "extra_query must be a mapping"),
        ({"extra_headers": ["a"]}, "extra_headers must be a mapping"),
        ({"extra_body": 1}, "extra_body must be a mapping"),
    ],
)
async def test_an_option_the_sdk_cannot_use_is_refused_by_name(
    build: Build, params: dict[str, Any], message: str
) -> None:
    with pytest.raises(RequestRefusedError, match=message):
        await send(build(FakeRouter(), {}), params)
