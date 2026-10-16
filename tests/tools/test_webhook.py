"""Webhook tools against a real HTTP server on localhost."""

import hashlib
import hmac
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import BaseModel

from tests.outputs import AnyResult, NoArguments
from unmute_core.state import CallState
from unmute_core.tools import (
    Accepted,
    Done,
    ToolError,
    ToolRegistry,
    Webhook,
    WebhookAuth,
    webhook_tool,
)
from unmute_core.tools._executors import _error_excerpt


class OrderInput(BaseModel):
    order_id: str
    count: int | None = None


class Handler(BaseHTTPRequestHandler):
    """Answers by path: echo the request, or misbehave on purpose."""

    def _answer(self) -> None:
        path = urlsplit(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if path == "/slow":
            time.sleep(1.5)
        if path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/echo")
            self.end_headers()
            return
        status, answer = {
            "/status/503": (503, b"{}"),
            "/error-json": (
                409,
                b'{"error": {"code": "SLOT_TAKEN", "message": "Slot taken"}}',
            ),
            "/error-text": (500, b"  Something\n  broke  "),
            "/bad-json": (200, b"not json"),
            "/empty": (204, b""),
            "/big": (200, b'"' + b"x" * 70_000 + b'"'),
        }.get(path, (200, None))
        if answer is None:
            answer = json.dumps(
                {
                    "method": self.command,
                    "path": path,
                    "query": parse_qs(urlsplit(self.path).query),
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body.decode(),
                }
            ).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(answer)))
        self.end_headers()
        self.wfile.write(answer)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _answer  # noqa: N815

    def log_message(self, format: str, *args: object) -> None:
        """Keep the test output quiet."""


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


class Customer(BaseModel):
    customer_id: str | None = None
    tier: int | None = None


def _call(tool, arguments, state=None):
    return ToolRegistry([tool]).call(tool.name, arguments, state)


def _tool(url, **options):
    options.setdefault("allow_private_network", True)
    options.setdefault("output", AnyResult)
    return webhook_tool("lookup_order", "Look up an order.", url, input=OrderInput, **options)


async def test_post_sends_the_arguments_as_json(server):
    result = await _call(_tool(f"{server}/echo"), '{"order_id": "A1", "count": 2}')
    assert result["method"] == "POST"
    assert json.loads(result["body"]) == {"count": 2, "order_id": "A1"}
    assert result["headers"]["content-type"] == "application/json"
    assert result["headers"]["host"] == server.removeprefix("http://")


async def test_get_sends_the_arguments_as_the_query(server):
    tool = _tool(f"{server}/echo?fixed=1", method="get")
    result = await _call(tool, {"order_id": "A 1", "count": 2})
    assert result["method"] == "GET"
    assert result["query"] == {"fixed": ["1"], "order_id": ["A 1"], "count": ["2"]}
    assert result["body"] == ""


async def test_headers_and_hmac_signature(server):
    tool = _tool(f"{server}/echo", headers={"Authorization": "Bearer t0k"}, hmac_secret="s3cret")
    result = await _call(tool, {"order_id": "A1"})
    expected = hmac.new(b"s3cret", result["body"].encode(), hashlib.sha256).hexdigest()
    assert result["headers"]["authorization"] == "Bearer t0k"
    assert result["headers"]["x-signature-256"] == expected


async def test_url_placeholders_and_locked_values_come_from_state(server):
    state = CallState(Customer)
    state.save({"customer_id": "c/7", "tier": 3})
    tool = _tool(
        f"{server}/customers/{{{{customer_id}}}}/orders",
        locked={"tier": "{{tier}}", "source": "voice"},
    )
    assert "tier" not in json.dumps(ToolRegistry([tool]).schemas())
    result = await _call(tool, {"order_id": "A1", "tier": 99}, state)
    assert result["path"] == "/customers/c%2F7/orders"
    assert json.loads(result["body"]) == {"order_id": "A1", "source": "voice", "tier": 3}


async def test_unset_url_placeholder_refuses_before_sending(server):
    tool = _tool(f"{server}/customers/{{{{customer_id}}}}")
    refused = await _call(tool, {"order_id": "A1"}, CallState(Customer))
    assert "customer_id not set yet" in refused["refused"]


async def test_same_origin_redirect_is_followed(server):
    result = await _call(_tool(f"{server}/redirect"), {"order_id": "A1"})
    assert result["path"] == "/echo"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("/status/503", "webhook returned HTTP 503"),
        ("/bad-json", "webhook returned invalid JSON"),
        ("/big", "webhook response exceeded the size limit"),
    ],
)
async def test_bad_answers_become_tool_errors(server, path, message):
    with pytest.raises(ToolError, match=message):
        await _call(_tool(f"{server}{path}"), {"order_id": "A1"})


async def test_a_json_error_body_reaches_the_model(server):
    with pytest.raises(ToolError) as caught:
        await _call(_tool(f"{server}/error-json"), {"order_id": "A1"})
    assert (
        caught.value.message
        == "webhook returned HTTP 409: Slot taken; SLOT_TAKEN. Do not claim success"
    )
    assert caught.value.event_text == "webhook returned HTTP 409"


async def test_a_text_error_body_reaches_the_model_and_the_sinks_stay_short(server):
    events = []
    registry = ToolRegistry([_tool(f"{server}/error-text")])
    registry.listen(lambda stage, data: events.append((stage, data)))
    with pytest.raises(ToolError, match=r"HTTP 500: Something broke. Do not claim success$"):
        await registry.call("lookup_order", {"order_id": "A1"})
    failed = [data for stage, data in events if stage == "tool.failed"]
    assert failed[0]["error"] == "webhook returned HTTP 500"


def test_an_error_excerpt_is_capped_and_empty_when_the_body_is():
    assert _error_excerpt(b"") == ""
    assert _error_excerpt(b"x" * 400) == "x" * 300
    assert _error_excerpt(b'{"error": "bad", "message": "worse", "code": 7}') == "bad; worse; 7"


async def test_empty_answer_reads_ok(server):
    assert await _call(_tool(f"{server}/empty", output=Done), {"order_id": "A1"}) == {"ok": True}


async def test_timeout_covers_the_whole_call(server):
    with pytest.raises(ToolError, match="webhook request timed out"):
        await _call(_tool(f"{server}/slow", timeout=1), {"order_id": "A1"})


async def test_request_size_is_capped(server):
    with pytest.raises(ToolError, match="request exceeded the size limit"):
        await _call(_tool(f"{server}/echo"), {"order_id": "x" * 70_000})


async def test_environment_proxies_are_ignored(server, monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    result = await _call(_tool(f"{server}/echo"), {"order_id": "A1"})
    assert result["path"] == "/echo"


async def test_default_webhook_refuses_localhost(server):
    port = server.rsplit(":", 1)[1]
    tool = webhook_tool(
        "lookup_order",
        "Look up an order.",
        f"https://127.0.0.1:{port}/echo",
        input=OrderInput,
        output=AnyResult,
    )
    with pytest.raises(ToolError, match="webhook destination"):
        await _call(tool, {"order_id": "A1"})


def test_default_webhook_refuses_plain_http_at_build(server):
    with pytest.raises(ValueError, match="scheme is not allowed"):
        webhook_tool(
            "lookup_order",
            "Look up an order.",
            f"{server}/echo",
            input=OrderInput,
            output=AnyResult,
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"url": "https://{{host}}/x"}, "path and query only"),
        ({"url": "/relative"}, "scheme and a host"),
        ({"url": "ftp://api.example.com/x"}, "scheme is not allowed"),
        ({"url": "http://api.example.com/x"}, "scheme is not allowed"),
        ({"url": "https://user:pw@api.example.com/x"}, "authority is invalid"),
        ({"url": "https://api.example.com/x#top"}, "authority is invalid"),
        ({"url": "https://api example.com/x"}, "authority is invalid"),
        ({"url": "https://api.example.com:0/x"}, "port is invalid"),
        ({"url": "https://api.example.com:http/x"}, "port is invalid"),
        ({"url": "https://api.example.com:70000/x"}, "port is invalid"),
        ({"headers": {"Host": "evil"}}, "set by the library"),
        ({"headers": {"X-A": "1\r\nX-B: 2"}}, "forbidden character"),
        ({"method": "GET", "hmac_secret": "s"}, "no body to sign"),
        ({"body": "envelope"}, "needs tool_name"),
        ({"body": "envelope", "tool_name": "t", "method": "GET"}, "use body='raw'"),
        (
            {"auth": WebhookAuth.api_key("TOKEN"), "headers": {"x-api-key": "1"}},
            "set by auth",
        ),
        (
            {"auth": WebhookAuth.api_key("TOKEN"), "headers": {"Authorization": "Basic x"}},
            "second credential",
        ),
    ],
)
def test_bad_webhook_config_fails_at_start(options, message):
    with pytest.raises(ValueError, match=message):
        Webhook(**({"url": "https://api.example.com/x"} | options))


async def test_auth_reads_its_token_from_the_environment_at_each_call(server, monkeypatch):
    bearer = _tool(f"{server}/echo", auth=WebhookAuth.bearer("ORDERS_TOKEN"))
    key = _tool(f"{server}/echo", auth=WebhookAuth.api_key("ORDERS_KEY"))
    monkeypatch.setenv("ORDERS_TOKEN", "t1")
    monkeypatch.setenv("ORDERS_KEY", "k1")
    assert (await _call(bearer, {"order_id": "A1"}))["headers"]["authorization"] == "Bearer t1"
    assert (await _call(key, {"order_id": "A1"}))["headers"]["x-api-key"] == "k1"
    monkeypatch.setenv("ORDERS_TOKEN", "t2")
    assert (await _call(bearer, {"order_id": "A1"}))["headers"]["authorization"] == "Bearer t2"


async def test_an_unset_token_refuses_the_call_and_logs_the_variable(server, monkeypatch, caplog):
    monkeypatch.delenv("ORDERS_TOKEN", raising=False)
    tool = _tool(f"{server}/echo", auth=WebhookAuth.bearer("ORDERS_TOKEN"))
    with pytest.raises(ToolError, match="could not authenticate"):
        await _call(tool, {"order_id": "A1"})
    assert "ORDERS_TOKEN is not set" in caplog.text


async def test_a_token_provider_gives_the_token_at_each_call(server):
    tokens = iter(["t1", "t2"])

    async def vault() -> str:
        return next(tokens)

    bearer = _tool(f"{server}/echo", auth=WebhookAuth.bearer(token=vault))
    key = _tool(f"{server}/echo", auth=WebhookAuth.api_key(header="X-Key", token=lambda: "k1"))
    assert (await _call(bearer, {"order_id": "A1"}))["headers"]["authorization"] == "Bearer t1"
    assert (await _call(bearer, {"order_id": "A1"}))["headers"]["authorization"] == "Bearer t2"
    assert (await _call(key, {"order_id": "A1"}))["headers"]["x-key"] == "k1"


@pytest.mark.parametrize("token", [lambda: "", lambda: 1 / 0])
async def test_a_provider_without_a_token_refuses_the_call(server, token, caplog):
    tool = _tool(f"{server}/echo", auth=WebhookAuth.bearer(token=token))
    with pytest.raises(ToolError, match="could not authenticate"):
        await _call(tool, {"order_id": "A1"})
    assert "token provider" in caplog.text


@pytest.mark.parametrize(
    "build", [lambda: WebhookAuth("bearer"), lambda: WebhookAuth.bearer("T", token=lambda: "x")]
)
def test_auth_needs_exactly_one_token_source(build):
    with pytest.raises(ValueError, match="exactly one of token_env and token"):
        build()


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: WebhookAuth.bearer("sk-live-123"), "UPPER_SNAKE"),
        (lambda: WebhookAuth("bearer", "TOKEN", "X-Key"), "bearer always sends Authorization"),
        (lambda: WebhookAuth.api_key("TOKEN", "Host"), "cannot be used"),
    ],
)
def test_bad_auth_fails_at_start(build, message):
    with pytest.raises(ValueError, match=message):
        build()


async def test_the_envelope_wraps_the_arguments_with_the_tool_and_the_call_facts(server):
    tool = _tool(f"{server}/echo", body="envelope")
    registry = ToolRegistry([tool])
    result = await registry.call(
        tool.name, {"order_id": "A1"}, context={"call_id": "c-1", "direction": "inbound"}
    )
    assert isinstance(result, dict)
    sent = json.loads(result["body"])
    assert sent.pop("timestamp") > 0
    assert sent == {
        "tool_name": "lookup_order",
        "description": "Look up an order.",
        "arguments": {"order_id": "A1"},
        "context": {"call_id": "c-1", "direction": "inbound"},
    }


async def test_a_webhook_that_does_not_wait_answers_at_once_and_sends_behind(server, caplog):
    tool = _tool(f"{server}/slow", wait_for_response=False, timeout=1, output=Accepted)
    assert await _call(tool, {"order_id": "A1"}) == {"ok": True, "accepted": True}
    executor = tool.executor
    assert isinstance(executor, Webhook)
    await executor.drain(timeout=3)
    assert "a send that did not wait failed" in caplog.text


def test_a_webhook_that_does_not_wait_must_say_its_output_is_accepted(server):
    with pytest.raises(ValueError, match="Pass output=Accepted"):
        _tool(f"{server}/slow", wait_for_response=False)


async def test_a_webhook_that_does_not_wait_still_refuses_an_unset_placeholder(server):
    tool = _tool(
        f"{server}/customers/{{{{customer_id}}}}", wait_for_response=False, output=Accepted
    )
    refused = await _call(tool, {"order_id": "A1"}, CallState(Customer))
    assert "customer_id not set yet" in refused["refused"]


@pytest.mark.parametrize("query", ["order_id=7", "order_id", "order_id="])
async def test_a_get_argument_that_clashes_with_a_url_query_value_is_refused(server, query):
    webhook = Webhook(f"{server}/echo?{query}", method="GET", allow_private_network=True)
    with pytest.raises(ToolError, match="'order_id' clashes"):
        await webhook.run({"order_id": "A1"}, None)


def test_a_get_tool_whose_argument_or_locked_value_is_a_url_query_name_is_refused_at_build():
    url = "https://api.example.com/orders?order_id=7"
    with pytest.raises(ValueError, match="Rename it"):
        webhook_tool(
            "lookup_order", "Look up.", url, input=OrderInput, output=AnyResult, method="GET"
        )
    with pytest.raises(ValueError, match="Rename it"):
        webhook_tool(
            "lookup_order",
            "Look up.",
            url,
            input=NoArguments,
            output=AnyResult,
            method="GET",
            locked={"order_id": "{{order_id}}"},
        )


def test_a_bad_config_error_never_repeats_a_secret():
    with pytest.raises(ValueError, match="no body to sign") as raised:
        Webhook("https://api.example.com/x", method="GET", hmac_secret="s3cr3t-value")
    assert "s3cr3t-value" not in str(raised.value)


def test_a_method_is_read_in_any_case():
    assert Webhook("https://api.example.com/x", method="patch").method == "PATCH"


def test_an_unknown_method_keeps_its_plain_error() -> None:
    with pytest.raises(
        ValueError, match=r"^webhook method 'fetch' must be one of \['DELETE'"
    ) as raised:
        Webhook("https://orders.test/a", method="fetch")
    assert type(raised.value) is ValueError


def test_a_range_error_names_the_public_class_and_the_field():
    with pytest.raises(ValueError) as caught:
        Webhook("https://api.example.com/x", timeout=120)
    assert str(caught.value).splitlines()[:2] == ["1 validation error for Webhook", "timeout"]
