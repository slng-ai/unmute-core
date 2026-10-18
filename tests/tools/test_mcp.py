"""An MCP server's settings: its address, transport and credential, the same on both frameworks."""

import pytest

from unmute_core.tools import McpServer, McpServerError, WebhookAuth


@pytest.mark.parametrize(
    ("url", "transport", "expected"),
    [
        ("https://tools.example.com/mcp", None, "streamable_http"),
        ("https://tools.example.com/mcp/", None, "streamable_http"),
        ("https://tools.example.com/sse", None, "sse"),
        ("https://tools.example.com/sse", "streamable_http", "streamable_http"),
    ],
)
def test_the_transport_is_the_one_set_or_the_one_the_path_implies(
    url: str, transport: str | None, expected: str
) -> None:
    server = McpServer("search", url=url, transport=transport)  # ty: ignore[invalid-argument-type]
    assert server.transport_for(url) == expected


@pytest.mark.parametrize(
    ("auth", "expected"),
    [
        (None, {}),
        (WebhookAuth.bearer("SEARCH_KEY"), {"Authorization": "Bearer s3cret"}),
        (WebhookAuth.api_key("SEARCH_KEY", "X-Search-Key"), {"X-Search-Key": "s3cret"}),
    ],
)
async def test_the_credential_rides_in_the_headers(
    monkeypatch: pytest.MonkeyPatch, auth: WebhookAuth | None, expected: dict[str, str]
) -> None:
    monkeypatch.setenv("SEARCH_KEY", "s3cret")
    server = McpServer("search", url="https://tools.example.com/mcp", auth=auth)
    assert await server.headers() == expected


async def test_a_missing_variable_is_named_before_the_call_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEARCH_URL", raising=False)
    monkeypatch.delenv("SEARCH_KEY", raising=False)
    server = McpServer("search", url_env="SEARCH_URL", auth=WebhookAuth.bearer("SEARCH_KEY"))
    with pytest.raises(McpServerError, match="'search': set the environment variable SEARCH_URL"):
        server.address()
    with pytest.raises(McpServerError, match="'search': set the environment variable SEARCH_KEY"):
        await server.headers()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one of url and url_env"),
        ({"url": "https://a.example/mcp", "url_env": "A_URL"}, "exactly one of url and url_env"),
        ({"url_env": "https://a.example/mcp"}, "UPPER_SNAKE"),
        ({"url_env": "A_URL", "tools": ("search", "search")}, "one tool name twice"),
    ],
)
def test_settings_that_cannot_reach_a_server_are_refused(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        McpServer("search", **kwargs)
