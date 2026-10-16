"""A fake Context Router behind the openai client's MockTransport, and the router's own fill rule.

``router_fill`` matches the SLNG Context Router's placeholder substitution. It is kept
apart from the library's code on purpose: the parity test checks the library against it.
"""

import json
import re
from typing import Any

from openai import AsyncOpenAI

from tests.openai_http import http

_ROUTER_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
_SYSTEM_LIKE_ROLES = {"system", "developer"}


def _substitute_text(text: str, variables: dict[str, str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return variables[name] if name in variables else match.group(0)

    return _ROUTER_PLACEHOLDER.sub(_replace, text)


def router_fill(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the messages the model sees after the router fills them."""
    variables = body.get("template_variables") or {}
    filled = []
    for message in body["messages"]:
        message = dict(message)
        if message.get("role") in _SYSTEM_LIKE_ROLES:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = _substitute_text(content, variables)
            elif isinstance(content, list):
                message["content"] = [
                    {**part, "text": _substitute_text(part["text"], variables)}
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                    else part
                    for part in content
                ]
        filled.append(message)
    return filled


def chunk(delta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


def sse(chunks: list[dict[str, Any]]) -> bytes:
    frames = [f"data: {json.dumps(item)}\n\n" for item in chunks]
    return ("".join(frames) + "data: [DONE]\n\n").encode()


class FakeRouter:
    """Records each request and answers with a fixed SSE stream or an error status."""

    def __init__(
        self,
        chunks: list[dict[str, Any]] | None = None,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        self.chunks = chunks if chunks is not None else [chunk({"content": "ok"})]
        self.status = status
        self.headers = headers or {}
        self.error = error or {"message": "no"}
        self.requests: list[http.Request] = []

    def handle(self, request: http.Request) -> http.Response:
        self.requests.append(request)
        if self.status != 200:
            return http.Response(self.status, json={"error": self.error}, headers=self.headers)
        return http.Response(
            200,
            content=sse(self.chunks),
            headers={"content-type": "text/event-stream", **self.headers},
        )

    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="test",
            base_url="https://router.test/v1",
            max_retries=0,
            http_client=http.AsyncClient(transport=http.MockTransport(self.handle)),
        )

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)
