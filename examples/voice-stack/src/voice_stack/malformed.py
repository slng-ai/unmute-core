"""Model trial only: log what Gemini was asked when it ends a reply on MALFORMED_FUNCTION_CALL.

Gemini sends no text and no tool call for such a reply, and both frameworks' plugins log only
the finish reason. This wraps the client so the log also shows the tools offered and the
last turns, to find which schemas a small model fails to fill.
"""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

TAIL_TURNS = 3
"""How many of the request's last turns the log shows."""


def watch(client: genai.Client) -> genai.Client:
    """Wrap the client's streaming call, so each malformed function call is logged in full.

    Args:
        client: The client both frameworks' Google plugins call through.

    Returns:
        The same client, its ``generate_content_stream`` wrapped.
    """
    models = client.aio.models
    models.generate_content_stream = _logged(models.generate_content_stream)  # ty: ignore[invalid-assignment]
    return client


def _logged(
    stream: Callable[..., Awaitable[AsyncIterator[types.GenerateContentResponse]]],
) -> Callable[..., Awaitable[AsyncIterator[types.GenerateContentResponse]]]:
    """Return ``stream`` that logs the request of any malformed function call it answers."""

    async def logged(**request: Any) -> AsyncIterator[types.GenerateContentResponse]:
        """Start the stream, and watch its responses."""
        return _watched(await stream(**request), request)

    return logged


async def _watched(
    responses: AsyncIterator[types.GenerateContentResponse], request: dict[str, Any]
) -> AsyncIterator[types.GenerateContentResponse]:
    """Pass every response on, and log the request when a candidate ended malformed."""
    async for response in responses:
        if any(
            candidate.finish_reason == types.FinishReason.MALFORMED_FUNCTION_CALL
            for candidate in response.candidates or ()
        ):
            logger.warning("MALFORMED %s", json.dumps(_record(request), default=str))
        yield response


def _record(request: dict[str, Any]) -> dict[str, Any]:
    """Return the tools a request offered and its last turns, as plain JSON data."""
    config = request.get("config")
    tools = [
        declaration.model_dump(exclude_none=True, mode="json")
        for tool in getattr(config, "tools", None) or ()
        for declaration in getattr(tool, "function_declarations", None) or ()
    ]
    tail = [
        turn.model_dump(exclude_none=True, mode="json") if isinstance(turn, types.Content) else turn
        for turn in list(request.get("contents") or [])[-TAIL_TURNS:]
    ]
    return {"tools": tools, "tail": tail}
