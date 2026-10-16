"""Model trial only: log what Gemini wrote when it ends a reply on MALFORMED_FUNCTION_CALL."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from google import genai

logger = logging.getLogger("voice_stack.malformed")


def watch(client: genai.Client) -> genai.Client:
    """Wrap the client's streaming call, and log each malformed function call in full."""
    models = client.aio.models
    stream = models.generate_content_stream

    async def logged(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream as before. Log the finish message, the tools and the last turns on a malformed call."""
        responses = await stream(*args, **kwargs)

        async def each() -> AsyncIterator[Any]:
            async for response in responses:
                for candidate in response.candidates or ():
                    if str(candidate.finish_reason).endswith("MALFORMED_FUNCTION_CALL"):
                        config = kwargs.get("config")
                        tools = [
                            decl.model_dump(exclude_none=True, mode="json")
                            for tool in (getattr(config, "tools", None) or ())
                            for decl in (getattr(tool, "function_declarations", None) or ())
                        ]
                        contents = kwargs.get("contents") or []
                        tail = [
                            c.model_dump(exclude_none=True, mode="json")
                            if hasattr(c, "model_dump")
                            else c
                            for c in list(contents)[-3:]
                        ]
                        print(
                            "MALFORMED "
                            + json.dumps(
                                {
                                    "finish_message": candidate.finish_message,
                                    "tools": tools,
                                    "tail": tail,
                                },
                                ensure_ascii=False,
                                default=str,
                            ),
                            flush=True,
                        )
                yield response

        return each()

    models.generate_content_stream = logged  # type: ignore[method-assign]
    return client
