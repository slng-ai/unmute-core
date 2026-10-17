"""What both smoke runs share.

The caller line, the scripted reply, the offline switch and the check that the agent
answered.
"""

import logging
import os
import socket
from collections.abc import AsyncIterator
from typing import Any

from unmute_core.model import ModelEvent, ModelRequest, TextDelta

REPLY = "Hello from the smoke test."
CALLER = "Hi, can you help me?"
PLACEHOLDERS = {
    "OPENAI_API_KEY": "smoke",
    "GOOGLE_API_KEY": "smoke",
    "SLNG_API_KEY": "smoke",
    "SLNG_ROUTER_URL": "https://router.invalid",
    "LANGFUSE_PUBLIC_KEY": "smoke",
    "LANGFUSE_SECRET_KEY": "smoke",
    "LANGFUSE_BASE_URL": "https://langfuse.invalid",
    "OTEL_EXPORTER_OTLP_TIMEOUT": "1",
    "UNMUTE_AGENT_ID": "smoke",
    "FARO_DB": ":memory:",
}
PROMPTS: list[str] = []
"""The system text of each request a scripted model answered."""


class LibraryErrors(logging.Handler):
    """Keeps every error the library logs. A Pipecat processor logs an error and goes on."""

    def __init__(self) -> None:
        """Start with no errors."""
        super().__init__(logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the message."""
        self.messages.append(f"{record.name}: {record.getMessage()}")


ERRORS = LibraryErrors()


def system_text(messages: list[dict[str, Any]]) -> str:
    """Return the system text of a chat request, in the OpenAI message shape."""
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") in ("system", "developer")
    )


class Scripted:
    """A ``ModelCall`` that answers every request with the scripted reply."""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Keep the request's system text, then answer with the reply."""
        PROMPTS.append(system_text(request.messages))
        yield TextDelta(REPLY)


def refuse_network(*args: object, **kwargs: object) -> None:
    """Fail any network call: an offline smoke sends nothing."""
    raise OSError("the offline smoke runs without network")


def go_offline() -> None:
    """Fill placeholder keys, block the network, and script what would call out.

    The plugins only check that a key is set when they are built. The smoke never lets
    them send a request. A ``.env`` file is not read, so a developer's keys change
    nothing. When installed, OpenAI embeddings become llama-index's mock embedder, and
    the SLNG router answers with the script.
    """
    for key, value in PLACEHOLDERS.items():
        os.environ.setdefault(key, value)
    try:
        import dotenv
    except ImportError:
        pass
    else:
        dotenv.load_dotenv = lambda *args, **kwargs: False  # ty: ignore[invalid-assignment]
    socket.socket.connect = refuse_network
    socket.getaddrinfo = refuse_network  # ty: ignore[invalid-assignment]
    try:
        from unmute_core.knowledge._offline import mock_openai_embeddings
    except ImportError:
        pass
    else:
        mock_openai_embeddings()
    try:
        from unmute_core.slng import RouterModelCall
    except ImportError:
        return
    RouterModelCall.stream = Scripted.stream  # ty: ignore[invalid-assignment]


def listen_for_errors() -> None:
    """Keep every error the library logs from now on, for ``check``."""
    logging.getLogger("unmute_core").addHandler(ERRORS)


def check(name: str, replies: list[str], *, live: bool) -> str:
    """Return the agent's reply, or exit with what went wrong.

    The agent must answer, the library must log no error, and offline every model
    request must carry the agent's instructions.
    """
    reply = " ".join(replies).strip()
    if not reply or (not live and REPLY not in reply):
        raise SystemExit(f"{name}: the agent did not answer the caller, got {replies!r}")
    if ERRORS.messages:
        raise SystemExit(f"{name}: the library logged errors: {ERRORS.messages!r}")
    if not live and (not PROMPTS or not all(PROMPTS)):
        raise SystemExit(f"{name}: a model request had no instructions: {PROMPTS!r}")
    return reply
