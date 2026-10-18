"""An MCP server's settings, the same on every framework.

Each adapter turns an ``McpServer`` into its framework's own client:
``unmute_core.livekit.mcp_toolset`` and ``unmute_core.pipecat.mcp_tools``.
"""

import re
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import AfterValidator, Field, model_validator
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.settings import env_value
from unmute_core.tools._executors import WebhookAuth
from unmute_core.tools._spec import ToolError

_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*")

McpTransport = Literal["sse", "streamable_http"]
"""How the client talks to the server."""


class McpServerError(RuntimeError):
    """An MCP server the agent needs cannot be reached, or its settings are incomplete."""


def missing_extra(framework: str) -> ImportError:
    """Return the error for an MCP server built without the ``mcp`` extra."""
    return ImportError(f'MCP tools need the mcp extra: install "unmute-core[{framework},mcp]"')


def _env_name(name: str) -> str:
    """Refuse an environment variable name that is not UPPER_SNAKE: it may be an address."""
    if name and not _ENV_NAME.fullmatch(name):
        raise ValueError("url_env must be an UPPER_SNAKE environment variable name")
    return name


def _unique(names: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Refuse a tool filter that names one tool twice."""
    if names is not None and len(set(names)) != len(names):
        raise ValueError("tools lists one tool name twice")
    return names


@original_errors
@dataclass(frozen=True, config=VALUE)
class McpServer:
    """One MCP server whose tools the agent offers.

    Attributes:
        name: The source's name. A LiveKit task lists it in ``native_tools``.
        url: The server's address. Use ``url_env`` to keep it out of the code.
        url_env: The environment variable that holds the address, in UPPER_SNAKE case.
        transport: ``sse`` or ``streamable_http``. None picks streamable HTTP for a path
            that ends in ``/mcp``, and SSE for any other.
        auth: The credential sent with every request. None sends none.
        tools: The server tools to offer. None offers every tool the server lists.
        timeout: Seconds to wait for the connection and for each call.
    """

    name: Annotated[str, Field(min_length=1, max_length=64)]
    url: str = ""
    url_env: Annotated[str, AfterValidator(_env_name)] = ""
    transport: McpTransport | None = None
    auth: WebhookAuth | None = None
    tools: Annotated[
        tuple[Annotated[str, Field(min_length=1)], ...] | None, AfterValidator(_unique)
    ] = None
    timeout: Annotated[float, Field(gt=0, le=300)] = 30

    @model_validator(mode="after")
    def _one_address(self) -> Self:
        """Refuse a server with no address source, or with two.

        Raises:
            ValueError: If neither or both of ``url`` and ``url_env`` are set.
        """
        if bool(self.url) == bool(self.url_env):
            raise ValueError("an MCP server needs exactly one of url and url_env")
        return self

    def address(self) -> str:
        """Return the server's address, read now when it comes from ``url_env``.

        Raises:
            McpServerError: If ``url_env`` is unset or empty.
        """
        url = self.url or env_value(self.url_env, str) or ""
        if not url:
            raise McpServerError(
                f"MCP server {self.name!r}: set the environment variable {self.url_env}"
            )
        return url

    def transport_for(self, url: str) -> McpTransport:
        """Return the transport to use for ``url``: the one set, or the one its path implies."""
        if self.transport is not None:
            return self.transport
        return "streamable_http" if urlsplit(url).path.rstrip("/").endswith("/mcp") else "sse"

    async def headers(self) -> dict[str, str]:
        """Return the headers to send, with the token read now.

        Raises:
            McpServerError: If the token is unset or empty, or its provider fails.
        """
        if self.auth is None:
            return {}
        try:
            return await self.auth.headers()
        except ToolError as error:
            what = (
                f"set the environment variable {self.auth.token_env}"
                if self.auth.token_env
                else "its token provider gave no token"
            )
            raise McpServerError(f"MCP server {self.name!r}: {what}") from error
