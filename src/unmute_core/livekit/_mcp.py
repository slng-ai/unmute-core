"""An MCP server as a LiveKit toolset, connected before the call starts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from unmute_core.tools._mcp import McpServer, McpServerError, missing_extra

if TYPE_CHECKING:
    from livekit.agents.llm.mcp import MCPToolResultContext, MCPToolset


def _text(result: MCPToolResultContext) -> str:
    """Return a tool result's text parts, joined, as Pipecat gives them to the model.

    LiveKit's own resolver sends each part as JSON. The model then reads a different
    result on each framework.
    """
    return "".join(part.text for part in result.result.content if part.type == "text")


async def mcp_toolset(server: McpServer) -> MCPToolset:
    """Return the server's tools as a LiveKit toolset, already connected.

    Pass it in ``native_tools``. A step gets it when its task lists ``server.name``.
    LiveKit only logs a toolset that fails to connect and starts the agent without its
    tools. This connects first, so a server that is down stops the call before it starts.

    Args:
        server: The server and the tools to offer.

    Raises:
        McpServerError: If the address or token is missing, or the server cannot be
            reached or listed.
        ImportError: If the ``mcp`` extra is not installed.
    """
    try:
        from livekit.agents import mcp
    except ImportError as error:
        raise missing_extra("livekit") from error
    url = server.address()
    toolset = mcp.MCPToolset(
        id=server.name,
        mcp_server=mcp.MCPServerHTTP(
            url,
            transport_type=server.transport_for(url),
            allowed_tools=list(server.tools) if server.tools is not None else None,
            headers=await server.headers(),
            timeout=server.timeout,
            client_session_timeout_seconds=server.timeout,
            tool_result_resolver=_text,
        ),
    )
    try:
        return await toolset.setup()
    except Exception as error:
        await toolset.aclose()
        raise McpServerError(f"MCP server {server.name!r} could not be reached: {error}") from error
