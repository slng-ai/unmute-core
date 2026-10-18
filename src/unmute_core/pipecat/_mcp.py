"""An MCP server as Pipecat tools, connected before the call starts."""

from pipecat.adapters.schemas.function_schema import FunctionSchema

from unmute_core.tools._mcp import McpServer, McpServerError, missing_extra


async def mcp_tools(server: McpServer) -> list[FunctionSchema]:
    """Return the server's tools as Pipecat tools, each with its handler, already connected.

    Pass them in ``native_tools``. The LLM service closes the connection when the pipeline
    ends. A step gets one when its task lists the tool's own name.

    Args:
        server: The server and the tools to offer.

    Raises:
        McpServerError: If the address or token is missing, or the server cannot be
            reached or listed.
        ImportError: If the ``mcp`` extra is not installed.
    """
    try:
        from pipecat.services.mcp_service import (
            MCPClient,
            SseServerParameters,
            StreamableHttpParameters,
        )
    except ImportError as error:
        raise missing_extra("pipecat") from error
    url = server.address()
    headers = await server.headers()
    params = (
        StreamableHttpParameters(url=url, headers=headers, timeout=server.timeout)
        if server.transport_for(url) == "streamable_http"
        else SseServerParameters(url=url, headers=headers, timeout=server.timeout)
    )
    client = MCPClient(
        server_params=params,
        tools_filter=list(server.tools) if server.tools is not None else None,
    )
    try:
        return list((await client.tools()).standard_tools)
    except Exception as error:
        await client.close()
        raise McpServerError(f"MCP server {server.name!r} could not be reached: {error}") from error
