"""A real MCP server on a free local port, for the MCP builder tests. No network leaves the host."""

import asyncio
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from mcp.server.mcpserver import MCPServer


def opening_hours(day: str) -> str:
    """Return the shop's opening hours on a day."""
    return f"{day}: 9 to 17"


def close_shop() -> str:
    """Close the shop for the day."""
    return "closed"


def closed_port() -> int:
    """Return a local port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@asynccontextmanager
async def shop_server(path: str = "/mcp") -> AsyncGenerator[str]:
    """Serve a shop's two tools over streamable HTTP. Yields the server's URL."""
    server = MCPServer("shop")
    server.tool()(opening_hours)
    server.tool()(close_shop)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    web = uvicorn.Server(
        uvicorn.Config(server.streamable_http_app(streamable_http_path=path), log_level="warning")
    )
    serving = asyncio.create_task(web.serve(sockets=[listener]))
    while not web.started and not serving.done():
        await asyncio.sleep(0)
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}{path}"
    finally:
        web.should_exit = True
        await serving
        listener.close()
