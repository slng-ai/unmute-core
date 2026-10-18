"""An MCP server's tools on a Pipecat call: offered as listed, answered, or refused at start."""

import pytest

from tests.mcp_server import closed_port, shop_server
from tests.pipecat.harness import Harness
from unmute_core.pipecat import mcp_tools
from unmute_core.tools import McpServer, McpServerError


async def test_a_caller_asks_the_hours_and_the_mcp_tool_answers() -> None:
    async with shop_server() as url:
        tools = await mcp_tools(McpServer("shop", url=url, tools=("opening_hours",)))
        h = Harness(native_tools=tools)
        h.user("When are you open on Monday?")
        result, _ = await h.call("opening_hours", {"day": "Monday"})
    assert "opening_hours" in h.tool_names()
    assert "close_shop" not in h.tool_names()
    assert result == "Monday: 9 to 17"


async def test_a_server_that_is_down_stops_the_call_before_it_starts() -> None:
    server = McpServer("shop", url=f"http://127.0.0.1:{closed_port()}/mcp", timeout=2)
    with pytest.raises(McpServerError, match="'shop' could not be reached"):
        await mcp_tools(server)
