"""An MCP server's tools on a LiveKit agent: offered as listed, answered, or refused at start."""

import pytest
from livekit.agents import AgentSession, llm

from tests.livekit.fake import ScriptedLLM, call
from tests.mcp_server import closed_port, shop_server
from unmute_core.agent import AgentSpec
from unmute_core.livekit import UnmuteAgent, mcp_toolset
from unmute_core.tools import McpServer, McpServerError


async def test_a_caller_asks_the_hours_and_the_mcp_tool_answers() -> None:
    async with shop_server() as url:
        shop = await mcp_toolset(McpServer("shop", url=url, tools=("opening_hours",)))
        model = ScriptedLLM(call("opening_hours", day="Monday"), "Nine to five on Monday.")
        agent = UnmuteAgent(AgentSpec("You run a shop."), native_tools=[shop])
        async with AgentSession(llm=model) as session:
            await session.start(agent)
            await session.run(user_input="When are you open on Monday?")
    first, second = model.requests
    assert first.tools == ["opening_hours"]
    outputs = [i.output for i in second.chat_ctx.items if isinstance(i, llm.FunctionCallOutput)]
    assert outputs == ["Monday: 9 to 17"]


async def test_a_server_that_is_down_stops_the_call_before_it_starts() -> None:
    server = McpServer("shop", url=f"http://127.0.0.1:{closed_port()}/mcp", timeout=2)
    with pytest.raises(McpServerError, match="'shop' could not be reached"):
        await mcp_toolset(server)
