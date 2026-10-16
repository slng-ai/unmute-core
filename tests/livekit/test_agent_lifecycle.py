"""What an agent owns over its session: the toolsets its steps borrow, the models it closes,
and the checks it runs when it is built.
"""

from collections.abc import AsyncIterator
from typing import cast

import pytest
from livekit.agents import AgentSession, llm
from livekit.agents.beta.tools import EndCallTool

from tests.livekit.fake import ScriptedLLM, call
from tests.tasks.job import Canned, job, plan, registry, state
from tests.test_handoff import KeyLimited
from unmute_core.agent import AgentSpec
from unmute_core.livekit import (
    ModelCallLLM,
    UnmuteAgent,
)
from unmute_core.model import ModelEvent, ModelRequest, TextDelta

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}


class Scripted:
    """A ``ModelCall`` that plays one script per request. An ``Exception`` in it is raised."""

    def __init__(self, *scripts: list[object], model: str = "scripted") -> None:
        self.scripts = list(scripts)
        self.requests: list[ModelRequest] = []
        self.model = model
        self.closed = 0

    async def aclose(self) -> None:
        """Count the close."""
        self.closed += 1

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Play the next script."""
        self.requests.append(request)
        for item in self.scripts.pop(0) if self.scripts else []:
            if isinstance(item, Exception):
                raise item
            yield cast(ModelEvent, item)


class Pinger(llm.Toolset):
    """A toolset of one tool, which closing empties, as a real toolset releases its tools."""

    def __init__(self) -> None:
        async def ping() -> str:
            """Ping."""
            return "pong"

        super().__init__(id="pinger", tools=[llm.function_tool(ping, name="ping")])

    async def aclose(self) -> None:
        """Release the tools."""
        self._tools = []


async def test_a_step_never_owns_the_main_agents_toolset() -> None:
    payload = job()
    payload["tasks"][0]["native_tools"] = ["pinger"]
    tasks = plan(payload, tools=registry(lookup_order=Canned(FOUND)))
    model = ScriptedLLM(call("find_order"), call("lookup_order"), "It is open.")
    agent = UnmuteAgent(
        AgentSpec("Orders.", tasks=tasks), state=state(tasks), native_tools=[Pinger()]
    )
    async with AgentSession(llm=model) as session:
        await session.start(agent)
        await session.run(user_input="Where is my order?")
        await session.run(user_input="And now?")
    main, step, *_, last = model.requests
    assert "ping" in main.tools
    assert "ping" in step.tools
    assert "ping" in last.tools


async def test_the_agent_closes_each_model_it_was_given_once_when_the_session_closes() -> None:
    shared = Scripted([TextDelta("Hi.")])
    own = Scripted()
    agent = UnmuteAgent(
        AgentSpec("You help.", model=shared, summarizer=shared, task_models={"x": own})
    )
    async with AgentSession() as session:
        await session.start(agent)
        await session.run(user_input="Hi.")
    assert agent._releasing is not None
    await agent._releasing
    assert (shared.closed, own.closed) == (1, 1)


async def test_the_llm_closes_its_model_call() -> None:
    model = Scripted()
    await ModelCallLLM(model).aclose()
    assert model.closed == 1


def test_a_placeholder_key_too_long_fails_at_build_only_for_a_model_call_with_a_limit() -> None:
    prompt = "Hello {{" + "a" * 70 + "}}."
    UnmuteAgent(AgentSpec(prompt))
    with pytest.raises(ValueError, match="agent instructions: the placeholder"):
        UnmuteAgent(AgentSpec(prompt, model=KeyLimited("x")))


def test_livekits_own_agent_options_reach_the_agent() -> None:
    history = llm.ChatContext()
    history.add_message(role="user", content="Hi.")
    agent = UnmuteAgent(AgentSpec("Orders."), chat_ctx=history, min_consecutive_speech_delay=0.4)
    assert [item.text_content for item in agent.chat_ctx.messages()] == ["Hi."]
    assert agent.min_consecutive_speech_delay == 0.4


def test_a_task_named_like_a_native_tool_is_refused_at_build() -> None:
    payload = job()
    payload["tasks"][1] = {**payload["tasks"][1], "name": "end_call", "when": "The caller is done."}
    payload |= {"groups": [], "task_only_tools": []}
    tasks = plan(payload)
    with pytest.raises(ValueError, match="the tool 'end_call' is offered twice"):
        UnmuteAgent(
            AgentSpec("Orders.", tasks=tasks), state=state(tasks), native_tools=[EndCallTool()]
        )
