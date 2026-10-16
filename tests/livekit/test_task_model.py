"""A task that names its own model runs its step on it, on a real text-only session."""

import pytest
from livekit.agents import AgentSession

from tests.livekit.fake import ScriptedLLM, call
from tests.tasks.job import JOB, Canned, plan, registry, state
from unmute_core.agent import AgentSpec
from unmute_core.livekit import LiveKitModel, UnmuteAgent

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}
FAST = {**JOB, "tasks": [{**JOB["tasks"][0], "model": "fast"}, *JOB["tasks"][1:]]}


async def test_the_step_asks_the_tasks_model_and_the_agent_asks_its_own() -> None:
    main = ScriptedLLM(call("find_order"), "Anything else?")
    fast = ScriptedLLM("Which order is it?")
    tasks = plan(FAST, tools=registry(lookup_order=Canned(FOUND)))
    agent = UnmuteAgent(
        AgentSpec("You help with orders.", tasks=tasks, task_models={"fast": LiveKitModel(fast)}),
        state=state(tasks),
    )
    async with AgentSession(llm=main) as session:
        await session.start(agent)
        await session.run(user_input="Where is my order?")
    assert len(main.requests) == 1
    assert fast.requests
    assert all("lookup_order" in request.tools for request in fast.requests)
    assert all("find_order" not in request.tools for request in fast.requests)


def test_a_task_model_needs_its_entry_in_task_models() -> None:
    tasks = plan(FAST)
    with pytest.raises(ValueError, match="runs on the model 'fast', which task_models does not"):
        UnmuteAgent(AgentSpec("Orders.", tasks=tasks), state=state(tasks))
