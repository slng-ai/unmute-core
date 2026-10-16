"""``save_values`` shows the values whenever State has some, and the tool when one is free."""

from typing import Any

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pydantic import BaseModel

from tests.pipecat.test_model import Scripted, service
from tests.tasks.job import SPECIALIST, plan, registry, state
from unmute_core.agent import AgentSpec
from unmute_core.pipecat import UnmuteFlows
from unmute_core.state import CallState
from unmute_core.tasks import TaskPlan, TasksSpec


class Owned(BaseModel):
    """One value, and a task saves it."""

    note: str | None = None


class Empty(BaseModel):
    """No values."""


class Mixed(Owned):
    """The owned value and one nothing saves."""

    mood: str | None = None


def flows(model: type[BaseModel], tasks: dict[str, Any] = SPECIALIST) -> UnmuteFlows:
    context = LLMContext()
    task_plan = TaskPlan(TasksSpec.model_validate(tasks), state=model, tools=registry())
    return UnmuteFlows(
        AgentSpec("You help.", tasks=task_plan, save_values=True),
        llm=service(Scripted()),
        context=context,
        aggregators=LLMContextAggregatorPair(context),
        state=CallState(task_plan.state),
    )


def shown(flow: UnmuteFlows) -> list[Any]:
    context = LLMContext([{"role": "user", "content": "Hi."}])
    flow._show_values(context)
    messages: list[Any] = context.get_messages()
    return [m["content"] for m in messages if m["role"] == "system"]


def test_values_all_owned_show_the_block_and_no_save_tool() -> None:
    flow = flows(Owned)
    assert "save_values" not in {tool["name"] for tool in flow.main_tools()}
    [block] = shown(flow)
    assert "- note, saved by the task take_note" in block


def test_a_free_value_adds_the_save_tool_and_the_block() -> None:
    flow = flows(Mixed)
    assert "save_values" in {tool["name"] for tool in flow.main_tools()}
    [block] = shown(flow)
    assert "- mood: " in block


def test_a_state_without_values_shows_no_block() -> None:
    assert shown(flows(Empty, {"mode": "record", "tasks": []})) == []


def test_the_job_state_still_builds_with_save_values() -> None:
    task_plan = plan({"mode": "record", "tasks": []})
    context = LLMContext()
    flow = UnmuteFlows(
        AgentSpec("You help.", tasks=task_plan, save_values=True),
        llm=service(Scripted()),
        context=context,
        aggregators=LLMContextAggregatorPair(context),
        state=state(task_plan),
    )
    assert "save_values" in {tool["name"] for tool in flow.main_tools()}
