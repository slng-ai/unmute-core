"""A main agent hands the call to another, on the fake pipeline around a real FlowManager.

What a handoff does on both adapters is in tests/conformance (``handoff*.json``). These
cases cover the Pipecat side: how the handoff call is answered, and the prompt filled on
the native path.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from google.genai import types
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams

from tests.pipecat.harness import MAIN_PROMPT, FakeAggregators, FakeLLM, Harness, runs_llm
from tests.tasks.job import JOB, Canned, specialist_plan
from tests.test_handoff import KeyLimited, Summarizer
from unmute_core.agent import AgentSpec
from unmute_core.handoff import HandoffContext
from unmute_core.pipecat import ModelCallLLMService, UnmuteFlows

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}
NOTES_PROMPT = "You take notes."
ORDERS_BACK_PROMPT = "You help with orders. Note: {{note}}"


def _desks() -> Harness:
    """A call on the orders desk, whose ``to_notes`` hands it to the notes desk and back."""
    h: Harness

    def handoff(name: str, to: Any) -> FunctionSchema:
        """A native tool whose handler hands the call to the desk ``to`` builds."""

        async def handler(params: FunctionCallParams) -> None:
            """Hand the call off."""
            spec, native = to()
            await h.flows.hand_off(params, spec, native_tools=native)

        return FunctionSchema(name, f"The {name} handoff.", {}, [], handler=handler)

    def notes() -> tuple[AgentSpec, list[FunctionSchema]]:
        return AgentSpec(NOTES_PROMPT, tasks=specialist_plan()), [handoff("to_orders", orders)]

    def orders() -> tuple[AgentSpec, list[FunctionSchema]]:
        return AgentSpec(ORDERS_BACK_PROMPT, tasks=h.tasks.plan), [handoff("to_notes", notes)]

    h = Harness(native_tools=[handoff("to_notes", notes)], lookup_order=Canned(FOUND))
    return h


async def test_the_handoff_call_is_answered_without_a_reply_and_the_new_agent_is_run() -> None:
    h = _desks()
    assert h.tool_names() == ["search_web", "normalize", "find_order", "order_flow", "to_notes"]
    h.user("I need a note taken.")
    result, properties = await h.call("to_notes", said="Putting you through.")

    assert result == {"status": "handed off"}
    assert not runs_llm(properties)
    assert h.system_instruction == NOTES_PROMPT
    assert h.runs == 1


async def test_a_value_the_specialist_saved_reaches_the_first_agent_after_the_way_back() -> None:
    h = _desks()
    h.user("Notes please.")
    await h.call("to_notes")
    h.user("Take a note.")
    await h.call("take_note")
    h.user("Call me back.")
    await h.call("finish", {"note": "call back"})
    h.user("Now my order.")
    await h.call("to_orders")

    assert h.system_instruction == "You help with orders. Note: call back"
    assert h.tool_names() == ["search_web", "normalize", "find_order", "order_flow", "to_notes"]
    assert MAIN_PROMPT not in h.system_instruction


def _summary_desk(summarizer: Summarizer | None) -> Harness:
    """A call whose ``to_notes`` hands off with a summary."""
    h: Harness

    async def handler(params: FunctionCallParams) -> None:
        """Hand the call off with a summary."""
        await h.flows.hand_off(
            params,
            AgentSpec(NOTES_PROMPT, tasks=specialist_plan()),
            context=HandoffContext(history="summary", summarizer=summarizer),
        )

    tool = FunctionSchema("to_notes", "The to_notes handoff.", {}, [], handler=handler)
    h = Harness(native_tools=[tool], lookup_order=Canned(FOUND))
    return h


async def test_a_summary_handoff_asks_its_own_summarizer() -> None:
    summarizer = Summarizer("Notes wanted.")
    h = _summary_desk(summarizer)
    h.user("I need a note taken.")
    await h.call("to_notes")

    [request] = summarizer.requests
    assert request.site == "summary"
    assert request.messages[1]["content"] == "user: I need a note taken."
    assert h.messages() == [
        {"role": "assistant", "content": "Summary of the conversation so far: Notes wanted."}
    ]


async def test_a_summary_no_model_can_write_falls_back_to_what_was_said() -> None:
    h = _summary_desk(None)
    h.user("I need a note taken.")
    await h.call("to_notes")

    assert h.messages() == [{"role": "user", "content": "I need a note taken."}]


async def test_the_first_summary_handoff_with_no_model_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    h = _summary_desk(None)
    h.user("I need a note taken.")
    with caplog.at_level("WARNING", logger="unmute_core.pipecat._side"):
        await h.call("to_notes")
        assert await h.flows._side.summary([], None) is None
    warnings = [r for r in caplog.records if "summary handoff has no summarizer" in r.message]
    assert len(warnings) == 1


def test_a_native_service_that_can_be_asked_says_so() -> None:
    h = Harness(llm=ModelCallLLMService(cast(Any, Summarizer("x"))))
    assert h.flows._side.can_ask
    assert not Harness().flows._side.can_ask


def test_a_record_plan_on_a_service_that_cannot_be_asked_warns_once_at_build(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="unmute_core.pipecat._tasks"):
        Harness(mode="record")
        Harness(mode="status")
    warnings = [r for r in caplog.records if "finish check" in r.message]
    assert len(warnings) == 1


def test_a_summary_history_with_no_summarizer_warns_once_at_build(
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = {**JOB, "tasks": [{**JOB["tasks"][0], "history": "summary"}, *JOB["tasks"][1:]]}
    with caplog.at_level("WARNING", logger="unmute_core.pipecat._tasks"):
        Harness(job, mode="status")
    assert len([r for r in caplog.records if "history 'summary'" in r.message]) == 1


async def test_a_service_that_cannot_be_asked_runs_the_finish_check_on_the_summarizer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    summarizer = Summarizer("x")
    note = {**JOB["tasks"][1], "when": "The caller wants to leave a note."}
    payload = {**JOB, "groups": [], "tasks": [note], "task_only_tools": []}
    with caplog.at_level("WARNING", logger="unmute_core.pipecat._tasks"):
        h = Harness(payload, mode="record", summarizer=summarizer)
    assert not [r for r in caplog.records if "finish check" in r.message]
    h.user("I want to leave a note.")
    await h.call("take_note")
    await h.reply("What is your note?")
    h.user("Please call me back.")
    await h.reply("Noted: please call me back.")
    await asyncio.sleep(0.05)
    assert [request.tool_choice for request in summarizer.requests] == ["required"]


class _GoogleModels:
    """A GenAI client's ``aio.models`` that finishes the step and keeps what it was sent."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.sent.append(kwargs)
        finish = types.FunctionCall(name="finish", args={"note": "Please call me back."})
        content = types.Content(role="model", parts=[types.Part(function_call=finish)])
        return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


async def test_a_google_service_runs_the_finish_check_on_its_own_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = GoogleLLMService(api_key="test", settings=GoogleLLMService.Settings(model="gemini-x"))
    models = _GoogleModels()
    llm._client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
    finishes: list[Any] = []

    async def run_function_calls(calls: list[Any]) -> None:
        finishes.extend(calls)

    llm.run_function_calls = run_function_calls  # ty: ignore[invalid-assignment]
    note = {**JOB["tasks"][1], "when": "The caller wants to leave a note."}
    payload = {**JOB, "groups": [], "tasks": [note], "task_only_tools": []}
    with caplog.at_level("WARNING", logger="unmute_core.pipecat._tasks"):
        h = Harness(payload, mode="record", llm=llm)
    assert not [r for r in caplog.records if "finish check" in r.message]
    h.user("I want to leave a note.")
    await h.call("take_note")
    await h.reply("What is your note?")
    h.user("Please call me back.")
    await h.reply("Noted: please call me back.")
    await asyncio.sleep(0.05)

    [sent] = models.sent
    assert sent["model"] == "gemini-x"
    assert sent["config"]["tool_config"] == {"function_calling_config": {"mode": "ANY"}}
    [finish] = finishes
    assert (finish.function_name, finish.arguments) == ("finish", {"note": "Please call me back."})


def test_a_service_that_can_be_asked_warns_about_nothing_at_build(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = ModelCallLLMService(cast(Any, Summarizer("x")))
    with caplog.at_level("WARNING", logger="unmute_core.pipecat._tasks"):
        Harness(mode="record", llm=llm)
    assert caplog.records == []


def test_a_task_handoff_needs_the_agents_tool() -> None:
    job = {**JOB, "tasks": [{**JOB["tasks"][0], "handoffs": ["to_nowhere"]}, *JOB["tasks"][1:]]}
    with pytest.raises(ValueError, match="lists the handoff 'to_nowhere'"):
        Harness(job)


def test_a_task_model_is_refused_on_pipecat() -> None:
    job = {**JOB, "tasks": [{**JOB["tasks"][0], "model": "fast"}, *JOB["tasks"][1:]]}
    with pytest.raises(ValueError, match="runs on the model 'fast', and Pipecat runs every task"):
        Harness(job, task_models={"fast": Summarizer("x")})


async def test_a_step_handoff_that_breaks_leaves_the_step_running() -> None:
    async def broken(params: FunctionCallParams) -> None:
        """A handoff whose handler fails before it hands off."""
        raise RuntimeError("the notes desk is down")

    job = {**JOB, "tasks": [{**JOB["tasks"][0], "handoffs": ["to_notes"]}, *JOB["tasks"][1:]]}
    tool = FunctionSchema("to_notes", "The to_notes handoff.", {}, [], handler=broken)
    h = Harness(job, native_tools=[tool], lookup_order=Canned(FOUND))
    h.user("Where is my order?")
    await h.call("find_order")
    h.user("Put me through to notes.")
    with pytest.raises(RuntimeError, match="down"):
        await h.call("to_notes")

    h.user("A-12.")
    result, _ = await h.call("finish", {"order_id": "A-12"})
    assert result == {"status": "completed"}


def test_an_agent_with_no_tasks_needs_no_task_stack() -> None:
    async def handler(params: FunctionCallParams) -> None:
        """Do nothing: the test never calls it."""

    handoff = FunctionSchema("to_notes", "The handoff.", {}, [], handler=handler)
    context = LLMContext()
    flows = UnmuteFlows(
        AgentSpec("You help with orders."),
        llm=cast(Any, FakeLLM()),
        context=context,
        aggregators=cast(Any, FakeAggregators(context)),
        native_tools=[handoff],
    )
    flows.install()

    assert flows.main_tools() == []
    assert isinstance(context.tools, ToolsSchema)
    assert [tool.name for tool in context.tools.standard_tools] == ["to_notes"]
    assert flows.state.names == ()
    assert flows.main_instruction() == "You help with orders."


def _transferring_job() -> dict[str, Any]:
    """The job, whose order flow takes a note and hands the caller to the notes desk."""
    group = {**JOB["groups"][0], "steps": ["take_note"], "then": "transfer", "then_target": "notes"}
    return {**JOB, "groups": [group]}


def test_a_group_transfer_needs_its_target() -> None:
    with pytest.raises(ValueError, match="transfers the call to 'notes', which then_targets"):
        Harness(_transferring_job())


async def test_a_completed_group_hands_the_call_to_its_target() -> None:
    h: Harness

    async def to_notes(params: FunctionCallParams) -> None:
        """Hand the call to the notes desk."""
        notes = AgentSpec(NOTES_PROMPT, name="notes", tasks=specialist_plan())
        await h.flows.hand_off(params, notes)

    h = Harness(_transferring_job(), then_targets={"notes": to_notes})
    h.user("Take a note.")
    await h.call("order_flow")
    h.user("Call me back.")
    runs = h.runs
    _, properties = await h.call("finish", {"note": "call back"}, said="Noted.")
    h.pump()

    assert h.state.plain("note") == "call back"
    assert h.system_instruction == NOTES_PROMPT
    assert h.runs == runs + 1
    assert not runs_llm(properties)
    assert h.messages() == [{"role": "user", "content": "Take a note."}]
    assert "order_flow" not in h.tool_names()


async def test_a_group_transfer_that_breaks_returns_to_the_main_agent() -> None:
    async def broken(params: FunctionCallParams) -> None:
        """A handoff whose handler fails before it hands off."""
        raise RuntimeError("the notes desk is down")

    h = Harness(_transferring_job(), then_targets={"notes": broken})
    h.user("Take a note.")
    await h.call("order_flow")
    h.user("Call me back.")
    _, properties = await h.call("finish", {"note": "call back"}, said="Noted.")
    h.pump()

    assert h.tool_names() == ["search_web", "normalize", "find_order", "order_flow"]
    assert h.system_instruction == h.flows.main_instruction()
    assert runs_llm(properties)
    assert h.main_reads()["status"] == "completed"


async def test_a_handoff_whose_native_tool_clashes_is_refused_before_anything_changes() -> None:
    h: Harness

    async def handler(params: FunctionCallParams) -> None:
        """Hand the call to an agent whose native tool repeats a registry tool."""
        clash = FunctionSchema("search_web", "A clash.", {}, [])
        await h.flows.hand_off(
            params, AgentSpec(NOTES_PROMPT, tasks=specialist_plan()), native_tools=[clash]
        )

    tool = FunctionSchema("to_notes", "The to_notes handoff.", {}, [], handler=handler)
    h = Harness(native_tools=[tool], lookup_order=Canned(FOUND))
    h.user("I need a note taken.")
    with pytest.raises(ValueError, match="the tool 'search_web' is offered twice on one agent"):
        await h.call("to_notes")
    assert h.system_instruction == MAIN_PROMPT


async def test_a_handoff_prompt_with_a_placeholder_too_long_for_its_model_call_is_refused() -> None:
    h: Harness
    refused: list[ValueError] = []

    async def to_notes(params: FunctionCallParams) -> None:
        """Hand the call to a desk whose prompt cannot go to the router."""
        notes = AgentSpec("Hi {{" + "a" * 70 + "}}", name="notes", tasks=specialist_plan())
        try:
            await h.flows.hand_off(params, notes)
        except ValueError as error:
            refused.append(error)
            raise

    h = Harness(
        _transferring_job(),
        then_targets={"notes": to_notes},
        llm=ModelCallLLMService(cast(Any, KeyLimited("x"))),
    )
    h.user("Take a note.")
    await h.call("order_flow")
    h.user("Call me back.")
    await h.call("finish", {"note": "call back"}, said="Noted.")

    [error] = refused
    assert "agent instructions: the placeholder" in str(error)
