"""A main agent hands the call to another, on a real text-only LiveKit session.

What a handoff does on both adapters is in tests/conformance (``handoff*.json``). These
cases cover the LiveKit side: the session's own handoff, the prompt filled on the native
path, the shared call span, and native tools.
"""

from collections.abc import Sequence
from typing import Any, cast

import pytest
from livekit.agents import Agent, AgentSession, function_tool, llm
from livekit.agents.beta.tools import EndCallTool

from tests.livekit.fake import ScriptedLLM, call
from tests.livekit.test_trace import Memory
from tests.tasks.job import JOB, Canned, plan, registry, specialist_plan, state
from tests.test_handoff import Summarizer
from unmute_core.agent import AgentSpec
from unmute_core.handoff import HandoffContext
from unmute_core.livekit import UnmuteAgent, shape_chat_ctx
from unmute_core.state import CallState
from unmute_core.telemetry import Telemetry

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}


class Orders(UnmuteAgent):
    """The job's agent, with a handoff to the notes desk."""

    def __init__(
        self,
        call_state: CallState,
        *,
        telemetry: Telemetry | None = None,
        native_tools: Sequence[llm.Tool | llm.Toolset] = (),
    ) -> None:
        super().__init__(
            AgentSpec(
                "You help with orders. Note: {{note}}",
                name="orders",
                tasks=plan(tools=registry(lookup_order=Canned(FOUND))),
            ),
            state=call_state,
            telemetry=telemetry,
            native_tools=native_tools,
        )

    @function_tool
    async def to_notes(self) -> UnmuteAgent | str:
        """Hand the caller to the notes desk."""
        assert self.state is not None
        return await self.hand_off(Notes(self.state))


class Notes(UnmuteAgent):
    """The specialist agent, with a handoff back to orders."""

    def __init__(self, call_state: CallState) -> None:
        super().__init__(
            AgentSpec("You take notes.", name="notes", tasks=specialist_plan()), state=call_state
        )

    @function_tool
    async def to_orders(self) -> UnmuteAgent | str:
        """Hand the caller back to orders."""
        assert self.state is not None
        return await self.hand_off(Orders(self.state))


def _state() -> CallState:
    """A call state both desks share."""
    return state(specialist_plan())


async def test_a_handoff_is_a_livekit_handoff_to_the_new_agent() -> None:
    model = ScriptedLLM(call("to_notes"), "Notes desk.")
    async with AgentSession(llm=model) as session:
        await session.start(Orders(_state()))
        result = await session.run(user_input="I need a note taken.")
        result.expect.contains_agent_handoff(new_agent_type=Notes)
        assert isinstance(session.current_agent, Notes)


async def test_a_value_the_specialist_saved_reaches_the_first_agent_after_the_way_back() -> None:
    model = ScriptedLLM(
        call("to_notes"),
        "Notes desk.",
        call("take_note"),
        "What should the note say?",
        call("finish", note="call back"),
        "Saved.",
        call("to_orders"),
        "Orders again.",
    )
    call_state = _state()
    async with AgentSession(llm=model) as session:
        await session.start(Orders(call_state))
        await session.run(user_input="Notes please.")
        await session.run(user_input="Take a note.")
        await session.run(user_input="Call me back.")
        await session.run(user_input="Now my order.")
        assert isinstance(session.current_agent, Orders)

    assert call_state.plain("note") == "call back"
    assert model.requests[-1].prompt == "You help with orders. Note: call back"


async def test_both_agents_record_into_one_call_span() -> None:
    sink = Memory()
    model = ScriptedLLM(call("to_notes"), "Notes desk.", call("to_orders"), "Orders again.")
    first = Orders(_state(), telemetry=Telemetry(sink, agent="desk", call_id="RM_1"))
    async with AgentSession(llm=model) as session:
        await session.start(first)
        await session.run(user_input="Notes please.")
        await session.run(user_input="Back to orders.")
    assert first._trace is not None and first._trace.closing is not None
    assert await first._trace.closing

    names = [span.name for span in sink.spans.get_finished_spans()]
    assert names.count("conversation") == 1
    assert names.count("llm") == 4
    assert {"to_notes", "to_orders"} <= {
        dict(span.attributes or {}).get("gen_ai.tool.name")
        for span in sink.spans.get_finished_spans()
    }
    handoffs = [
        dict(span.attributes or {})["input.value"]
        for span in sink.spans.get_finished_spans()
        if span.name == "handoff"
    ]
    assert handoffs == ['{"from": "orders", "to": "notes"}', '{"from": "notes", "to": "orders"}']


async def test_native_tools_reach_the_main_agent_and_no_step() -> None:
    model = ScriptedLLM(call("find_order"), "Which order?")
    async with AgentSession(llm=model) as session:
        await session.start(Orders(_state(), native_tools=[EndCallTool()]))
        await session.run(user_input="Where is my order?")

    main, step = model.requests[0], model.requests[1]
    assert "end_call" in main.tools
    assert "end_call" not in step.tools


async def test_a_summary_handoff_asks_its_own_summarizer() -> None:
    summarizer = Summarizer("Notes wanted.")

    class Desk(UnmuteAgent):
        """Orders, handing off with a summary from its own summarizer."""

        @function_tool
        async def to_notes(self) -> UnmuteAgent | str:
            """Hand the caller to the notes desk."""
            assert self.state is not None
            context = HandoffContext(history="summary", summarizer=summarizer)
            return await self.hand_off(Notes(self.state), context=context)

    model = ScriptedLLM(call("to_notes"), "Notes desk.")
    call_state = _state()
    async with AgentSession(llm=model) as session:
        await session.start(Desk(AgentSpec("You help with orders."), state=call_state))
        await session.run(user_input="I need a note taken.")

    [request] = summarizer.requests
    assert request.site == "summary"
    assert request.messages[1]["content"] == "user: I need a note taken."
    assert model.requests[1].texts == [
        ("assistant", "Summary of the conversation so far: Notes wanted.")
    ]


def test_a_task_handoff_needs_the_agents_tool() -> None:
    job = {**JOB, "tasks": [{**JOB["tasks"][0], "handoffs": ["to_nowhere"]}, *JOB["tasks"][1:]]}
    with pytest.raises(ValueError, match="lists the handoff 'to_nowhere'"):
        UnmuteAgent(AgentSpec("Orders.", tasks=plan(job)), state=_state())


def _chat() -> llm.ChatContext:
    """A call with a tool call between two caller turns."""
    return llm.ChatContext(
        [
            llm.ChatMessage(role="system", content=["Prompt."]),
            llm.ChatMessage(role="user", content=["Find A-12."]),
            llm.FunctionCall(call_id="c1", name="lookup_order", arguments="{}"),
            llm.FunctionCallOutput(call_id="c1", name="lookup_order", output="{}", is_error=False),
            llm.ChatMessage(role="assistant", content=["It is open."]),
            llm.ChatMessage(role="user", content=["Thanks."]),
        ]
    )


def _kinds(ctx: llm.ChatContext) -> list[str]:
    return [item.type if item.type != "message" else item.role for item in ctx.items]


def test_each_history_keeps_what_it_says() -> None:
    chat = _chat()
    assert _kinds(shape_chat_ctx(chat, HandoffContext(history="reset"))) == []
    assert _kinds(shape_chat_ctx(chat)) == ["user", "assistant", "user"]
    assert _kinds(
        shape_chat_ctx(chat, HandoffContext(history="summary", summarizer=Summarizer("x")))
    ) == ["user", "assistant", "user"]
    assert _kinds(shape_chat_ctx(chat, HandoffContext(history="full"))) == [
        "user",
        "function_call",
        "function_call_output",
        "assistant",
        "user",
    ]
    assert _kinds(
        shape_chat_ctx(chat, HandoffContext(history="full", include_tool_calls=False))
    ) == [
        "user",
        "assistant",
        "user",
    ]


def test_last_n_keeps_a_call_with_its_result_and_drops_an_orphan() -> None:
    chat = _chat()
    assert _kinds(shape_chat_ctx(chat, HandoffContext(history="last_n", max_messages=4))) == [
        "function_call",
        "function_call_output",
        "assistant",
        "user",
    ]
    assert _kinds(shape_chat_ctx(chat, HandoffContext(history="last_n", max_messages=3))) == [
        "assistant",
        "user",
    ]


async def test_a_step_handoff_tool_that_declines_leaves_the_step_running() -> None:
    job = {**JOB, "tasks": [{**JOB["tasks"][0], "handoffs": ["to_notes"]}, *JOB["tasks"][1:]]}

    class Closed(UnmuteAgent):
        """Orders, whose notes desk is closed."""

        @function_tool
        async def to_notes(self) -> str:
            """Hand the caller to the notes desk."""
            return "The notes desk is closed."

    model = ScriptedLLM(call("find_order"), call("to_notes"), "Sorry, notes is closed.")
    agent = Closed(
        AgentSpec(
            "You help with orders.", tasks=plan(job, tools=registry(lookup_order=Canned(FOUND)))
        ),
        state=_state(),
    )
    async with AgentSession(llm=model) as session:
        await session.start(agent)
        await session.run(user_input="Notes, please.")
        assert agent.tasks is not None and agent.tasks.active == "find_order"

    assert model.requests[2].prompt.startswith("Find order")


def _transferring_job() -> dict[str, Any]:
    """The job, whose order flow takes a note and hands the caller to the notes desk."""
    group = {**JOB["groups"][0], "steps": ["take_note"], "then": "transfer", "then_target": "notes"}
    return {**JOB, "groups": [group]}


def test_a_group_transfer_needs_its_target() -> None:
    with pytest.raises(ValueError, match="transfers the call to 'notes', which then_targets"):
        UnmuteAgent(AgentSpec("Orders.", tasks=plan(_transferring_job())), state=_state())


async def test_a_group_transfer_that_breaks_returns_to_the_main_agent() -> None:
    async def broken(owner: UnmuteAgent) -> object:
        raise RuntimeError("down")

    agent = UnmuteAgent(
        AgentSpec("You help with orders.", tasks=plan(_transferring_job())),
        state=_state(),
        then_targets={"notes": broken},
    )
    model = ScriptedLLM(
        call("order_flow"), "Your note?", call("finish", "Noted.", note="x"), "Notes is down."
    )
    async with AgentSession(llm=model) as session:
        await session.start(agent)
        await session.run(user_input="Take a note.")
        await session.run(user_input="Call me back.")
        assert session.current_agent is agent

    assert model.requests[3].prompt.startswith("You help with orders.")


async def test_a_plain_agent_cannot_take_a_hand_off_and_the_error_names_the_way() -> None:
    with pytest.raises(TypeError, match=r"For a plain Agent.*shape_chat_ctx"):
        await UnmuteAgent(AgentSpec("Orders.")).hand_off(cast(Any, Agent(instructions="Notes.")))
