"""A real text-only LiveKit session runs Unmute tasks, with a scripted model.

The cases run on LiveKit 1.8's own ``AgentSession``, not on a fake session, so a framework
change shows up in them. The rules the core owns are tested in tests/tasks/test_run.py.
"""

import asyncio
import json
from typing import Any

import pytest
from livekit.agents import AgentSession, llm

from tests.livekit.fake import Reply, ScriptedLLM, call
from tests.tasks.job import Canned, job, plan, registry, state
from unmute_core.agent import AgentSpec
from unmute_core.livekit import UnmuteAgent, _step
from unmute_core.state import CallState
from unmute_core.tasks import TaskPlan
from unmute_core.tasks._texts import REFUSED_BUSY, REFUSED_RERUN

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}


def _job(mode: str = "status") -> dict[str, Any]:
    """The test job, with ``take_note`` callable on its own too."""
    payload = job(mode=mode)
    payload["tasks"][1]["when"] = "The caller wants a note taken."
    return payload


def _plan(payload: dict[str, Any] | None = None, **executors: Canned) -> TaskPlan:
    executors.setdefault("lookup_order", Canned(FOUND))
    return plan(payload or _job(), tools=registry(**executors))


async def _talk(
    task_plan: TaskPlan, model: ScriptedLLM, *turns: str, events: list | None = None
) -> tuple[CallState, UnmuteAgent]:
    """Run a session through the caller's turns, and return the state and the agent."""
    call_state = state(task_plan)
    agent = UnmuteAgent(AgentSpec("You help with orders.", tasks=task_plan), state=call_state)
    if events is not None and agent.tasks is not None:
        agent.tasks.listen(lambda name, data: events.append((name, data)))
    async with AgentSession(llm=model) as session:
        await session.start(agent)
        for turn in turns:
            await session.run(user_input=turn)
    return call_state, agent


def _outputs(request: Any, name: str) -> list[Any]:
    """The outputs of every call to ``name`` in a request, as JSON when they are JSON."""
    calls = {
        item.call_id
        for item in request.chat_ctx.items
        if isinstance(item, llm.FunctionCall) and item.name == name
    }
    outputs = [
        item.output
        for item in request.chat_ctx.items
        if isinstance(item, llm.FunctionCallOutput) and item.call_id in calls
    ]
    return [json.loads(output) if output.startswith(("{", "[")) else output for output in outputs]


async def test_a_step_reads_only_its_own_prompt_filled_and_no_running_call() -> None:
    task_plan = _plan()
    model = ScriptedLLM(call("order_flow"), "Which order?")
    call_state = state(task_plan)
    call_state.save({"order_id": "B-7"}, step="find_order")
    agent = UnmuteAgent(AgentSpec("You help with orders.", tasks=task_plan), state=call_state)
    async with AgentSession(llm=model) as session:
        await session.start(agent)
        await session.run(user_input="Where is my order?")

    step = model.requests[1]
    assert [m.role for m in step.chat_ctx.messages()].count("system") == 1
    assert "Find order B-7." in step.prompt
    assert "You help with orders." not in step.prompt
    assert step.calls == []


async def test_a_listening_step_says_its_line_and_waits_for_the_caller() -> None:
    payload = _job()
    payload["tasks"][1]["opening"] = "listen"
    payload["tasks"][1]["announce"] = ["Go ahead with your note."]
    model = ScriptedLLM(call("take_note"), call("finish", note="hi"), "Saved.")
    call_state, _ = await _talk(_plan(payload), model, "Take a note.", "Hi.")

    assert call_state.plain("note") == "hi"
    step = model.requests[1]
    assert step.texts[-2:] == [("assistant", "Go ahead with your note."), ("user", "Hi.")]


async def test_a_nudge_while_a_step_waits_is_heard_by_the_step() -> None:
    """``idle_nudges`` can speak while a step holds the call, with no model request."""
    payload = _job()
    payload["tasks"][1]["opening"] = "listen"
    payload["tasks"][1]["announce"] = ["Go ahead with your note."]
    task_plan = _plan(payload)
    call_state = state(task_plan)
    model = ScriptedLLM(call("take_note"), call("finish", note="hi"), "Ok.")
    agent = UnmuteAgent(AgentSpec("You help with orders.", tasks=task_plan), state=call_state)
    async with AgentSession(llm=model) as session:
        await session.start(agent)
        await session.run(user_input="Take a note.")
        await session.say("Are you still there?")
        await session.run(user_input="Hi.")

    assert call_state.plain("note") == "hi"
    assert ("assistant", "Are you still there?") in model.requests[1].texts


async def test_a_call_closed_during_a_step_ends_with_no_reply() -> None:
    """A hang-up mid-flow must not let the main agent answer the cancelled step."""
    payload = _job()
    payload["tasks"][1]["opening"] = "listen"
    payload["tasks"][1]["announce"] = ["Go ahead with your note."]
    task_plan = _plan(payload)
    model = ScriptedLLM(call("take_note"), "Sorry, that did not work.")
    agent = UnmuteAgent(AgentSpec("You help with orders.", tasks=task_plan), state=state(task_plan))
    closed = asyncio.Event()
    async with AgentSession(llm=model) as session:
        session.on("close", lambda _event: closed.set())
        await session.start(agent)
        await session.run(user_input="Take a note.")
        session.shutdown()
        async with asyncio.timeout(3):
            await closed.wait()

    assert len(model.requests) == 1


async def test_a_delegate_that_just_returned_is_refused_until_the_caller_speaks() -> None:
    model = ScriptedLLM(call("find_order"), call("lookup_order"), call("find_order"), "It is open.")
    await _talk(_plan(), model, "Where is order A-12?")

    assert _outputs(model.requests[-1], "find_order") == [{"status": "completed"}, REFUSED_RERUN]
    assert len(model.requests) == 4


async def test_a_delegate_beside_a_running_one_starts_nothing_in_record_mode() -> None:
    """LiveKit 1.8 queues the second task. The core refuses it before it starts."""
    both = Reply("", [("find_order", {}), ("take_note", {})])
    model = ScriptedLLM(both, call("lookup_order"), "Anything else?")
    await _talk(_plan(_job("record")), model, "Find order A-12 and take a note.")

    owner = model.requests[-1]
    assert _outputs(owner, "take_note") == [REFUSED_BUSY]
    assert _outputs(owner, "find_order")[0]["status"] == "completed"


async def test_a_hand_back_carries_the_turn_it_answers_to_the_owner() -> None:
    model = ScriptedLLM(
        call("take_note"),
        "What is the note?",
        call("finish", unserved_request="Book a table"),
        "Sure, let me book that.",
    )
    await _talk(_plan(), model, "Take a note.", "Actually, book me a table.")

    owner = model.requests[-1]
    assert ("user", "Actually, book me a table.") in owner.texts
    assert _outputs(owner, "take_note")[0]["status"] == "unserved"


async def test_a_receipt_line_is_the_last_word_and_no_reply_follows() -> None:
    payload = job(mode="status")
    payload["tasks"][0]["finish"][0]["say"] = ["Order {{result.order.id}} is open."]
    model = ScriptedLLM(call("find_order"), call("lookup_order"), "never asked")
    _, agent = await _talk(_plan(payload), model, "Where is order A-12?")

    assert len(model.requests) == 2
    assert agent.chat_ctx.messages()[-1].text_content == "Order A-12 is open."


async def test_an_empty_step_reply_is_retried_with_a_note() -> None:
    model = ScriptedLLM(call("take_note"), "", "What is the note?")
    await _talk(_plan(), model, "Take a note.")

    first, retry = model.requests[1], model.requests[2]
    assert "The previous response was empty." not in first.prompt
    assert "The previous response was empty." in retry.prompt


async def test_after_a_tool_result_an_empty_retry_may_only_finish() -> None:
    model = ScriptedLLM(
        call("find_order"), call("lookup_order"), "", call("finish", order_id="A-1"), "Ok."
    )
    await _talk(_plan(lookup_order=Canned({"found": False})), model, "Find order A-1.")

    assert model.requests[3].tools == ["finish"]


async def test_a_silent_finish_after_a_tool_result_is_refused_once() -> None:
    payload = job()
    model = ScriptedLLM(
        call("find_order"),
        "Which order?",
        call("lookup_order"),
        call("finish", order_id="A-1"),
        Reply("I could not find it.", [("finish", {"order_id": "A-1"})]),
        "Anything else?",
    )
    _, _ = await _talk(
        _plan(payload, lookup_order=Canned({"found": False})), model, "Find it.", "A-1."
    )

    refusal = _outputs(model.requests[4], "finish")[0]
    assert "has not heard what lookup_order returned" in refusal["refused"]
    owner = _outputs(model.requests[-1], "find_order")[0]
    assert owner["status"] == "completed"
    assert owner["steps"][0]["said"] == "I could not find it."


async def test_a_via_tools_success_moves_the_caller_into_its_task() -> None:
    payload = job(mode="status")
    payload["tasks"][1]["tools"] = ["search_web"]
    payload["tasks"][1]["via"] = ["search_web"]
    model = ScriptedLLM(call("search_web"), call("finish", note="found it"), "Noted.")
    call_state, _ = await _talk(_plan(payload), model, "Search and note it.")

    assert call_state.plain("note") == "found it"
    step = model.requests[1]
    assert step.calls == ["search_web"]


class _Played:
    """A line's speech that has already ended, cut off or played to its end."""

    def __init__(self, cut_off: bool, played: str) -> None:
        self.interrupted = cut_off
        self.chat_items = [llm.ChatMessage(role="assistant", content=[played])] if played else []

    def add_done_callback(self, callback: Any) -> None:
        callback(self)


@pytest.mark.parametrize(
    ("cut_off", "played", "asked", "looked_up"),
    [
        (True, "Shall I close", 2, False),
        (True, "Shall I close order A-12", 1, True),
        (False, "Shall I close order A-12?", 1, True),
    ],
)
async def test_a_confirm_line_cut_off_before_its_value_is_asked_again(
    monkeypatch: pytest.MonkeyPatch, cut_off: bool, played: str, asked: int, looked_up: bool
) -> None:
    payload = job(mode="status")
    payload["tasks"][0]["finish"][0]["confirm"] = ["Shall I close order {{input.order_id}}?"]
    lookup = Canned(FOUND)
    task_plan = _plan(payload, lookup_order=lookup)
    spoken = _step.say
    said: list[str] = []

    def say(session: AgentSession, text: str, **options: Any) -> Any:
        said.append(text)
        speech = spoken(session, text, **options)
        return _Played(cut_off, played) if text.startswith("Shall I close") else speech

    monkeypatch.setattr(_step, "say", say)
    model = ScriptedLLM(
        call("find_order"),
        call("lookup_order", order_id="A-12"),
        call("lookup_order", order_id="A-12"),
        "Okay.",
    )
    await _talk(task_plan, model, "Please close order A-12.", "Yes.")

    assert said.count("Shall I close order A-12?") == asked
    assert bool(lookup.calls) is looked_up
