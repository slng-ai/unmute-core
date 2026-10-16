"""One call's runs: the finish rules of a step, and the order a delegate runs its steps in.

The finish rules cover both modes: ``status`` follows Unmute's finish and closing tools
(unmute@4a738ad templates/livekit_v1/agents.py.tmpl:1039-1199), and ``record`` hands back a
record of each step and checks a finish against what the caller said. Each rule runs once per
mode where the modes differ.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel

from tests.outputs import AnyResult
from tests.tasks.job import Canned, State, job, plan, registry, state
from unmute_core.state import CallInfo, CallState
from unmute_core.tasks import (
    AfterReply,
    CallTasks,
    DelegateRun,
    FinishCheck,
    StepRun,
    TaskPlan,
    TasksSpec,
    decided_finish,
)
from unmute_core.tasks import _finish as finish_module
from unmute_core.tasks._texts import (
    BEGIN_STEP,
    GO_BACK_TOO_EARLY,
    REENTRY_REDO,
    REFUSED_BUSY,
    REFUSED_HANDOFF_BESIDE,
    REFUSED_HANDOFF_ENDED,
    REFUSED_RERUN,
    SAVE_BEFORE_CALLER,
    STEP_BOUNDARY,
    UNSERVED_ON_REENTRY,
    WAIT_FOR_CALLER,
)
from unmute_core.tools import Assign, ToolError, ToolExecutor, ToolRegistry, ToolSpec

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}


def _tasks(
    payload: dict[str, Any] | None = None,
    *,
    mode: str = "record",
    confirm: dict[str, str] | None = None,
    events: list[tuple[str, dict[str, Any]]] | None = None,
    **executors: ToolExecutor,
) -> CallTasks:
    payload = payload or job()
    payload["mode"] = mode
    task_plan = plan(payload, confirm=confirm, tools=registry(**executors))
    tasks = CallTasks(task_plan, state(task_plan))
    if events is not None:
        tasks.listen(lambda name, data: events.append((name, data)))
    return tasks


def _start(tasks: CallTasks, name: str, turns: int = 1) -> DelegateRun:
    run = tasks.start(name, caller_turns=turns)
    assert isinstance(run, DelegateRun)
    return run


def _step(run: DelegateRun, turns: int = 1) -> StepRun:
    step = run.next_step()
    assert step is not None
    step.opened(caller_turns=turns)
    return step


@pytest.mark.parametrize("mode", ["status", "record"])
async def test_a_hand_back_keeps_confirmed_values_only_in_record_mode(mode: str) -> None:
    tasks = _tasks(mode=mode)
    run = _start(tasks, "take_note")
    step = _step(run)
    assert (
        await step.finish({"note": "call back", "unserved_request": "A refund"}, caller_turns=2)
        is None
    )
    run.ended(step)
    result = run.result(caller_turns=2)
    if mode == "status":
        assert result == {"status": "unserved"}
        assert tasks.state.plain("note") is None
    else:
        assert result["status"] == "unserved"
        assert result["steps"] == [
            {
                "step": "take_note",
                "outcome": "handed back",
                "saved": {"note": "call back"},
                "not_done": "A refund",
            }
        ]
    assert step.carries_turn


@pytest.mark.parametrize("mode", ["status", "record"])
async def test_a_save_before_the_caller_spoke_is_refused_only_in_record_mode(mode: str) -> None:
    tasks = _tasks(mode=mode)
    step = _step(_start(tasks, "take_note"), turns=2)
    refusal = await step.finish({"note": "made up"}, caller_turns=2)
    if mode == "status":
        assert refusal is None and tasks.state.plain("note") == "made up"
        return
    assert refusal == SAVE_BEFORE_CALLER and not step.done
    assert step.answer_refusal
    assert await step.finish({"note": "said"}, caller_turns=3) is None
    assert tasks.state.plain("note") == "said"


async def test_a_refused_save_from_a_reply_that_spoke_gets_no_second_reply() -> None:
    step = _step(_start(_tasks(), "take_note"), turns=2)
    refusal = await step.finish({"note": "read back"}, caller_turns=2, spoke=True)
    assert refusal == SAVE_BEFORE_CALLER and not step.answer_refusal
    refusal = await step.finish({"note": 7}, caller_turns=3, spoke=True)
    assert refusal is not None and step.answer_refusal


async def test_a_refused_save_held_is_saved_unconfirmed_when_the_caller_hangs_up() -> None:
    payload = job()
    payload["tasks"][1]["when"] = "The caller wants a note taken."
    tasks = _tasks(payload, confirm={"note": "take_note"})
    step = _step(_start(tasks, "take_note"), turns=2)
    assert await step.finish({"note": "read back"}, caller_turns=2, spoke=True) is not None
    assert step.hold_save(caller_spoke_since=lambda: False)
    assert not step.hold_save(caller_spoke_since=lambda: False)
    assert tasks.state.plain("note") is None
    assert await tasks.state.settle_held() == ["take_note"]
    assert tasks.state.plain("note") == "read back"
    assert not tasks.state.is_confirmed("note")


async def test_only_a_save_refused_before_the_caller_spoke_is_held() -> None:
    step = _step(_start(_tasks(), "take_note"), turns=2)
    assert await step.finish({"note": 7}, caller_turns=3) is not None
    assert not step.hold_save(caller_spoke_since=lambda: False)


async def test_a_silent_finish_after_a_result_is_refused_once_then_recorded() -> None:
    tasks = _tasks()
    step = _step(_start(tasks, "take_note"))
    unheard = ("search_web", '{"late": true}')
    refusal = await step.finish({"note": "x"}, caller_turns=2, unheard=unheard)
    assert refusal == (
        "Not recorded: the caller has not heard what search_web returned. Tell them what it "
        "means for them, and call finish in that same reply."
    )
    assert await step.finish({"note": "x"}, caller_turns=2, unheard=step.refused_unheard) is None
    assert step.not_heard == 'search_web returned: {"late": true}'
    status = _step(_start(_tasks(mode="status"), "take_note"))
    assert await status.finish({"note": "x"}, caller_turns=2, unheard=unheard) is None


async def test_a_refused_finish_reads_each_modes_wording() -> None:
    record = _step(_start(_tasks(), "find_order"))
    first = await record.finish({"order_id": "A-12", "status": "lost"}, caller_turns=2)
    again = await record.finish({"order_id": "A-12", "status": "lost"}, caller_turns=2)
    assert first is not None and "Convert what the caller already told you" in first
    assert again is not None and "These are the values that were just refused" in again
    status = _step(_start(_tasks(mode="status"), "find_order"))
    refusal = await status.finish({"status": "lost"}, caller_turns=2)
    assert refusal == (
        "Not recorded: status: Input should be 'open' or 'closed'. Ask again, then call finish "
        "with a value that fits."
    )


async def test_a_closing_tool_that_succeeds_ends_the_step_and_saves_its_result() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    lookup = Canned({"found": False}, FOUND)
    tasks = _tasks(events=events, lookup_order=lookup)
    run = _start(tasks, "find_order")
    step = _step(run)
    assert await step.call_tool("lookup_order", {"q": "A"}) == {"found": False}
    assert not step.done
    assert await step.call_tool("lookup_order", {"q": "A"}) is None
    assert step.outcome is not None and step.outcome.saved == ("order_id", "status")
    assert step.carries_turn
    assert await step.call_tool("lookup_order", {}) == {
        "refused": "lookup_order already succeeded in this step. Call finish to record it; a new "
        "action is a new request from the caller."
    }
    run.ended(step)
    assert (
        "task.ended_on_tool",
        {"task": "find_order", "task_run_id": run.run_id, "saved": ["order_id", "status"]},
    ) in events


async def test_a_refused_closing_save_of_a_hidden_result_shows_only_the_refusal() -> None:
    tasks = _tasks(lookup_order=Canned({"found": True, "order": {"id": "A-12"}, "status": "lost"}))
    object.__setattr__(tasks.plan.tools.get("lookup_order"), "show_result", False)
    step = _step(_start(tasks, "find_order"))
    answer = await step.call_tool("lookup_order", {})
    assert isinstance(answer, dict)
    assert set(answer) == {"refused"}
    assert step.result == {"found": True, "order": {"id": "A-12"}, "status": "lost"}


async def test_a_refused_closing_save_is_repaired_with_the_tools_own_values() -> None:
    tasks = _tasks(lookup_order=Canned({"found": True, "order": {"id": "A-12"}, "status": "lost"}))
    step = _step(_start(tasks, "find_order"))
    answer = await step.call_tool("lookup_order", {})
    assert isinstance(answer, dict) and answer["refused"].startswith(
        "Not recorded: status: Input should be 'open' or 'closed'. Call finish with the values "
        "this tool returned"
    )
    assert await step.finish({"order_id": "made-up", "status": "open"}, caller_turns=2) is None
    assert tasks.state.plain("order_id") == "A-12"


class Gate:
    """A tool executor that waits until it is let through, then answers."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.open = asyncio.Event()

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        await self.open.wait()
        return self.result


@pytest.mark.parametrize("mode", ["status", "record"])
async def test_a_finish_waits_for_a_closing_tool_that_still_runs(mode: str) -> None:
    gate = Gate(FOUND)
    tasks = _tasks(mode=mode, lookup_order=gate)
    step = _step(_start(tasks, "find_order"))
    action = asyncio.create_task(step.call_tool("lookup_order", {}))
    await asyncio.sleep(0)
    finish = asyncio.create_task(step.finish({"order_id": "made-up"}, caller_turns=2))
    await asyncio.sleep(0)
    assert not finish.done()
    gate.open.set()
    assert await action is None
    assert await finish is None
    assert tasks.state.plain("order_id") == "A-12"


@pytest.mark.parametrize("mode", ["status", "record"])
async def test_a_finish_sent_before_a_failed_closing_tool_is_refused_in_record_mode(
    mode: str,
) -> None:
    gate = Gate({"found": False})
    tasks = _tasks(mode=mode, lookup_order=gate)
    step = _step(_start(tasks, "find_order"))
    action = asyncio.create_task(step.call_tool("lookup_order", {}))
    await asyncio.sleep(0)
    finish = asyncio.create_task(step.finish({"order_id": "A-1"}, caller_turns=2))
    await asyncio.sleep(0)
    gate.open.set()
    assert await action == {"found": False}
    if mode == "status":
        assert await finish is None and tasks.state.plain("order_id") == "A-1"
    else:
        assert await finish == (
            "Not recorded: lookup_order did not succeed, and this finish was sent before its "
            "result. Read that result, then continue with the caller."
        )


@pytest.mark.parametrize("mode", ["status", "record"])
async def test_a_closing_tool_that_never_returns(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finish_module, "TERMINAL_WAIT_SECONDS", 0.01)
    gate = Gate(FOUND)
    tasks = _tasks(mode=mode, lookup_order=gate)
    step = _step(_start(tasks, "find_order"))
    action = asyncio.create_task(step.call_tool("lookup_order", {}))
    await asyncio.sleep(0)
    second = asyncio.create_task(step.call_tool("lookup_order", {}))
    refusal = await step.finish({"order_id": "A-1"}, caller_turns=2)
    if mode == "record":
        assert await second == {
            "refused": "lookup_order is still running in this step, so lookup_order did not run. "
            "Wait for its result; a new action is a new request from the caller."
        }
        assert refusal is not None and refusal.startswith(
            "Not recorded: lookup_order is still running."
        )
        assert await step.finish({"unserved_request": "Too slow"}, caller_turns=2) is None
    else:
        assert refusal is None and tasks.state.plain("order_id") == "A-1"
    gate.open.set()
    await asyncio.gather(action, second)
    # The late success saves nothing: the step had ended.
    assert step.outcome is not None and step.outcome.saved != ("order_id", "status")


async def test_a_group_goes_back_and_the_step_run_again_is_told_why() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    tasks = _tasks(events=events)
    run = _start(tasks, "order_flow")
    first = _step(run)
    assert first.go_back_tool() is None and STEP_BOUNDARY not in first.instructions()
    assert await first.finish({"order_id": "A-12", "status": "open"}, caller_turns=2) is None
    run.ended(first)
    second = _step(run, turns=2)
    assert STEP_BOUNDARY in second.instructions()
    second.answered()
    assert STEP_BOUNDARY not in second.instructions()
    assert "call `go_back_to_step`" in second.instructions()
    assert second.go_back({"steps": ["find_order"]}, caller_turns=2) == GO_BACK_TOO_EARLY
    assert second.go_back({"steps": ["find_order"]}, caller_turns=3) is None
    run.ended(second)
    again = _step(run, turns=3)
    assert again.task.name == "find_order" and again.history == "messages"
    assert again.instructions().endswith(REENTRY_REDO)
    assert (
        await again.finish({"unserved_request": "change it"}, caller_turns=3) == UNSERVED_ON_REENTRY
    )
    assert await again.finish({"order_id": "A-13", "status": "open"}, caller_turns=4) is None
    run.ended(again)
    resumed = _step(run, turns=4)
    assert resumed.task.name == "take_note"
    assert await resumed.finish({"note": "x"}, caller_turns=5) is None
    run.ended(resumed)
    assert run.next_step() is None
    result = run.result(caller_turns=5)
    assert [row["step"] for row in result["steps"]] == ["find_order", "find_order", "take_note"]
    assert [name for name, _ in events] == [
        "group.started",
        "group.step_started",
        "task.started",
        "task.completed",
        "group.step_started",
        "task.started",
        "group.went_back",
        "group.step_started",
        "task.started",
        "task.completed",
        "group.step_started",
        "task.started",
        "task.completed",
        "group.completed",
    ]


async def test_a_status_mode_group_offers_no_go_back_and_opens_with_unmutes_line() -> None:
    run = _start(_tasks(mode="status"), "order_flow")
    first = _step(run)
    await first.finish({"order_id": "A-12"}, caller_turns=1)
    run.ended(first)
    second = _step(run)
    assert second.go_back_tool() is None
    assert second.go_back({"steps": ["find_order"]}, caller_turns=5) is not None
    assert second.opening_message == BEGIN_STEP and STEP_BOUNDARY not in second.instructions()
    assert "go_back_to_step" not in second.instructions()


def _refund_job() -> dict[str, Any]:
    payload = job()
    payload["tasks"].append(
        {
            "name": "refund",
            "when": "The caller wants a refund.",
            "instructions": "Refund.",
            "requires": ["order_id"],
        }
    )
    return payload


async def test_a_task_runs_its_prerequisite_first() -> None:
    tasks = _tasks(_refund_job())
    run = _start(tasks, "refund")
    assert run.announce == "Let me look."
    first = _step(run)
    assert first.task.name == "find_order" and first.announce is None
    assert "the refund step comes next" in first.instructions()
    await first.finish({"order_id": "A-12", "status": "open"}, caller_turns=2)
    run.ended(first)
    then = _step(run, turns=2)
    assert then.task.name == "refund" and STEP_BOUNDARY in then.instructions()
    await then.finish({}, caller_turns=3)
    run.ended(then)
    assert run.next_step() is None and run.status == "completed"
    steps = run.result(caller_turns=3)["steps"]
    assert [(row["step"], row["outcome"]) for row in steps] == [
        ("find_order", "completed"),
        ("refund", "completed"),
    ]
    met = _start(tasks, "refund", turns=4)
    assert _step(met).task.name == "refund"


async def test_a_prerequisite_that_hands_back_stops_the_task() -> None:
    run = _start(_tasks(_refund_job()), "refund")
    first = _step(run)
    await first.finish({"unserved_request": "Something else"}, caller_turns=2)
    run.ended(first)
    assert run.next_step() is None and run.status == "unserved"


async def test_a_group_step_without_its_required_value_fails_without_starting() -> None:
    payload = job()
    payload["tasks"][1]["requires"] = ["order_id"]
    payload["groups"][0]["non_blocking"] = ["find_order"]
    events: list[tuple[str, dict[str, Any]]] = []
    run = _start(_tasks(payload, events=events), "order_flow")
    first = _step(run)
    first.fail()
    run.ended(first)
    assert run.next_step() is None and run.status == "failed"
    assert [name for name, _ in events][-4:] == [
        "group.step_started",
        "task.requires",
        "task.failed",
        "group.stopped",
    ]


async def test_a_confirmed_step_is_skipped_and_entering_it_withdraws_its_confirmation() -> None:
    payload = job()
    payload["groups"][0]["skip_when_confirmed"] = {"find_order": "order_id"}
    tasks = _tasks(payload, confirm={"order_id": "find_order"})
    tasks.state.save({"order_id": "A-12"}, step="find_order")
    run = _start(tasks, "order_flow")
    assert _step(run).task.name == "take_note"
    assert tasks.state.is_confirmed("order_id")
    run.result(caller_turns=1)
    alone = _start(tasks, "find_order", turns=2)
    _step(alone)
    assert not tasks.state.is_confirmed("order_id")


async def test_a_via_tool_enters_its_task_only_on_success() -> None:
    payload = job()
    payload["tasks"][1] |= {"tools": ["search_web"], "via": ["search_web"]}
    tasks = _tasks(payload)
    assert "When this succeeds, the caller moves into a step" in tasks.via_description(
        "search_web", "Search."
    )
    assert tasks.enter_via("search_web", {"error": "timeout"}, caller_turns=1) is None
    assert tasks.enter_via("lookup_order", {"ok": True}, caller_turns=1) is None
    run = tasks.enter_via("search_web", {"hits": 3}, caller_turns=1)
    assert run is not None and run.announce is None
    step = _step(run)
    assert step.task.name == "take_note" and step.announce is None and not step.listens
    assert tasks.enter_via("search_web", {"hits": 3}, caller_turns=1) is None


@pytest.mark.parametrize("mode", ["status", "record"])
async def test_one_run_at_a_time_and_no_rerun_before_the_caller_speaks(mode: str) -> None:
    tasks = _tasks(mode=mode)
    run = _start(tasks, "take_note")
    beside = tasks.start("find_order", caller_turns=1)
    assert (beside == REFUSED_BUSY) == (mode == "record")
    step = _step(run)
    await step.finish({"note": "x"}, caller_turns=2)
    run.ended(step)
    run.result(caller_turns=2)
    assert tasks.start("take_note", caller_turns=2) == REFUSED_RERUN
    assert isinstance(tasks.start("take_note", caller_turns=3), DelegateRun)


async def test_a_busy_refusal_is_reported() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    tasks = _tasks(events=events)
    _start(tasks, "take_note")
    assert tasks.start("find_order", caller_turns=1) == REFUSED_BUSY
    name, data = events[-1]
    assert (name, data["task"]) == ("task.refused_busy", "find_order")


async def test_a_receipt_line_closes_a_run_that_completed_on_it() -> None:
    payload = job()
    payload["tasks"][0]["finish"][0]["say"] = ["Found {{result.order.id}}; it is {{status}}."]
    tasks = _tasks(payload, lookup_order=Canned(FOUND))
    run = _start(tasks, "find_order")
    step = _step(run)
    await step.call_tool("lookup_order", {})
    run.ended(step)
    assert run.closing == "Found A-12; it is open."
    run.result(caller_turns=2)
    assert run.closing == "Found A-12; it is open."


def test_delegate_tools_and_openings_follow_the_mode() -> None:
    payload = job()
    payload["tasks"][1] |= {
        "when": "The caller leaves a note.",
        "opening": "listen",
        "announce": ["Go ahead."],
    }
    tasks = _tasks(payload, mode="status")
    tools = tasks.delegate_tools()
    assert [tool["name"] for tool in tools] == ["find_order", "take_note", "order_flow"]
    assert tools[0]["description"].startswith(
        "The caller asks about an order. The caller's words right above the status"
    )
    assert tools[1]["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    run = _start(tasks, "take_note")
    assert run.announce is None
    step = _step(run)
    assert step.listens and step.announce == "Go ahead." and step.opening_message is None
    assert step.site == "task:take_note"
    quiet = payload["tasks"][1] | {"announce": []}
    payload["tasks"][1] = quiet
    silent = _step(_start(_tasks(payload, mode="status"), "take_note"))
    assert silent.opening_message == WAIT_FOR_CALLER
    assert step.finish_tool()["description"] == (
        "Save this step's values and finish, or return unserved without saving."
    )


async def test_a_step_that_stays_empty_says_the_lines_the_call_gives_it() -> None:
    tasks = _tasks()
    tasks.gave_up_line = "Lo siento, no pude."
    tasks.gave_up_after_tool_line = "Lo siento, revisa el estado."
    task = _step(_start(tasks, "find_order"))
    assert task.gave_up(finish_only=False) == "Lo siento, no pude."
    assert task.gave_up(finish_only=True) == "Lo siento, revisa el estado."


async def test_a_step_that_stays_empty_apologises_unless_its_group_can_do_without_it() -> None:
    """A step that ends with nothing saved says sorry, unless it is non-blocking in its group."""
    task = _step(_start(_tasks(), "find_order"))
    assert task.gave_up(finish_only=False) == "Sorry, I couldn't complete that. Please try again."
    assert "after the last tool result" in (task.gave_up(finish_only=True) or "")
    assert not task.done

    payload = job()
    payload["groups"][0]["non_blocking"] = ["find_order"]
    run = _start(_tasks(payload), "order_flow")
    first = _step(run)
    assert first.gave_up(finish_only=False) is None
    assert first.outcome is not None and first.outcome.status == "failed"
    run.ended(first)
    assert _step(run).task.name == "take_note"


def test_two_sites_that_share_a_line_never_say_it_twice_in_a_row() -> None:
    tasks = _tasks()
    lines = ("One sec.", "Let me look.")
    picked = [tasks.pick_line(site, lines) for site in ("task:a", "task:b") * 10]
    assert all(first != second for first, second in pairwise(picked))


def _opening_job() -> dict[str, Any]:
    """The job with ``order_flow`` taking the order and ``find_order`` opening on a search."""
    payload = job(mode="status")
    payload["groups"][0]["arguments"] = ["order_id", "status"]
    find_order = payload["tasks"][0]
    find_order["tools"] = ["lookup_order", "search_web"]
    find_order["opens_with"] = {
        "tool": "search_web",
        "arguments": {"q": "order_id"},
        "say": ["I found {{result.title}}. Is that the one?"],
    }
    return payload


def _opening_tasks(**executors: ToolExecutor) -> CallTasks:
    task_plan = plan(_opening_job(), tools=registry(**executors))
    return CallTasks(task_plan, state(task_plan))


def test_a_delegate_takes_its_arguments_typed_from_the_state() -> None:
    [tool] = [row for row in _opening_tasks().delegate_tools() if row["name"] == "order_flow"]
    properties = tool["parameters"]["properties"]
    assert set(properties) == {"order_id", "status"}
    assert tool["parameters"]["required"] == []
    assert "closed" in str(properties["status"])


def test_starting_a_delegate_saves_its_arguments_and_clears_the_ones_left_out() -> None:
    tasks = _opening_tasks()
    run = tasks.start(
        "order_flow", caller_turns=1, arguments={"order_id": "A-12", "status": "open"}
    )
    assert isinstance(run, DelegateRun)
    assert (tasks.state.plain("order_id"), tasks.state.plain("status")) == ("A-12", "open")
    run.status = "completed"
    run.result(caller_turns=1)

    tasks.start("order_flow", caller_turns=2, arguments={"order_id": "B-7", "status": ""})
    assert (tasks.state.plain("order_id"), tasks.state.plain("status")) == ("B-7", None)


def test_an_argument_that_does_not_fit_refuses_the_start() -> None:
    tasks = _opening_tasks()
    refused = tasks.start("order_flow", caller_turns=1, arguments={"status": "lost"})
    assert isinstance(refused, dict)
    assert refused["refused"].startswith("order_flow did not start")
    assert tasks.active is None


async def test_an_opening_line_reads_the_whole_result_of_a_tool_that_hides_it() -> None:
    task_plan = plan(_opening_job(), tools=registry(search_web=Canned({"title": "order A-12"})))
    hidden = task_plan.tools.get("search_web")
    object.__setattr__(hidden, "show_result", False)
    tasks = CallTasks(task_plan, state(task_plan))
    step = _first_step(tasks, {"order_id": "A-12"})

    opening = await step.open()

    assert opening is not None
    assert opening.line == "I found order A-12. Is that the one?"


async def test_an_opening_tool_needs_its_values() -> None:
    step = _first_step(_opening_tasks(search_web=Canned({"title": "order A-12"})), {})

    assert await step.open() is None
    assert not step.listens and step.announce == "Let me look."


async def test_an_opening_leaves_out_an_unset_value_its_tool_does_not_require() -> None:
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}, "near": {"type": "string"}},
        "required": ["near"],
    }
    search = Canned({"title": "every order"})
    tools = registry()
    payload = _opening_job()
    payload["tasks"][0]["opens_with"]["arguments"] = {"q": "order_id", "near": "status"}
    search_spec = ToolSpec("search_web", "Search.", schema, AnyResult, search)
    task_plan = plan(
        payload,
        tools=ToolRegistry([tools.get("lookup_order"), search_spec, tools.get("normalize")]),
    )
    step = _first_step(CallTasks(task_plan, state(task_plan)), {"status": "open"})

    opening = await step.open()

    assert opening is not None and opening.line == "I found every order. Is that the one?"
    assert search.calls == [{"near": "open"}]
    no_status = _first_step(CallTasks(task_plan, state(task_plan)), {"order_id": "A-12"})
    assert await no_status.open() is None


def _first_step(tasks: CallTasks, arguments: dict[str, str]) -> StepRun:
    run = tasks.start("order_flow", caller_turns=1, arguments=arguments)
    assert isinstance(run, DelegateRun)
    step = run.next_step()
    assert step is not None
    return step


@pytest.mark.parametrize(
    ("result", "note"),
    [
        pytest.param({"title": "Order A-12 is open"}, "Order A-12 is open", id="saved-at-once"),
        pytest.param({"error": "no such order"}, None, id="a-failure-saves-nothing"),
    ],
)
async def test_a_step_tool_saves_its_assign_rows_as_soon_as_it_returns(
    result: dict[str, str], note: str | None
) -> None:
    tools = registry(search_web=Canned(result))
    tools = ToolRegistry(
        [tools.get(name) for name in ("lookup_order", "search_web", "normalize")],
        state=State,
        assign={"search_web": [Assign(variable="note", path=("title",))]},
    )
    task_plan = plan(_opening_job(), tools=tools)
    tasks = CallTasks(task_plan, state(task_plan))
    step = _first_step(tasks, {})

    await step.call_tool("search_web", {"q": "A-12"})

    assert not step.done
    assert tasks.state.plain("note") == note


async def test_a_tool_reply_is_the_line_code_says_for_a_matching_result() -> None:
    payload = _opening_job()
    payload["tasks"][0]["replies"] = [
        {
            "tool": "search_web",
            "when": {"state": ["open"]},
            "say": ["Order {{result.order}} is open. Shall I close it?"],
        }
    ]
    search = Canned({"order": "A-12", "state": "open"}, {"order": "A-12", "state": "closed"})
    task_plan = plan(payload, tools=registry(search_web=search))
    step = _first_step(CallTasks(task_plan, state(task_plan)), {})

    assert await step.call_tool("search_web", {"q": "A-12"}) == {"order": "A-12", "state": "open"}
    assert step.reply == "Order A-12 is open. Shall I close it?"
    await step.call_tool("search_web", {"q": "A-12"})
    assert step.reply is None


class Checked(Canned):
    """A lookup that validates its input as a Python tool does: upper case, ``A-`` orders only."""

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        order = str(arguments.get("order_id", "")).upper()
        if not order.startswith("A-"):
            raise ToolError("order_id: there is no such order")
        return {"order_id": order}


def _confirm_step(
    lookup: ToolExecutor,
    *,
    line: str = "Shall I close order {{input.order_id}}?",
    schema: dict[str, Any] | None = None,
    language: str = "en",
    held: dict[str, str] | None = None,
    timezone: str | None = None,
) -> StepRun:
    payload = _opening_job()
    payload["tasks"][0]["finish"][0]["confirm"] = [line]
    tools = registry(lookup_order=lookup)
    if schema is not None:
        lookup_spec = ToolSpec("lookup_order", "Close an order.", schema, AnyResult, lookup)
        tools = ToolRegistry([lookup_spec, tools.get("search_web"), tools.get("normalize")])
    task_plan = plan(payload, tools=tools)
    tasks = CallTasks(task_plan, state(task_plan), language=language, timezone=timezone)
    return _first_step(tasks, held or {})


async def test_a_confirm_line_cut_off_or_a_changed_input_is_asked_again() -> None:
    lookup = Checked({"found": False})
    step = _confirm_step(lookup)
    await step.call_tool("lookup_order", {"order_id": "A-12"})
    assert step.reply is not None

    step.reply_played(step.reply, heard=False)
    again = await step.call_tool("lookup_order", {"order_id": "A-12"})
    assert isinstance(again, dict) and step.reply == again["asked"] == "Shall I close order A-12?"
    step.reply_played(step.reply, heard=True)
    await step.call_tool("lookup_order", {"order_id": "A-13"})
    assert step.reply == "Shall I close order A-13?"
    assert lookup.calls == []


@pytest.mark.parametrize(
    ("line", "played", "runs"),
    [
        pytest.param(
            "Shall I close order {{input.order_id}} now, today?",
            "Shall I close order A-",
            False,
            id="cut-inside-the-value",
        ),
        pytest.param(
            "Shall I close order {{input.order_id}} now, today?",
            "Shall I close order A-12 now,",
            True,
            id="cut-after-the-value",
        ),
        pytest.param("Shall I close it?", "Shall I close", False, id="no-value-cut-early"),
    ],
)
async def test_a_confirm_line_cut_off_counts_as_heard_once_its_values_were_played(
    line: str, played: str, runs: bool
) -> None:
    lookup = Checked({"found": False})
    step = _confirm_step(lookup, line=line)
    await step.call_tool("lookup_order", {"order_id": "A-12"})
    assert step.reply is not None

    step.reply_played(step.reply, heard=False, played=played)
    await step.call_tool("lookup_order", {"order_id": "A-12"})

    assert lookup.calls == ([{"order_id": "A-12"}] if runs else [])


async def test_a_yes_runs_the_tool_only_while_its_locked_values_are_the_ones_asked() -> None:
    lookup = Canned({"found": False})
    payload = _opening_job()
    payload["tasks"][0]["finish"][0]["confirm"] = ["Shall I close order {{input.order_id}}?"]
    schema = {
        "type": "object",
        "properties": {"order_id": {"type": "string"}, "note": {"type": "string"}},
    }
    lookup_spec = ToolSpec(
        "lookup_order", "Close an order.", schema, AnyResult, lookup, locked={"note": "{{note}}"}
    )
    tools = registry()
    tools = ToolRegistry([lookup_spec, tools.get("search_web"), tools.get("normalize")])
    task_plan = plan(payload, tools=tools)
    tasks = CallTasks(task_plan, state(task_plan))
    tasks.state.save({"note": "fragile"})
    step = _first_step(tasks, {})
    await step.call_tool("lookup_order", {"order_id": "A-12"})
    assert step.reply is not None
    step.reply_played(step.reply, heard=True)

    tasks.state.save({"note": "urgent"})
    again = await step.call_tool("lookup_order", {"order_id": "A-12"})

    assert isinstance(again, dict) and again["asked"] == "Shall I close order A-12?"
    assert lookup.calls == []
    step.reply_played(step.reply, heard=True)
    await step.call_tool("lookup_order", {"order_id": "A-12"})
    assert lookup.calls == [{"order_id": "A-12", "note": "urgent"}]


class _Due(BaseModel):
    due: str


class _Shaped(State):
    order: _Due | None = None


async def test_a_confirm_line_reads_a_field_of_a_shape_in_the_state() -> None:
    payload = _opening_job()
    payload["tasks"][0]["finish"][0]["confirm"] = ["Close {{input.order_id}}, due {{order.due}}?"]
    task_plan = TaskPlan(TasksSpec.model_validate(payload), state=_Shaped, tools=registry())
    tasks = CallTasks(task_plan, CallState(_Shaped))
    tasks.state.save({"order": {"due": "Friday"}})
    step = _first_step(tasks, {})

    await step.call_tool("lookup_order", {"order_id": "A-12"})

    assert step.reply == "Close A-12, due Friday?"


def test_a_confirm_line_that_misnames_a_field_of_a_shape_is_refused_when_built() -> None:
    payload = _opening_job()
    payload["tasks"][0]["finish"][0]["confirm"] = ["Due {{order.dew}}?"]
    with pytest.raises(ValueError, match="'dew'"):
        TaskPlan(TasksSpec.model_validate(payload), state=_Shaped, tools=registry())


async def test_an_input_the_tool_refuses_is_never_asked() -> None:
    step = _confirm_step(Checked())

    with pytest.raises(ToolError, match="no such order"):
        await step.call_tool("lookup_order", {"order_id": "B-7"})
    assert step.reply is None


async def test_a_step_whose_tool_needs_a_yes_saves_nothing_that_tool_did_not_return() -> None:
    step = _confirm_step(Checked())

    refused = await step.finish({"order_id": "A-12", "status": "closed"}, caller_turns=2)

    assert isinstance(refused, str)
    assert refused.startswith("Not recorded: order_id and status come from `lookup_order`")
    assert step.outcome is None
    assert await step.finish({"unserved_request": "A refund."}, caller_turns=2) is None


async def test_a_value_already_held_still_finishes_a_step_whose_tool_needs_a_yes() -> None:
    step = _confirm_step(Checked(), held={"order_id": "A-12", "status": "open"})

    assert await step.finish({"order_id": "A-12", "status": "open"}, caller_turns=2) is None


async def test_a_confirm_line_says_a_date_in_the_call_language() -> None:
    schema = {"type": "object", "properties": {"day": {"type": "string", "format": "date"}}}
    step = _confirm_step(
        Canned(), line="¿Lo cierro el {{input.day}}?", schema=schema, language="es"
    )

    await step.call_tool("lookup_order", {"day": "2026-10-16"})

    assert step.reply == "¿Lo cierro el viernes, 16 de octubre de 2026?"


async def test_a_confirm_line_names_the_day_from_today_where_the_business_is() -> None:
    schema = {
        "type": "object",
        "properties": {
            "day": {"type": "string", "format": "date"},
            "service": {"enum": ["hair-color", "haircut"]},
        },
    }
    step = _confirm_step(
        Canned(),
        line="Shall I book your {{input.service}} for {{input.day}}?",
        schema=schema,
        timezone="Europe/Madrid",
    )
    tomorrow = datetime.now(ZoneInfo("Europe/Madrid")).date() + timedelta(days=1)

    await step.call_tool("lookup_order", {"day": tomorrow.isoformat(), "service": "hair-color"})

    assert step.reply is not None
    assert step.reply.startswith("Shall I book your hair color for tomorrow, ")


def test_an_unknown_timezone_is_refused_by_name() -> None:
    task_plan = plan(job())
    with pytest.raises(ValueError, match="not an IANA name"):
        CallTasks(task_plan, state(task_plan), timezone="Mars/Olympus")


def _confirm_opening(
    lookup: ToolExecutor, held: dict[str, str], *, confirm: dict[str, str] | None = None
) -> tuple[CallTasks, StepRun]:
    """``find_order`` opens by asking to confirm ``lookup_order`` on the held order."""
    payload = _opening_job()
    find_order = payload["tasks"][0]
    find_order["finish"][0]["confirm"] = ["Shall I close order {{input.order_id}}?"]
    find_order["opening"] = "listen"
    find_order["opens_with"] = {"tool": "lookup_order", "arguments": {"order_id": "order_id"}}
    task_plan = plan(payload, tools=registry(lookup_order=lookup), confirm=confirm)
    tasks = CallTasks(task_plan, state(task_plan))
    return tasks, _first_step(tasks, held)


async def test_an_opening_question_cut_off_is_asked_again_before_anything_runs() -> None:
    lookup = Checked({"found": True, "order": {"id": "A-12"}, "status": "open"})
    _, step = _confirm_opening(lookup, {"order_id": "A-12"})
    opening = await step.open()
    assert opening is not None and opening.line is not None

    step.reply_played(opening.line, heard=False)
    again = await step.call_tool("lookup_order", {"order_id": "A-12"})

    assert isinstance(again, dict) and again["asked"] == "Shall I close order A-12?"
    assert lookup.calls == []


async def test_an_opening_question_waits_for_its_value_and_skips_a_confirmed_one() -> None:
    _, empty = _confirm_opening(Checked(), {})
    assert await empty.open() is None
    assert empty.listens and empty.announce == "Let me look."

    tasks, step = _confirm_opening(
        Checked(), {"order_id": "A-12"}, confirm={"order_id": "find_order"}
    )
    tasks.state.save({"order_id": "A-12"}, step="find_order")
    assert tasks.state.is_confirmed("order_id")
    assert await step.open() is None


async def test_a_step_opening_on_a_confirm_tool_with_no_arguments_asks_its_line() -> None:
    lookup = Canned({"found": True, "order": {"id": "A-12"}, "status": "closed"})
    payload = _opening_job()
    find_order = payload["tasks"][0]
    find_order["finish"][0]["confirm"] = ["Shall I close your order?"]
    find_order["opening"] = "listen"
    find_order["opens_with"] = {"tool": "lookup_order"}
    task_plan = plan(payload, tools=registry(lookup_order=lookup))
    step = _first_step(CallTasks(task_plan, state(task_plan)), {})

    opening = await step.open()

    assert opening is not None and opening.line == "Shall I close your order?"
    assert lookup.calls == []


def _ended(
    step: StepRun,
    *,
    spoke: bool = True,
    called: bool = False,
    completed: bool = True,
    answers_result: bool = False,
    caller_turns: int = 3,
) -> AfterReply | None:
    """Tell the step a reply ended; the defaults are a spoken, completed answer to the caller."""
    return step.reply_ended(
        spoke=spoke,
        called=called,
        completed=completed,
        answers_result=answers_result,
        caller_turns=caller_turns,
    )


@pytest.mark.parametrize(
    ("reply", "due"),
    [
        ({}, True),
        ({"caller_turns": 2}, False),
        ({"called": True}, False),
        ({"spoke": False}, False),
        ({"completed": False}, False),
    ],
)
def test_a_finish_check_follows_words_without_a_call_once_the_caller_spoke(
    reply: dict[str, Any], due: bool
) -> None:
    """Once the caller spoke, a reply of words without a tool call is checked for a finish."""
    step = _step(_start(_tasks(), "take_note"), turns=2)
    check = (_ended(step, **reply) or AfterReply()).check
    assert (check is not None) is due
    if check is not None:
        assert [tool["name"] for tool in check.tools] == ["finish", "not_yet"]


def test_two_finish_checks_share_no_tool_declaration() -> None:
    first, second = (_ended(_step(_start(_tasks(), "take_note"), turns=2)) for _ in range(2))
    assert first is not None and first.check is not None
    assert second is not None and second.check is not None
    first.check.tools[1]["parameters"]["properties"]["x"] = {}
    assert second.check.tools[1]["parameters"]["properties"] == {}


def test_a_finish_check_runs_once_per_caller_turn_and_only_in_record_mode() -> None:
    step = _step(_start(_tasks(), "take_note"), turns=2)
    assert _ended(step) is not None
    assert _ended(step) is None
    assert _ended(step, caller_turns=4) is not None
    quiet = _step(_start(_tasks(mode="status"), "take_note"), turns=2)
    assert _ended(quiet) is None


async def test_a_step_with_its_own_tools_is_checked_only_after_one_succeeded() -> None:
    """A step with its own tools is not checked for a finish until one of them succeeded."""
    payload = job()
    payload["tasks"][1]["tools"] = ["search_web"]
    tasks = _tasks(payload, search_web=Canned({"error": "down"}, {"hits": 1}))
    step = _step(_start(tasks, "take_note"), turns=2)
    assert _ended(step) is None
    await step.call_tool("search_web", {})
    assert _ended(step, caller_turns=4) is None
    await step.call_tool("search_web", {})
    assert _ended(step, caller_turns=5) is not None


async def test_a_held_finish_goes_out_with_the_reply_that_says_the_result() -> None:
    step = _step(_start(_tasks(), "take_note"), turns=2)
    unheard = ("search_web", '{"hits": 1}')
    refused = await step.finish({"note": "x"}, caller_turns=3, unheard=unheard)
    assert refused is not None
    assert _ended(step, spoke=False, answers_result=True) is None
    after = _ended(step, answers_result=True)
    assert after is not None and after.finish == {"note": "x"}
    assert (_ended(step, answers_result=True, caller_turns=4) or AfterReply()).finish is None


@pytest.mark.parametrize(
    "reply",
    [{"completed": False}, {"answers_result": False}, {"called": True}],
)
async def test_a_held_finish_is_dropped_by_a_cut_a_new_caller_turn_or_a_call(
    reply: dict[str, Any],
) -> None:
    step = _step(_start(_tasks(), "take_note"), turns=2)
    await step.finish({"note": "x"}, caller_turns=3, unheard=("search_web", "{}"))
    assert (_ended(step, **{"answers_result": True} | reply) or AfterReply()).finish is None
    assert (_ended(step, answers_result=True, caller_turns=9) or AfterReply()).finish is None


def test_a_check_reads_only_a_finish_it_chose() -> None:
    assert decided_finish([("finish", '{"note": "x"}')]) == {"note": "x"}
    assert decided_finish([("not_yet", "{}")]) is None
    assert decided_finish([("finish", "not json")]) is None
    assert decided_finish([]) is None


def _ending_group() -> dict[str, Any]:
    payload = job()
    payload["groups"][0] |= {"steps": ["take_note"], "then": "end"}
    return payload


def _transferring_group() -> dict[str, Any]:
    payload = job()
    payload["groups"][0] |= {"steps": ["take_note"], "then": "transfer", "then_target": "billing"}
    return payload


@pytest.mark.parametrize("group", [_ending_group, _transferring_group], ids=["end", "transfer"])
async def test_a_group_that_hands_the_caller_back_returns_whatever_its_then(
    group: Callable[[], dict[str, Any]],
) -> None:
    run = _start(_tasks(group()), "order_flow")
    step = _step(run)
    await step.finish({"note": "x", "unserved_request": "A refund"}, caller_turns=2)
    run.ended(step)
    run.result(caller_turns=2)
    assert not run.ends_call
    assert run.transfers_to is None


async def test_a_tool_asked_to_confirm_reads_the_call_values_its_locked_arguments_name() -> None:
    lookup = Checked({"found": False})
    schema = {
        "type": "object",
        "properties": {"order_id": {"type": "string"}, "call": {"type": "string"}},
        "required": ["order_id"],
    }
    spec = ToolSpec(
        "lookup_order",
        "Close an order.",
        schema,
        AnyResult,
        lookup,
        {"call": "{{call.carrier_call_id}}"},
    )
    tools = registry()
    payload = _opening_job()
    payload["tasks"][0]["finish"][0]["confirm"] = ["Shall I close order {{input.order_id}}?"]
    task_plan = plan(
        payload, tools=ToolRegistry([spec, tools.get("search_web"), tools.get("normalize")])
    )
    call_state = CallState(task_plan.state, call=CallInfo(carrier_call_id="c-1"))
    tasks = CallTasks(task_plan, call_state)
    step = _first_step(tasks, {})
    asked = await step.call_tool("lookup_order", {"order_id": "a-12"})
    assert isinstance(asked, dict) and asked["status"] == "not_run_yet"
    step.reply_played(asked["asked"], heard=True)
    await step.call_tool("lookup_order", {"order_id": "A-12"})
    assert lookup.calls == [{"order_id": "A-12", "call": "c-1"}]


async def test_a_held_save_the_step_hands_back_unserved_is_not_reported_made() -> None:
    tasks = _tasks()
    step = _step(_start(tasks, "take_note"), turns=2)
    await step.finish({"note": "a call back"}, caller_turns=2, spoke=True)

    async def hand_back(check: FinishCheck) -> dict[str, Any]:
        return {"note": "x", "unserved_request": "A refund"}

    step.hold_save(caller_spoke_since=lambda: True, decide=hand_back)
    assert await tasks.state.settle_held() == []
    assert tasks.state.plain("note") is None


async def test_a_step_that_hands_off_ends_its_group_with_no_group_event() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    tasks = _tasks(events=events)
    run = _start(tasks, "order_flow")
    step = _step(run)
    assert step.may_hand_off(alone=True) is None
    assert not step.done
    step.hand_off()
    assert step.done and step.handed_off
    run.ended(step)
    assert run.next_step() is None and run.handed_off and run.status == "unserved"
    run.result(caller_turns=1)
    assert tasks.active is None
    assert [name for name, _ in events] == [
        "group.started",
        "group.step_started",
        "task.started",
        "task.handed_off",
    ]


async def test_a_step_handoff_beside_another_call_or_after_the_end_is_refused() -> None:
    tasks = _tasks()
    run = _start(tasks, "find_order")
    step = _step(run)
    assert step.may_hand_off(alone=False) == REFUSED_HANDOFF_BESIDE
    assert not step.done
    assert await step.finish({"order_id": "A-12", "status": "open"}, caller_turns=2) is None
    assert step.may_hand_off(alone=True) == REFUSED_HANDOFF_ENDED
    step.hand_off()
    assert not step.handed_off
