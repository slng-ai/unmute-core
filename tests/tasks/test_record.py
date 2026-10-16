"""The run record, what later steps read of it, spoken lines, events and the re-run guard.

The cases cover the record, served steps, receipts, events and the guard.
"""

from typing import Any

from tests.tasks.job import job, plan, state
from unmute_core.state import CallState
from unmute_core.tasks import (
    FinishRule,
    GroupProgress,
    RunRecord,
    SaveOutcome,
    ServedRequests,
    caller_turns,
)
from unmute_core.tasks._record import UNSET, task_event
from unmute_core.tasks._run import ReentryGuard
from unmute_core.tasks._texts import (
    REFUSED_RERUN,
    REFUSED_RERUN_HANDED_BACK,
    SERVED_HEADING,
    SERVED_RULE,
)
from unmute_core.tasks._values import save_task_values
from unmute_core.tools._lines import ReceiptLines, render_line


def _confirming() -> tuple[Any, CallState]:
    """order_id is confirmed by find_order, and the group may skip find_order."""
    payload = job()
    payload["groups"][0]["skip_when_confirmed"] = {"find_order": "order_id"}
    task_plan = plan(payload, confirm={"order_id": "find_order"})
    return task_plan, state(task_plan)


def _finish(
    task_plan: Any,
    call_state: CallState,
    step: str,
    hand_back_saves: bool = False,
    /,
    **values: Any,
) -> SaveOutcome:
    return save_task_values(
        task_plan.tasks[step],
        values,
        state=call_state,
        model=task_plan.result_model(step),
        hand_back_saves=hand_back_saves,
    )


def test_the_record_of_a_completed_group_lists_each_step_and_what_it_saved() -> None:
    task_plan, call_state = _confirming()
    progress = GroupProgress(task_plan.groups["order_flow"], task_plan.steps("order_flow"))
    record = RunRecord.of_group(progress)
    while (index := progress.next_step()) is not None:
        task = progress.task
        values = (
            {"order_id": "A-12", "status": "open"}
            if task.name == "find_order"
            else {"note": "call back"}
        )
        outcome = _finish(task_plan, call_state, task.name, **values)
        record.ended(task.name, outcome, index=index)
        progress.record(index, outcome)
    result = record.build(progress.status(), call_state)
    assert result == {
        "status": "completed",
        "task": "order_flow",
        "steps": [
            {
                "step": "find_order",
                "outcome": "completed",
                "saved": {"order_id": "A-12", "status": "open"},
                "confirmed": ["order_id"],
            },
            {"step": "take_note", "outcome": "completed", "saved": {"note": "call back"}},
        ],
    }
    assert next(iter(result)) == "status"


def test_the_record_of_a_hand_back_carries_the_request_and_what_did_not_fit() -> None:
    task_plan, call_state = _confirming()
    record = RunRecord("order_flow")
    record.ended(
        "find_order",
        _finish(task_plan, call_state, "find_order", order_id="A-12", status="open"),
        index=0,
    )
    record.ended(
        "take_note",
        _finish(task_plan, call_state, "take_note", unserved_request="Change it to five"),
        index=1,
    )
    assert record.build("unserved", call_state)["steps"][1] == {
        "step": "take_note",
        "outcome": "handed back",
        "not_done": "Change it to five",
    }
    other_plan = plan()
    other_state = state(other_plan)
    partial = RunRecord("find_order")
    partial.ended(
        "find_order",
        _finish(
            other_plan,
            other_state,
            "find_order",
            True,
            order_id="A-12",
            status="lost",
            unserved_request="A refund",
        ),
    )
    assert partial.build("unserved", other_state)["steps"] == [
        {
            "step": "find_order",
            "outcome": "handed back",
            "saved": {"order_id": "A-12"},
            "not_saved": ["status"],
            "not_done": "A refund",
        }
    ]


def test_a_skipped_step_is_listed_where_the_group_would_have_run_it() -> None:
    task_plan, call_state = _confirming()
    call_state.save({"order_id": "A-12"}, step="find_order")
    progress = GroupProgress(
        task_plan.groups["order_flow"],
        task_plan.steps("order_flow"),
        task_plan.skipped_steps("order_flow", call_state.is_confirmed),
    )
    record = RunRecord.of_group(progress)
    assert progress.next_step() == 1
    record.ended("take_note", _finish(task_plan, call_state, "take_note", note="hi"), index=1)
    assert record.build("completed", call_state)["steps"] == [
        {"step": "find_order", "outcome": "skipped"},
        {"step": "take_note", "outcome": "completed", "saved": {"note": "hi"}},
    ]
    later = RunRecord("g", skipped={2: "c"})
    later.ended("a", SaveOutcome("completed"), index=0)
    later.ended("b", SaveOutcome("failed"), index=1)
    assert [row["step"] for row in later.build("failed", call_state)["steps"]] == ["a", "b"]
    assert later.build("completed", call_state)["steps"][-1] == {"step": "c", "outcome": "skipped"}


def test_a_step_run_again_after_going_back_is_listed_again() -> None:
    task_plan, call_state = _confirming()
    progress = GroupProgress(task_plan.groups["order_flow"], task_plan.steps("order_flow"))
    record = RunRecord.of_group(progress)
    orders = iter(["A-12", "A-13"])
    went_back = False
    while (index := progress.next_step()) is not None:
        task = progress.task
        if task.name == "take_note" and not went_back:
            went_back = True
            progress.go_back(["find_order"])
            continue
        values = (
            {"order_id": next(orders), "status": "open"}
            if task.name == "find_order"
            else {"note": "x"}
        )
        outcome = _finish(task_plan, call_state, task.name, **values)
        record.ended(task.name, outcome, index=index)
        progress.record(index, outcome)
    steps = record.build("completed", call_state)["steps"]
    assert [(row["step"], row["outcome"]) for row in steps] == [
        ("find_order", "completed"),
        ("find_order", "completed"),
        ("take_note", "completed"),
    ]
    assert [row["saved"]["order_id"] for row in steps[:2]] == ["A-13", "A-13"]


def test_the_record_of_a_single_task_has_one_step_and_its_provider() -> None:
    task_plan, call_state = _confirming()
    record = RunRecord.of_task("take_note", provider=None, provider_runs=False)
    record.ended("take_note", _finish(task_plan, call_state, "take_note", note="call back"))
    assert record.build("completed", call_state) == {
        "status": "completed",
        "task": "take_note",
        "steps": [{"step": "take_note", "outcome": "completed", "saved": {"note": "call back"}}],
    }
    met = RunRecord.of_task("refund", provider="find_order", provider_runs=False)
    met.ended("refund", SaveOutcome("completed"))
    assert met.build("completed", call_state)["steps"] == [
        {"step": "find_order", "outcome": "skipped"},
        {"step": "refund", "outcome": "completed"},
    ]
    ran = RunRecord.of_task("refund", provider="find_order", provider_runs=True)
    ran.ended("find_order", SaveOutcome("failed"))
    assert ran.build("failed", call_state)["steps"] == [{"step": "find_order", "outcome": "failed"}]


def test_a_step_records_what_it_said_or_the_result_it_left_unheard() -> None:
    _, call_state = _confirming()
    record = RunRecord("order_status")
    record.ended("a", SaveOutcome("completed"), said=" Found A-12; it's open. ")
    record.ended("b", SaveOutcome("completed"), said="  ")
    record.ended("c", SaveOutcome("completed"), not_heard="lookup_order returned: {'late': True}")
    record.ended("d", SaveOutcome("completed"), said="It arrives Wednesday.", not_heard="ignored")
    assert record.build("completed", call_state)["steps"] == [
        {"step": "a", "outcome": "completed", "said": "Found A-12; it's open."},
        {"step": "b", "outcome": "completed"},
        {"step": "c", "outcome": "completed", "not_heard": "lookup_order returned: {'late': True}"},
        {"step": "d", "outcome": "completed", "said": "It arrives Wednesday."},
    ]


def test_a_saved_value_no_longer_settled_reads_unset_and_values_stay_whole() -> None:
    task_plan, call_state = _confirming()
    record = RunRecord("order_flow")
    record.ended(
        "find_order",
        _finish(task_plan, call_state, "find_order", order_id="A-12", status="open"),
        index=0,
    )
    record.ended("take_note", _finish(task_plan, call_state, "take_note", note="x" * 4050), index=1)
    call_state.save({"order_id": "B-1"})
    steps = record.build("completed", call_state)["steps"]
    assert steps[0] == {
        "step": "find_order",
        "outcome": "completed",
        "saved": {"order_id": UNSET, "status": "open"},
    }
    assert steps[1]["saved"]["note"] == "x" * 4050


def test_a_step_reads_what_earlier_runs_did_but_not_its_own_task() -> None:
    task_plan, call_state = _confirming()
    served = ServedRequests(task_plan.tasks)
    assert served.section("take_note") == ""
    first = RunRecord.of_task("find_order", provider=None, provider_runs=False)
    first.ended(
        "find_order", _finish(task_plan, call_state, "find_order", order_id="A-12", status="open")
    )
    served.record(first)
    found = "- find_order (The caller asks about an order): saved order_id and status."
    assert served.section("take_note") == "\n".join([SERVED_HEADING, found, SERVED_RULE])
    assert served.section("find_order") == ""
    group = RunRecord("order_flow", skipped={2: "take_note"})
    group.ended(
        "find_order",
        _finish(task_plan, call_state, "find_order", order_id="A-12", status="open"),
        index=0,
    )
    group.ended(
        "take_note",
        _finish(task_plan, call_state, "take_note", unserved_request="A refund"),
        index=1,
    )
    group.ended("refund", SaveOutcome("failed"), index=3)
    assert group.served() == [("find_order", ("order_id", "status"))]
    kept = RunRecord.of_task("take_note", provider=None, provider_runs=False)
    kept.ended("take_note", SaveOutcome("unserved", saved=("note",), unserved_request="A refund"))
    done = RunRecord.of_task("refund", provider=None, provider_runs=False)
    done.ended("refund", SaveOutcome("completed"))
    for run in (group, kept, done):
        served.record(run)
    assert served.section("find_order") == "\n".join(
        [SERVED_HEADING, "- take_note: saved note.", "- refund: done.", SERVED_RULE]
    )
    again = RunRecord.of_task("refund", provider=None, provider_runs=False)
    again.ended("refund", SaveOutcome("completed", saved=("note",)))
    served.record(again)
    assert served.section("find_order") == "\n".join(
        [SERVED_HEADING, "- take_note: saved note.", "- refund: saved note.", SERVED_RULE]
    )


def test_receipt_lines_fill_from_the_result_and_saved_values_and_rotate() -> None:
    words = {"result.order.id": "A-12", "result.paid": True, "status": "open"}
    assert render_line("{{result.order.id}} is {{status}}, paid: {{result.paid}}", words) == (
        "A-12 is open, paid: yes"
    )
    assert render_line("Ref {{result.ref}}", {"result.ref": ""}) is None
    rule = FinishRule(
        tool="lookup_order",
        success={"found": (True,)},
        say=("First {{result.id}}.", "Second {{result.id}}.", "Only {{x}}."),
    )
    lines = ReceiptLines()
    assert [lines.pick("k", rule.say, {"result.id": "7"}) for _ in range(3)] == [
        "First 7.",
        "Second 7.",
        "First 7.",
    ]


def test_task_events_carry_names_never_values() -> None:
    assert task_event("task.completed", task="find_order", run_id="r1", saved=["order_id"]) == (
        "task.completed",
        {"task": "find_order", "task_run_id": "r1", "saved": ["order_id"]},
    )


def test_reentry_guard_refuses_until_the_caller_speaks() -> None:
    guard = ReentryGuard()
    messages = [{"role": "user"}, {"role": "assistant"}]
    guard.record("find_order", caller_turns(messages))
    assert guard.refuses("find_order", caller_turns(messages))
    messages.append({"role": "user"})
    assert not guard.refuses("find_order", caller_turns(messages))


def test_a_refused_rerun_says_how_the_last_run_ended_in_record_mode() -> None:
    guard = ReentryGuard()
    guard.record("new_booking", 1, "unserved")
    guard.record("find_order", 1, "completed")
    guard.record("broke", 1)
    assert guard.refusal("new_booking", record=True) == REFUSED_RERUN_HANDED_BACK
    assert guard.refusal("new_booking", record=False) == REFUSED_RERUN
    for name in ("find_order", "broke"):
        assert guard.refusal(name, record=True) == REFUSED_RERUN
    guard.record("new_booking", 3, "completed")
    assert guard.refusal("new_booking", record=True) == REFUSED_RERUN
    guard.refusal("find_order", record=True)["refused"] = "x"
    assert guard.refusal("find_order", record=True) == REFUSED_RERUN
