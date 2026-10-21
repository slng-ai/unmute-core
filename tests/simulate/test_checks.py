"""The deterministic checks on a simulated call."""

from typing import Any

import pytest

from tests.simulate.calls import records, said, save, scenario, task, tool
from unmute_core.tasks import EMPTY_REPLY_GAVE_UP
from unmute_core.testing.simulate import run_checks
from unmute_core.testing.simulate._checks import (
    check_ended_naturally,
    check_fabrication,
    check_fallback_lines,
    check_injected_arguments,
    check_no_wrong_call_end_save,
    check_owner_rewrites,
    check_requirements,
    check_run_records,
    check_save_after_caller_spoke,
    check_served_not_handed_back,
    events_of,
    match_value,
    spoken_digits,
)

APPOINTMENT = {"service": "check-up and cleaning", "day": "2026-10-27", "time": "15:00"}


@pytest.mark.parametrize(
    ("matcher", "stored", "ok"),
    [
        ({"absent": True}, None, True),
        ({"absent": True}, "x", False),
        ({"phone": "+34612345678"}, "+34612345678", True),
        ({"phone": "+34612345678"}, "+34 612 345 678", True),
        ({"phone": "+34612345678"}, "+44612345678", False),
        ({"phone": "+34612345678"}, None, False),
        ({"bool": True}, True, True),
        ({"bool": True}, "true", True),
        ({"bool": False}, True, False),
        ({"exact": "new_customer"}, "new_customer", True),
        ({"exact": "16:30"}, "16:30:00", False),
        ({"any": [{"exact": "16:30"}, {"exact": "16:30:00"}]}, "16:30:00", True),
        ({"exact_ci": "A48213"}, "a-48 213", True),
        ({"present": r"REC-\d{4}-\d{3}"}, "REC-5678-001", True),
        ({"present": r"REC-\d{4}-\d{3}"}, "REC-56-1", False),
        ({"text": [["robin"], ["vega"]]}, "Robin Vega", True),
        ({"text": [["robin"], ["vega"]]}, "Robin", False),
        ({"fields": {"email": {"exact": "r@x.com"}}}, {"email": "r@x.com", "name": "R"}, True),
        ({"fields": {"email": {"exact": "r@x.com"}}}, '{"email": "r@x.com"}', True),
        ({"fields": {"email": {"exact": "r@x.com"}}}, "r@x.com", False),
        (
            {"appointment": {"service": ["clean"], "day": "2026-10-27", "time": "15:00"}},
            APPOINTMENT,
            True,
        ),
        (
            {"appointment": {"service": ["clean"], "day": "2026-10-27", "time": "16:00"}},
            APPOINTMENT,
            False,
        ),
    ],
)
def test_a_matcher_says_whether_a_saved_value_meets_it(
    matcher: dict[str, Any], stored: object, ok: bool
) -> None:
    assert match_value(matcher, stored)[0] is ok


def test_an_unknown_matcher_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown matcher"):
        match_value({"roughly": 1}, "x")


@pytest.mark.parametrize(
    ("line", "digits"),
    [
        ("plus three four, six one two", "34612"),
        ("six double oh, one one one", "600111"),
        ("triple seven and 8", "7778"),
        ("oh, I'm not sure", "0"),
    ],
)
def test_spoken_digits_reads_digits_said_as_words(line: str, digits: str) -> None:
    assert spoken_digits(line) == digits


BOOKING = scenario(
    known={
        "phone_number": ["+34612345678"],
        "service": ["clean", "check"],
        "day": ["2026-10-27"],
        "time": ["15:00"],
    }
)


@pytest.mark.parametrize(
    ("events", "passed", "category"),
    [
        (
            [said("caller", "plus three four six one two three four five six seven eight")],
            True,
            "",
        ),
        ([save("verify_identity", phone_number="+34612345678")], True, ""),
        ([save("verify_identity", phone_number="+34699999999")], False, "fabricated_value"),
        (
            [
                said("caller", "It's six nine nine, nine nine nine, nine nine nine."),
                save("verify_identity", phone_number="+34699999999"),
            ],
            True,
            "caller_off_script_value",
        ),
        (
            [
                tool("find_appointments", {"day": "2026-10-29", "time": "11:00", "service": "x"}),
                save("change_or_cancel", appointment={"day": "2026-10-29", "time": "11:00"}),
            ],
            True,
            "",
        ),
        (
            [
                save("take_booking", appointment={**APPOINTMENT, "day": "2026-10-29"}),
                tool("new_booking", {"steps": [{"saved": {"appointment": "2026-10-29"}}]}),
            ],
            False,
            "fabricated_value",
        ),
    ],
    ids=[
        "nothing_saved",
        "a_known_number",
        "an_invented_number",
        "a_number_the_caller_made_up",
        "a_value_read_off_an_earlier_tool_result",
        "a_later_run_record_does_not_excuse_a_save",
    ],
)
def test_a_saved_value_the_caller_never_gave_is_fabricated(
    events: list[dict[str, Any]], passed: bool, category: str
) -> None:
    result = check_fabrication(events_of(records(*events)), BOOKING)
    assert (result.passed, result.category) == (passed, category)


@pytest.mark.parametrize(
    ("events", "passed", "category"),
    [
        (
            [
                task("task.started", task="verify_identity"),
                said("caller", "+34 612 345 678"),
                save("verify_identity", phone_number="+34612345678"),
            ],
            True,
            "",
        ),
        (
            [
                said("caller", "+34 612 345 678"),
                task("task.started", task="verify_identity"),
                save("verify_identity", phone_number="+34612345678"),
            ],
            False,
            "opening_save_answered_just_before",
        ),
        (
            [
                said("caller", "I'd like to book."),
                task("task.started", task="open_record"),
                tool("create_customer_record", {"record_id": "REC-5678-001"}),
                save("open_record", record_id="REC-5678-001"),
            ],
            True,
            "",
        ),
        (
            [
                said("caller", "Remind me, please."),
                task("task.started", task="reminder"),
                {"kind": "call_end_saves"},
                save("reminder", wants_reminder=True),
            ],
            True,
            "",
        ),
    ],
    ids=[
        "after_the_caller_spoke",
        "before_the_caller_spoke",
        "a_value_its_own_tool_returned",
        "a_save_held_for_the_calls_end",
    ],
)
def test_a_step_saves_only_after_the_caller_spoke_in_it(
    events: list[dict[str, Any]], passed: bool, category: str
) -> None:
    result = check_save_after_caller_spoke(events_of(records(*events)))
    assert (result.passed, result.category) == (passed, category)


@pytest.mark.parametrize(
    ("applied", "confirming_task", "passed"),
    [
        ({"wants_reminder": True}, None, True),
        ({"wants_reminder": False}, None, False),
        ({"phone_number": "+34612345678"}, "verify_identity", False),
        ({"call": {"end_reason": "caller_hung_up"}}, None, True),
    ],
    ids=["kept_as_said", "against_the_scenario", "reported_as_confirmed", "the_call_itself"],
)
def test_a_save_at_the_calls_end_is_never_a_wrong_one(
    applied: dict[str, Any], confirming_task: str | None, passed: bool
) -> None:
    held = save("reminder", **applied)
    held["confirmed"] = dict.fromkeys(applied, True)
    call = records({"kind": "call_end_saves"}, held)
    call[0]["variables"] = {
        name: {"stored": value, "confirming_task": confirming_task}
        for name, value in applied.items()
    }
    expect = scenario(expect={"wants_reminder": {"bool": True}}, tags=["early_answer"])
    result = check_no_wrong_call_end_save(call[0], events_of(call), expect)
    assert (result.passed, result.core) == (passed, True)


def _group_run(status: str = "completed", **outcomes: str) -> list[dict[str, Any]]:
    """Return the task events of a two-step group run, and the record its delegate returned."""
    run = {"group": "new_booking", "task_run_id": "r1"}
    return [
        task("group.started", **run),
        task("task.started", task="verify_identity", step=0, **run),
        task("task.completed", task="verify_identity", step=0, saved=["phone_number"], **run),
        task("task.started", task="take_booking", step=1, **run),
        task("task.completed", task="take_booking", step=1, saved=["appointment"], **run),
        task("group.completed", **run),
        tool(
            "new_booking",
            {
                "status": status,
                "task": "new_booking",
                "steps": [
                    {
                        "step": "verify_identity",
                        "outcome": outcomes.get("verify_identity", "completed"),
                        "saved": {"phone_number": "+34612345678"},
                    },
                    {
                        "step": "take_booking",
                        "outcome": "completed",
                        "saved": {"appointment": "{}"},
                    },
                ],
            },
        ),
    ]


@pytest.mark.parametrize(
    ("events", "passed"),
    [
        (_group_run(), True),
        (_group_run(status="unserved"), False),
        (_group_run(verify_identity="skipped"), False),
        ([*_group_run()[:-1], tool("new_booking", None)], True),
        (_group_run()[:-1], True),
        (
            [*_group_run(), tool("new_booking", {"status": "completed", "task": "new_booking"})],
            False,
        ),
    ],
    ids=[
        "a_matching_record",
        "another_status",
        "another_step_outcome",
        "a_record_no_model_read",
        "no_delegate_call_seen",
        "a_record_with_no_run",
    ],
)
def test_a_run_record_matches_its_task_events(events: list[dict[str, Any]], passed: bool) -> None:
    result = check_run_records(events_of(records(*events)), {"new_booking"})
    assert result.passed is passed, result.detail


@pytest.mark.parametrize(
    ("requires", "events", "requests", "variables", "problems"),
    [
        ({"task_started": "change_or_cancel"}, [], [], {}, "change_or_cancel never started"),
        (
            {"task_started": "change_or_cancel"},
            [task("task.started", task="change_or_cancel", task_run_id="r")],
            [],
            {},
            "",
        ),
        ({"mock_called": "availability"}, [], [], {}, "the availability mock was never called"),
        ({"mock_called": "availability"}, [], [{"mock": "availability"}], {}, ""),
        ({"tools_called": ["send_sms"]}, [], [{"mock": "sms", "tool": "send_sms"}], {}, ""),
        ({"tools_called": ["end_call"]}, [tool("end_call", {})], [], {}, ""),
        ({"tools_called": ["end_call"]}, [], [], {}, "end_call was never called"),
        (
            {"time_in_availability": True},
            [],
            [],
            {"appointment": {"day": "2026-10-27", "time": "15:00"}},
            "",
        ),
        (
            {"time_in_availability": True},
            [],
            [],
            {"appointment": {"day": "2026-10-27", "time": "16:00"}},
            "booked 16:00 on 2026-10-27, free: ('10:00', '13:30', '15:00', '17:30')",
        ),
        (
            {"any_of": [{"value": {"appointment.time": "17:00"}}]},
            [],
            [],
            {"appointment": {"time": "17:00"}},
            "",
        ),
        (
            {"any_of": [{"task_started_after_first_group": "change_or_cancel"}]},
            [
                task("task.started", task="change_or_cancel", task_run_id="a"),
                task("group.completed", group="new_booking", task_run_id="b"),
            ],
            [],
            {},
            "none of [{'task_started_after_first_group': 'change_or_cancel'}] happened",
        ),
    ],
)
def test_a_scenario_requirement_must_happen_on_the_way(
    requires: dict[str, Any],
    events: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    variables: dict[str, Any],
    problems: str,
) -> None:
    call = records(*events, variables=variables, requests=requests)
    result = check_requirements(call[0], call, scenario(requires=requires))
    assert result is not None
    assert (result.passed, result.detail) == (not problems, problems)


def test_a_scenario_with_no_requirements_has_no_requirement_check() -> None:
    call = records()
    only_injected = scenario(requires={"injected": {"customer_records": {}}})
    assert check_requirements(call[0], call, scenario()) is None
    assert check_requirements(call[0], call, only_injected) is None


@pytest.mark.parametrize(
    ("params", "problems"),
    [
        ({"phone": "+34600111222", "name": "Robin Vega"}, ""),
        (
            {"phone": "+34600111999", "name": "Robin Vega"},
            "customer_records phone='+34600111999', saved caller_phone='+34600111222'",
        ),
    ],
)
def test_a_hidden_argument_is_the_value_the_call_saved(
    params: dict[str, str], problems: str
) -> None:
    call = records(
        variables={"caller_phone": "+34600111222", "contact": {"name": "Robin Vega"}},
        requests=[{"mock": "customer_records", "params": params}],
    )
    injected = {"customer_records": {"phone": "caller_phone", "name": "contact.name"}}
    result = check_injected_arguments(call[0], call, scenario(requires={"injected": injected}))
    assert result is not None
    assert (result.passed, result.detail) == (not problems, problems)


@pytest.mark.parametrize(
    ("request_text", "passed"),
    [
        ("leave a message for Dr. Ruiz", False),
        ("change the message for Dr. Ruiz", True),
        ("book a cleaning", True),
    ],
)
def test_a_served_request_is_never_handed_back(request_text: str, passed: bool) -> None:
    events = [
        task("task.completed", task="take_message", saved=["message"]),
        tool("finish", None, unserved_request=request_text),
    ]
    served = scenario(served={"take_message": ["message"]})
    result = check_served_not_handed_back(events_of(records(*events)), served)
    assert result is not None
    assert result.passed is passed


@pytest.mark.parametrize(
    ("events", "category"),
    [
        ([save("verify_identity", phone_number="+34612345678")], ""),
        (
            [
                save("verify_identity", phone_number="+34612345678"),
                save(None, origin="tool", phone_number="+34699999999"),
            ],
            "owner_rewrote_task_value",
        ),
        ([save(None, origin="tool", phone_number="+34612345678")], "owner_wrote_task_value_first"),
        ([save(None, origin="prefetch", phone_number="+34612345678")], ""),
    ],
    ids=["a_task_save", "a_rewrite", "written_before_the_task", "the_prefetch_is_not_the_owner"],
)
def test_the_main_agent_does_not_write_a_value_a_task_owns(
    events: list[dict[str, Any]], category: str
) -> None:
    result = check_owner_rewrites(events_of(records(*events)), {"phone_number"})
    assert (result.passed, result.category, result.core) == (not category, category, False)


def test_a_fallback_line_is_counted_apart() -> None:
    events = events_of(records(said("agent", f"Thank you. {EMPTY_REPLY_GAVE_UP}")))
    result = check_fallback_lines(events)
    assert (result.passed, result.core, result.category) == (
        False,
        False,
        "empty_response_fallback",
    )


@pytest.mark.parametrize(
    ("ended_by", "turns", "passed"),
    [
        ("caller_hangup", 4, True),
        ("agent_hangup", 4, True),
        ("turn_cap", 18, False),
        ("terminal_failure", 2, False),
    ],
)
def test_a_call_ends_naturally_when_one_side_hangs_up(
    ended_by: str, turns: int, passed: bool
) -> None:
    result = check_ended_naturally({"ended_by": ended_by, "turns": turns}, scenario())
    assert result.passed is passed


def test_every_check_runs_and_one_with_nothing_to_check_is_left_out() -> None:
    call = records(
        said("caller", "Hi."),
        variables={"phone_number": "+34612345678"},
        errors=[{"kind": "call_failed", "error": "llm failed"}],
    )
    expect = scenario(expect={"phone_number": {"phone": "+34612345678"}})
    results = {r.id: r for r in run_checks(call, expect)}
    assert list(results) == [
        "no_errors",
        "saved_values_match",
        "no_fabricated_save",
        "no_save_before_caller_spoke_in_step",
        "no_wrong_call_end_save",
        "run_record_matches_events",
        "ended_naturally",
        "owner_does_not_rewrite_task_values",
        "no_empty_response_fallback",
    ]
    assert not results["no_errors"].passed
    assert results["no_errors"].category == "call_failed"
    assert results["saved_values_match"].passed
