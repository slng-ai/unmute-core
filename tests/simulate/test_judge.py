"""The judge: its transcript, how its reply is read, and which calls it leaves unscored."""

import json
from typing import Any

import pytest

from tests.simulate.calls import records, said, save, scenario, task, tool
from unmute_core.testing.simulate import CheckResult, Endpoint
from unmute_core.testing.simulate._agent import load_agent
from unmute_core.testing.simulate._endpoints import Chat
from unmute_core.testing.simulate._judge import (
    JUDGE_CHECKS,
    UNAVAILABLE,
    annotated_transcript,
    judge_call,
    parse_verdicts,
)
from unmute_core.testing.simulate._run import left_out


def reply(**verdicts: str) -> str:
    """Return a judge reply: every check passes, except those given."""
    return json.dumps(
        {
            check: {"verdict": verdicts.get(check, "pass"), "turns": ["T2"], "evidence": "q"}
            for check in JUDGE_CHECKS
        }
    )


def by_id(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {result.id.removeprefix("judge."): result for result in results}


def test_a_full_reply_rules_on_every_check() -> None:
    results = by_id(parse_verdicts(reply(no_premature_claim="fail")))
    assert list(results) == list(JUDGE_CHECKS)
    failed = results["no_premature_claim"]
    assert (failed.passed, failed.core, failed.category) == (False, True, "no_premature_claim")
    assert (failed.detail, failed.evidence) == ("q", ["T2"])
    assert all(r.passed for check, r in results.items() if check != "no_premature_claim")


@pytest.mark.parametrize(
    ("raw", "unavailable"),
    [
        ("not json at all", set(JUDGE_CHECKS)),
        ("[1, 2]", set(JUDGE_CHECKS)),
        (
            json.dumps({k: v for k, v in json.loads(reply()).items() if k != "caller_goal_served"}),
            {"caller_goal_served"},
        ),
        (reply(no_invented_facts="maybe"), {"no_invented_facts"}),
        (
            json.dumps({**json.loads(reply()), "no_wrap_up_mid_group": "pass"}),
            {"no_wrap_up_mid_group"},
        ),
    ],
    ids=["not_json", "not_an_object", "a_check_missing", "an_unknown_verdict", "a_bare_string"],
)
def test_a_check_with_no_verdict_is_unavailable_not_failed(raw: str, unavailable: set[str]) -> None:
    results = by_id(parse_verdicts(raw))
    assert {check for check, r in results.items() if r.category == UNAVAILABLE} == unavailable
    assert all(not results[check].passed for check in unavailable)


def test_turns_that_are_not_a_list_are_no_evidence() -> None:
    raw = json.loads(reply())
    raw["no_premature_claim"]["turns"] = "T3"
    assert by_id(parse_verdicts(json.dumps(raw)))["no_premature_claim"].evidence == []


def judged(**verdicts: str) -> list[CheckResult]:
    return parse_verdicts(reply(**verdicts))


@pytest.mark.parametrize(
    ("errors", "results", "reason"),
    [
        ([], judged(), None),
        ([], judged(no_premature_claim="fail"), None),
        ([], judged(caller_followed_script="fail"), "caller"),
        ([], parse_verdicts("{}"), "judge"),
        ([{"kind": "model_error"}], judged(), None),
        ([{"kind": "call_failed"}], judged(), None),
        (
            [{"kind": "environment_upstream_error"}],
            judged(caller_followed_script="fail"),
            "environment",
        ),
        ([{"kind": "harness_turn_timeout"}], parse_verdicts("{}"), "harness"),
    ],
    ids=[
        "scored",
        "an_agent_failure_is_scored",
        "the_caller_left_its_brief",
        "the_judge_gave_no_verdict",
        "a_model_error_is_the_agents",
        "a_failed_call_is_the_agents",
        "the_endpoint_failed_first",
        "the_simulator_broke",
    ],
)
def test_a_call_is_left_out_only_for_what_is_not_the_agents_fault(
    errors: list[dict[str, Any]], results: list[CheckResult], reason: str | None
) -> None:
    assert left_out({"errors": errors}, results) == reason


def test_the_transcript_numbers_caller_turns_and_shows_unheard_events() -> None:
    call = records(
        said("agent", "Hello, Maple Street Dental."),
        said("caller", "I'd like to book."),
        task("group.started", group="new_booking"),
        task("group.step_started", group="new_booking", task="verify_identity"),
        task("task.started", group="new_booking", task="verify_identity", step=0),
        said("agent", "What's your number?"),
        said("caller", "+34 612 345 678. Bye. [HANGUP]"),
        save("verify_identity", phone_number="+34612345678"),
        save(None, origin="call", call={"end_reason": "caller_hung_up"}),
        tool("finish", None, phone_number="+34612345678"),
        tool("new_booking", {"status": "completed"}),
        said("agent", "Thanks, goodbye.", unheard=True),
        {"kind": "hangup", "source": "caller"},
    )
    assert annotated_transcript(call, "Hello, Maple Street Dental.").splitlines() == [
        "T0",
        "  AGENT (greeting): Hello, Maple Street Dental.",
        "T1",
        "  CALLER: I'd like to book.",
        "  [group.started new_booking]",
        "  [task.started verify_identity (group new_booking, step 0)]",
        "  AGENT (fixed line): What's your number?",
        "T2",
        "  CALLER: +34 612 345 678. Bye. [HANGUP]",
        "  [SAVE by verify_identity: phone_number=+34612345678]",
        '  [tool new_booking({}) returned: {"status": "completed"}]',
        "  AGENT (fixed line, unheard: the caller had already hung up): Thanks, goodbye.",
        "  [call ended by caller]",
    ]


def test_a_line_is_put_down_to_the_model_reply_that_said_it() -> None:
    call = records(
        said("caller", "Hi."),
        {"kind": "llm", "site": "task:take_booking", "text": "Which day suits you?"},
        said("agent", "Which day suits you?"),
        {"kind": "llm", "site": "main", "text": "All booked."},
        said("agent", "All booked."),
    )
    lines = annotated_transcript(call, "Hello.").splitlines()
    assert lines[-2:] == [
        "  AGENT (step take_booking): Which day suits you?",
        "  AGENT (main agent): All booked.",
    ]


class ScriptedJudge(Chat):
    """A judge that gives each reply in turn, and keeps what it was asked."""

    def __init__(self, *replies: str | Exception) -> None:
        super().__init__(Endpoint(model="judge"))
        self.replies = list(replies)
        self.asked: list[dict[str, Any]] = []

    async def complete(self, messages: list[dict[str, str]], **params: Any) -> str:
        self.asked.append({"messages": messages, **params})
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


async def test_the_judge_is_asked_again_for_a_reply_that_is_not_the_json_asked_for() -> None:
    judge = ScriptedJudge("Sure! Here are my verdicts.", reply())
    results, kept = await judge_call(
        judge,
        scenario(goal="Book a cleaning."),
        records(said("caller", "Hi.")),
        load_agent("booking"),
    )
    assert all(r.passed for r in results)
    assert len(judge.asked) == 2
    assert judge.asked[0]["temperature"] == 0
    assert judge.asked[0]["response_format"] == {"type": "json_object"}
    prompt = judge.asked[0]["messages"][1]["content"]
    assert "Caller's goal: Book a cleaning." in prompt
    assert "Group new_booking" in prompt
    assert kept["raw"] == reply()
    assert kept["transcript"].startswith("T0\nT1\n  CALLER: Hi.")


async def test_a_judge_that_fails_leaves_every_check_unavailable() -> None:
    judge = ScriptedJudge(RuntimeError("the endpoint is down"))
    results, kept = await judge_call(judge, scenario(), records(), load_agent("verify"))
    assert {r.category for r in results} == {UNAVAILABLE}
    assert kept["raw"].startswith("judge error: RuntimeError")
