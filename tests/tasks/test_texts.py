"""The texts a task shows the model, in both modes.

``record`` cases pin the prompts of record mode. ``status`` cases pin Unmute's text:
unmute@4a738ad internal/generate/task_prompt.go, livekit_v1_build.go
(delegateReturnFinality) and the golden internal/generate/testdata/golden/livekit_v1_remy.txt.
"""

from typing import Any

import pytest

from tests.tasks.job import filled, job, plan, state
from unmute_core.tasks import (
    EMPTY_REPLY_ATTEMPTS,
    CallStartResults,
    FinishRule,
    empty_reply_note,
)
from unmute_core.tasks._texts import (
    FINISH_DESCRIPTION_RECORD,
    INPUTS_HEADING,
    SERVED_HEADING,
    SPOKEN_FORM,
    delegate_description,
    finish_contract,
    render_instructions,
    via_description,
)

UNMUTE_ESCAPE = (
    "`unserved_request` is for a request this step cannot serve. Do this step's own work "
    "first, and never use it to skip that work: the caller's original reason for being here "
    "is not an unserved request. If a handoff here covers what they want, call that handoff "
    "instead. Only when no tool and no handoff here can serve what the caller is asking, call "
    "`finish` with their request in `unserved_request`, in their own words, rather than "
    "refusing or explaining what you cannot do here. The agent that owns this step reads that "
    "status and takes the caller from there."
)


def test_status_tail_is_unmutes_byte_for_byte() -> None:
    # livekit_v1_remy.txt, prompts/tasks/confirm_booking.md
    assert finish_contract("finish", [], record=False) == (
        "\n\nWhen this step is complete, call `finish`.\n\n" + UNMUTE_ESCAPE
    )
    assert finish_contract("finish", ["tier"], record=False).startswith(
        "\n\nWhen this step is complete, call `finish` with: tier.\n\n`unserved_request`"
    )
    rule = FinishRule(tool="save_booking", success={"status": ("booked",)})
    tail = finish_contract(
        "finish", [], record=False, ends_on={"save_booking": [rule], "find": [rule]}
    )
    assert tail == (
        "\n\nWhen this step is complete, call `finish`.\n\nWhen `find` or `save_booking` returns "
        "a successful result, this step ends by itself and saves it; do not call `finish` after "
        "it. `finish` is for a request this step cannot serve, for values you already hold, and "
        "for recording a result whose save was refused.\n\n" + UNMUTE_ESCAPE
    )


def test_status_delegate_reads_unmutes_docstring() -> None:
    # livekit_v1_remy.txt:703, the do_event delegate
    assert delegate_description(
        "The caller is ready to plan their event; run the events flow.",
        record=False,
        carries_turn=False,
    ) == (
        "The caller is ready to plan their event; run the events flow. When this flow finishes "
        "it returns a status. Continue with the caller. Do not run this flow again for the same "
        "request. A completed status means the step finished. Read any saved values through your "
        "own prompt references. An unserved status means the step could not help. Ask the caller "
        "what they need, then use your tools or a handoff."
    )
    carried = delegate_description("Book it.", record=False, carries_turn=True)
    assert carried.startswith(
        "Book it. The caller's words right above the status are what the step answered; it "
        "served the part it owns, so act only on what remains. When this flow finishes it "
        "returns a status."
    )
    assert delegate_description(None, record=False, carries_turn=False).startswith(
        "Run this flow. When"
    )
    assert via_description("Search.", record=False, carries_turn=False).startswith(
        "Search. When this succeeds, the caller moves into a step that continues from its "
        "result, and the call returns a status. Continue with the caller. A completed status"
    )


@pytest.mark.parametrize(
    ("then", "ending"),
    [
        # unmute@d70247b internal/generate/livekit_v1_build.go:1125
        ({"transfers_to": "billing"}, "when it finishes the caller is handed to the billing."),
        ({"ends_call": True}, "when it finishes the call ends."),
    ],
    ids=["transfer", "end"],
)
def test_a_group_that_ends_or_transfers_says_it_does_not_return_with_unmutes_words(
    then: dict[str, Any], ending: str
) -> None:
    unmute = f"Take the payment. This flow does not return to you: {ending}"
    status = delegate_description("Take the payment.", record=False, carries_turn=False, **then)
    record = delegate_description("Take the payment.", record=True, carries_turn=False, **then)
    for text in (status, record):
        assert text.startswith(
            unmute + " Only a flow that hands the caller back, or fails, returns."
        )
    assert "When this flow finishes it returns" not in status + record


def _render(task_plan: Any, name: str, **kwargs: Any) -> str:
    task = task_plan.tasks[name]
    return render_instructions(
        task,
        record=True,
        finish_name="finish",
        result_names=[row.argument for row in task.assign],
        **kwargs,
    )


def test_instructions_render_saved_values_and_the_record_tail() -> None:
    task_plan = plan()
    call_state = state(task_plan)
    task = task_plan.tasks["find_order"]
    site = "task:find_order"
    prompt = _render(task_plan, "find_order")
    assert prompt.startswith("Find order {{order_id}}.")
    assert filled(prompt, call_state, site).startswith("Find order none recorded yet..")
    call_state.save({"order_id": "A-12"})
    prompt = _render(task_plan, "find_order", ends_on={"lookup_order": task.finish})
    text = filled(prompt, call_state, site)
    assert text.startswith("Find order A-12.")
    assert "When this step is complete, call `finish` with: order_id, status." in text
    assert (
        "When `lookup_order` returns a result where `found` is `true`, this step ends by itself "
        "and saves it; do not call `finish` after it. Any other result leaves this step open: "
        "tell the caller what it means for them, carry on with the step, and call `finish` "
        "yourself once it is complete."
    ) in text
    assert text.endswith("takes the caller from there.\n\n" + SPOKEN_FORM)


def test_record_tail_keeps_unmutes_escape_sentences() -> None:
    tail = finish_contract("finish", [], record=True)
    assert tail.startswith("\n\nWhen this step is complete, call `finish`. Saving")
    assert (
        "`unserved_request` is for a request this step cannot serve. Do this step's own work "
        "first, and never use it to skip that work: the caller's original reason for being "
        "here is not an unserved request."
    ) in tail
    assert (
        "If the caller confirmed this step's values before asking, pass those values in the "
        "same call: they are saved before the caller is handed back."
    ) in tail


def test_record_delegate_describes_the_run_record() -> None:
    assert delegate_description("The caller asks.", record=True, carries_turn=False) == (
        "The caller asks. When this flow finishes, the call returns what it did. Continue with "
        "the caller. Do not run this flow again for the same request. The result lists each "
        "step that ran: what it saved and confirmed, and, for a step that handed the caller "
        "back, what it could not serve, in the caller's words. The caller has already heard "
        "what a step said, so do not repeat it; a step's not_heard is a tool's result the "
        "caller has not heard, so tell them what it means for them. Reply only with what still "
        "needs saying, and speak of what the flow did in the steps' own terms: what a step "
        "saved is recorded, and booked, sent or paid only where a step said so. A completed "
        "status means the flow finished. An unserved status means a step handed the caller back "
        "without doing that request: serve it with your own tools or another task that covers "
        "it, or ask the caller. Until one of them has done it, it is still undone, and that is "
        "what the caller hears from you. To change or undo something a task saved, start the "
        "task that saved it or one that makes that change; a step that can start on its own is "
        "also a task you can start. A failed status means the step broke before it finished and "
        "saved nothing further. Tell the caller it did not go through, then use your tools or a "
        "handoff."
    )
    assert delegate_description(None, record=True, carries_turn=False).startswith(
        "Run this flow. When"
    )
    assert delegate_description("x", record=True, carries_turn=True).endswith(
        "The caller's words right above the result are what the step answered; it served the "
        "part it owns, so act only on what remains."
    )
    assert via_description("Search.", record=True, carries_turn=False).startswith(
        "Search. When this succeeds, the caller moves into a step that continues from its "
        "result, and the call returns what that step did. Continue with the caller. The "
        "result lists each step that ran"
    )


def test_record_finish_says_it_ends_the_step_so_the_closing_line_goes_with_it() -> None:
    assert FINISH_DESCRIPTION_RECORD == (
        "Save this step's values and finish, or hand the caller back with unserved_request, "
        "passing any of this step's values they already confirmed so they are saved first. It "
        "ends the step at once, so say anything the caller should hear in the same reply."
    )
    rules = {"lookup_order": plan().tasks["find_order"].finish}
    tail = finish_contract("finish", ["note"], record=True, ends_on=rules)
    assert tail.startswith(
        "\n\nWhen this step is complete, call `finish` with: note. Saving ends this step at "
        "once, so say any confirmation, what a tool's result means for the caller, or what "
        "happens next in the same reply as that call.\n\nWhen `lookup_order` returns a result "
        "where"
    )
    more = finish_contract("finish", ["note"], record=True, next_step="reminder")
    assert (
        "in the same reply as that call. The call goes on after this step: the reminder step "
        "comes next. Close with what this step did and keep the goodbye for the end of the "
        "call.\n\n"
    ) in more
    assert "The call goes on" not in tail
    assert "call `go_back_to_step`: only that step can change it" in finish_contract(
        "finish", [], record=True, go_back="go_back_to_step"
    )


def test_a_tool_that_ends_the_step_is_named_with_what_it_must_return() -> None:
    tail = finish_contract(
        "finish",
        ["order_id"],
        record=True,
        ends_on={
            "lookup_order": [
                FinishRule(tool="lookup_order", success={"found": (True,), "late": (False,)}),
                FinishRule(tool="lookup_order", success={"status": ("shipped", "delivered")}),
            ],
            "book": [FinishRule(tool="book", success={"booked": (True,)})],
        },
    )
    assert (
        "When `book` returns a result where `booked` is `true` or `lookup_order` returns a "
        "result where `found` is `true` and `late` is `false`, or where `status` is `shipped` "
        "or `delivered`, this step ends by itself"
    ) in tail


def test_a_tool_code_confirms_and_reports_is_one_the_model_never_does_in_words() -> None:
    book = FinishRule(
        tool="book",
        success={"status": ("booked",)},
        confirm=("Shall I book {{input.day}}?",),
        say=("Booked for {{result.day}}.",),
    )
    tail = finish_contract("finish", [], record=True, ends_on={"book": [book]})
    assert (
        "Code asks the caller to confirm before `book` runs, so never ask that yourself: call "
        "it as soon as the caller has given what it needs. It answers that it has not run yet. "
        "When the caller says yes, call it again with the same input. When `book` succeeds, "
        "code tells the caller. Never say yourself that it is done: until it has succeeded, it "
        "is not."
    ) in tail
    plain = finish_contract(
        "finish",
        [],
        record=True,
        ends_on={"book": [book.model_copy(update={"confirm": (), "say": ()})]},
    )
    assert "Code asks" not in plain and "code tells the caller" not in plain


def _inputs_plan() -> Any:
    payload = job()
    payload["groups"][0]["inputs"] = [
        {"kind": "tool", "name": "normalize_timestamp"},
        {"kind": "input", "name": "salon_name", "value": "Luna Salon"},
    ]
    payload["tasks"][1]["inputs"] = [
        {"kind": "memory", "name": "order_id"},
        {"kind": "input", "name": "salon_name", "value": "Luna Salon"},
    ]
    return plan(payload)


def test_inputs_render_after_the_prompt_and_before_the_tail() -> None:
    task_plan = _inputs_plan()
    call_state = state(task_plan)
    results = CallStartResults()
    step = task_plan.tasks["take_note"]

    def render() -> str:
        prompt = _render(
            task_plan,
            "take_note",
            inputs=task_plan.inputs_for(step, "order_flow"),
            call_start=results,
        )
        return filled(prompt, call_state, "task:take_note")

    tail = finish_contract("finish", ["note"], record=True)
    assert render() == (
        "Take a note.\n\n"
        "Your inputs:\n"
        "- normalize_timestamp, from the call start: not available yet\n"
        "- salon_name: Luna Salon\n"
        "- order_id: none recorded yet." + tail
    )
    results.record(
        "normalize_timestamp",
        "[SYSTEM] Background data — never read aloud or share raw data with user.\n"
        "Instructions for the model:\nUse the timezone.\n\n"
        'JSON:\n{"ok": true, "result": {"timezone": "Europe/Madrid"}}',
    )
    call_state.save({"order_id": "A-12"})
    assert render() == (
        "Take a note.\n\n"
        "Your inputs:\n"
        "- normalize_timestamp, from the call start: [SYSTEM] Background data — never read "
        "aloud or share raw data with user.\n"
        "  Instructions for the model:\n"
        "  Use the timezone.\n"
        "\n"
        "  JSON:\n"
        '  {"ok": true, "result": {"timezone": "Europe/Madrid"}}\n'
        "- salon_name: Luna Salon\n"
        "- order_id: A-12" + tail
    )


def test_literal_inputs_are_never_filled_as_templates() -> None:
    payload = job()
    payload["tasks"][1]["inputs"] = [{"kind": "input", "name": "greeting", "value": "Hi {{name}}"}]
    task_plan = plan(payload)
    prompt = _render(
        task_plan, "take_note", inputs=task_plan.inputs_for(task_plan.tasks["take_note"])
    )
    assert "- greeting: Hi { {name}}" in filled(prompt, state(task_plan), "task:take_note")


def test_an_empty_list_memory_input_reads_as_an_empty_list() -> None:
    payload = job()
    payload["tasks"][1]["inputs"] = [{"kind": "memory", "name": "tags"}]
    task_plan = plan(payload)
    prompt = _render(
        task_plan, "take_note", inputs=task_plan.inputs_for(task_plan.tasks["take_note"])
    )
    assert "- tags: []\n" in filled(prompt, state(task_plan), "task:take_note")


def test_a_step_without_inputs_or_earlier_runs_renders_exactly_as_before() -> None:
    task_plan = _inputs_plan()
    task = task_plan.tasks["find_order"]
    before = _render(task_plan, "find_order", ends_on={"lookup_order": task.finish})
    results = CallStartResults()
    results.record("normalize_timestamp", "[SYSTEM] Background data")
    assert (
        _render(
            task_plan,
            "find_order",
            ends_on={"lookup_order": task.finish},
            inputs=task_plan.inputs_for(task),
            call_start=results,
            served="",
        )
        == before
    )
    assert INPUTS_HEADING not in before


def test_what_earlier_runs_did_renders_after_the_inputs() -> None:
    task_plan = _inputs_plan()
    inputs = task_plan.inputs_for(task_plan.tasks["take_note"], "order_flow")
    tail = finish_contract("finish", ["note"], record=True)
    section = SERVED_HEADING + "\n- find_order: saved order_id."
    with_section = _render(task_plan, "take_note", inputs=inputs, served=section)
    without = _render(task_plan, "take_note", inputs=inputs)
    assert with_section == without.removesuffix(tail) + "\n\n" + section + tail


def test_every_empty_reply_retry_reads_a_different_note() -> None:
    """The Responses API reuses an answer when the request has not changed."""
    notes = {
        empty_reply_note(retry=retry, finish_only=finish_only)
        for retry in range(1, EMPTY_REPLY_ATTEMPTS)
        for finish_only in (False, True)
    }
    assert len(notes) == 4
    assert "never ask again" in empty_reply_note(retry=1, finish_only=False)
