"""``DevRecorder``: the records of one call, and that each one is a record the reader takes."""

import json
from typing import Any

import pytest

from tests.dev.contract import problems
from tests.state.unmute_state import State
from unmute_core.dev import (
    DevRecorder,
    call_id_from,
    known_name,
    metrics_enabled,
    separator,
    stdout_sink,
)
from unmute_core.state import CallState, StateOrigin


def valid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check every record against the contract and return them."""
    assert [problems(json.dumps(record)) for record in records] == [[] for _ in records]
    return records


def test_the_call_record_opens_the_call_and_the_end_closes_it() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    recorder.end_call(reason="hung up")
    first, last = valid(sent)[0], sent[-1]
    assert first["call"] == {
        "target": "pipecat",
        "state": "open",
        "input_boundaries": "partial",
        "model_calls": "known",
        "generated_text": "available",
    }
    assert (last["call"]["state"], last["call"]["reason"], last["revision"]) == (
        "ended",
        "hung up",
        3,
    )
    assert all(record["order"] == 1 for record in sent)


def test_sending_a_record_again_raises_its_revision_and_keeps_its_order() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    recorder.put("exchange", "input-1", {"role": "input", "state": "open", "link_status": "known"})
    recorder.update("exchange", "input-1", state="ended")
    recorder.update("exchange", "input-1", state="ended")
    recorder.update("exchange", "nothing", state="ended")
    valid(sent)
    assert [(r["id"], r["revision"], r["order"]) for r in sent[1:]] == [
        ("input-1", 1, 2),
        ("input-1", 2, 2),
    ]


def test_nothing_is_sent_after_the_end() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    recorder.end_call()
    count = len(sent)
    recorder.end_call()
    recorder.put("exchange", "late", {"role": "input", "state": "open", "link_status": "known"})
    assert recorder.closed
    assert len(sent) == count


def test_a_sink_that_raises_never_fails_the_call() -> None:
    def broken(record: dict[str, Any]) -> None:
        raise OSError("closed")

    recorder = DevRecorder("call-1", broken)
    recorder.measure("m", "first_response", 0.2, "x")
    recorder.end_call()
    assert recorder.closed


@pytest.mark.parametrize("call_id", ["", "has space", "a" * 129, "dot.ted"])
def test_a_call_id_the_reader_refuses_is_refused_early(call_id: str) -> None:
    with pytest.raises(ValueError, match="dev call id"):
        DevRecorder(call_id)


def test_a_measurement_is_measured_pending_or_unavailable_and_never_zero_when_absent() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    recorder.put("operation", "llm-1", {"type": "llm", "name": "LLM", "state": "running"})
    kw: dict[str, Any] = {"scope": "operation", "operation_id": "llm-1"}
    recorder.measure("a", "first_response", 0.25, "s", **kw)
    recorder.measure("b", "first_response", None, "s", pending=True, **kw)
    recorder.measure("c", "first_response", None, "s", reason="never came", **kw)
    recorder.measure("d", "first_response", float("nan"), "s", **kw)
    recorder.measure("e", "reply_latency", -1, "s")
    valid(sent)
    by_id = {r["id"]: r["measurement"] for r in sent if r["kind"] == "measurement"}
    assert [(i, m["state"], m.get("value")) for i, m in by_id.items()] == [
        ("a", "measured", 0.25),
        ("b", "pending", None),
        ("c", "unavailable", None),
        ("d", "unavailable", None),
        ("e", "unavailable", None),
    ]


def test_ending_the_call_settles_what_is_still_open() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    recorder.put(
        "exchange",
        "response-1",
        {"role": "response", "state": "open", "link_status": "unavailable"},
    )
    recorder.put(
        "operation",
        "llm-1",
        {"exchange_id": "response-1", "type": "llm", "name": "LLM", "state": "running"},
    )
    recorder.measure(
        "llm-1:first",
        "first_response",
        None,
        "s",
        scope="operation",
        operation_id="llm-1",
        pending=True,
    )
    recorder.put(
        "exchange",
        "response-2",
        {"role": "response", "state": "open", "link_status": "unavailable"},
    )
    for kind, identity, payload in [
        (
            "operation",
            "tts-1",
            {"exchange_id": "response-2", "type": "tts", "name": "TTS", "state": "ended"},
        ),
        (
            "text",
            "t",
            {
                "exchange_id": "response-2",
                "message_id": "t",
                "speaker": "assistant",
                "text": "Hi",
                "state": "final",
                "origin": "generated",
                "separator_before": "",
            },
        ),
    ]:
        recorder.put(kind, identity, payload)
    recorder.end_call(error=True)
    valid(sent)
    final = {(r["kind"], r["id"]): r[r["kind"]] for r in sent}
    assert final["operation", "llm-1"]["state"] == "incomplete"
    assert final["exchange", "response-1"]["state"] == "incomplete"
    assert final["exchange", "response-2"]["state"] == "ended"
    assert final["measurement", "llm-1:first"]["state"] == "unavailable"
    assert final["call", "call"]["state"] == "error"


def test_a_state_change_is_a_state_record_with_a_long_value_cut_at_4_kib() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    state = CallState(State, confirm={"caller_phone": "confirm_number"})
    state.save({"count": 1}, origin=StateOrigin("prefetch", "today"))
    recorder.watch_state(state)
    state.save({"caller_phone": "+34600111222"})
    state.save({"caller_phone": "+34600111222"}, step="confirm_number")
    valid(sent)
    rows = [r["state"] for r in sent if r["kind"] == "state"]
    assert rows[0] == {
        "origin": {"kind": "start", "name": ""},
        "changes": [{"field": "count", "value": 1, "by": {"kind": "prefetch", "name": "today"}}],
    }
    assert rows[1]["changes"][0]["confirmed"] is False
    assert rows[2]["changes"][0]["confirmed"] is True
    assert [r["id"] for r in sent if r["kind"] == "state"] == ["state-1", "state-2", "state-3"]


def test_the_save_tool_and_an_event_tool_get_origins_the_reader_takes() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    state = CallState(State)
    recorder.watch_state(state)
    state.save({"count": 1}, origin=StateOrigin("tool", "save_values"))
    state.save({"count": 2}, origin=StateOrigin("event", "lookup_customer"))
    rows = [r["state"] for r in valid(sent) if r["kind"] == "state"]
    assert [(row["origin"], row["changes"][0]["by"]) for row in rows] == [
        ({"kind": "step", "name": "save_values"},) * 2,
        ({"kind": "prefetch", "name": "lookup_customer"},) * 2,
    ]


def test_a_value_over_4_kib_is_a_cut_string_with_truncated_set() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    state = CallState(State)
    recorder.watch_state(state)
    state.save({"caller_reason": ["create_booking"] * 600}, step="s")
    state.save({"count": 7}, step="s")
    valid(sent)
    long, short = (r["state"]["changes"][0] for r in sent if r["kind"] == "state")
    assert long["truncated"] is True
    assert isinstance(long["value"], str) and len(long["value"].encode()) == 4096
    assert "truncated" not in short and short["value"] == 7


def test_task_events_become_one_handoff_operation_per_run() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    recorder.response = "response-1"
    run = {"task_run_id": "r1"}
    recorder.task_event("task.started", {"task": "verify", **run})
    recorder.task_event("task.value_refused", {"task": "verify", **run})
    recorder.task_event("task.completed", {"task": "verify", **run})
    recorder.task_event("group.started", {"group": "booking", "task_run_id": "r2"})
    recorder.task_event("task.started", {"task": "ask", "group": "booking", "task_run_id": "r2"})
    recorder.task_event("task.failed", {"task": "ask", "group": "booking", "task_run_id": "r2"})
    recorder.task_event("group.stopped", {"group": "booking", "task_run_id": "r2"})
    recorder.task_event("task.started", {"task": "pay", "task_run_id": "r3"})
    recorder.task_event("task.failed", {"task": "pay", "task_run_id": "r3"})
    valid(sent)
    ops = {r["id"]: r["operation"] for r in sent if r["kind"] == "operation"}
    assert [(o["name"], o["state"], o.get("exchange_id")) for o in ops.values()] == [
        ("verify", "returned", "response-1"),
        ("booking", "cancelled", "response-1"),
        ("pay", "failed", "response-1"),
    ]
    assert ops["handoff-1"]["type"] == "handoff"
    assert "reason" in ops["handoff-3"]


def test_the_default_sink_prints_a_sentinel_line(capsys: pytest.CaptureFixture[str]) -> None:
    DevRecorder("call-1")
    line = capsys.readouterr().out.strip()
    assert line.startswith("UNMUTE_METRIC {")
    assert problems(line.removeprefix("UNMUTE_METRIC ")) == []
    stdout_sink({"kind": "call"})


def test_the_dev_loop_switch_and_call_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNMUTE_DEV_METRICS", raising=False)
    assert not metrics_enabled()
    monkeypatch.setenv("UNMUTE_DEV_METRICS", "0")
    assert not metrics_enabled()
    monkeypatch.setenv("UNMUTE_DEV_METRICS", "1")
    assert metrics_enabled()
    assert call_id_from({"unmute_dev_call_id": "abc-1"}) == "abc-1"
    assert call_id_from({"unmute_dev_call_id": "no good"}) is None
    assert call_id_from(None) is None


def test_a_bad_dev_switch_warns_and_stays_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("UNMUTE_DEV_METRICS", "false")
    assert not metrics_enabled()
    monkeypatch.setenv("UNMUTE_DEV_METRICS", "2")
    assert not metrics_enabled()
    assert "UNMUTE_DEV_METRICS must be one of" in caplog.text


def test_a_control_tool_is_a_handoff_operation_with_no_tool_duration() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    response = recorder.open_response()
    operation = recorder.control_started("transfer_to_sales", exchange_id=response, parent="llm-9")
    recorder.control_ended(operation, "failed", "The transfer failed.")
    latest = {(r["kind"], r["id"]): r[r["kind"]] for r in valid(sent)}
    assert latest["operation", operation] == {
        "type": "handoff",
        "name": "transfer_to_sales",
        "state": "failed",
        "exchange_id": response,
        "parent_operation_id": "llm-9",
        "reason": "The transfer failed.",
    }
    assert not [k for k in latest if k[0] == "measurement"]


def test_limited_marks_the_call_record() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    recorder.limited(model_calls="partial", generated_text="limited")
    call = valid(sent)[-1]["call"]
    assert (call["model_calls"], call["generated_text"]) == ("partial", "limited")


def test_the_builders_open_exchanges_and_text_rows() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    heard = recorder.open_input()
    answer = recorder.open_response(heard)
    follow_up = recorder.direct_text(answer, "Hello.", interrupted=True)
    assert recorder.response == answer
    loose = recorder.open_response()
    latest = {(r["kind"], r["id"]): r[r["kind"]] for r in valid(sent)}
    assert latest["exchange", heard] == {"role": "input", "state": "open", "link_status": "known"}
    assert latest["exchange", answer] == {
        "role": "response",
        "state": "open",
        "playback": "not_started",
        "link_status": "known",
        "input_id": heard,
    }
    assert latest["text", follow_up]["state"] == "incomplete"
    assert latest["exchange", loose]["link_status"] == "unavailable"


def test_separator_and_known_name() -> None:
    assert separator("Hello", "there") == " "
    assert separator("Hello ", "there") == ""
    assert separator("Hello", "there", spacing=True) == ""
    assert separator("", "there") == ""
    assert (known_name("gpt-4.1"), known_name("unknown"), known_name(None)) == (
        "gpt-4.1",
        None,
        None,
    )


def test_close_response_leaves_no_response_open() -> None:
    recorder = DevRecorder("call-1", lambda record: None)
    recorder.open_response()
    assert recorder.response == "response-1"
    recorder.close_response()
    assert recorder.response is None


def test_a_dev_switch_with_spaces_around_the_word_still_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNMUTE_DEV_METRICS", " On ")
    assert metrics_enabled()


def test_an_operation_keeps_its_keys_in_order_and_leaves_out_what_is_empty() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    exchange = recorder.open_response()
    plain = recorder.operation("llm", "gpt", exchange_id=exchange, model="m", provider="")
    first = recorder.operation(
        "tts", "voice", exchange_id=exchange, model="m", provider="p", provider_first=True
    )
    keys = {
        record["id"]: list(record["operation"]) for record in valid(sent) if "operation" in record
    }
    assert keys[plain] == ["type", "name", "state", "exchange_id", "model"]
    assert keys[first] == ["type", "name", "state", "provider", "model", "exchange_id"]


def test_a_breakdown_with_a_bad_part_is_dropped_and_uses_no_id() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    part = {
        "key": "wait",
        "label": "wait",
        "owner": "vad",
        "owner_kind": "setting",
        "start_time": 1.0,
        "duration_secs": 0.2,
    }
    for bad in ({"duration_secs": -1.0}, {"start_time": float("nan")}, {"label": ""}):
        recorder.breakdown("user_silence", 0.5, [part, part | bad])
    recorder.breakdown(None, 0.5, [part])
    recorder.breakdown("user_silence", 0.5, [])
    assert [record["kind"] for record in sent] == ["call"]
    recorder.breakdown("user_silence", 0.2, [part])
    sent_breakdown = valid(sent)[-1]
    assert sent_breakdown["id"] == "breakdown-1"
    assert list(sent_breakdown["breakdown"]) == ["measured_from", "total_secs", "parts"]
    assert list(sent_breakdown["breakdown"]["parts"][0]) == list(part)


def test_answer_text_grows_row_by_row_and_settles_with_its_request() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    exchange = recorder.open_response()
    operation = recorder.operation("llm", "gpt", exchange_id=exchange)
    recorder.pending_request(operation, exchange, "node")
    row = recorder.new_id("answer-text")
    for chunk in ("Hello", "there", "friend"):
        recorder.append_text(row, exchange, chunk, spacing=False)
    texts = [r["text"]["text"] for r in valid(sent) if r["kind"] == "text"]
    assert texts == ["Hello", "Hello there", "Hello there friend"]
    recorder.settle_request(
        operation,
        "ended",
        timings={"first_response": 0.1, "request_duration": 0.4},
        source="node",
        unavailable="never",
        answer=row,
    )
    latest = {(r["kind"], r["id"]): r[r["kind"]] for r in valid(sent)}
    assert latest["text", row]["state"] == "final"
    assert latest["operation", operation]["state"] == "ended"
    assert latest["measurement", f"{operation}:first_response"]["value"] == 0.1
    assert latest["measurement", f"{operation}:request_duration"]["value"] == 0.4


def test_a_request_that_did_not_end_leaves_its_timings_unavailable_with_the_given_reason() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    exchange = recorder.open_response()
    operation = recorder.operation("llm", "gpt", exchange_id=exchange)
    recorder.pending_request(operation, exchange, "node")
    recorder.settle_request(
        operation,
        "cancelled",
        timings={"request_duration": 0.4},
        source="node",
        unavailable="Cut off.",
    )
    latest = {(r["kind"], r["id"]): r[r["kind"]] for r in valid(sent)}
    for metric in ("first_response", "request_duration"):
        assert latest["measurement", f"{operation}:{metric}"]["state"] == "unavailable"
        assert latest["measurement", f"{operation}:{metric}"]["reason"] == "Cut off."


def test_user_text_joins_segments_and_typed_text_waits_for_an_empty_input() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    spoken = recorder.open_input()
    recorder.user_text(spoken, "Hello", final=False)
    recorder.user_text(spoken, "Hello", final=True)
    recorder.user_text(spoken, "there", final=True)
    recorder.typed_text(spoken, "ignored")
    typed = recorder.open_input()
    recorder.typed_text(typed, "  ")
    recorder.typed_text(typed, "Typed words")
    recorder.typed_text(typed, "Again")
    rows = [r["text"] for r in valid(sent) if r["kind"] == "text"]
    assert [(r["exchange_id"], r["text"], r["state"], r["separator_before"]) for r in rows] == [
        (spoken, "Hello", "provisional", ""),
        (spoken, "Hello", "final", ""),
        (spoken, "there", "final", " "),
        (typed, "Typed words", "final", ""),
    ]
    assert rows[0]["message_id"] == rows[1]["message_id"] == rows[2]["message_id"]
    assert rows[3]["origin"] == "transcription"


def test_a_tool_is_timed_and_a_control_tool_is_not() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    exchange = recorder.open_response()
    tool = recorder.tool_started("lookup", exchange_id=exchange, parent=None, control=False)
    control = recorder.tool_started("transfer", exchange_id=exchange, parent=None, control=True)
    recorder.tool_ended(tool, "returned", duration=0.3, source="s", control=False)
    recorder.tool_ended(control, "failed", duration=0.3, source="s", control=True, reason="No.")
    latest = {(r["kind"], r["id"]): r[r["kind"]] for r in valid(sent)}
    assert (tool, control) == ("tool-2", "handoff-3")
    assert latest["operation", tool]["state"] == "returned"
    assert latest["measurement", f"{tool}:duration"]["metric"] == "tool_duration"
    assert latest["operation", control]["reason"] == "No."
    assert ("measurement", f"{control}:duration") not in latest


def test_an_exchange_settles_by_whether_it_was_cut_and_whether_it_spoke() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    ids = [recorder.open_response() for _ in range(3)]
    recorder.settle_exchange(ids[0], cut=False, spoke=True)
    recorder.settle_exchange(ids[1], cut=True, spoke=True)
    recorder.settle_exchange(ids[2], cut=True, spoke=False)
    latest = {r["id"]: r["exchange"] for r in valid(sent) if r["kind"] == "exchange"}
    assert [(latest[i]["state"], latest[i]["playback"]) for i in ids] == [
        ("ended", "ended"),
        ("interrupted", "interrupted"),
        ("interrupted", "unknown"),
    ]


def test_a_measurement_that_is_not_a_finite_time_is_unavailable() -> None:
    sent: list[dict[str, Any]] = []
    recorder = DevRecorder("call-1", sent.append)
    for index, value in enumerate((0, 2, 0.5, -1, float("nan"), float("inf"), True, None)):
        recorder.measure(f"m-{index}", "first_response", value, "s", scope="call")  # type: ignore[arg-type]
    states = [r["measurement"] for r in valid(sent)[1:]]
    assert [m["state"] for m in states] == ["measured"] * 3 + ["unavailable"] * 5
    assert [m.get("value") for m in states[:3]] == [0, 2, 0.5]
