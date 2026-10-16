"""A group's step order: in order, stopping, non-blocking steps, and going back."""

from typing import Literal

import pytest

from tests.tasks.job import plan
from unmute_core.tasks import GO_BACK_TOOL_NAME, GroupProgress, SaveOutcome
from unmute_core.tasks._progress import go_back_targets
from unmute_core.tasks._texts import REENTRY_REDO, REENTRY_RESUME

Status = Literal["completed", "unserved", "failed"]


def _progress(*, non_blocking: tuple[str, ...] = (), steps: int = 3) -> GroupProgress:
    task_plan = plan()
    base = task_plan.tasks["take_note"]
    tasks = [base.model_copy(update={"name": f"step_{n}"}) for n in range(steps)]
    group = task_plan.groups["order_flow"].model_copy(
        update={"steps": tuple(task.name for task in tasks), "non_blocking": non_blocking}
    )
    return GroupProgress(group, tasks)


def _run_through(progress: GroupProgress, outcomes: dict[int, Status]) -> list[int]:
    order = []
    while (index := progress.next_step()) is not None:
        order.append(index)
        if progress.record(index, SaveOutcome(outcomes.get(index, "completed"))) == "stop":
            break
    return order


def test_a_group_runs_its_steps_in_order_and_completes() -> None:
    progress = _progress()
    assert _run_through(progress, {}) == [0, 1, 2]
    assert progress.status() == "completed"


def test_an_unserved_step_stops_the_group_even_when_non_blocking() -> None:
    progress = _progress(non_blocking=("step_1",))
    assert _run_through(progress, {1: "unserved"}) == [0, 1]
    assert progress.status() == "unserved"


def test_a_blocking_failure_stops_the_group_and_a_non_blocking_one_does_not() -> None:
    blocking = _progress()
    assert _run_through(blocking, {1: "failed"}) == [0, 1]
    assert blocking.status() == "failed"
    lenient = _progress(non_blocking=("step_1",))
    assert _run_through(lenient, {1: "failed"}) == [0, 1, 2]
    assert lenient.status() == "completed"


def test_going_back_runs_the_targets_then_the_interrupted_step_then_the_rest() -> None:
    progress = _progress(steps=4)
    order = []
    for _ in range(3):
        index = progress.next_step()
        assert index is not None
        order.append(index)
        if index < 2:
            progress.record(index, SaveOutcome("completed"))
    assert progress.go_back(["step_1", "step_0"]) == [1, 0]
    while (index := progress.next_step()) is not None:
        order.append(index)
        assert progress.reentered == (index in (0, 1, 2))
        progress.record(index, SaveOutcome("completed"))
    assert order == [0, 1, 2, 1, 0, 2, 3]


def test_going_back_offers_only_entered_steps_other_than_the_running_one() -> None:
    progress = _progress()
    progress.next_step()
    assert progress.go_back_tool() is None
    progress.record(0, SaveOutcome("completed"))
    progress.next_step()
    tool = progress.go_back_tool()
    assert tool is not None and tool["name"] == GO_BACK_TOOL_NAME
    assert tool["parameters"]["properties"]["steps"]["items"]["enum"] == ["step_0"]
    assert '"step_0": "step 0. It saves: note."' in tool["description"]
    with pytest.raises(ValueError, match="cannot go back to 'step_2'"):
        progress.go_back(["step_2"])
    with pytest.raises(ValueError, match="at least one"):
        progress.go_back([])
    assert progress.current == 1


def test_going_back_twice_never_offers_a_step_waiting_to_resume() -> None:
    progress = _progress()
    for _ in range(3):
        index = progress.next_step()
        assert index is not None
        if index < 2:
            progress.record(index, SaveOutcome("completed"))
    assert progress.go_back(["step_0"]) == [0]
    assert progress.next_step() == 0 and progress.reentered
    tool = progress.go_back_tool()
    assert tool is not None
    assert tool["parameters"]["properties"]["steps"]["items"]["enum"] == ["step_1"]
    with pytest.raises(ValueError, match="cannot go back to 'step_2'"):
        progress.go_back(["step_2"])
    assert progress.go_back(["step_1"]) == [1]
    order = []
    while (index := progress.next_step()) is not None:
        order.append(index)
        progress.record(index, SaveOutcome("completed"))
    assert order == [1, 0, 2]
    assert progress.status() == "completed"


def test_going_back_waits_for_the_caller_to_speak_in_the_step() -> None:
    progress = _progress()
    progress.next_step()
    progress.mark_opened(caller_turns=2)
    assert not progress.may_go_back(2)
    assert progress.may_go_back(3)


def test_a_step_entered_again_is_told_why_and_ends_unserved_only_on_a_new_turn() -> None:
    progress = _progress()
    progress.next_step()
    progress.mark_opened(caller_turns=1)
    assert progress.reentry_note() is None
    assert not progress.refuses_unserved(1)
    progress.record(0, SaveOutcome("completed"))
    progress.next_step()
    progress.mark_opened(caller_turns=1)
    assert progress.go_back(["step_0"]) == [0]
    assert progress.next_step() == 0
    progress.mark_opened(caller_turns=2)
    assert progress.reentry_note() == REENTRY_REDO
    assert progress.refuses_unserved(2)
    assert not progress.refuses_unserved(3)
    progress.record(0, SaveOutcome("completed"))
    assert progress.next_step() == 1
    progress.mark_opened(caller_turns=3)
    assert progress.reentry_note() == REENTRY_RESUME
    assert progress.refuses_unserved(3)
    progress.record(1, SaveOutcome("completed"))
    assert progress.next_step() == 2
    assert progress.reentry_note() is None


def test_a_step_gone_back_from_before_its_redo_ended_still_redoes() -> None:
    progress = _progress()
    for _ in range(3):
        index = progress.next_step()
        assert index is not None
        if index < 2:
            progress.record(index, SaveOutcome("completed"))
    progress.go_back(["step_0"])
    assert progress.next_step() == 0
    progress.go_back(["step_1"])
    notes = []
    while (index := progress.next_step()) is not None:
        notes.append((index, progress.reentry_note()))
        progress.record(index, SaveOutcome("completed"))
    assert notes == [(1, REENTRY_REDO), (0, REENTRY_REDO), (2, REENTRY_RESUME)]


def test_go_back_targets_read_the_steps_argument() -> None:
    assert go_back_targets({"steps": ["a", "b"]}) == ["a", "b"]
    assert go_back_targets({"steps": "a"}) == ["a"]
    assert go_back_targets({"steps": [1, "a"]}) == ["a"]
    assert go_back_targets(None) == []


def test_a_refused_go_back_names_the_steps_it_can_go_back_to() -> None:
    progress = _progress()
    progress.next_step()
    progress.record(0, SaveOutcome("completed"))
    progress.next_step()
    with pytest.raises(ValueError, match=r"cannot go back to 'step_2'. Go back to one of: step_0"):
        progress.go_back(["step_2"])


@pytest.mark.parametrize(
    ("arguments", "targets"),
    [
        ({"steps": "a"}, ["a"]),
        ({"steps": ["a", 3, "b"]}, ["a", "b"]),
        ({"steps": 3}, []),
        ({}, []),
        ("steps", []),
    ],
)
def test_go_back_targets_read_what_the_model_wrote(arguments: object, targets: list[str]) -> None:
    assert go_back_targets(arguments) == targets
