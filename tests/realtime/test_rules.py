"""What a realtime model can do, and the instructions it reads, on any framework."""

import pytest

from tests.tasks.job import State
from unmute_core.handoff import HandoffContext
from unmute_core.hooks import Hooks, PromptInfo
from unmute_core.model import MissingVariablesError
from unmute_core.realtime import (
    check_realtime,
    check_realtime_handoff,
    realtime_instructions,
    say_line,
)
from unmute_core.state import CallState


def test_a_fixed_line_becomes_an_instruction_to_say_it() -> None:
    assert say_line("Hello.") == (
        "Say this to the caller now, in these words as far as you can: Hello."
    )


def test_a_plain_agent_runs_on_any_realtime_model() -> None:
    check_realtime(fixed_instructions=False, save_values=True, hooks=Hooks())
    check_realtime(fixed_instructions=True)


def test_tasks_are_refused() -> None:
    with pytest.raises(ValueError, match="tasks do not run on a realtime model"):
        check_realtime(fixed_instructions=False, tasks=True)


@pytest.mark.parametrize("point", ["before_model", "wrap_model", "after_model"])
def test_model_hooks_are_refused_and_prompt_hooks_are_not(point: str) -> None:
    hooks = Hooks()
    hooks.prompt(lambda text, _info: text)
    check_realtime(fixed_instructions=False, hooks=hooks)
    getattr(hooks, point)(lambda *_args: None)
    with pytest.raises(ValueError, match=f"the {point} hooks never run"):
        check_realtime(fixed_instructions=False, hooks=hooks)


def test_save_values_needs_instructions_that_change() -> None:
    with pytest.raises(ValueError, match="save_values needs instructions that change"):
        check_realtime(fixed_instructions=True, save_values=True)


def test_a_handoff_keeps_the_whole_conversation() -> None:
    check_realtime_handoff(HandoffContext(history="full"), fixed_instructions=False)
    for context in (
        HandoffContext(),
        HandoffContext(history="summary"),
        HandoffContext(history="last_n", max_messages=4),
        HandoffContext(history="full", include_tool_calls=False),
    ):
        with pytest.raises(ValueError, match='Pass HandoffContext\\(history="full"\\)'):
            check_realtime_handoff(context, fixed_instructions=False)


def test_a_model_with_fixed_instructions_hands_off_nothing() -> None:
    with pytest.raises(ValueError, match="cannot hand the call to another agent"):
        check_realtime_handoff(HandoffContext(history="full"), fixed_instructions=True)


def test_the_instructions_are_filled_hooked_and_followed_by_their_blocks() -> None:
    state = CallState(State)
    state.save({"order_id": "A-12"})
    hooks = Hooks()
    seen: list[PromptInfo] = []

    @hooks.prompt
    def shout(text: str, info: PromptInfo) -> str:
        seen.append(info)
        return text + " Be brief."

    text = realtime_instructions(
        "Help with order {{order_id}}.",
        state,
        hooks=hooks,
        model="gpt-realtime",
        blocks=("Notes: none.", None, "Values: order_id"),
    )

    assert text == "Help with order A-12. Be brief.\n\nNotes: none.\n\nValues: order_id"
    assert seen == [PromptInfo(kind="agent", site="", model="gpt-realtime")]


def test_a_value_the_state_lacks_is_an_error() -> None:
    with pytest.raises(MissingVariablesError):
        realtime_instructions("Help {{nobody}}.", CallState(State))
