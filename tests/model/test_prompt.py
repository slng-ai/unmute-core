"""The prompt fill follows the router's rule, and the variables come from the call state."""

import copy

import pytest

from tests.state.unmute_state import State
from unmute_core.model import (
    MissingVariablesError,
    fill_messages,
    render_prompt,
    router_messages,
    template_variables,
)
from unmute_core.model._prompt import folded_system
from unmute_core.state import CallState


def system(text: str) -> dict:
    return {"role": "system", "content": text}


@pytest.mark.parametrize(
    ("text", "variables", "expected"),
    [
        ("Hi {{name}}.", {"name": "Ana"}, "Hi Ana."),
        ("Hi {{ name }}.", {"name": "Ana"}, "Hi Ana."),
        ("Hi {{\n name\t}}.", {"name": "Ana"}, "Hi Ana."),
        ("{{a}}{{a}} {{B_2}}", {"a": "x", "B_2": "y"}, "xx y"),
        ("Keep {{a-b}} and {name} and ${name}.", {}, "Keep {{a-b}} and {name} and ${name}."),
        ("A path: {{a.b}} {{a__b}}", {"a__b": "x"}, "A path: x x"),
        ("One pass: {{a}}", {"a": "{{b}}", "b": "no"}, "One pass: {{b}}"),
        ("Backslash: {{a}}", {"a": r"\1 \n"}, r"Backslash: \1 \n"),
        ("No placeholders.", {"unused": "fine"}, "No placeholders."),
    ],
)
def test_fill_follows_the_router_rule(text: str, variables: dict, expected: str) -> None:
    assert fill_messages([system(text)], variables) == [system(expected)]


def test_only_system_and_developer_text_is_filled() -> None:
    image = {"type": "image_url", "image_url": {"url": "{{a}}"}}
    messages = [
        system("S {{a}}"),
        {"role": "developer", "content": [{"type": "text", "text": "D {{a}}"}, image]},
        {"role": "user", "content": "U {{a}}"},
        {"role": "assistant", "content": "A {{a}}"},
        {"role": "tool", "content": "T {{a}}", "tool_call_id": "1"},
    ]
    before = copy.deepcopy(messages)

    filled = fill_messages(messages, {"a": "x"})

    assert messages == before
    assert filled == [
        system("S x"),
        {"role": "developer", "content": [{"type": "text", "text": "D x"}, image]},
        *before[2:],
    ]


def test_missing_names_are_refused_sorted_like_the_router_422() -> None:
    with pytest.raises(MissingVariablesError, match=r"variables: a, b$") as raised:
        fill_messages([system("{{b}} {{a}} {{b}} {{c}}")], {"c": "x"})
    assert raised.value.names == ["a", "b"]


def test_a_placeholder_in_a_user_message_needs_no_value() -> None:
    user = {"role": "user", "content": "{{a}}"}
    assert fill_messages([user], {}) == [user]


@pytest.fixture
def state() -> CallState:
    state = CallState(State, confirm={"caller_phone": "confirm_number"})
    state.save({"count": 2, "caller_phone": "+34600111222"})
    state.save(
        {
            "last_appointment": {
                "scheduled_date": "2026-09-12",
                "scheduled_time": "10:00",
                "appointment_type": "haircut",
            }
        }
    )
    return state


def test_variables_render_from_the_state(state: CallState) -> None:
    messages = [
        system("{{count}} {{caller_phone}} {{last_appointment__scheduled_date}} {{reminder_email}}")
    ]

    assert template_variables(messages, state) == {
        "count": "2",
        "caller_phone": "none recorded yet.",
        "last_appointment__scheduled_date": "2026-09-12",
        "reminder_email": "none recorded yet.",
    }
    at_task = template_variables(messages, state, "task:confirm_number")
    assert at_task["caller_phone"] == "+34600111222"


def test_both_path_forms_read_the_state(state: CallState) -> None:
    messages = [
        system("{{last_appointment.scheduled_date}} {{ last_appointment__scheduled_date }}")
    ]
    variables = template_variables(messages, state)
    assert variables == {"last_appointment__scheduled_date": "2026-09-12"}
    assert fill_messages(messages, variables) == [system("2026-09-12 2026-09-12")]


def test_a_prefixed_name_reads_only_a_value_of_that_name(state: CallState) -> None:
    with pytest.raises(MissingVariablesError, match="rtv_count"):
        template_variables([system("{{rtv_count}}")], state)


def test_the_router_gets_every_path_in_its_own_form() -> None:
    messages = [
        system("{{ a.b }} and {{c}}"),
        {"role": "developer", "content": [{"type": "text", "text": "{{d.e.f}}"}]},
        {"role": "user", "content": "{{a.b}}"},
    ]
    assert router_messages(messages) == [
        system("{{a__b}} and {{c}}"),
        {"role": "developer", "content": [{"type": "text", "text": "{{d__e__f}}"}]},
        {"role": "user", "content": "{{a.b}}"},
    ]


def test_only_names_the_prompt_uses_are_sent(state: CallState) -> None:
    messages = [system("No placeholders."), {"role": "user", "content": "{{count}}"}]
    assert template_variables(messages, state) == {}
    assert template_variables([system("Hi")], None) == {}


def test_a_name_the_state_lacks_is_refused(state: CallState) -> None:
    with pytest.raises(MissingVariablesError, match=r"unknown, wrong$"):
        template_variables([system("{{count}} {{wrong}} {{unknown}}")], state)
    with pytest.raises(MissingVariablesError, match=r"count$"):
        template_variables([system("{{count}}")], None)


def test_render_prompt_fills_one_prompt_from_the_state(state: CallState) -> None:
    prompt = "You have {{count}} visits. Last: {{last_appointment.scheduled_date}}."
    assert render_prompt(prompt, state) == "You have 2 visits. Last: 2026-09-12."
    assert render_prompt("{{caller_phone}}", state, "task:confirm_number") == "+34600111222"
    assert render_prompt("No placeholders.", None) == "No placeholders."
    with pytest.raises(MissingVariablesError, match="wrong"):
        render_prompt("{{wrong}}", state)


def test_folded_system_joins_system_and_developer_text_in_order() -> None:
    messages = [
        {"role": "user", "content": "a"},
        {"role": "system", "content": "one"},
        {"role": "assistant", "content": "b"},
        {"role": "developer", "content": [{"type": "text", "text": "two"}, {"type": "image"}]},
    ]
    assert folded_system(messages) == [
        {"role": "system", "content": "one\n\ntwo"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_folded_system_leaves_one_system_message_alone() -> None:
    messages = [{"role": "system", "content": "one"}, {"role": "user", "content": "a"}]
    assert folded_system(messages) == messages
