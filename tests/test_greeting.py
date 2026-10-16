"""How a call opens: Unmute's ``conversation.greeting`` rules, and the line it says."""

import pytest
from pydantic import BaseModel

from unmute_core.greeting import GREETING_HOLD_SECONDS, GREETING_INSTRUCTION, Greeting
from unmute_core.prefetch import CallValue, Clock, Prefetch
from unmute_core.state import CallState


class Customer(BaseModel):
    name: str | None = None
    today: str | None = None
    profile: dict[str, str] | None = None


PREFETCH = Prefetch(
    [
        CallValue("who", "calling_number", ["name"]),
        Clock("clock", "Europe/Madrid", {"today": "date"}),
    ]
)


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"speaks_first": "user", "text": "Hi"}, "requires speaks_first: agent"),
        ({"text": "Hi {{ name"}, "not {{name}}"),
    ],
)
def test_a_bad_greeting_is_refused(fields, error):
    with pytest.raises(ValueError, match=error):
        Greeting(**fields)


def test_a_line_may_name_only_values_prefetch_or_call_start_fills():
    Greeting(text="Hi {{name}}, today is {{today}}.").check(PREFETCH)
    Greeting(text="You are calling from {{call.calling_number}}.").check(None)
    Greeting(text="Hi {{customer_name}}.").check(None, {"customer_name": "required"})
    with pytest.raises(ValueError, match=r"names \{\{profile.city\}\}, which neither"):
        Greeting(text="Hi from {{profile.city}}").check(PREFETCH)
    with pytest.raises(ValueError, match="neither a prefetch entry nor call_start fills"):
        Greeting(text="Hi {{name}}").check(None)


def test_the_line_is_said_with_its_values_in_either_placeholder_form():
    state = CallState(Customer)
    state.save({"name": "Ana", "profile": {"city": "Madrid"}})
    assert Greeting(text="Hi {{name}} from {{profile__city}}.").line(state) == (
        "Hi Ana from Madrid."
    )


def test_a_line_whose_value_is_still_empty_is_worded_by_the_model_instead():
    assert Greeting(text="Hi {{name}}.").line(CallState(Customer)) is None
    assert Greeting(text="Hi {{name}}.").line(None) is None


def test_a_line_whose_value_waits_for_confirmation_is_worded_by_the_model_instead():
    state = CallState(Customer, confirm={"name": "verify_name"})
    state.save({"name": "Ana"})
    assert Greeting(text="Hi {{name}}.").line(state) is None


def test_no_text_is_worded_by_the_model_and_a_caller_first_call_says_nothing():
    assert Greeting().line(None) is None
    assert Greeting().instruction == GREETING_INSTRUCTION
    assert Greeting(speaks_first="user").line(None) is None
    assert Greeting(text="Hello!").line(None) == "Hello!"


def test_a_greeting_holds_the_callers_audio_for_at_most_twenty_seconds() -> None:
    assert GREETING_HOLD_SECONDS == 20.0
