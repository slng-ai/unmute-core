"""Numbers and SIP addresses in the forms each dial or transfer takes."""

import pytest

from unmute_core.telephony import (
    check_destination,
    is_e164,
    refer_uri,
    sip_reason,
    sip_uri,
    sip_user,
)


@pytest.mark.parametrize(
    ("number", "valid"),
    [
        ("+34910000000", True),
        ("+14155550123", True),
        ("+123456789012345", True),
        ("+1234567", True),
        ("+12", True),
        ("+1", False),
        ("+1234567890123456", False),
        ("34910000000", False),
        ("+0910000000", False),
        ("+3491000000a", False),
        (None, False),
    ],
)
def test_e164_is_a_plus_then_2_to_15_digits(number: object, valid: bool) -> None:
    assert is_e164(number) is valid


@pytest.mark.parametrize(
    ("destination", "uri"),
    [
        ("+34910000000", "tel:+34910000000"),
        ("tel:+34910000000", "tel:+34910000000"),
        ("sip:+34910000000@trunk.example", "sip:+34910000000@trunk.example"),
        ("sips:desk@pbx.example", "sips:desk@pbx.example"),
    ],
)
def test_a_refer_takes_a_uri_and_a_bare_number_gets_tel(destination: str, uri: str) -> None:
    assert refer_uri(destination) == uri


@pytest.mark.parametrize(
    ("destination", "user"),
    [
        ("+34910000000", "+34910000000"),
        ("tel:+34910000000", "+34910000000"),
        ("sip:+34910000000@trunk.example", "+34910000000"),
        ("sips:desk@pbx.example", "desk"),
    ],
)
def test_a_dial_takes_the_user_part_only(destination: str, user: str) -> None:
    assert sip_user(destination) == user


def test_a_number_goes_through_the_trunk_and_a_sip_uri_stays() -> None:
    assert sip_uri("+34910000000", "trunk.example") == "sip:+34910000000@trunk.example"
    assert sip_uri("tel:+34910000000", "trunk.example") == "sip:+34910000000@trunk.example"
    assert sip_uri("sip:desk@pbx.example", "trunk.example") == "sip:desk@pbx.example"


@pytest.mark.parametrize(
    ("code", "reason"),
    [(486, "busy"), (603, "busy"), (480, "no answer"), (487, "no answer"), (503, "sip 503")],
)
def test_a_sip_status_becomes_words(code: int, reason: str) -> None:
    assert sip_reason(code) == reason
    assert sip_reason(None) == "failed"


@pytest.mark.parametrize("destination", ["", "910000000", "tel:", "mailto:a@b", "sip: "])
def test_a_destination_must_be_a_number_or_a_uri(destination: str) -> None:
    with pytest.raises(ValueError, match=r"E\.164 number"):
        check_destination(destination)
