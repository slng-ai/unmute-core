"""The router headers and body field follow the router's own checks."""

import pytest

from unmute_core.slng import RouterHeaders, router_body, tracking_ids
from unmute_core.state import CallInfo


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("salon", "salon"),
        ("  salon:v3  ", "salon:v3"),
        ("a" * 128, "a" * 128),
        ("9f1c-uuid_like.id/x", "9f1c-uuid_like.id/x"),
    ],
)
def test_a_good_id_is_kept_stripped(value: str, expected: str) -> None:
    assert RouterHeaders(agent_id=value, session_id="call-1").agent_id == expected


@pytest.mark.parametrize("value", ["", "   ", "a" * 257, *[f"a{c}b" for c in ", \t{}|"]])
@pytest.mark.parametrize("field", ["agent_id", "session_id", "llm_config_id"])
def test_a_bad_id_is_refused_by_name(field: str, value: str) -> None:
    ids = {"agent_id": "salon", "session_id": "call-1", field: value}
    with pytest.raises(ValueError, match=field):
        RouterHeaders(**ids)


@pytest.mark.parametrize("field", ["agent_id", "session_id"])
def test_an_agent_or_session_id_over_128_characters_is_refused(field: str) -> None:
    ids = {"agent_id": "salon", "session_id": "call-1", field: "a" * 129}
    with pytest.raises(ValueError, match=field):
        RouterHeaders(**ids)


def test_speech_gets_the_calls_two_ids() -> None:
    call = CallInfo(agent_id="salon", session_id="call-1")
    assert tracking_ids(call) == {"external_agent_id": "salon", "external_session_id": "call-1"}


@pytest.mark.parametrize("value", ["café-agent", "日本語"])
@pytest.mark.parametrize("field", ["agent_id", "session_id", "llm_config_id"])
def test_a_non_ascii_id_is_refused_with_the_fix(field: str, value: str) -> None:
    ids = {"agent_id": "salon", "session_id": "call-1", field: value}
    with pytest.raises(ValueError, match=rf"{field} must be printable ASCII.*use ASCII letters"):
        RouterHeaders(**ids)


def test_newlines_are_refused() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        RouterHeaders(agent_id="sal\non", session_id="call-1")


HEADERS = RouterHeaders(agent_id="salon", session_id="call-1")


def test_build_sends_the_identity() -> None:
    assert HEADERS.build() == {"X-Slng-Agent-Id": "salon", "X-Slng-Session-Id": "call-1"}


def test_build_adds_the_config_and_the_prompt_site() -> None:
    headers = RouterHeaders(agent_id="salon", session_id="call-1", llm_config_id="cfg")
    assert headers.build("task:verify_customer") == {
        "X-Slng-Agent-Id": "salon",
        "X-Slng-Session-Id": "call-1",
        "X-Slng-Llm-Config-Id": "cfg",
        "X-Slng-Prompt-Site": "task:verify_customer",
    }


@pytest.mark.parametrize("site", ["task:a", "task:a1_b", "task:" + "a" * 64])
def test_a_good_site_is_sent(site: str) -> None:
    assert HEADERS.build(site)["X-Slng-Prompt-Site"] == site


@pytest.mark.parametrize(
    "site", ["main", "a", "task:", "task:A", "task:1a", "task:a-b", "task:" + "a" * 65, " task:a"]
)
def test_a_bad_site_is_refused(site: str) -> None:
    with pytest.raises(ValueError, match="prompt site"):
        HEADERS.build(site)


def test_a_summary_keeps_the_agent_id_and_has_its_own_prompt_site() -> None:
    assert HEADERS.build("summary") == {
        "X-Slng-Agent-Id": "salon",
        "X-Slng-Session-Id": "call-1",
        "X-Slng-Prompt-Site": "task:summary",
    }


def test_the_overlay_never_replaces_the_identity() -> None:
    overlay = {"traceparent": "00-abc-def-01", "x-slng-agent-id": "other", "X-World": "eu"}
    assert HEADERS.build("task:a", overlay) == {
        "traceparent": "00-abc-def-01",
        "X-World": "eu",
        "X-Slng-Agent-Id": "salon",
        "X-Slng-Session-Id": "call-1",
        "X-Slng-Prompt-Site": "task:a",
    }


def test_router_body_holds_the_variables() -> None:
    assert router_body({"name": "Ana", "B_2": ""}) == {
        "template_variables": {"name": "Ana", "B_2": ""}
    }


@pytest.mark.parametrize(
    ("variables", "match"),
    [
        ({f"v{i}": "" for i in range(65)}, "at most 64"),
        ({"a.b": "x"}, r"'a\.b'"),
        ({"": "x"}, "''"),
        ({"a" * 65: "x"}, "letters, digits"),
        ({"name": "x" * 4001}, "4001 characters"),
    ],
)
def test_router_body_refuses_what_the_router_would(variables: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        router_body(variables)


def test_router_body_allows_the_limits() -> None:
    variables = {f"v{i}": "x" * 4000 for i in range(64)}
    assert router_body(variables)["template_variables"] == variables
