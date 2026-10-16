"""Gemini thought signatures: the placeholder on replayed tool calls, and ``ToolCall.extra``."""

from typing import Any

from unmute_core.model import (
    SKIP_THOUGHT_SIGNATURE,
    ToolCall,
    has_signature,
    is_gemini,
    replayed_extra,
    signed_tool_calls,
)

SIGNED = {"google": {"thought_signature": "sig-1"}}
SKIPPED = {"google": {"thought_signature": SKIP_THOUGHT_SIGNATURE}}


def tool_call(call_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    call: dict[str, Any] = {
        "id": call_id,
        "type": "function",
        "function": {"name": "find", "arguments": "{}"},
    }
    return call if extra is None else {**call, "extra_content": extra}


def calls_message(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def test_a_gemini_request_signs_the_first_unsigned_call_of_each_message() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        calls_message(tool_call("a"), tool_call("b")),
        {"role": "tool", "tool_call_id": "a", "content": "{}"},
        {"role": "tool", "tool_call_id": "b", "content": "{}"},
    ]

    signed = signed_tool_calls(messages, gemini=True)

    first, second = signed[1]["tool_calls"]
    assert first["extra_content"] == SKIPPED
    assert "extra_content" not in second
    assert signed[0] is messages[0]
    assert "extra_content" not in messages[1]["tool_calls"][0]


def test_a_signed_first_call_keeps_its_signature() -> None:
    messages = [calls_message(tool_call("a", SIGNED), tool_call("b"))]
    assert signed_tool_calls(messages, gemini=True) == messages


def test_a_placeholder_keeps_the_calls_other_provider_data() -> None:
    extra = {"google": {"other": 1}, "xai": {"x": 2}}
    [message] = signed_tool_calls([calls_message(tool_call("a", extra))], gemini=True)
    assert message["tool_calls"][0]["extra_content"] == {
        "google": {"other": 1, "thought_signature": SKIP_THOUGHT_SIGNATURE},
        "xai": {"x": 2},
    }


def test_another_model_is_left_alone_until_the_history_holds_a_signature() -> None:
    unsigned = [calls_message(tool_call("a"))]
    assert signed_tool_calls(unsigned, gemini=False) is unsigned

    mixed = [calls_message(tool_call("a", SIGNED)), calls_message(tool_call("b"))]
    assert signed_tool_calls(mixed, gemini=False)[1]["tool_calls"][0]["extra_content"] == SKIPPED


def test_only_assistant_tool_calls_count() -> None:
    messages = [{"role": "user", "content": "hi", "tool_calls": [tool_call("a")]}]
    assert signed_tool_calls(messages, gemini=True) == messages


def test_gemini_is_read_from_the_model_name() -> None:
    assert is_gemini("google/gemini-3-flash-preview")
    assert is_gemini("Gemini-3-Pro")
    assert not is_gemini("gpt-4.1-mini")
    assert not is_gemini("slng/auto")


def test_a_signature_is_a_non_empty_google_thought_signature() -> None:
    assert has_signature(SIGNED)
    assert not has_signature({"google": {"thought_signature": ""}})
    assert not has_signature({"google": "sig"})
    assert not has_signature(None)


def test_a_replayed_call_sends_back_only_the_keys_livekit_sends() -> None:
    extra = {"google": SIGNED["google"], "xai": {"id": "x"}, "openai": {"a": 1}, "livekit": {}}
    assert replayed_extra(extra) == {"google": SIGNED["google"], "xai": {"id": "x"}}
    assert replayed_extra({"openai": {"a": 1}}) == {}


def test_a_tool_call_without_extras_equals_one_built_before_extras_existed() -> None:
    plain = ToolCall("a", "find", "{}")
    assert plain == ToolCall("a", "find", "{}", {})
    assert plain.extra == {}
    assert hash(ToolCall("a", "find", "{}", SIGNED)) == hash(plain)
    assert ToolCall("a", "find", "{}", SIGNED) != plain
