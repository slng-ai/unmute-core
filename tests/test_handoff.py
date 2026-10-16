"""What the next agent hears: the context rules and the summary request."""

from collections.abc import AsyncIterator

import pytest

from unmute_core.handoff import (
    SUMMARY_PROMPT,
    HandoffContext,
    shape_history,
    summary_message,
    summary_request,
    transcript,
)
from unmute_core.model import ModelEvent, ModelRequest, TextDelta
from unmute_core.tasks import STEP_START_CUE


class Summarizer:
    """A ``ModelCall`` that answers every request with one summary, and keeps the requests."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Answer with the summary."""
        self.requests.append(request)
        yield TextDelta(self.text)


class KeyLimited(Summarizer):
    """A ``Summarizer`` that fills the prompt on its own side and takes keys of 64 characters."""

    variable_key_max = 64


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"history": "last_n"}, "last_n needs max_messages"),
        ({"history": "full", "max_messages": 3}, "only valid with last_n"),
        ({"history": "messages", "summarizer": Summarizer("x")}, "only valid with summary"),
    ],
)
def test_a_context_that_does_not_fit_is_refused(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HandoffContext(**kwargs)


def test_the_default_keeps_what_was_said() -> None:
    assert HandoffContext() == HandoffContext(history="messages", include_tool_calls=True)


def test_the_transcript_holds_only_what_was_said() -> None:
    messages = [
        {"role": "system", "content": "You help callers."},
        {"role": "user", "content": STEP_START_CUE},
        {"role": "user", "content": "I need my order."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": '{"found": true}'},
        {"role": "assistant", "content": [{"type": "text", "text": "It ships today."}]},
        object(),
    ]
    assert transcript(messages) == "user: I need my order.\nassistant: It ships today."


def test_the_summary_request_and_message() -> None:
    request = summary_request([{"role": "user", "content": "Hi."}])
    assert request.site == "summary" and request.tools == []
    assert request.messages == [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": "user: Hi."},
    ]
    assert summary_message(" They want a refund. ") == {
        "role": "assistant",
        "content": "Summary of the conversation so far: They want a refund.",
    }


CALL = [
    {"role": "system", "content": "You are the front desk."},
    {"role": "user", "content": STEP_START_CUE},
    {"role": "user", "content": "Where is my order?"},
    {
        "role": "assistant",
        "content": "Let me look.",
        "extra_content": {"google": {"thought_signature": "x"}},
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "find", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": '{"status": "late"}'},
    {
        "role": "assistant",
        "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "finish", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "c2", "content": "saved"},
    {"role": "assistant", "content": "It is late."},
]
SPOKEN = [
    {"role": "user", "content": "Where is my order?"},
    {"role": "assistant", "content": "Let me look."},
    {"role": "assistant", "content": "It is late."},
]
WITH_TOOLS = [
    {"role": "user", "content": "Where is my order?"},
    {"role": "assistant", "content": "Let me look.", "tool_calls": CALL[3]["tool_calls"]},
    CALL[4],
    {"role": "assistant", "content": "It is late."},
]


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (HandoffContext(), SPOKEN),
        (HandoffContext(history="summary"), SPOKEN),
        (HandoffContext(history="full"), WITH_TOOLS),
        (HandoffContext(history="full", include_tool_calls=False), SPOKEN),
        (HandoffContext(history="last_n", max_messages=2), [CALL[4], WITH_TOOLS[3]][1:]),
        (HandoffContext(history="last_n", max_messages=3), WITH_TOOLS[1:]),
        (HandoffContext(history="reset"), []),
    ],
    ids=["messages", "summary", "full", "full_no_tools", "last_n_orphan", "last_n", "reset"],
)
def test_shape_history_keeps_what_each_policy_says(
    context: HandoffContext, expected: list[dict]
) -> None:
    before = [dict(message) for message in CALL]
    assert shape_history(CALL, context) == expected
    assert before == CALL


UNKNOWN_CALL = [
    {"role": "user", "content": "Where is my order?"},
    {
        "role": "assistant",
        "content": "Let me look.",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "find", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": '{"status": "late"}'},
    {"role": "assistant", "content": "It is late."},
]


def test_a_call_to_a_tool_the_next_agent_lacks_is_dropped_with_its_reply() -> None:
    kept = shape_history(UNKNOWN_CALL, HandoffContext(history="full"), tools={"note"})
    assert kept == [
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "Let me look."},
        {"role": "assistant", "content": "It is late."},
    ]


def test_a_call_to_a_tool_the_next_agent_has_stays() -> None:
    context = HandoffContext(history="full")
    assert shape_history(UNKNOWN_CALL, context, tools={"find"}) == shape_history(
        UNKNOWN_CALL, context
    )


def test_last_n_counts_its_window_before_unknown_calls_drop() -> None:
    context = HandoffContext(history="last_n", max_messages=3)
    kept = shape_history(UNKNOWN_CALL, context, tools=set())
    assert kept == [
        {"role": "assistant", "content": "Let me look."},
        {"role": "assistant", "content": "It is late."},
    ]
