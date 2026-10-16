"""What a step hears of the call, and tools as Pipecat schemas.

The history cases check that a tool call is dropped together with its reply, and that
dropping finish calls drops each finish and its reply but keeps speech.
"""

from typing import Any, cast

from tests.tasks.job import Canned, registry
from unmute_core.pipecat import tools_schema
from unmute_core.pipecat._history import (
    last_n,
    shape_history,
    shaped,
    speech_only,
    trailing_results,
    without_calls_to,
    without_instructions,
)
from unmute_core.pipecat._tools import inline_refs
from unmute_core.tasks import TaskSpec


def _call(call_id: str, name: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


CHAT: list[dict[str, Any]] = [
    {"role": "system", "content": "Prompt."},
    {"role": "user", "content": "Find my order."},
    {"role": "assistant", "content": "Looking.", "tool_calls": [_call("c1", "lookup")]},
    {"role": "tool", "tool_call_id": "c1", "content": '{"found": true}'},
    {"role": "assistant", "content": "Found it."},
    {"role": "developer", "content": "Note."},
]


def test_history_drops_a_tool_call_together_with_its_reply() -> None:
    assert without_instructions(CHAT) == CHAT[1:5]
    assert speech_only(CHAT) == [
        CHAT[1],
        {"role": "assistant", "content": "Looking."},
        CHAT[4],
    ]
    assert last_n(CHAT[1:5], 2) == [CHAT[4]]
    assert last_n(CHAT[1:5], 3) == CHAT[2:5]
    assert last_n(CHAT, 0) == []


def test_without_calls_to_drops_a_finish_and_its_reply_and_keeps_speech() -> None:
    chat = [
        {"role": "user", "content": "A-12."},
        {"role": "assistant", "content": "Saved.", "tool_calls": [_call("f1", "finish")]},
        {"role": "tool", "tool_call_id": "f1", "content": "{}"},
        {"role": "assistant", "tool_calls": [_call("f2", "finish"), _call("l1", "lookup")]},
        {"role": "tool", "tool_call_id": "f2", "content": "{}"},
        {"role": "tool", "tool_call_id": "l1", "content": "{}"},
    ]
    assert without_calls_to(chat, {"finish"}) == [
        chat[0],
        {"role": "assistant", "content": "Saved."},
        {"role": "assistant", "tool_calls": [_call("l1", "lookup")]},
        chat[5],
    ]


def test_trailing_results_name_the_tools_a_request_answers() -> None:
    assert trailing_results(CHAT[:4]) == [("lookup", '{"found": true}')]
    assert trailing_results(CHAT) == []


def test_refs_are_inlined_for_pipecats_function_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"when": {"$ref": "#/$defs/Slot", "description": "The slot."}},
        "$defs": {"Slot": {"type": "string", "enum": ["am", "pm"]}},
    }
    assert inline_refs(schema) == {
        "type": "object",
        "properties": {
            "when": {"type": "string", "enum": ["am", "pm"], "description": "The slot."}
        },
    }


async def test_an_agent_without_tasks_gets_each_tool_with_its_handler() -> None:
    results: list[Any] = []
    tools = tools_schema(registry(search_web=Canned({"hits": 1})))
    assert [schema.name for schema in tools.standard_tools] == [
        "lookup_order",
        "search_web",
        "normalize",
    ]

    async def report(result: Any, **_: Any) -> None:
        results.append(result)

    worker = type("Worker", (), {})()
    params = type(
        "Params",
        (),
        {
            "arguments": {},
            "tool_call_id": "call_1",
            "llm": worker,
            "pipeline_worker": worker,
            "result_callback": staticmethod(report),
        },
    )
    handler = next(s.handler for s in tools.standard_tools if s.name == "search_web")
    assert handler is not None
    await handler(cast(Any, params))
    assert results == [{"hits": 1}]


def test_a_steps_history_carries_no_reasoning_of_earlier_replies() -> None:
    """A spoken reply's provider data stays behind; a call keeps its own."""
    signature = {"google": {"thought_signature": "sig"}}
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "book", "arguments": "{}"},
        "extra_content": signature,
    }
    messages = [
        {"role": "system", "content": "owner"},
        {"role": "user", "content": "Book me in."},
        {"role": "assistant", "content": "One moment.", "extra_content": signature},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Done.", "extra_content": signature},
    ]
    spoken = [
        {"role": "assistant", "content": "One moment."},
        {"role": "assistant", "content": "Done."},
    ]
    task = TaskSpec(name="book", when="w", instructions="Book.")
    full = shape_history(task.model_copy(update={"history": "full"}), messages)
    assert [m for m in full if m["role"] == "assistant" and m["content"]] == spoken
    assert full[2]["tool_calls"][0]["extra_content"] == signature
    tail = shape_history(task.model_copy(update={"history": "last_n", "max_messages": 3}), messages)
    assert tail[-1] == {"role": "assistant", "content": "Done."}
    assert tail[0]["tool_calls"][0]["extra_content"] == signature
    said = shape_history(task.model_copy(update={"history": "messages"}), messages)
    assert said[1:] == spoken
    assert messages[2]["extra_content"] == signature


def test_each_history_keeps_what_it_says() -> None:
    assert shaped(CHAT, "reset") == []
    spoken = [CHAT[1], {"role": "assistant", "content": "Looking."}, CHAT[4]]
    assert shaped(CHAT, "messages") == spoken
    assert shaped(CHAT, "summary") == spoken
    assert shaped(CHAT, "full") == CHAT[1:5]
    assert shaped(CHAT, "full", include_tool_calls=False) == spoken
    assert shaped(CHAT, "last_n", max_messages=2, include_tool_calls=False) == spoken[1:]
