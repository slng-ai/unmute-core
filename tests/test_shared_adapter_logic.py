"""The framework-neutral logic both adapters call: the same answer on either one."""

import copy
import logging
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.outputs import AnyResult
from tests.tasks.job import SPECIALIST, State, registry, specialist_plan
from tests.telemetry.conftest import attrs
from unmute_core.agent import AgentSpec
from unmute_core.call import transcript_messages
from unmute_core.handoff import newest_caller
from unmute_core.model import (
    STEP_START_CUE,
    ModelAnswer,
    TextDelta,
    ToolCall,
    Usage,
    collect,
)
from unmute_core.state import CallState
from unmute_core.tasks import TaskPlan, TasksSpec
from unmute_core.telemetry import Sink, Telemetry, TurnTracker, log_activated, log_no_simulation
from unmute_core.tools import ToolInterruption, ToolSpec
from unmute_core.turn import InterruptionRule, word_count


@pytest.mark.parametrize(
    ("floor", "min_words", "text"),
    [(0, -1, "min_words must be 0 or more, got -1"), (1, 0, "min_words must be 1 or more, got 0")],
)
def test_an_interruption_rule_keeps_each_frameworks_floor(floor, min_words, text):
    with pytest.raises(ValueError, match=text):
        InterruptionRule("keep", min_words, min_words_floor=floor)


def test_a_speech_time_belongs_to_stop():
    with pytest.raises(ValueError, match="applies to stop only"):
        InterruptionRule("keep", 1, 0.5)
    assert InterruptionRule("stop", 1, 0.5).min_speech_s == 0.5


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("No wait, Saturday please", 4),
        ("Yes.", 1),
        ("はい", 1),
        ("ちょっと待ってください", 6),
        ("请等一下", 2),
        ("ครับ", 1),
        ("รอสักครู่ครับ", 4),
        ("OK 好的", 2),
        ("", 0),
    ],
)
def test_words_are_counted_also_without_spaces(text, words):
    assert word_count(text) == words


class _Nothing:
    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        return None


def _tool(name: str, interruption: ToolInterruption) -> ToolSpec:
    return ToolSpec(
        name,
        "does it",
        {"type": "object", "properties": {}},
        AnyResult,
        _Nothing(),
        interruption=interruption,
    )


def test_a_registry_says_which_calls_the_callers_speech_leaves_running():
    from unmute_core.tools import ToolRegistry

    registry = ToolRegistry([_tool("slow", "continue"), _tool("quick", "cancel")])
    assert registry.keeps_running("slow")
    assert not registry.keeps_running("quick")
    assert not registry.keeps_running("unknown")


def test_a_tool_call_reads_as_an_openai_entry_without_provider_data():
    call = ToolCall("c1", "find", '{"a": 1}', {"google": {}})
    assert call.to_openai() == {
        "id": "c1",
        "type": "function",
        "function": {"name": "find", "arguments": '{"a": 1}'},
    }


async def test_collect_joins_the_text_and_keeps_calls_and_usage():
    async def events():
        yield TextDelta("Hel")
        yield ToolCall("c1", "find", "{}")
        yield TextDelta("lo")
        yield Usage(3, 4)

    answer = await collect(events())
    assert answer == ModelAnswer("Hello", (ToolCall("c1", "find", "{}"),), Usage(3, 4))


async def test_collect_names_an_event_that_is_not_a_model_event():
    async def events():
        yield "plain text"

    with pytest.raises(TypeError, match="streamed str; stream TextDelta, ToolCall or Usage"):
        await collect(events())


MESSAGES = [
    {"role": "system", "content": "prompt"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": ""},
    {"role": "assistant", "content": [{"type": "text"}]},
    {"role": "user", "content": STEP_START_CUE},
    "not a dict",
]


def test_the_transcript_drops_empty_text_only_when_asked():
    assert transcript_messages(MESSAGES) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": STEP_START_CUE},
    ]
    assert transcript_messages(MESSAGES, drop_empty=True) == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": STEP_START_CUE},
    ]


def test_the_newest_caller_passes_the_cue_only_when_told_to():
    assert newest_caller(MESSAGES, skip_cue=True) == {"role": "user", "content": "hi"}
    assert newest_caller(MESSAGES, skip_cue=False) == {"role": "user", "content": STEP_START_CUE}
    assert newest_caller([{"role": "assistant", "content": "x"}], skip_cue=True) is None


def test_a_state_sets_and_clears_its_suppliers():
    from pydantic import BaseModel

    class Booking(BaseModel):
        phone: str | None = None

    state = CallState(Booking)
    state.set_suppliers({"phone": "collect"})
    assert state.suppliers == {"phone": "collect"}
    state.clear_suppliers()
    assert state.suppliers == {}


def test_both_adapters_log_the_same_coval_lines(caplog):
    log = logging.getLogger("test.coval")
    with caplog.at_level(logging.INFO, logger="test.coval"):
        log_no_simulation(log)
        log_no_simulation(log, "browser call")
        log_activated(log, "sim_1", "metadata")
        log_activated(log, "sim_1", "metadata", "sip call")
    assert [record.getMessage() for record in caplog.records] == [
        "No Coval simulation placed this call, so its trace is filed as a Coval "
        "conversation when the call ends",
        "No Coval simulation placed this browser call, so its trace is filed as a Coval "
        "conversation when the call ends",
        "Coval tracing active for simulation sim_1 via metadata",
        "Coval tracing active for simulation sim_1 via metadata (sip call)",
    ]


class _Memory(Sink):
    def __init__(self) -> None:
        self.spans = InMemorySpanExporter()

    def exporter(self):
        return self.spans


async def test_a_turn_belongs_to_the_caller_until_the_agent_answers():
    sink = _Memory()
    telemetry = Telemetry(sink, agent="salon", call_id="RM_abc")
    telemetry.start_call()
    turns = TurnTracker(telemetry)
    assert not turns.caller_open
    first = turns.start()
    assert turns.caller_open
    assert turns.open(answering=False) is first
    assert turns.caller_open
    assert turns.open() is first
    assert not turns.caller_open
    turns.end(user_text="hi", agent_lines=["hello"])
    assert turns.span is None
    turns.end(user_text="ignored")
    turns.open(answering=False)
    turns.end()
    await telemetry.close()
    spans = [s for s in sink.spans.get_finished_spans() if s.name == "turn"]
    assert [attrs(s)["turn.number"] for s in spans] == [1, 2]


def _with_model(name: str) -> TaskPlan:
    payload = copy.deepcopy(SPECIALIST)
    payload["tasks"][0]["model"] = name
    return TaskPlan(TasksSpec.model_validate(payload), state=State, tools=registry())


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"tasks": specialist_plan(), "tools": registry()}, "Pass tools= or tasks=, not both"),
        ({"tasks": _with_model("fast")}, "runs on the model 'fast', which task_models does not"),
    ],
)
def test_an_agent_that_cannot_run_is_refused_with_the_fix(fields: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AgentSpec("You help callers.", **fields)
