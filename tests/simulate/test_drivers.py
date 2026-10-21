"""A whole simulated call on each framework, offline: a scripted agent model and caller.

The agent model answers by where the request comes from: the main agent starts the
verify task and says goodbye once it read the task's record; the step asks for the number
and saves it once the caller gave one.
"""

import importlib
import json
import re
from collections.abc import AsyncIterator
from typing import Any

import pytest

from unmute_core.model import ModelEvent, ModelRequest, TextDelta, ToolCall
from unmute_core.testing.simulate import Endpoint, run_checks
from unmute_core.testing.simulate._agent import build
from unmute_core.testing.simulate._call import run_call
from unmute_core.testing.simulate._caller import SimCaller
from unmute_core.testing.simulate._endpoints import Chat
from unmute_core.testing.simulate._record import Recorder
from unmute_core.testing.simulate._run import DRIVERS, Framework
from unmute_core.testing.simulate._scenario import load_scenario, scenario_paths

GOODBYE = "You're verified. The clinic will ring you about your session. Goodbye."


class ScriptedAgent:
    """The agent's model: the main agent delegates, the step asks, then saves."""

    model = "scripted"

    def __init__(self) -> None:
        self.sites: list[str] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        site = request.site or "main"
        self.sites.append(site)
        number = len(self.sites)
        heard = [str(m.get("content")) for m in request.messages if m.get("role") == "user"]
        if site == "main":
            if any(m.get("role") == "tool" for m in request.messages):
                yield TextDelta(GOODBYE)
            else:
                yield ToolCall(id=f"call{number}", name="verify_identity", arguments="{}")
        elif heard and re.search(r"\d|three", heard[-1]):
            saved = json.dumps({"phone_number": "+34699120345"})
            yield ToolCall(id=f"call{number}", name="finish", arguments=saved)
        else:
            yield TextDelta("What's the phone number on your booking?")


class ScriptedCaller(Chat):
    """The caller's model: each line in turn."""

    def __init__(self, *lines: str) -> None:
        super().__init__(Endpoint(model="caller"))
        self.lines = list(lines)

    async def complete(self, messages: list[dict[str, str]], **params: Any) -> str:
        return self.lines.pop(0)


@pytest.mark.parametrize(
    ("framework", "package"), [("livekit", "livekit.agents"), ("pipecat", "pipecat")]
)
async def test_a_simulated_call_runs_to_the_end_and_passes_its_checks(
    framework: Framework, package: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip(package)
    driver_module = importlib.import_module(f"unmute_core.{framework}._simulate")
    monkeypatch.setattr(driver_module, "SETTLE_S", 0.3)
    scenario, agent = load_scenario(scenario_paths(["verify_only"])[0])
    model = ScriptedAgent()
    call = build(agent, model, line=scenario.line)
    recorder = Recorder(call.tool_names - call.via_tools().keys())
    recorder.attach(call.hooks)
    recorder.watch(call.state)
    driver = getattr(driver_module, DRIVERS[framework])(call, recorder)
    caller = ScriptedCaller(
        "When is my next session?",
        "It's plus three four, six nine nine, one two zero, three four five.",
        "Thanks, bye. [HANGUP]",
    )
    records = await run_call(
        driver, SimCaller(caller, scenario.caller_brief()), scenario, call, recorder
    )

    record = records[0]
    assert (record["ended_by"], record["turns"], record["errors"]) == ("caller_hangup", 3, [])
    assert record["transcript"][:4] == [
        {"role": "agent", "text": agent.greeting},
        {"role": "caller", "text": "When is my next session?"},
        {"role": "agent", "text": "What's the phone number on your booking?"},
        {
            "role": "caller",
            "text": "It's plus three four, six nine nine, one two zero, three four five.",
        },
    ]
    assert {"role": "agent", "text": GOODBYE} in record["transcript"]
    assert record["final_values"]["phone_number"] == "+34699120345"
    kinds = [r.get("kind") or r.get("stage") for r in records[1:-1]]
    events = [r for r in records if r["type"] == "event"]
    stages = [e["stage"] for e in events if e["kind"] == "task_event"]
    assert stages == ["task.started", "task.completed"]
    delegate = next(e for e in events if e.get("name") == "verify_identity")
    completed = next(e for e in events if e.get("stage") == "task.completed")
    assert events.index(delegate) > events.index(completed), kinds
    assert delegate["output"]["status"] == "completed"
    assert next(e for e in events if e["kind"] == "state_write" and e["writer"])["writer"] == (
        "verify_identity"
    )
    results = run_checks(
        records,
        scenario,
        delegates=call.delegates(),
        providers=call.providers(),
        via_tools=call.via_tools(),
        task_values=call.task_values(),
    )
    assert [r.id for r in results if not r.passed] == []
    assert model.sites[:3] == ["main", "task:verify_identity", "task:verify_identity"]
