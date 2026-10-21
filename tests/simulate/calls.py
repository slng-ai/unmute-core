"""Records of made-up simulated calls, and run directories of them, for the simulator tests.

A timeline is written as events in order; each gets the next second as its time.
"""

import json
from pathlib import Path
from typing import Any

from unmute_core.testing.simulate import Scenario


def scenario(**fields: Any) -> Scenario:
    """Return a scenario with a placeholder caller, and the given fields."""
    return Scenario.model_validate(
        {
            "id": "s",
            "agent": "booking",
            "persona": "a caller",
            "goal": "book",
            "facts": {},
            "moves": [],
            **fields,
        }
    )


def said(who: str, text: str, **data: Any) -> dict[str, Any]:
    """Return a line of the caller or the agent."""
    return {"kind": f"{who}_said", "text": text, **data}


def task(stage: str, **data: Any) -> dict[str, Any]:
    """Return a task or group event."""
    return {"kind": "task_event", "stage": stage, "data": data}


def save(writer: str | None, origin: str = "step", **applied: Any) -> dict[str, Any]:
    """Return a save by a step (its name) or by something else (``origin``)."""
    return {
        "kind": "state_write",
        "applied": applied,
        "confirmed": dict.fromkeys(applied),
        "writer": writer,
        "origin": origin,
    }


def tool(name: str, output: Any = None, **arguments: Any) -> dict[str, Any]:
    """Return a tool call with its arguments and the result a model read."""
    return {"kind": "tool_call", "name": name, "arguments": arguments, "output": output}


def records(
    *events: dict[str, Any],
    variables: dict[str, Any] | None = None,
    requests: list[dict[str, Any]] | None = None,
    **call: Any,
) -> list[dict[str, Any]]:
    """Return a call's records: the call record, the timeline and the mock requests.

    Args:
        events: The timeline, in order.
        variables: Each saved value as the call left it.
        requests: The mock requests.
        call: Fields of the call record, like ``ended_by``.
    """
    record = {
        "type": "call",
        "scenario": "s",
        "ended_by": "caller_hangup",
        "turns": sum(1 for e in events if e["kind"] == "caller_said"),
        "errors": [],
        "variables": {
            name: {"stored": value, "confirmed": None, "confirming_task": None}
            for name, value in (variables or {}).items()
        },
        **call,
    }
    timeline = [{"type": "event", "t": float(t), **e} for t, e in enumerate(events)]
    return [record, *timeline, {"type": "mock_requests", "requests": requests or []}]


def write_call(
    out: Path,
    scenario_id: str,
    index: int,
    *,
    passed: bool = True,
    left_out: str | None = None,
    failed: tuple[str, ...] = (),
) -> Path:
    """Write one checked call into a run directory, and return its file."""
    path = out / "calls" / scenario_id / f"{index:02d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    results = [{"id": check, "passed": False, "core": True, "evidence": []} for check in failed]
    lines = [
        {"type": "call", "scenario": scenario_id, "ended_by": "caller_hangup", "turns": 3},
        {
            "type": "checks",
            "scenario": scenario_id,
            "passed": passed,
            "valid": left_out is None,
            "left_out": left_out,
            "results": results,
        },
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path
