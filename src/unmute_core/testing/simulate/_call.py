"""One simulated call: the caller and the agent take turns until one of them hangs up.

The loop is the same on both frameworks. A framework driver (``TextCall``) opens the call,
types each caller line into the agent, and returns what the agent said once it has gone
quiet. Rules, as on a real call:

* The caller ends the call by writing ``[HANGUP]``. Their last words still reach the
  agent, and what the agent says after them is marked unheard.
* The agent ends the call with a tool whose effect ends it.
* A caller reply with no words is sent as "Hello?".
* ``turn_cap`` caller turns cut the call off. Such a call did not end naturally.
"""

import asyncio
import logging
import time
from typing import Any, Protocol

from unmute_core.testing.simulate._agent import AgentCall
from unmute_core.testing.simulate._caller import SimCaller, hangs_up
from unmute_core.testing.simulate._record import HARNESS, Recorder
from unmute_core.testing.simulate._scenario import Scenario

logger = logging.getLogger(__name__)

TURN_TIMEOUT_S = 120.0
"""How long one turn of the agent may take before the simulator gives up on the call."""


class TextCall(Protocol):
    """A framework driver: the agent of one call, spoken to in text.

    Attributes:
        ended: True once the agent's side ended the call.
    """

    ended: bool

    async def start(self) -> list[str]:
        """Open the call and return what the agent says before the caller speaks."""
        ...

    async def say(self, line: str) -> list[str]:
        """Type one caller line and return what the agent says once it has gone quiet."""
        ...

    async def hang_up(self) -> None:
        """Leave the call as the caller. The call ends and its held saves are made."""
        ...

    async def close(self) -> None:
        """End the call if it is still open, and release everything. Never raises."""
        ...


async def run_call(
    driver: TextCall, caller: SimCaller, scenario: Scenario, call: AgentCall, recorder: Recorder
) -> list[dict[str, Any]]:
    """Run one call to its end and return its records.

    A failure of the simulator itself, like a turn that never settles, is kept as a
    ``harness`` error, so the call is left out of the pass rate.
    """
    began = time.monotonic()
    ended_by = "harness_error"
    try:
        said = await asyncio.wait_for(driver.start(), TURN_TIMEOUT_S)
        for line in said:
            recorder.add("agent_said", text=line)
        ended_by = await _turns(driver, caller, scenario, recorder, " ".join(said))
    except TimeoutError:
        recorder.error(f"{HARNESS}_turn_timeout", error="the agent never went quiet")
    except Exception as error:
        logger.warning("the simulator broke the call to %s", scenario.agent, exc_info=True)
        recorder.error(f"{HARNESS}_error", error=repr(error)[:300])
    finally:
        await driver.close()
        recorder.unread()
    turns = sum(1 for event in recorder.events if event["kind"] == "caller_said")
    if any(error["kind"] == "call_failed" for error in recorder.errors) and ended_by in (
        "agent_hangup",
        "harness_error",
    ):
        ended_by = "terminal_failure"
    record = {
        "type": "call",
        "scenario": scenario.id,
        "agent": call.agent.name,
        "ended_by": ended_by,
        "turns": turns,
        "duration_s": round(time.monotonic() - began, 1),
        "final_values": call.state.dump(),
        "variables": recorder.variables(call.state),
        "errors": recorder.errors,
        "transcript": recorder.transcript(),
    }
    return [
        record,
        *({"type": "event", **event} for event in recorder.events),
        {"type": "mock_requests", "requests": call.mocks.requests},
    ]


async def _turns(
    driver: TextCall, caller: SimCaller, scenario: Scenario, recorder: Recorder, said: str
) -> str:
    """Take turns until someone hangs up or the cap is hit. Return who ended the call."""
    turns = 0
    while True:
        if driver.ended:
            recorder.add("hangup", source="agent")
            return "agent_hangup"
        if turns >= scenario.turn_cap:
            recorder.add("hangup", source="turn_cap")
            await driver.hang_up()
            return "turn_cap"
        text, hanging_up = hangs_up(await caller.reply(said))
        if not text and not hanging_up:
            text = "Hello?"
        if text:
            turns += 1
            recorder.add("caller_said", text=text)
            lines = await asyncio.wait_for(driver.say(text), TURN_TIMEOUT_S)
            for line in lines:
                recorder.add("agent_said", text=line, **({"unheard": True} if hanging_up else {}))
            said = " ".join(lines)
        if hanging_up:
            recorder.add("hangup", source="caller")
            await driver.hang_up()
            return "caller_hangup"
