"""Run a project's own agent through scripted calls: whole calls on its real model, or one turn.

Run it inside the project's environment, from the project folder::

    python -I -m unmute_core.testing converse livekit agent scenarios/*.json
    python -I -m unmute_core.testing converse pipecat bot scenarios/*.json
    python -I -m unmute_core.testing smoke livekit agent        # one turn, no network
    python -I -m unmute_core.testing smoke pipecat bot --live   # one turn, real model

The project runs as written. ``unmute_core.livekit.testing`` and
``unmute_core.pipecat.testing`` say what a project must have, and how each call runs.

A scenario is a JSON file: a whole call, typed by a scripted caller::

    {
      "name": "a whole claim",
      "call_start": {"call": {"direction": "inbound", "calling_number": "+447700900123"}},
      "turns": [
        {"caller": "KP482913, born 5 June 1979.", "expect": ["Is your policy number"]},
        {"caller": "Yes.", "expect": ["Thank you, Daniel"], "max_requests": 2}
      ],
      "state": {"claim_number": "KC100231"}
    }

``Scenario`` documents every key. Code-spoken lines are word for word, so they make exact
checks. The model's own words are checked loosely, if at all. ``ConverseSettings`` lists
the ``CONVERSE_*`` variables that change how a run runs, like ``CONVERSE_AUDIO=1`` for a
caller whose lines are spoken.
"""

from unmute_core.testing._scenario import (
    Call,
    ConverseSettings,
    Scenario,
    ScenarioTurn,
    Turn,
    check,
    load,
)

__all__ = ["Call", "ConverseSettings", "Scenario", "ScenarioTurn", "Turn", "check", "load"]
