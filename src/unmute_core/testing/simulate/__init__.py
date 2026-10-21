"""The call simulator: many simulated callers ring an agent, and each call is checked.

A scripted scenario (``python -m unmute_core.testing converse``) types the caller's exact lines and
proves one path. The simulator puts a model on the caller's side instead, with a persona,
a goal, the facts it may give and the moves it makes, and runs each scenario many times.
Every call is checked by code and by a judge, and a run reports pass rates per scenario
and per check. That is how a change to the runtime or to a prompt is measured: two runs,
before and after, compared check by check with a Fisher exact test::

    python -m unmute_core.testing scenarios
    python -m unmute_core.testing simulate livekit --calls 10 --out runs/before
    python -m unmute_core.testing simulate pipecat booking_happy tag:intake --calls 3
    python -m unmute_core.testing report runs/after --compare runs/before
    python -m unmute_core.testing parity runs/livekit runs/pipecat

Three models take part, each on any OpenAI-compatible endpoint: the agent's, the
caller's and the judge's. ``--model`` sets all three, ``--agent-model``,
``--caller-model`` and ``--judge-model`` one each, and ``SimulateSettings`` lists the
``SIMULATE_*`` variables, keys included. The caller and the judge should run on another
model than the agent, so the two sides do not share one model's habits.

The call is text, on the framework's own agent: ``UnmuteAgent`` in a LiveKit
``AgentSession``, or ``UnmuteFlows`` on a Pipecat pipeline. The bundled agents are made-up
businesses (Maple Street Dental, Northwind Outfitters, Lakeside Bikes) whose tools answer
from deterministic mocks, so a check knows what each tool returned.

Each call gets:

* deterministic checks (``_checks``): the values left saved match the scenario, no saved
  value is one the caller never gave, a step saves only after the caller spoke in it,
  each run record matches its task events, the call ended naturally, and what the
  scenario requires happened.
* the judge's checks (``_judge``): it reads the transcript with every event the caller
  cannot hear (task starts and ends, saves and who made them, tool results) in place, and
  rules on what code cannot see, like a booking claimed before it was saved.

A call passes when it passes every core check. A call is **scored** only when it is
evidence about the agent. These calls are left out of every pass rate, counted apart and
named in the summary:

* ``environment``: the agent's endpoint failed: a rate limit, a 5xx, a timeout or a lost
  link that remained after retries. A rate limit before any answer is retried first. A
  request the endpoint refused as invalid (a 400) is the agent's error and fails the call.
* ``harness``: the simulator itself broke, like a turn that never settled.
* ``judge``: the judge gave no verdict.
* ``caller``: the simulated caller left its brief: it gave a number, a date or a service
  its facts do not hold, or did the opposite of a move. The judge rules on this
  (``caller_followed_script``). A saved value the caller made up is reported the same way
  by the fabrication check: ``caller_off_script_value``, not ``fabricated_value``.

A rerun into the same directory skips calls with a verdict and runs the left-out
``environment`` and ``harness`` calls again. ``--retry-failed N`` runs failed calls again
too, keeping each earlier attempt beside it; a call that passed only on a retry is
reported as flaky.

``--gate`` makes the run a CI check: it fails when a planned call has no result, when
fewer than ``--gate-min-scored`` (75%) of the planned calls are scored, so a broken
environment is never a pass, or when the scored calls' pass rate is below
``--gate-min-pass`` (100%).
"""

from unmute_core.testing.simulate._checks import CheckResult, run_checks
from unmute_core.testing.simulate._endpoints import Endpoint, SimulateSettings
from unmute_core.testing.simulate._gate import Verdict, evaluate
from unmute_core.testing.simulate._run import Runner, RunOptions
from unmute_core.testing.simulate._scenario import Scenario, load_scenario

__all__ = [
    "CheckResult",
    "Endpoint",
    "RunOptions",
    "Runner",
    "Scenario",
    "SimulateSettings",
    "Verdict",
    "evaluate",
    "load_scenario",
    "run_checks",
]
