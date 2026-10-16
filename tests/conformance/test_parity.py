"""Every scenario in ``scenarios/`` runs on every adapter, and each one does what it says.

Both adapters must match the same expected data, so a behaviour change in one of them
fails here. A behaviour change on purpose updates the scenario that shows it.
"""

import re
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

import pytest

from tests.conformance import livekit_driver, pipecat_driver
from tests.conformance.scenario import Scenario, expected, load_scenarios

type Driver = Callable[[Scenario], Awaitable[dict[str, Any]]]


@pytest.mark.parametrize(
    "driver",
    [
        livekit_driver.run,
        livekit_driver.run_plain,
        pipecat_driver.run,
        partial(pipecat_driver.run, voice=True),
        livekit_driver.run_realtime,
        pipecat_driver.run_realtime,
    ],
    ids=[
        "livekit",
        "livekit-plain",
        "pipecat",
        "pipecat-voice",
        "livekit-realtime",
        "pipecat-realtime",
    ],
)
@pytest.mark.parametrize("scenario", load_scenarios(), ids=lambda scenario: scenario.name)
async def test_the_scenario_runs_the_same_on_every_adapter(
    scenario: Scenario, driver: Driver
) -> None:
    realtime = driver in (livekit_driver.run_realtime, pipecat_driver.run_realtime)
    if scenario.expect.build_error is not None:
        with pytest.raises(ValueError, match=re.escape(scenario.expect.build_error)):
            await driver(scenario)
        return
    assert await driver(scenario) == expected(scenario, realtime=realtime)
