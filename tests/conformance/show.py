"""Print what each adapter does in one scenario: ``uv run python -m tests.conformance.show <name>``.

Use it to write or update a scenario's expected block. Read the output before you copy it.
"""

import asyncio
import json
import sys

from loguru import logger

from tests.conformance import livekit_driver, pipecat_driver
from tests.conformance.scenario import load_scenarios


async def main(name: str) -> None:
    """Run the named scenario on each adapter and print its outcome as JSON."""
    logger.remove()
    scenario = next(scenario for scenario in load_scenarios() if scenario.name == name)
    for driver in (livekit_driver, pipecat_driver):
        outcome = await driver.run(scenario)
        print(f"== {driver.__name__.rsplit('.', 1)[-1]}")
        print(json.dumps(outcome, indent=2, default=lambda call: call.model_dump()))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
