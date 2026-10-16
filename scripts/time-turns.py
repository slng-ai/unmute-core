"""Time one call turn on each adapter with the conformance drivers.

Runs every conformance scenario on LiveKit, Pipecat and Pipecat with voice, with logs off,
and prints the median CPU and wall time per turn as JSON. CPU time is what the library
adds. Wall time includes the drivers' waits.

Usage: uv run python scripts/time-turns.py [repeats]
"""

import asyncio
import json
import logging
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conformance import livekit_driver, pipecat_driver
from tests.conformance.scenario import Scenario, load_scenarios

DRIVERS: dict[str, Callable[[Scenario], Awaitable[dict[str, Any]]]] = {
    "livekit": livekit_driver.run,
    "pipecat": pipecat_driver.run,
    "pipecat-voice": partial(pipecat_driver.run, voice=True),
}


async def _one_pass(drive: Callable[[Scenario], Awaitable[Any]], scenarios: list[Scenario]) -> int:
    """Run every scenario once and return how many turns ran."""
    turns = 0
    for scenario in scenarios:
        try:
            await drive(scenario)
        except BaseException:  # noqa: BLE001 - a driver skips a scenario by raising pytest.skip
            continue
        turns += len(scenario.turns)
    return turns


async def main(repeats: int) -> dict[str, dict[str, float]]:
    """Time every driver ``repeats`` times and return the medians per turn."""
    scenarios = load_scenarios()
    results: dict[str, dict[str, float]] = {}
    for name, drive in DRIVERS.items():
        cpu: list[float] = []
        wall: list[float] = []
        turns = 0
        for _ in range(repeats):
            cpu_start, wall_start = time.process_time(), time.perf_counter()
            turns = await _one_pass(drive, scenarios)
            cpu.append(time.process_time() - cpu_start)
            wall.append(time.perf_counter() - wall_start)
        results[name] = {
            "turns": turns,
            "cpu_ms_per_turn": round(statistics.median(cpu) / turns * 1000, 3),
            "wall_ms_per_turn": round(statistics.median(wall) / turns * 1000, 3),
            "cpu_spread_ms": round((max(cpu) - min(cpu)) / turns * 1000, 3),
        }
    return results


if __name__ == "__main__":
    logger.remove()
    logging.disable(logging.CRITICAL)
    print(json.dumps(asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)), indent=1))
