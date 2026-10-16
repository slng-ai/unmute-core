"""Generated Pipecat agent for salon-concierge.

Compiled by `unmute`; do not edit by hand. Prompts, model routes, and the agent
graph are baked in from the package. Secret values are read from the environment
and never written here. The agency model uses the Pipecat workers API: a main
PipelineWorker owns the transport + STT, each agent is an LLMWorker with its own
LLM and voice, and agent_transfer is activate_worker(). Tasks and task groups
run as Pipecat Flows on the owning agent: a delegate tool snapshots the shared
context, a FlowManager walks the steps as nodes, and control returns with only
a completed or unserved status.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

from call import run_bot, transport_params
from settings import require_env
from utils import knowledge

load_dotenv()

_LOGGING_CONFIGURED = False


def _configure_logging() -> None:
    """Send log output to stderr at UNMUTE_LOG_LEVEL, once per process."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("UNMUTE_LOG_LEVEL", "INFO").upper())
    _LOGGING_CONFIGURED = True


# Read, split and embed every knowledge base at module import, which is once per
# container rather than once per session.
#
# Pipecat Cloud imports this module and then calls bot() for each session, and it
# keeps a warm container for about five minutes after one ends, so this cost is
# amortised across every session that lands on it. It is deliberately not inside
# bot(): there it would run per session and a caller would wait for it.
#
# A failure here raises during import, so the container never reaches ready and
# the platform refuses the start. That is the intent: an agent that cannot read
# its documents should not take calls.
_configure_logging()
knowledge.build_indexes()

# Checked at import, so a container without its keys refuses to start rather
# than start and go quiet.
#
# A check inside run_bot alone runs once per session. The container then
# reports healthy, the platform marks the deployment ready, and the browser
# gets a valid answer to its offer. The failure happens in a background task
# that only the log sees, and the caller hears silence.
#
# run_bot still calls it, because a session that somehow starts without them
# should fail before the caller hears anything either.
require_env()


async def bot(runner_args: RunnerArguments) -> None:
    """Handle one session: build its transport, then run the agent on it.

    This is the entry point the Pipecat runner and Pipecat Cloud call.

    Args:
        runner_args: The arguments the runner started this session with.
    """
    _configure_logging()

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
