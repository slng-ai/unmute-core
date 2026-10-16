"""Kestrel on Pipecat.

The agent is the shared ``kestrel`` package's ``APP``, the same one ``livekit/agent.py``
runs. ``voice_stack.pipecat`` builds the pipeline around it: SLNG speech in eu-north,
Gemini on Vertex EU, the turn pace and the greeting.

    uv run bot.py -t webrtc        # then open http://localhost:7860
"""

from dotenv import load_dotenv
from kestrel.agent import APP
from pipecat.runner.types import RunnerArguments
from voice_stack.pipecat import run_bot

load_dotenv()


async def bot(runner_args: RunnerArguments) -> None:
    """Run one call. The Pipecat runner calls this for each session."""
    await run_bot(APP, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
