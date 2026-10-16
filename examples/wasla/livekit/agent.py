"""Wasla on LiveKit Agents.

The agent is the shared ``wasla`` package's ``APP``, the same one ``pipecat/bot.py``
runs. ``voice_stack.livekit`` builds the LiveKit side around it: SLNG speech in eu-north,
Gemini on Vertex EU, the turn pace and the greeting.

    lk agent console agent.py      # talk through your microphone
    uv run agent.py dev            # a worker for your LiveKit project
"""

from dotenv import load_dotenv
from livekit.agents import cli
from voice_stack.livekit import server_for
from wasla.agent import APP

load_dotenv()

server, entrypoint = server_for(APP)

if __name__ == "__main__":
    cli.run_app(server)
