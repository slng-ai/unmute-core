"""Every test call, and every timed turn, gets an agent id, as a deploy sets ``UNMUTE_AGENT_ID``."""

import os

os.environ.setdefault("UNMUTE_AGENT_ID", "test-agent")
