"""Every fixed value this agent runs with, in one place.

Compiled from the package: names, limits, timeouts, and the environment
variables each kind of session needs. Nothing here reads a secret.
"""

from __future__ import annotations

import os

MAIN_NAME = "main"
# The longest a handoff or a finish waits for a running terminal tool to settle.
TERMINAL_SETTLE_TIMEOUT_SECS = 30.0
# Twilio Media Streams is 8 kHz mono in both directions.
PHONE_SAMPLE_RATE_HZ = 8000

# What every session needs, whatever channel it arrives on. What a *phone call*
# adds is CALL_REQUIRED_ENV below, checked separately, so this package still runs
# in the browser with nothing but model keys (V10/B3).
REQUIRED_ENV = [
    "GOOGLE_API_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "OPENAI_API_KEY",
    "SLNG_API_KEY",
]


def require_env() -> None:
    """Fail at startup, naming every required environment variable that is unset.

    Raises:
        RuntimeError: When at least one name in REQUIRED_ENV has no value.
    """
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


# What a phone call adds to REQUIRED_ENV above, checked only on a phone call: a
# browser or console session on this package reads none of these and must not be
# asked for them. The check runs before the caller hears anything, so a missing
# value can never first show up as a failed transfer on a call somebody is
# paying for.
CALL_REQUIRED_ENV = [
    "MANAGER_PHONE_NUMBER",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE_NUMBER",
]
