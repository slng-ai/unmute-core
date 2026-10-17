"""Why a LiveKit call ended: its close reasons become the shared end reasons."""

import pytest

from unmute_core.call import (
    CALLER_HUNG_UP,
    ENDED,
    ERROR,
)
from unmute_core.livekit._trace import end_reason


@pytest.mark.parametrize(
    ("close", "reason"),
    [
        ("participant_disconnected", CALLER_HUNG_UP),
        ("error", ERROR),
        ("user_initiated", ENDED),
        ("job_shutdown", ENDED),
    ],
)
def test_livekits_close_reasons_become_the_shared_end_reasons(close: str, reason: str) -> None:
    assert end_reason(close) == reason
