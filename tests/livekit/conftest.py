"""Skip the LiveKit tests when the ``livekit`` extra is not installed."""

import pytest

pytest.importorskip("livekit.agents")
