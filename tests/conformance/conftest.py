"""Skip the conformance suite unless both the ``livekit`` and ``pipecat`` extras are installed."""

import pytest

pytest.importorskip("livekit.agents")
pytest.importorskip("pipecat")
