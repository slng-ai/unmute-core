"""The turn pace: each framework's floor and ceiling, and the options each adapter builds.

The rows are Unmute's (unmute@868ae23 internal/target/pace.go).
"""

import pytest

pytest.importorskip("livekit.agents")
pytest.importorskip("pipecat")

from unmute_core.livekit import endpointing
from unmute_core.pipecat import smart_turn_params, vad_params
from unmute_core.turn import TurnWindow, turn_window


@pytest.mark.parametrize(
    ("pace", "window"),
    [
        ("snappy", TurnWindow(0.25, 0.25, 1.2, "dynamic")),
        ("balanced", TurnWindow(0.3, 0.3, 1.6, "dynamic")),
        ("patient", TurnWindow(0.4, 0.5, 2.5, "fixed")),
        (None, TurnWindow(0.3, 0.3, 1.6, "dynamic")),
    ],
)
def test_livekit_moves_the_floor_and_the_ceiling(pace, window):
    assert turn_window(pace) == window


@pytest.mark.parametrize(
    ("pace", "ceiling"), [("snappy", 1.2), ("balanced", 1.6), ("patient", 3.0)]
)
def test_pipecat_moves_only_the_ceiling(pace, ceiling):
    window = turn_window(pace, "pipecat")
    assert (window.vad_silence, window.max_delay) == (0.2, ceiling)


def test_an_unknown_pace_is_refused():
    with pytest.raises(
        ValueError, match="turn_pace 'brisk' is not one of snappy, balanced, patient"
    ):
        turn_window("brisk")


def test_each_adapter_builds_its_own_options():
    assert endpointing(turn_window("patient")) == {
        "mode": "fixed",
        "min_delay": 0.5,
        "max_delay": 2.5,
    }
    window = turn_window("snappy", "pipecat")
    assert vad_params(window).stop_secs == 0.2
    assert smart_turn_params(window).stop_secs == 1.2


def test_a_pace_error_names_the_function_and_the_field():
    with pytest.raises(ValueError) as caught:
        turn_window("brisk")
    assert str(caught.value).splitlines()[:2] == ["1 validation error for turn_window", "pace"]


def test_a_framework_error_names_the_function_and_the_field():
    with pytest.raises(ValueError) as caught:
        turn_window("snappy", "tornado")  # ty: ignore[invalid-argument-type]
    assert str(caught.value).splitlines()[:2] == [
        "1 validation error for turn_window",
        "framework",
    ]
