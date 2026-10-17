"""Announce lines: one of several, never the same line twice in a row."""

from itertools import pairwise

import pytest

from unmute_core.tools import LinePicker


def test_a_site_never_repeats_its_last_line():
    pick = LinePicker()
    lines = ["A.", "B."]
    said = [pick.pick("tool:lookup", lines) for _ in range(10)]
    assert all(first != second for first, second in pairwise(said))


@pytest.mark.parametrize(
    ("picks", "said"),
    [
        (
            [("task:a", ["Let me look."]), ("tool:b", ["Let me look.", "One moment."])],
            ["Let me look.", "One moment."],
        ),
        ([("tool:x", ["Hold on."]), ("tool:x", ["Hold on."])], ["Hold on.", "Hold on."]),
        ([("tool:x", [])], [None]),
    ],
    ids=["two-sites-share-a-line", "a-single-line-repeats", "no-lines-say-nothing"],
)
def test_a_line_is_picked_for_each_site(
    picks: list[tuple[str, list[str]]], said: list[str | None]
) -> None:
    pick = LinePicker()
    assert [pick.pick(site, lines) for site, lines in picks] == said
