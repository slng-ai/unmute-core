"""Announce lines: one of several, never the same line twice in a row."""

from itertools import pairwise

from unmute_core.tools import LinePicker


def test_a_site_never_repeats_its_last_line():
    pick = LinePicker()
    lines = ["A.", "B."]
    said = [pick.pick("tool:lookup", lines) for _ in range(10)]
    assert all(first != second for first, second in pairwise(said))


def test_two_sites_sharing_a_line_do_not_say_it_twice_in_a_row():
    pick = LinePicker()
    assert pick.pick("task:a", ["Let me look."]) == "Let me look."
    assert pick.pick("tool:b", ["Let me look.", "One moment."]) == "One moment."


def test_a_single_line_is_said_each_time_and_no_lines_say_nothing():
    pick = LinePicker()
    assert [pick.pick("tool:x", ["Hold on."]) for _ in range(2)] == ["Hold on.", "Hold on."]
    assert pick.pick("tool:x", []) is None
