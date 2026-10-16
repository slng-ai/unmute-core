"""Spoken lines picked from alternatives: a task's opening line, or one before a slow tool.

Unmute writes several alternatives so a caller does not hear the same words each time
(unmute@868ae23 internal/generate/announce.go). The pick never repeats the line a site used
last, nor the last line said anywhere.
"""

import random
from collections.abc import Sequence


def checked_lines(lines: str | Sequence[str], *, placeholders: bool = False) -> tuple[str, ...]:
    """Return announce lines stripped: one line, or alternatives.

    Args:
        lines: One line, or alternatives.
        placeholders: Allow ``{{name}}`` and ``{{call.<field>}}``, filled as the line is said.

    Raises:
        ValueError: If a line is blank, or holds a placeholder where none is allowed.
    """
    stripped = tuple(line.strip() for line in ((lines,) if isinstance(lines, str) else lines))
    if any(not line for line in stripped):
        raise ValueError("an announce line must not be blank")
    if not placeholders and any("{{" in line or "}}" in line for line in stripped):
        raise ValueError("an announce line is spoken as written and takes no placeholders")
    return stripped


class LinePicker:
    """Picks one line of several for each site of one call. Build one per call."""

    def __init__(self) -> None:
        """Start a call with no line said."""
        self._last_line: dict[str, str] = {}
        self._last_spoken: str | None = None

    def pick(self, site: str, lines: Sequence[str]) -> str | None:
        """Pick one line, or None for no lines.

        Never the line this site used last, and never the last line spoken at any site, so
        two sites that share a line do not say it twice in a row.

        Args:
            site: Who speaks it, like ``task:find_order`` or ``tool:lookup``.
            lines: The alternatives.
        """
        if not lines:
            return None
        fresh = [line for line in lines if line != self._last_spoken]
        choices = [line for line in fresh if line != self._last_line.get(site)] or fresh or lines
        self._last_line[site] = self._last_spoken = line = random.choice(choices)
        return line
