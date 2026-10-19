"""Reasoning tags kept out of what the agent says.

A reasoning model's ``<think>...</think>`` block is not speech, and neither is the tag
itself. With reasoning switched off, some models still send a lone ``</think>`` before or
after the answer, and a voice would read it out.

This filter drops a complete reasoning block, and a lone opening or closing tag, from
streamed text. Text outside a block is kept as it is: deleting an answer over a stray tag
would be worse than the tag. A block left open when the stream ends is reasoning cut short,
and is dropped.

Some models mark their reasoning in their own way, like ``<|channel>thought`` ...
``<channel|>``. Give the filter those markers as pairs, or look up a model's pair with
``reasoning_markers``.
"""

import functools
import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType

REASONING_TAGS = ("think", "thought", "thinking", "reasoning")
Marker = tuple[str, str]
"""An opening and a closing marker, matched as written, like ``("<|channel>thought", "<channel|>")``."""
MODEL_REASONING_MARKERS: Mapping[str, Marker] = MappingProxyType(
    {"gemma-4-31b-it": ("<|channel>thought", "<channel|>")}
)
"""The markers of models that use their own, by model name without a provider prefix. They are
the ones LiveKit's own inference models strip."""
_NAMES = "|".join(REASONING_TAGS)
_TAG_SOURCE = rf"(?i:<(?P<slash>/?)(?P<name>{_NAMES})\b[^<>]*>)"
_TAG = re.compile(_TAG_SOURCE)
_MAX_PARTIAL = 64
"""The longest tag a chunk can end in the middle of, like ``</reasoning ...>``."""


def reasoning_markers(model: str) -> tuple[Marker, ...]:
    """Return the reasoning markers a model uses besides the standard tags.

    Args:
        model: The model name the request goes to. A provider prefix is dropped, as LiveKit
            does, so ``google/gemma-4-31b-it`` and ``gemma-4-31b-it`` are the same model.

    Returns:
        The model's opening and closing marker in a tuple, or an empty tuple for a model
        that uses only the standard tags.
    """
    marker = MODEL_REASONING_MARKERS.get(model.rsplit("/", 1)[-1])
    return (marker,) if marker is not None else ()


@functools.lru_cache(maxsize=64)
def _pattern(markers: tuple[Marker, ...]) -> re.Pattern[str]:
    """Return the pattern that finds the standard tags and the given markers."""
    if not markers:
        return _TAG
    literals = "|".join(
        f"(?P<open{index}>{re.escape(opening)})|(?P<close{index}>{re.escape(closing)})"
        for index, (opening, closing) in enumerate(markers)
    )
    return re.compile(f"{_TAG_SOURCE}|{literals}")


def _read(match: re.Match[str]) -> tuple[bool, str]:
    """Return whether a match closes a block, and the key of the block it belongs to."""
    if (name := match["name"]) is not None:
        return match["slash"] == "/", name.lower()
    group = match.lastgroup or ""
    return group.startswith("close"), "marker" + group.removeprefix("open").removeprefix("close")


def _could_be_tag(text: str) -> bool:
    """Return True when text from a ``<``, with no ``>`` yet, can still become a tag."""
    body = text[1:].lstrip("/").lower()
    name = re.match(r"[a-z]*", body)
    word = name[0] if name else ""
    if len(word) == len(body):
        return any(tag.startswith(word) for tag in REASONING_TAGS)
    return word in REASONING_TAGS and not body[len(word)].isalnum()


class ReasoningFilter:
    """Feed streamed text in order, and get back what may be spoken.

    One filter per model reply: it holds back a tag cut between two chunks, and remembers
    an open block. It always drops the standard tags in ``REASONING_TAGS``, ignoring case.
    Markers given besides them are matched as written, and a block one opens ends only at
    its own closing marker.
    """

    def __init__(self, markers: Iterable[Marker] = ()) -> None:
        """Start outside any block, with nothing held back.

        Args:
            markers: Extra pairs of an opening and a closing marker, like
                ``("<|channel>thought", "<channel|>")``. A model-specific pair comes from
                ``reasoning_markers``.

        Raises:
            ValueError: If a marker is empty, or a pair has the same opening and closing marker.
        """
        self._markers = tuple(markers)
        for opening, closing in self._markers:
            # a pair that cannot open and close is a mistake the caller should hear about now
            if not opening or not closing or opening == closing:
                raise ValueError(
                    f"a reasoning marker pair needs two different, non-empty markers, "
                    f"got {(opening, closing)!r}"
                )
        self._pattern = _pattern(self._markers)
        self._literals = tuple(marker for pair in self._markers for marker in pair)
        self._pending = ""
        self._inside: str | None = None

    def feed(self, text: str) -> str:
        """Return the speakable part of the next chunk.

        Args:
            text: The next piece of the reply.
        """
        data = self._pending + text
        self._pending = ""
        out: list[str] = []
        position = 0
        for match in self._pattern.finditer(data):
            closing, key = _read(match)
            if self._inside is None:
                out.append(data[position : match.start()])
                if not closing:
                    self._inside = key
            elif closing and key == self._inside:
                self._inside = None
            position = match.end()
        rest = data[position:]
        if (start := self._held_from(rest)) is not None:
            self._pending = rest[start:]
            rest = rest[:start]
        if self._inside is None:
            out.append(rest)
        return "".join(out)

    def _held_from(self, rest: str) -> int | None:
        """Return where the end of ``rest`` could still become a tag or marker, or None."""
        starts: list[int] = []
        start = rest.rfind("<")
        if (
            start != -1
            and ">" not in rest[start:]
            and len(rest) - start <= _MAX_PARTIAL
            and _could_be_tag(rest[start:])
        ):
            starts.append(start)
        starts.extend(
            len(rest) - size
            for marker in self._literals
            for size in range(min(len(rest), len(marker) - 1), 0, -1)
            if rest.endswith(marker[:size])
        )
        return min(starts, default=None)

    def flush(self) -> str:
        """Return the text held back at the end of the stream, unless a block is open."""
        held, self._pending = self._pending, ""
        return "" if self._inside is not None else held
