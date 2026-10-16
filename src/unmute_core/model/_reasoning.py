"""Reasoning tags kept out of what the agent says.

A reasoning model's ``<think>...</think>`` block is not speech, and neither is the tag
itself. With reasoning switched off, some models still send a lone ``</think>`` before or
after the answer, and a voice would read it out.

This filter drops a complete reasoning block, and a lone opening or closing tag, from
streamed text. Text outside a block is kept as it is: deleting an answer over a stray tag
would be worse than the tag. A block left open when the stream ends is reasoning cut short,
and is dropped.
"""

import re

REASONING_TAGS = ("think", "thought", "thinking", "reasoning")
_TAG = re.compile(r"<(/?)(" + "|".join(REASONING_TAGS) + r")\b[^<>]*>", flags=re.IGNORECASE)
_MAX_PARTIAL = 64
"""The longest tag a chunk can end in the middle of, like ``</reasoning ...>``."""


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
    an open block.
    """

    def __init__(self) -> None:
        """Start outside any block, with nothing held back."""
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
        for match in _TAG.finditer(data):
            closing, name = match[1] == "/", match[2].lower()
            if self._inside is None:
                out.append(data[position : match.start()])
                if not closing:
                    self._inside = name
            elif closing and name == self._inside:
                self._inside = None
            position = match.end()
        rest = data[position:]
        start = rest.rfind("<")
        if (
            start != -1
            and ">" not in rest[start:]
            and len(rest) - start <= _MAX_PARTIAL
            and _could_be_tag(rest[start:])
        ):
            self._pending = rest[start:]
            rest = rest[:start]
        if self._inside is None:
            out.append(rest)
        return "".join(out)

    def flush(self) -> str:
        """Return the text held back at the end of the stream, unless a block is open."""
        held, self._pending = self._pending, ""
        return "" if self._inside is not None else held
