"""Pick a handler for a pipeline frame by its class, in one dictionary lookup.

An observer sees every frame of a call, and most are audio. A chain of ``isinstance``
checks tests an audio frame against every branch before it finds none. ``FrameTable``
answers from a dictionary keyed by the frame's exact class. A class it has not met is
matched once, against the rows in order, and the answer is kept.
"""

from collections.abc import Callable, Iterable

from pipecat.frames.frames import Frame
from pipecat.observers.base_observer import FramePushed

type Row[Owner] = tuple[type[Frame] | tuple[type[Frame], ...], Callable[[Owner, FramePushed], None]]


class FrameTable[Owner]:
    """Maps a frame class to the first handler whose classes it is an instance of.

    The first-match rule is the one an ``if / elif`` chain of ``isinstance`` has: rows are
    tried in the order given, and a subclass of a listed class takes that row's handler.
    """

    def __init__(self, rows: Iterable[Row[Owner]]) -> None:
        """Keep the rows, in the order they are tried."""
        self._rows = tuple(rows)
        self._found: dict[type[Frame], Callable[[Owner, FramePushed], None] | None] = {}

    def handler(self, frame: Frame) -> Callable[[Owner, FramePushed], None] | None:
        """Return the handler for this frame's class, or None when no row matches."""
        kind = type(frame)
        if kind not in self._found:
            self._found[kind] = next(
                (handler for classes, handler in self._rows if issubclass(kind, classes)), None
            )
        return self._found[kind]
