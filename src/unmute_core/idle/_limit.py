"""End a call that runs past its longest allowed length.

Ported from Unmute's ``conversation.max_duration`` (unmute@868ae23
templates/livekit_v1/call.py.tmpl:440-449, templates/pipecat_v1/call.py.tmpl:3-28): no
closing line, the call ends as it stands.
"""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated

from pydantic import Field, validate_call

from unmute_core._pydantic import named_errors

logger = logging.getLogger(__name__)


class CallLimit:
    """Ends one call after ``seconds``. Start it once the call has opened."""

    @named_errors("CallLimit")
    @validate_call
    def __init__(
        self,
        seconds: Annotated[int | float, Field(gt=0)],
        end: Callable[[], Awaitable[object] | object],
    ) -> None:
        """Keep the limit. Nothing runs before ``start``.

        Args:
            seconds: The longest the call may run.
            end: Ends the call, like ``session.shutdown``.

        Raises:
            ValueError: If ``seconds`` is not above zero.
        """
        self.seconds = seconds
        self._end = end
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        """True from ``start`` until the call ends or the limit is cancelled."""
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start counting. A second call does nothing."""
        if self._task is None:
            self._task = asyncio.create_task(self._wait())

    def cancel(self) -> None:
        """Stop counting: the call ended first."""
        if self._task is not None:
            self._task.cancel()

    async def _wait(self) -> None:
        """Wait out the limit, then end the call."""
        await asyncio.sleep(self.seconds)
        logger.info("the call reached its max duration of %ss; ending it", self.seconds)
        if inspect.isawaitable(ended := self._end()):
            await ended
