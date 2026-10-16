"""Holding a LiveKit job's shutdown until the work a session's close starts is done."""

import asyncio
import contextlib
from collections.abc import Callable, Iterable
from typing import Any

from livekit.agents import get_job_context

CLOSE_WAIT_SECONDS = 15.0
"""How long the job's shutdown waits for the close to finish."""


class CloseGuard:
    """Holds the job's shutdown until the session closed and its close work finished.

    LiveKit runs the job's shutdown callbacks beside the session's own close, so the callback
    waits for the close first, then for the tasks the close started. It gives up after
    ``CLOSE_WAIT_SECONDS``.
    """

    def __init__(
        self,
        pending: Callable[[], Iterable[asyncio.Future[Any] | None]],
        *,
        reraise: bool = False,
    ) -> None:
        """Keep what to wait for once the session has closed.

        Args:
            pending: Returns the tasks the close started, read after the close, so a task
                made by the close is seen. None entries are skipped.
            reraise: Raise a task's failure into the job's shutdown, which logs it.
        """
        self._closed = asyncio.Event()
        self._pending = pending
        self._reraise = reraise

    def closed(self) -> None:
        """Note that the session closed."""
        self._closed.set()

    async def wait(self) -> None:
        """Wait for the close and its tasks, or give up after ``CLOSE_WAIT_SECONDS``."""
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(CLOSE_WAIT_SECONDS):
                await self._closed.wait()
                if tasks := [task for task in self._pending() if task is not None]:
                    await asyncio.wait(tasks)
                    if self._reraise:
                        for task in tasks:
                            task.result()

    def hold_job(self) -> None:
        """Make the job's shutdown wait for this guard. Does nothing outside a job."""
        if (job := get_job_context(required=False)) is not None:
            job.add_shutdown_callback(self.wait)
