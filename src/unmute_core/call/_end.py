"""The end of a call: settle the saves, run the call-end tools, tell the observers, drain."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import SkipValidation
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.call._events import EventToolRunner
from unmute_core.hooks import Hooks
from unmute_core.prefetch import Lookup, Prefetch
from unmute_core.state import CallState
from unmute_core.tools import ToolRegistry, ToolSpec
from unmute_core.tools._spec import Draining

logger = logging.getLogger(__name__)


@original_errors
@dataclass(frozen=True, config=VALUE)
class CallEnd:
    """How a call ended.

    Attributes:
        reason: Why it ended, like ``participant_disconnected``. ``error`` after a failure.
        values: Every State value as JSON data, or None without a state.
        confirmed: The values the caller agreed to.
        transcript: The chat as ``{"role", "content"}`` messages, caller and agent only.
        duration_s: Seconds since ``CallHooks`` was built.
    """

    reason: str
    values: SkipValidation[dict[str, Any] | None]
    confirmed: list[str]
    transcript: SkipValidation[list[dict[str, Any]]]
    duration_s: float


class EndRun:
    """Run a call's end once, and stop the background lookups when the call is over.

    It owns the end's task, the registries whose webhooks drain at the end, and the
    background lookups task. ``CallHooks`` builds it and reports the end through it.
    """

    def __init__(
        self,
        *,
        state: CallState | None,
        hooks: Hooks,
        prefetch: Prefetch | None,
        registries: list[ToolRegistry],
        runner: EventToolRunner,
        timeout_s: float,
        build_end: Callable[[str], CallEnd],
    ) -> None:
        """Keep what the end needs.

        Args:
            state: The call's state, or None.
            hooks: The call's hooks, told ``call.ended``.
            prefetch: The prefetch whose tools drain too, or None.
            registries: The registries to drain. The list may grow while the call runs.
            runner: The call's tools on events: the end waits for them and fires ``call_end``.
            timeout_s: The longest each of the two end phases may take.
            build_end: Read how the call ended now, from the reason.
        """
        self._state = state
        self._hooks = hooks
        self._prefetch = prefetch
        self._registries = registries
        self._runner = runner
        self._timeout_s = timeout_s
        self._build_end = build_end
        self.task: asyncio.Task[CallEnd] | None = None
        """The end's task once it started."""
        self.lookups: asyncio.Future[None] | None = None
        """The background lookups, once started."""

    def stop_lookups(self) -> None:
        """Stop the background lookups still running: the call no longer needs them."""
        if (task := self.lookups) is not None and not task.done():
            task.cancel()

    def start(self, reason: str, *, failed: bool) -> asyncio.Task[CallEnd]:
        """Start the end once, and return its task."""
        if self.task is None:
            self.task = asyncio.ensure_future(self._end("error" if failed else reason))
        return self.task

    async def _end(self, reason: str) -> CallEnd:
        """Run the end once."""
        self._runner.began_end(reason)
        self.stop_lookups()
        await self._run_end_tools()
        ended = self._build_end(reason)
        await self._notify_and_drain(ended)
        return ended

    async def _run_end_tools(self) -> None:
        """Settle the held saves, wait for background runs, run the call-end tools."""
        try:
            async with asyncio.timeout(self._timeout_s):
                if self._state is not None:
                    await _logged(self._state.settle_held(), "settling the held saves")
                if pending := list(self._runner.pending):
                    await asyncio.wait(pending)
                await _logged(self._runner.fire("call_end"), "the call-end tools")
        except TimeoutError:
            logger.warning(
                "the call-end tools took over %.0f s and were cut short", self._timeout_s
            )

    async def _notify_and_drain(self, ended: object) -> None:
        """Tell the ``call.ended`` observers, then drain the webhooks."""
        try:
            async with asyncio.timeout(self._timeout_s):
                await self._hooks.notify("call.ended", ended)
                for executor in self._drainable():
                    await _logged(executor.drain(), "draining a webhook")
        except TimeoutError:
            logger.warning(
                "the call.ended observers took over %.0f s and were cut short", self._timeout_s
            )

    def _drainable(self) -> list[Draining]:
        """Every executor with a ``drain``, once each, the prefetch tools' too."""
        lookups = self._prefetch.entries if self._prefetch is not None else ()
        tools: list[ToolSpec] = [
            *(tool for registry in self._registries for tool in map(registry.get, registry.names)),
            *(e.run for e in lookups if isinstance(e, Lookup) and isinstance(e.run, ToolSpec)),
        ]
        executors = {
            id(tool.executor): tool.executor
            for tool in tools
            if isinstance(tool.executor, Draining)
        }
        return list(executors.values())


async def _logged(work: Awaitable[object], what: str) -> None:
    """Await work and log, never raise, when it fails."""
    try:
        await work
    except Exception:
        logger.exception("%s failed at the end of the call", what)
