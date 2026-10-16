"""Keeping a realtime model's instructions current: the saves, the caller and the notes."""

import asyncio
import contextlib
from collections.abc import Callable, Mapping

from livekit.agents import llm
from livekit.agents.voice import Agent, AgentSession, ConversationItemAddedEvent

from unmute_core.call import CallHooks
from unmute_core.livekit._realtime import realtime_model
from unmute_core.realtime import SETTLE_WAIT_SECONDS, check_realtime, realtime_instructions
from unmute_core.state import CallState
from unmute_core.tools import values_block


class RealtimeSync:
    """Sends a realtime model the agent's instructions again as the call moves.

    Owns the listeners on the call state and the session, and the tasks that send the
    instructions.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        prompt: str,
        state: CallState | None,
        call: CallHooks | None,
        save_values: bool,
        owners: Mapping[str, str],
        has_delegates: Callable[[], bool],
    ) -> None:
        """Keep what the instructions read.

        Args:
            agent: The agent whose instructions are sent.
            prompt: The agent's prompt, with its placeholders.
            state: The call state, or None.
            call: The call's hooks, or None.
            save_values: The agent shows its State values, as ``save_values`` asks.
            owners: Which task owns each value, for the values block.
            has_delegates: Whether the agent has tasks to delegate to, asked on entry.
        """
        self._agent = agent
        self._prompt = prompt
        self._state = state
        self._call = call
        self._save_values = save_values
        self._owners = owners
        self._has_delegates = has_delegates
        self._unlisten: Callable[[], None] | None = None
        self._instructing: set[asyncio.Task[None]] = set()

    def text(self, session: AgentSession) -> str:
        """Return the instructions a realtime model reads now."""
        model = realtime_model(session)
        call = self._call
        values = self._save_values and self._state is not None and bool(self._state.names)
        return realtime_instructions(
            self._prompt,
            self._state,
            hooks=call.hooks if call is not None else None,
            model=model.model if model is not None else "unknown",
            blocks=(
                call.notes_block() if call is not None else None,
                values_block(self._state, owners=self._owners) if values and self._state else None,
            ),
        )

    def enter(self, model: llm.RealtimeModel) -> None:
        """Refuse what the realtime model cannot do, then follow the saves and the caller.

        LiveKit read the filled instructions as the agent entered. A model that can take
        new instructions hears them again after each save. Each caller message runs the
        call's tools on events, as a cascade request would, and their notes follow. The
        model answers the caller meanwhile: no request waits for them.

        Raises:
            ValueError: If the agent has tasks or model hooks, or ``save_values`` on a
                model that fixes its instructions.
        """
        fixed = not model.capabilities.mutable_instructions
        call = self._call
        check_realtime(
            fixed_instructions=fixed,
            tasks=self._has_delegates(),
            save_values=self._save_values,
            hooks=call.hooks if call is not None else None,
        )
        if self._unlisten is not None:
            return
        session = self._agent.session
        agent = self._agent

        async def instruct(caller: str | None) -> None:
            """Run the tools on events for a caller message, then send the instructions."""
            if call is not None and caller is not None:
                await call.before_request(caller)
            if not fixed and session.current_agent is agent:
                await agent.update_instructions(self.text(session))

        def soon(caller: str | None) -> None:
            """Start ``instruct`` beside the reply."""
            task = asyncio.create_task(instruct(caller))
            self._instructing.add(task)
            task.add_done_callback(self._instructing.discard)

        def heard(event: ConversationItemAddedEvent) -> None:
            """Pass a caller message on, while this agent holds the call."""
            item = event.item
            if (
                isinstance(item, llm.ChatMessage)
                and item.role == "user"
                and session.current_agent is agent
            ):
                soon(item.text_content)

        unlisten = self._state.listen(lambda _event: soon(None)) if self._state else None
        session.on("conversation_item_added", heard)

        def stop() -> None:
            """Stop following the saves and the caller."""
            if unlisten is not None:
                unlisten()
            session.off("conversation_item_added", heard)

        self._unlisten = stop

    async def settled(self) -> None:
        """Wait until the instructions the saves and the caller asked for have been sent.

        Waits at most ``SETTLE_WAIT_SECONDS``: a stalled connection must not hold the
        greeting back.
        """
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(SETTLE_WAIT_SECONDS):
                while self._instructing:
                    await asyncio.gather(*self._instructing, return_exceptions=True)

    def close(self) -> None:
        """Stop following the call, once the session closed."""
        if self._unlisten is not None:
            self._unlisten()
