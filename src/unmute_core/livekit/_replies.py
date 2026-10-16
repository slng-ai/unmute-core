"""What a task step's model replies said and called, as its finish needs to know them."""

import asyncio
import contextlib
from dataclasses import dataclass, field

from livekit.agents import llm

from unmute_core.tasks import FINISH_TOOL_NAME, StepRun

REPLY_SETTLE_SECONDS = 10.0
"""How long a finish waits for the reply that made it to end."""


@dataclass
class Reply:
    """One model reply of a step, as a finish in it needs to know it.

    Attributes:
        unheard: The step tool and result text this reply answers, or None.
        spoke: The reply said something.
        called: The reply called a tool.
        text: What it said, piece by piece.
        done: Set once the reply has ended.
    """

    unheard: tuple[str, str] | None
    spoke: bool = False
    called: bool = False
    text: list[str] = field(default_factory=list)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    async def settled(self) -> bool:
        """Wait for the reply to end. False when it has not ended in time."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.done.wait(), timeout=REPLY_SETTLE_SECONDS)
        return self.done.is_set()


class ReplyTracker:
    """Follows a step's replies: which tool calls each made and which results were heard.

    Owns the ids of the step's own calls, the reply that made each finish, and the finish
    the core refused before the caller heard anything.
    """

    def __init__(self, step: StepRun) -> None:
        """Keep the step the replies belong to.

        Args:
            step: The core's step, which decides what a result left unheard is.
        """
        self._step = step
        self._calls: set[str] = set()
        self._replies: dict[str, Reply] = {}
        self._refused_call: str | None = None

    def unheard(self, chat_ctx: llm.ChatContext) -> tuple[set[str], tuple[str, str] | None]:
        """Return the step's own calls this request answers, and the result left unheard."""
        answered = [
            item
            for item in chat_ctx.items
            if isinstance(item, llm.FunctionCallOutput) and item.call_id in self._calls
        ]
        unheard = self._step.unheard(
            [(item.name, item.output) for item in answered],
            refused=any(item.call_id == self._refused_call for item in answered),
        )
        return {item.call_id for item in answered}, unheard

    def read(self, chunk: llm.ChatChunk | str, reply: Reply) -> None:
        """Note what one chunk of a reply said and which calls it made."""
        if isinstance(chunk, str):
            reply.spoke = reply.spoke or bool(chunk.strip())
            reply.text.append(chunk)
            return
        if chunk.delta is None:
            return
        reply.text.append(chunk.delta.content or "")
        reply.spoke = reply.spoke or bool((chunk.delta.content or "").strip())
        for call in chunk.delta.tool_calls:
            reply.called = True
            self._calls.add(call.call_id)
            if call.name == FINISH_TOOL_NAME:
                self._replies[call.call_id] = reply

    def reply_of(self, call_id: str) -> Reply | None:
        """Return, and forget, the reply that made a finish call."""
        return self._replies.pop(call_id, None)

    def refuse(self, call_id: str) -> None:
        """Note that the finish call was refused before the caller heard the result."""
        self._refused_call = call_id

    def answered(self, call_ids: set[str]) -> None:
        """Forget the calls a request answered, once its reply was not empty."""
        self._calls -= call_ids
