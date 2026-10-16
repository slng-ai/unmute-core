"""What an agent or a step hears of a LiveKit chat: its caller turns, and its shaped history.

``CallerHistory`` reads a chat's caller turns once, however many questions a request asks.
The history functions are pure: they return a new ``ChatContext`` and leave the source alone.
"""

from collections.abc import Sequence
from functools import cached_property

from livekit.agents import llm

from unmute_core.handoff import HandoffContext, summary_message
from unmute_core.model import STEP_START_CUE
from unmute_core.tasks import FINISH_TOOL_NAME, StepRun


def is_cue(item: llm.ChatItem) -> bool:
    """True for the step-start cue, which the caller never said."""
    return isinstance(item, llm.ChatMessage) and item.text_content == STEP_START_CUE


class CallerHistory:
    """The caller's turns in a LiveKit chat, found with one scan of its messages.

    Attributes:
        turns: How many caller turns the chat holds. The step-start cue counts as one.
        newest: The caller's newest message, or None. The step-start cue is not one.
        newest_id: Its ID, or an empty string when there is none.
    """

    def __init__(self, ctx: llm.ChatContext) -> None:
        """Keep the chat to read. Nothing is scanned until a value is asked for.

        Args:
            ctx: The chat. It must not change while this object is used.
        """
        self._ctx = ctx

    @cached_property
    def _callers(self) -> list[llm.ChatMessage]:
        """The chat's messages with the ``user`` role, oldest first."""
        return [message for message in self._ctx.messages() if message.role == "user"]

    @cached_property
    def turns(self) -> int:
        """How many caller turns the chat holds."""
        return len(self._callers)

    @cached_property
    def newest(self) -> llm.ChatMessage | None:
        """The caller's newest message, or None."""
        return next((m for m in reversed(self._callers) if not is_cue(m)), None)

    @property
    def newest_id(self) -> str:
        """The ID of the caller's newest turn, or an empty string when there is none."""
        return self.newest.id if self.newest is not None else ""

    def withholds_handoffs(self, opening_turn: str | None) -> bool:
        """Return True while a request still answers the caller turn an opening started on.

        An opening reply leaves out the handoff tools: a model that could hand the call on
        before it said anything would ping-pong. The first request that answers a newer
        caller turn offers them again. ``None`` means nothing is withheld.
        """
        return opening_turn is not None and self.newest_id == opening_turn


def _without_finish_calls(items: Sequence[llm.ChatItem]) -> list[llm.ChatItem]:
    """Leave out every earlier step's finish call and its output.

    Every step's finish is named ``finish``. A step that opens on the step before's finish
    reads it as its own, already made, and says goodbye instead of doing its work.
    """
    finished = {
        item.call_id
        for item in items
        if isinstance(item, llm.FunctionCall) and item.name == FINISH_TOOL_NAME
    }
    return [
        item
        for item in items
        if not (
            isinstance(item, llm.FunctionCall | llm.FunctionCallOutput) and item.call_id in finished
        )
    ]


def _without_text_signatures(items: Sequence[llm.ChatItem]) -> list[llm.ChatItem]:
    """Drop provider data from earlier spoken turns, like a Gemini thought signature.

    LiveKit sends it back as that message's ``extra_content``
    (llm/_provider_format/openai.py:79). It carries an earlier turn's reasoning into a step
    that never made it: with it kept, a verify step handed the caller back 13 of 32 times,
    against 0 of 32 without. Function calls keep theirs, because Gemini refuses a call
    replayed without its signature.
    """
    return [
        item.model_copy(update={"extra": {}})
        if isinstance(item, llm.ChatMessage) and item.role == "assistant" and item.extra
        else item
        for item in items
    ]


def spoken_history(source: llm.ChatContext) -> llm.ChatContext:
    """Return the caller's and the agent's messages, without tool calls, prompts or the cue."""
    messages = [m for m in source.messages() if m.role in ("user", "assistant") and not is_cue(m)]
    return llm.ChatContext(_without_text_signatures(messages))


def shape_chat_ctx(
    chat_ctx: llm.ChatContext, context: HandoffContext | None = None
) -> llm.ChatContext:
    """Return what the next agent hears of a call, for a handoff on a plain LiveKit agent.

    Give it to the next agent as ``Agent(chat_ctx=...)``, and return that agent from a tool.
    It keeps what ``context`` keeps, as ``UnmuteAgent.hand_off`` does. With ``summary``, it
    keeps what was said: ask your own model for the summary.

    Args:
        chat_ctx: The chat the call comes from, like ``self.chat_ctx`` in the tool.
        context: What the next agent hears. None keeps what was said, without tool calls.
    """
    context = context or HandoffContext()
    return shaped_history(
        chat_ctx,
        context.history,
        max_messages=context.max_messages,
        include_tool_calls=context.include_tool_calls,
    )


def shaped_history(
    source: llm.ChatContext,
    history: str,
    *,
    max_messages: int | None = None,
    include_tool_calls: bool = True,
) -> llm.ChatContext:
    """Return what an agent or a step hears of ``source``, by a ``History`` value.

    No system message crosses, and no step-start cue. Every earlier step's finish is left
    out: the next agent or step would read it as its own work. ``summary`` gives what was
    said. The caller puts the summary in its place when there is one.

    Args:
        source: The chat the call comes from.
        history: ``messages``, ``full``, ``last_n``, ``summary`` or ``reset``.
        max_messages: How many items ``last_n`` keeps.
        include_tool_calls: ``full`` and ``last_n`` keep tool calls and their results.
    """
    if history == "reset":
        return llm.ChatContext()
    if history in ("messages", "summary"):
        return spoken_history(source)
    copied = source.copy(
        exclude_instructions=True,
        exclude_config_update=True,
        exclude_handoff=True,
        exclude_function_call=not include_tool_calls,
    )
    items = _without_finish_calls(
        [
            item
            for item in copied.items
            if not (isinstance(item, llm.ChatMessage) and item.role in ("system", "developer"))
            and not is_cue(item)
        ]
    )
    if history == "last_n":
        items = _last_items(items, max_messages or 1)
    return llm.ChatContext(_without_text_signatures(items))


def _last_items(items: Sequence[llm.ChatItem], limit: int) -> list[llm.ChatItem]:
    """Return the newest ``limit`` items, without a tool call or result whose pair fell out.

    The same window as Pipecat's ``last_n``, which keeps a call whose result is inside.
    LiveKit's own ``truncate`` drops it (llm/chat_context.py:708).
    """
    window = list(items[-limit:])
    calls = {item.call_id for item in window if isinstance(item, llm.FunctionCall)}
    outputs = {item.call_id for item in window if isinstance(item, llm.FunctionCallOutput)}
    return [
        item
        for item in window
        if not (isinstance(item, llm.FunctionCall) and item.call_id not in outputs)
        and not (isinstance(item, llm.FunctionCallOutput) and item.call_id not in calls)
    ]


def summary_history(summary: str) -> llm.ChatContext:
    """Return the one message an agent or a step hears in place of the call."""
    return llm.ChatContext(
        [llm.ChatMessage(role="assistant", content=[summary_message(summary)["content"]])]
    )


def step_history(source: llm.ChatContext, step: StepRun) -> llm.ChatContext:
    """Return the chat a step opens on, by the core's ``entry_history`` rule.

    A step reads its own prompt. A fresh step with nothing to open on, no line to say and
    a reply to give opens on the step-start cue, as on Pipecat.

    Args:
        source: The chat the step comes from: the owner's, or its group's so far.
        step: The step.
    """
    if step.history in ("reset", "caller_turn"):
        turn = CallerHistory(source).newest if step.history == "caller_turn" else None
        if turn is not None:
            return llm.ChatContext([turn])
        if step.announce or step.listens:
            return llm.ChatContext()
        return llm.ChatContext([llm.ChatMessage(role="user", content=[STEP_START_CUE])])
    return shaped_history(
        source,
        step.history,
        max_messages=step.task.max_messages,
        include_tool_calls=step.task.include_tool_calls,
    )
