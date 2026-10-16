"""The handoff tools an opening reply leaves out, for an agent and for a task step."""

from collections.abc import Collection

from livekit.agents import llm

from unmute_core.livekit._history import CallerHistory


class HandoffGate:
    """Withholds handoff tools from the first reply after an opening.

    A model that could hand the call on before it said anything would ping-pong. The gate
    opens again on the first request that answers a newer caller turn, also when the opening
    was cut off before its request.

    Attributes:
        opens: The agent speaks first when it enters: it takes a call handed to it.
    """

    def __init__(self, names: Collection[str], *, armed: bool) -> None:
        """Keep the names to withhold.

        Args:
            names: The ids of the handoff tools.
            armed: Withhold from the start. An agent is armed, with no turn yet. A step
                arms itself when it opens, through ``start_opening``.
        """
        self._names = frozenset(names)
        self._withholding = armed and bool(self._names)
        self._opening_turn: str | None = "" if armed else None
        self.opens = False

    def start_opening(self, chat_ctx: llm.ChatContext) -> None:
        """Take the caller turn the opening answers, as the agent or step enters."""
        self._withholding = bool(self._names)
        self._opening_turn = CallerHistory(chat_ctx).newest_id

    def offered(self, chat_ctx: llm.ChatContext, tools: list[llm.Tool]) -> list[llm.Tool]:
        """Return the tools a request offers, without the handoffs while they are withheld.

        The reply's tool follow-ups answer the caller turn the opening entered on, and leave
        them out too. A realtime model sends no request here, and keeps every tool.
        """
        if not self._withholding:
            return tools
        if not CallerHistory(chat_ctx).withholds_handoffs(self._opening_turn):
            self._withholding = False
            return tools
        return [tool for tool in tools if tool.id not in self._names]
