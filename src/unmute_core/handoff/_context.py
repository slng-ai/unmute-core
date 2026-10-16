"""What the next agent hears of the call, and the summary that can stand in for it.

The rules and texts follow Unmute's ``utils_context.py`` (both templates) and its context
checks (``ir/validate.go``). Adapters shape their own framework's history by
``HandoffContext``. The summary is one plain model request, the same on every framework.
"""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

from pydantic import Field, SkipValidation, model_validator
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.model import STEP_START_CUE, SUMMARY_SITE, ModelCall, ModelRequest

History = Literal["messages", "full", "last_n", "summary", "reset"]
"""What the next agent or a task step hears of the call so far.

``messages`` keeps what was said, without tool calls. ``full`` keeps tool calls too.
``last_n`` keeps the newest ``max_messages`` items. ``summary`` replaces the call with one
summary. ``reset`` keeps nothing.
"""

SUMMARY_PROMPT = (
    "Summarize this conversation for the agent taking over. Keep names, facts, and open "
    "commitments. Reply with the summary only."
)
SUMMARY_PREFIX = "Summary of the conversation so far: "
SUMMARY_TIMEOUT_SECONDS = 8.0
"""The longest a summary may take. A slower one falls back to ``messages``."""


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class HandoffContext:
    """What the agent that takes the call hears of it.

    Attributes:
        history: See ``History``.
        max_messages: How many items ``last_n`` keeps. Required for ``last_n`` only.
        include_tool_calls: ``full`` and ``last_n`` keep tool calls and their results.
        summarizer: The model that writes a ``summary``. None asks the agent's own model.
    """

    history: History = "messages"
    max_messages: Annotated[int, Field(ge=1)] | None = None
    include_tool_calls: bool = True
    summarizer: SkipValidation[ModelCall | None] = None

    @model_validator(mode="after")
    def _fits_the_history(self) -> Self:
        """Refuse a ``max_messages`` or a summarizer that does not fit the history.

        Raises:
            ValueError: If ``last_n`` has no ``max_messages``, or another history has one,
                or a summarizer comes without ``summary``.
        """
        if self.history == "last_n" and self.max_messages is None:
            raise ValueError("history last_n needs max_messages of 1 or more.")
        if self.history != "last_n" and self.max_messages is not None:
            raise ValueError(f"max_messages is only valid with last_n, not {self.history}.")
        if self.history != "summary" and self.summarizer is not None:
            raise ValueError(f"summarizer is only valid with summary, not {self.history}.")
        return self


def _text(content: object) -> str:
    """Return a message's text, from a string or a list of text parts."""
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
            if isinstance(part, str | dict)
        )
    return content if isinstance(content, str) else ""


def transcript(messages: Sequence[Any]) -> str:
    """Return what the caller and the agent said, one ``role: text`` line each.

    Prompts, tool calls, tool results and the step-start cue are left out: a summary is of
    the conversation, not of the agent's instructions.

    Args:
        messages: The call in OpenAI chat format. Other objects are skipped.
    """
    return "\n".join(
        f"{message['role']}: {text}"
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in ("user", "assistant")
        and (text := _text(message.get("content")).strip())
        and text != STEP_START_CUE
    )


def summary_request(messages: Sequence[Any]) -> ModelRequest:
    """Return the request that summarizes the call so far.

    Args:
        messages: The call in OpenAI chat format.
    """
    return ModelRequest(
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": transcript(messages)},
        ],
        site=SUMMARY_SITE,
    )


def summary_message(summary: str) -> dict[str, str]:
    """Return the one message the next agent hears in place of the call."""
    return {"role": "assistant", "content": SUMMARY_PREFIX + summary.strip()}
