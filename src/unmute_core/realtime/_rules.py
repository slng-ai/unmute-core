"""What a realtime model can do, the same on every framework, and the texts it reads.

``SAY_LINE`` generalises Unmute's realtime greeting (unmute@2c78a5a
templates/livekit_v1/agents.py.tmpl:303-314) to every fixed line. Unmute refuses tasks,
handoffs and call state on realtime models (internal/ir/validate.go:3592-3645). The
library runs handoffs and call state where both frameworks can, and refuses the rest.
"""

from collections.abc import Sequence

from unmute_core.handoff import HandoffContext
from unmute_core.hooks import Hooks, PromptInfo
from unmute_core.model import render_prompt
from unmute_core.state import CallState

SAY_LINE = "Say this to the caller now, in these words as far as you can: {line}"
"""What a realtime model with no TTS is told, to speak one of the library's fixed lines."""
SETTLE_WAIT_SECONDS = 5.0
"""The longest a worded greeting waits for a realtime model's new instructions to be sent."""
MODEL_HOOKS = ("before_model", "wrap_model", "after_model")
"""The hook points a realtime model never reaches: no request passes through the library."""


def say_line(line: str) -> str:
    """Return the instruction that has a realtime model speak ``line``."""
    return SAY_LINE.format(line=line)


def check_realtime(
    *,
    fixed_instructions: bool,
    tasks: bool = False,
    save_values: bool = False,
    hooks: Hooks | None = None,
) -> None:
    """Refuse what an agent on a realtime model cannot do. Both adapters call it.

    Args:
        fixed_instructions: The model fixes its instructions when the session starts, as
            OpenAI's live model does.
        tasks: The agent has tasks.
        save_values: The agent has the ``save_values`` tool.
        hooks: The call's hooks.

    Raises:
        ValueError: Naming what to remove, or to run on a cascade model.
    """
    if tasks:
        raise ValueError(
            "tasks do not run on a realtime model: it keeps the conversation on its own "
            "server, so a task step cannot have its own history or a finish check. Run the "
            "agent on a cascade model (speech to text, LLM, text to speech)"
        )
    if hooks is not None and (found := [point for point in MODEL_HOOKS if hooks.has(point)]):
        raise ValueError(
            f"the {', '.join(found)} hooks never run on a realtime model: no model request "
            "passes through the library. Remove them, or run the agent on a cascade model. "
            "prompt hooks do run"
        )
    if fixed_instructions and save_values:
        raise ValueError(
            "save_values needs instructions that change during the call, and this model "
            "fixes them when the session starts. Remove save_values, or use a realtime model"
        )


def check_realtime_handoff(context: HandoffContext, *, fixed_instructions: bool) -> None:
    """Refuse a handoff a realtime model cannot follow.

    Args:
        context: How much of the call the next agent hears.
        fixed_instructions: The model fixes its instructions when the session starts.

    Raises:
        ValueError: If the model fixes its instructions, or ``context`` keeps less than
            the whole conversation.
    """
    if fixed_instructions:
        raise ValueError(
            "this model fixes its instructions when the session starts, so it cannot hand "
            "the call to another agent. Remove the handoff, or use a realtime model"
        )
    if context.history != "full" or not context.include_tool_calls:
        raise ValueError(
            "a handoff on a realtime model keeps the whole conversation, tool calls "
            "included: the model holds it on its own server. Pass "
            f'HandoffContext(history="full"), not history={context.history!r}'
        )


def realtime_instructions(
    prompt: str,
    state: CallState | None,
    *,
    hooks: Hooks | None = None,
    model: str = "unknown",
    blocks: Sequence[str | None] = (),
) -> str:
    """Return the instructions a realtime model reads: the prompt, filled, and its blocks.

    A cascade request carries the call's notes and values as system messages of their
    own. A realtime model reads one text, so they follow the prompt. Build it again after
    a save, and send it to the model.

    Args:
        prompt: The agent's prompt, placeholders still in it.
        state: The call state the placeholders read.
        hooks: The call's hooks. Their ``prompt`` hooks run on the unfilled prompt.
        model: The model's name, for the ``prompt`` hooks.
        blocks: Text that follows the prompt, like ``values_block``. None is skipped.

    Raises:
        MissingVariablesError: If the prompt names a value the state does not have.
    """
    if hooks is not None:
        prompt = hooks.run_prompt(prompt, PromptInfo(kind="agent", site="", model=model))
    return "\n\n".join([render_prompt(prompt, state), *(block for block in blocks if block)])
