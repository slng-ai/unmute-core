"""Hooks: your code before, around and after prompts, model requests, tool calls and events.

```python
hooks = Hooks()
hooks.add(PromptSuffix("/no_think"))

@hooks.before_tool
def no_refunds(run: ToolRun) -> Refuse | None:
    return Refuse("Refunds are not open today.") if run.name == "refund" else None

call = CallHooks(state, hooks=hooks)
```
"""

from unmute_core.hooks._event_tool import EventTool
from unmute_core.hooks._hooks import (
    EVENTS,
    POINTS,
    EventPattern,
    FailureHandler,
    Hooks,
    ModelHandler,
    ModelResponse,
    Observer,
    PromptInfo,
    PromptKind,
    Refuse,
    ToolHandler,
    ToolRun,
)
from unmute_core.hooks._prompt_suffix import (
    PROMPT_SUFFIX_MAX,
    PromptSuffix,
    add_prompt_suffix,
    check_prompt_suffix,
)

__all__ = [
    "EVENTS",
    "POINTS",
    "PROMPT_SUFFIX_MAX",
    "EventPattern",
    "EventTool",
    "FailureHandler",
    "Hooks",
    "ModelHandler",
    "ModelResponse",
    "Observer",
    "PromptInfo",
    "PromptKind",
    "PromptSuffix",
    "Refuse",
    "ToolHandler",
    "ToolRun",
    "add_prompt_suffix",
    "check_prompt_suffix",
]
