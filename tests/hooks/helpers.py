"""Hooks the tests build often."""

from typing import Any

from unmute_core.hooks import EventTool, Hooks


def checks(*check: Any) -> Hooks:
    """Hooks that run each check before the greeting."""
    hooks = Hooks()
    for each in check:
        hooks.before_call(each)
    return hooks


def on_end(observer: Any, *event_tools: EventTool) -> Hooks:
    """Hooks that tell ``observer`` how the call ended, with the given tools on events."""
    hooks = Hooks(*event_tools)
    hooks.on("call.ended")(lambda _event, end: observer(end))
    return hooks
