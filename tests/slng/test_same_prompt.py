"""The model sees the same final prompt on the native path and through the router.

The native path fills the prompt in the library. The router path sends it unfilled, and the
router fills it with its own rule, copied in ``router_fake.router_fill``. Both must agree.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from tests.slng.router_fake import FakeRouter, router_fill
from tests.state.unmute_state import State
from unmute_core.model import (
    ModelEvent,
    ModelRequest,
    NativeModelCall,
    TextDelta,
    router_messages,
    template_variables,
)
from unmute_core.slng import RouterModelCall
from unmute_core.state import CallInfo, CallState

PROMPT = (
    "You book haircuts. The caller rang {{ count }} times.\n"
    "Their number: {{caller_phone}}. Last booking: {{last_appointment__scheduled_date}}.\n"
    "At {{ last_appointment.scheduled_time }}.\n"
    "Reminder email: {{reminder_email}}. Literal {{not-a-name}} stays."
)


@pytest.fixture
def state() -> CallState:
    state = CallState(State, confirm={"caller_phone": "confirm_number"})
    state.save({"count": 3, "caller_phone": "+34600111222"})
    state.save(
        {
            "last_appointment": {
                "scheduled_date": "2026-09-12",
                "scheduled_time": "10:00",
                "appointment_type": "haircut",
            }
        }
    )
    return state


def messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": PROMPT},
        {"role": "developer", "content": [{"type": "text", "text": "Step: {{count}}"}]},
        {"role": "user", "content": "My number is {{caller_phone}}"},
    ]


async def native_prompt(request: ModelRequest) -> list[dict[str, Any]]:
    seen: list[ModelRequest] = []

    async def framework_llm(filled: ModelRequest) -> AsyncIterator[ModelEvent]:
        seen.append(filled)
        yield TextDelta("ok")

    [event async for event in NativeModelCall(framework_llm).stream(request)]
    return seen[0].messages


async def router_prompt(request: ModelRequest) -> list[dict[str, Any]]:
    router = FakeRouter()
    call = RouterModelCall(
        CallInfo(agent_id="salon", session_id="call-1"),
        model="m",
        base_url="unused",
        client=router.client(),
    )
    [event async for event in call.stream(request)]
    assert router.body["messages"] == router_messages(request.messages)
    return router_fill(router.body)


@pytest.mark.parametrize("site", ["", "task:confirm_number", "task:book"])
async def test_both_paths_give_the_model_the_same_prompt(state: CallState, site: str) -> None:
    chat = messages()
    request = ModelRequest(
        messages=chat, site=site, variables=template_variables(chat, state, site)
    )

    native = await native_prompt(request)
    routed = await router_prompt(request)

    assert native == routed
    assert "{{" not in native[0]["content"].replace("{{not-a-name}}", "")
    phone = "+34600111222" if site == "task:confirm_number" else "none recorded yet."
    assert f"Their number: {phone}." in native[0]["content"]
    assert "At 10:00:00." in native[0]["content"]
    assert native[2]["content"] == "My number is {{caller_phone}}"
