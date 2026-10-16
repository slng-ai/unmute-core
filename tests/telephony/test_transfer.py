"""A transfer is a tool: it says its line, transfers, and ends the call or keeps helping."""

import asyncio
from dataclasses import dataclass
from typing import ClassVar

import pytest

from unmute_core.telephony import (
    NO_PHONE_CALLER,
    NoPhoneCaller,
    Transfer,
    TransferError,
    transfer_tool,
)
from unmute_core.tools import ToolError, ToolRegistry

BILLING = Transfer("send_to_billing", "+34910000000", ring_timeout_s=25)


class FakeLine:
    """A phone line that records what it was asked, and fails when told to."""

    warm: ClassVar[bool] = True

    def __init__(self, fails: Exception | None = None) -> None:
        self.fails = fails
        self.transfers: list[Transfer] = []
        self.said: list[str] = []

    async def transfer(self, transfer: Transfer) -> None:
        self.transfers.append(transfer)
        if self.fails is not None:
            raise self.fails

    async def say(self, line: str) -> None:
        self.said.append(line)


class ColdOnly(FakeLine):
    """A line with no warm transfer, like every Pipecat line."""

    warm: ClassVar[bool] = False


async def call(transfer: Transfer, line: FakeLine) -> tuple[ToolRegistry, object]:
    """Run the transfer tool through a registry, as an adapter does."""
    registry = ToolRegistry([transfer_tool(transfer, line)])
    return registry, await registry.call(transfer.name, "{}")


async def test_a_transfer_that_goes_through_ends_the_call() -> None:
    line = FakeLine()
    registry, result = await call(BILLING, line)
    assert result == {"transferred": True, "ended": False, "reason": None}
    assert registry.ends_call("send_to_billing", result)
    assert line.transfers == [BILLING]
    assert line.said == []


def test_the_tool_says_its_line_first_and_the_caller_cannot_cut_it() -> None:
    tool = transfer_tool(BILLING, FakeLine())
    assert tool.announce == ("Putting you through now, one moment.",)
    assert tool.announce_wait
    assert tool.interruption == "continue"
    assert tool.model_parameters["properties"] == {}
    warm = transfer_tool(Transfer("escalate", "+34910000000", mode="warm"), FakeLine())
    assert warm.announce == ("One moment while I bring a colleague on the line.",)
    own = transfer_tool(Transfer("a", "+34910000000", announce="Un momento."), FakeLine())
    assert own.announce == ("Un momento.",)


async def test_a_call_with_no_phone_caller_keeps_helping() -> None:
    with pytest.raises(ToolError) as raised:
        await call(BILLING, FakeLine(NoPhoneCaller()))
    assert raised.value.message == NO_PHONE_CALLER


@pytest.mark.parametrize(
    ("mode", "text"),
    [
        ("cold", "The transfer failed (busy). Tell the caller and keep helping them."),
        ("warm", "Nobody took the transfer (busy). Tell the caller and keep helping them."),
    ],
)
async def test_nobody_taking_it_returns_to_the_caller(mode: str, text: str) -> None:
    transfer = Transfer("send_to_billing", "+34910000000", mode=mode)  # ty: ignore[invalid-argument-type]
    line = FakeLine(TransferError("busy"))
    with pytest.raises(ToolError) as raised:
        await call(transfer, line)
    assert raised.value.message == text
    assert line.said == []


async def test_nobody_taking_it_says_the_line_and_hangs_up() -> None:
    transfer = Transfer(
        "send_to_billing", "+34910000000", on_unavailable="hangup", unavailable_line="Adiós."
    )
    line = FakeLine(TransferError("no answer"))
    registry, result = await call(transfer, line)
    assert result == {"transferred": False, "ended": True, "reason": "no answer"}
    assert registry.ends_call("send_to_billing", result)
    assert line.said == ["Adiós."]


def test_a_warm_transfer_needs_a_line_that_makes_one() -> None:
    warm = Transfer("escalate", "+34910000000", mode="warm", briefing="VIP caller.")
    assert transfer_tool(warm, FakeLine()).name == "escalate"
    with pytest.raises(ValueError, match="ColdOnly has no warm transfer"):
        transfer_tool(warm, ColdOnly())


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"destination": "910000000"}, "E.164 number"),
        ({"briefing": "VIP"}, "briefing is legal with a warm transfer only"),
        ({"mode": "warm", "destination": "sip:@trunk.example"}, "names no one to call"),
    ],
)
def test_a_bad_transfer_is_refused_when_built(fields: dict[str, object], message: str) -> None:
    values: dict[str, object] = {"name": "a", "destination": "+34910000000"} | fields
    with pytest.raises(ValueError, match=message):
        Transfer(**values)  # ty: ignore[invalid-argument-type]


@dataclass(slots=True)
class SlottedLine:
    warm: ClassVar[bool] = False

    async def transfer(self, transfer: Transfer) -> None: ...

    async def say(self, line: str) -> None: ...


def test_a_line_the_library_cannot_track_is_refused_when_built() -> None:
    with pytest.raises(TypeError, match="SlottedLine must be hashable and weak-referenceable"):
        transfer_tool(Transfer("t", "+34600111222"), SlottedLine())


class SlowLine(FakeLine):
    """A line whose transfer waits until the test lets it go."""

    def __init__(self, fails: Exception | None = None) -> None:
        super().__init__(fails)
        self.go = asyncio.Event()

    async def transfer(self, transfer: Transfer) -> None:
        await self.go.wait()
        await super().transfer(transfer)


async def test_one_transfer_at_a_time_and_a_retry_after_one_nobody_took() -> None:
    line = SlowLine(TransferError("busy"))
    sales = Transfer("send_to_sales", "+34910000001")
    registry = ToolRegistry([transfer_tool(BILLING, line), transfer_tool(sales, line)])
    first = asyncio.ensure_future(registry.call("send_to_billing", "{}"))
    await asyncio.sleep(0)
    for name in ("send_to_billing", "send_to_sales"):
        with pytest.raises(ToolError, match="already under way"):
            await asyncio.wait_for(registry.call(name, "{}"), 1)
    line.go.set()
    with pytest.raises(ToolError, match="busy"):
        await first
    line.fails = None
    assert await registry.call("send_to_sales", "{}") == {
        "transferred": True,
        "ended": False,
        "reason": None,
    }
    with pytest.raises(ToolError, match="already under way"):
        await registry.call("send_to_billing", "{}")
    assert [transfer.name for transfer in line.transfers] == ["send_to_billing", "send_to_sales"]


async def test_another_call_is_not_held_up_by_this_ones_transfer() -> None:
    held, other = SlowLine(), FakeLine()
    first = asyncio.ensure_future(
        ToolRegistry([transfer_tool(BILLING, held)]).call(BILLING.name, "{}")
    )
    await asyncio.sleep(0)
    second = await ToolRegistry([transfer_tool(BILLING, other)]).call(BILLING.name, "{}")
    assert second == {"transferred": True, "ended": False, "reason": None}
    held.go.set()
    assert await first == {"transferred": True, "ended": False, "reason": None}
