"""The deterministic services the bundled agents' tools call.

A check knows what a tool returned, because every mock answers the same way each time:

* ``availability``: the free times on a day, the same for each weekday.
* ``orders``: two known orders, one on time and one late.
* ``appointments``: the caller's one upcoming appointment, two days from today.
* ``customer_records``: opens a record keyed on the caller's number, or reports a
  duplicate.
* ``sms``: a text message queued, the way a carrier answers.
* ``hook``: a customer's own webhook, reached.
* ``fixed``: the result the agent file gives.

Each call is kept in the call's ``MockLog``, with every argument the tool ran with, the
locked ones included. The model never sees those, so the checks read them here.
"""

import datetime as dt
import uuid
from collections.abc import Callable, Mapping
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from unmute_core.state import CallState

AVAILABLE_TIMES: Mapping[int, tuple[str, ...]] = {
    0: ("09:00", "11:30", "16:00"),
    1: ("10:00", "13:30", "15:00", "17:30"),
    2: ("09:30", "12:00", "16:30"),
    3: ("10:30", "14:00", "18:00"),
    4: ("09:00", "11:00", "15:30"),
    5: ("10:00", "12:30"),
    6: (),
}
"""Free times per weekday, Monday first. The rest of each day is taken."""

ORDERS: Mapping[str, Mapping[str, object]] = {
    "A48213": {
        "found": True,
        "order_id": "A48213",
        "status": "in transit",
        "carrier": "DHL",
        "location": "Madrid sorting centre",
        "eta": "Friday",
        "late": False,
    },
    "B77105": {
        "found": True,
        "order_id": "B77105",
        "status": "delayed",
        "carrier": "SEUR",
        "location": "held at the Valencia depot",
        "eta": "next Wednesday",
        "late": True,
    },
}
"""The orders the order lookup knows."""

APPOINTMENT_NUMBER = "34612345678"
"""The digits of the one number with a booked appointment."""

ENQUIRIES = ("new_customer", "existing_customer", "complaint", "other")
"""The enquiry words a customer record takes."""


class MockResult(BaseModel):
    """Any result a mock returns. The checks read it whole, so no field is dropped."""

    model_config = ConfigDict(extra="allow")


def _digits(value: object) -> str:
    """Return only the digits of a value."""
    return "".join(ch for ch in str(value or "") if ch.isdigit())


class MockLog:
    """Every mock call of one call, in order, and the records the call opened.

    Attributes:
        timezone: The agent's time zone: today is today there.
        requests: Each call: the mock, the tool, its arguments and its result.
    """

    def __init__(self, timezone: str) -> None:
        """Start a call with no requests."""
        self.timezone = timezone
        self.requests: list[dict[str, Any]] = []
        self._records: dict[str, dict[str, Any]] = {}

    def today(self) -> dt.date:
        """Return today's date in the agent's time zone."""
        return dt.datetime.now(ZoneInfo(self.timezone)).date()

    def answer(
        self, mock: str, tool: str, arguments: Mapping[str, Any], fixed: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Answer one tool call, and keep it.

        Args:
            mock: The mock's name, one of ``MOCKS``.
            tool: The tool that called it.
            arguments: Every argument the tool ran with, the locked ones included.
            fixed: The result a ``fixed`` mock returns.

        Returns:
            The result.
        """
        result = MOCKS[mock](self, arguments, fixed)
        self.requests.append(
            {"mock": mock, "tool": tool, "params": dict(arguments), "result": result}
        )
        return result

    def availability(self, arguments: Mapping[str, Any], fixed: Mapping[str, Any]) -> dict:
        """Return the free times on the day asked for."""
        day = str(arguments.get("day") or arguments.get("date") or "")
        try:
            weekday = dt.date.fromisoformat(day).weekday()
        except ValueError:
            return {"ok": False, "error": "day must be written year-month-day, like 2026-03-19"}
        return {
            "ok": True,
            "day": day,
            "service": arguments.get("service"),
            "available_times": list(AVAILABLE_TIMES[weekday]),
        }

    def orders(self, arguments: Mapping[str, Any], fixed: Mapping[str, Any]) -> dict:
        """Return where a known order is, or that it was not found."""
        order_id = str(arguments.get("order_id") or "").strip().upper().replace(" ", "")
        return dict(ORDERS.get(order_id, {"found": False, "order_id": order_id}))

    def appointments(self, arguments: Mapping[str, Any], fixed: Mapping[str, Any]) -> dict:
        """Return the caller's upcoming appointment, when the number is the known one."""
        phone = _digits(arguments.get("phone") or arguments.get("phone_number"))
        found = bool(phone) and (
            phone.endswith(APPOINTMENT_NUMBER[-9:]) or APPOINTMENT_NUMBER.endswith(phone[-9:])
        )
        day = (self.today() + dt.timedelta(days=2)).isoformat()
        appointment = {
            "id": "APT-5521",
            "service": "check-up and cleaning",
            "day": day,
            "time": "11:00",
        }
        return {"ok": True, "found": found, "appointments": [appointment] if found else []}

    def customer_records(self, arguments: Mapping[str, Any], fixed: Mapping[str, Any]) -> dict:
        """Open a customer record, or report the one the number already has."""
        phone = _digits(arguments.get("phone"))
        enquiry = str(arguments.get("enquiry") or "")
        if not phone or enquiry not in ENQUIRIES:
            return {"record_id": "", "opened_on": "", "enquiry": enquiry, "status": "invalid"}
        if (existing := self._records.get(phone)) is not None:
            return {**existing, "status": "duplicate"}
        record = {
            "record_id": f"REC-{phone[-4:]}-{len(self._records) + 1:03d}",
            "opened_on": self.today().isoformat(),
            "enquiry": enquiry,
            "status": "created",
        }
        self._records[phone] = record
        return dict(record)

    def sms(self, arguments: Mapping[str, Any], fixed: Mapping[str, Any]) -> dict:
        """Queue a text message, as a carrier's API answers."""
        to = str(arguments.get("to") or "")
        return {"sid": "SM" + uuid.uuid5(uuid.NAMESPACE_URL, to).hex, "status": "queued", "to": to}

    def hook(self, arguments: Mapping[str, Any], fixed: Mapping[str, Any]) -> dict:
        """Say a customer's own webhook was reached."""
        return {"ok": True}

    def fixed(self, arguments: Mapping[str, Any], fixed: Mapping[str, Any]) -> dict:
        """Return the result the agent file gives."""
        return dict(fixed)


MOCKS: Mapping[str, Callable[[MockLog, Mapping[str, Any], Mapping[str, Any]], dict]] = {
    "availability": MockLog.availability,
    "orders": MockLog.orders,
    "appointments": MockLog.appointments,
    "customer_records": MockLog.customer_records,
    "sms": MockLog.sms,
    "hook": MockLog.hook,
    "fixed": MockLog.fixed,
}
"""Each mock by the name an agent file gives it."""


class MockTool:
    """The executor of one mocked tool: it answers from the call's ``MockLog``.

    Attributes:
        mock: The mock's name.
        tool: The tool's name.
        fixed: The result a ``fixed`` mock returns.
        log: The call's log, set once the call starts.
    """

    def __init__(self, mock: str, tool: str, fixed: Mapping[str, Any], log: MockLog) -> None:
        """Keep the mock, the tool and the call's log."""
        self.mock = mock
        self.tool = tool
        self.fixed = dict(fixed)
        self.log = log

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        """Answer the call from the mock."""
        return self.log.answer(self.mock, self.tool, arguments, self.fixed)
