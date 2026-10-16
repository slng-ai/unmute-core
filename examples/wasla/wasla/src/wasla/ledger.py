"""Wasla's collections ledger and the tools the agent calls on it.

``Ledger`` stands in for the lender's backend: three invented accounts, and what each call
records about them, in memory. The outcome codes are the ones collections teams use:

| Code | Meaning |
|---|---|
| ``PTP`` | a promise to pay, with a day |
| ``CB`` | a callback day and time |
| ``WN`` | wrong number: the number is taken off the account |
| ``HARDSHIP``, ``LEGAL``, ``DISPUTE``, ``RTP`` | a concern, in the holder's own words |

Every tool takes typed arguments and returns a typed result that carries the words code
says. The account id and the call's id are locked: the model never sends them. A write is
keyed by the call, so a second yes records nothing new.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from threading import Lock
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from pydantic import AfterValidator, BaseModel, Field
from unmute_core.state import BirthDate
from unmute_core.tools import ToolError

from wasla import words
from wasla.state import ConcernKind

TIMEZONE = "Asia/Dubai"
ZONE = ZoneInfo(TIMEZONE)
WINDOW_DAYS = 10
"""A promise or a callback may name a day up to this many days from today."""
TRIES = 3
"""Wrong dates of birth before the account is locked for today."""
SUPPORT = (
    "Wasla support is on eight hundred, nine two seven, five two, from nine AM to six PM, "
    "or by email at help at wasla dot a e."
)
CODES: dict[str, str] = {
    "hardship": "HARDSHIP",
    "legal-matter": "LEGAL",
    "purchase-dispute": "DISPUTE",
    "refusal-to-pay": "RTP",
}
SEED: tuple[dict[str, Any], ...] = (
    {
        "account_id": "WA-1001",
        "first_name": "Layla",
        "last_name": "Haddad",
        "birth_date": date(1990, 3, 14),
        "amount": Decimal("434.50"),
        "merchant": "Falcon Electronics",
        "purchase_date": date(2026, 8, 21),
        "due_date": date(2026, 9, 20),
    },
    {
        "account_id": "WA-1002",
        "first_name": "Omar",
        "last_name": "Khalil",
        "birth_date": date(1985, 11, 2),
        "amount": Decimal("1249.00"),
        "merchant": "Oasis Furniture",
        "purchase_date": date(2026, 8, 2),
        "due_date": date(2026, 9, 2),
    },
    {
        "account_id": "WA-1003",
        "first_name": "Sara",
        "last_name": "Nasser",
        "birth_date": date(1998, 7, 30),
        "amount": Decimal("89.75"),
        "merchant": "Saffron Fashion",
        "purchase_date": date(2026, 9, 12),
        "due_date": date(2026, 10, 12),
    },
)
"""Invented people, amounts and shops. Any resemblance to a real one is a coincidence."""


def today() -> date:
    """Return today's date on the Dubai clock. Tests replace it."""
    return datetime.now(ZONE).date()


def in_window(value: date) -> date:
    """Refuse a day in the past or after the deadline, with words for the holder.

    Raises:
        ValueError: When the day is outside today to today plus ten days.
    """
    if value < today():
        raise ValueError("Say: That day has already gone. Which other day would suit you?")
    deadline = today() + timedelta(days=WINDOW_DAYS)
    if value > deadline:
        raise ValueError(
            f"{value.isoformat()} is after the deadline, {deadline.isoformat()}. Say: I can "
            f"only note a day up to {words.day(deadline)}. Which day by then works for you?"
        )
    return value


WindowDay = Annotated[
    date,
    AfterValidator(in_window),
    Field(description="The day, as YYYY-MM-DD, from today up to the deadline."),
]


class Opened(BaseModel):
    """What ``open_account`` returns: who the call is for, and the last day it may name."""

    first_name: str
    last_name: str
    deadline: date


class Verified(BaseModel):
    """What ``verify_holder`` returns: the account's details only when the date matched."""

    status: Literal["verified", "mismatch", "locked"]
    birth_date: date | None = None
    tries_left: int = 0
    amount_words: str | None = None
    merchant: str | None = None
    purchase_spoken: str | None = None
    due_spoken: str | None = None


class GoodTime(BaseModel):
    """What ``good_time`` returns: whether to go on, and what code says next."""

    good_time: bool
    said: str


class Recorded(BaseModel):
    """What every write returns: the outcome code, its reference, and the words for it."""

    status: Literal["PTP", "CB", "WN", "HARDSHIP", "LEGAL", "DISPUTE", "RTP"]
    reference: str
    kind: ConcernKind | None = None
    when: str = ""


class Guide(BaseModel):
    """What ``payment_policy`` returns."""

    policy: str


@dataclass
class Ledger:
    """The accounts, every record a call made, and the wrong dates of birth so far.

    Thread-safe: tools may run in threads.
    """

    accounts: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {a["account_id"]: a for a in SEED}
    )
    records: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    wrong_tries: dict[str, int] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def _account(self, account_id: str) -> dict[str, Any]:
        """Return one account, or refuse a call about an account that does not exist."""
        found = self.accounts.get(account_id)
        if found is None:
            raise ToolError(f"no account {account_id}")
        return found

    def _write(self, call: str, code: str, account_id: str, **fields: object) -> dict[str, Any]:
        """Record one outcome once per call: the same call and code again returns it."""
        with self.lock:
            key = (call, code)
            if key not in self.records:
                reference = f"{code}-{len(self.records) + 1:04d}"
                self.records[key] = {
                    "code": code,
                    "account_id": account_id,
                    "reference": reference,
                    **fields,
                }
            return self.records[key]

    def codes(self, account_id: str) -> list[str]:
        """Return the outcome codes recorded on an account, in order."""
        return [r["code"] for r in self.records.values() if r["account_id"] == account_id]

    def open_account(self, account_id: str) -> Opened:
        """Read whose account a call is about, before the greeting. Nothing else of it."""
        found = self._account(account_id)
        return Opened(
            first_name=found["first_name"],
            last_name=found["last_name"],
            deadline=today() + timedelta(days=WINDOW_DAYS),
        )

    def verify_holder(
        self,
        account_id: str,
        birth_date: Annotated[
            BirthDate, Field(description="Their date of birth as YYYY-MM-DD. Never guess a part.")
        ],
    ) -> Verified:
        """Check the holder's date of birth. Call it as soon as they say it.

        It first reads the date back, so never read it back yourself. Call it again with the
        same date after their yes.
        """
        found = self._account(account_id)
        with self.lock:
            tries = self.wrong_tries.get(account_id, 0)
            if tries >= TRIES:
                return Verified(status="locked")
            if found["birth_date"] == birth_date:
                self.wrong_tries.pop(account_id, None)
                return Verified(
                    status="verified",
                    birth_date=birth_date,
                    amount_words=words.amount(found["amount"]),
                    merchant=found["merchant"],
                    purchase_spoken=words.day(found["purchase_date"]),
                    due_spoken=words.day(found["due_date"]),
                )
            self.wrong_tries[account_id] = tries = tries + 1
            if tries >= TRIES:
                return Verified(status="locked")
            return Verified(status="mismatch", tries_left=TRIES - tries)

    def good_time(
        self,
        good_time: Annotated[
            bool, Field(description="Whether now is a good time for them to talk.")
        ],
        amount_words: str,
        merchant: str,
    ) -> GoodTime:
        """Note whether now is a good time to talk about the account. Code says what follows."""
        if not good_time:
            return GoodTime(good_time=False, said="No problem at all.")
        return GoodTime(
            good_time=True,
            said=(
                "Thank you. This call is recorded for training and quality. I'm calling about "
                f"{amount_words} for your purchase from {merchant}, which is now overdue. Could "
                "you make the payment today?"
            ),
        )

    def record_promise(self, day: WindowDay, account_id: str, call: str) -> Recorded:
        """Record the holder's promise to pay the full amount on a day.

        It first reads the promise back, so never read it back yourself. Call it again with
        the same day after their yes.
        """
        record = self._write(call, "PTP", account_id, day=day.isoformat())
        when = "today" if day == today() else f"on {words.day(day)}"
        return Recorded(status="PTP", reference=record["reference"], when=when)

    def schedule_callback(
        self,
        day: WindowDay,
        at: Annotated[time, Field(description="The time, HH:MM on a 24-hour clock.")],
        account_id: str,
        call: str,
    ) -> Recorded:
        """Note a day and time to call back. Call it once you have both.

        It first reads them back, so never read them back yourself. Call it again with the
        same values after their yes.
        """
        record = self._write(call, "CB", account_id, day=day.isoformat(), at=at.isoformat())
        return Recorded(
            status="CB",
            reference=record["reference"],
            when=f"on {words.day(day)} at {words.clock(at)}",
        )

    def mark_wrong_number(self, account_id: str, call: str) -> Recorded:
        """Take this number off the list, so it is not called again. Call it after their yes."""
        record = self._write(call, "WN", account_id)
        return Recorded(status="WN", reference=record["reference"])

    def note_concern(
        self,
        kind: Annotated[
            ConcernKind,
            Field(
                description=(
                    "hardship: illness, disability, losing a job, money trouble. "
                    "legal-matter: police, a lawyer, a regulator, a threat. "
                    "purchase-dispute: they never made or got the purchase, or suspect "
                    "fraud. refusal-to-pay: they will not pay in this period."
                )
            ),
        ],
        note: Annotated[
            str,
            AfterValidator(lambda text: text.strip().rstrip(".?! ")),
            Field(min_length=1, description="What they said about it, in their own words."),
        ],
        account_id: str,
        call: str,
    ) -> Recorded:
        """Note what stops the holder from promising a payment, for the right team.

        It first reads the note back, so never read it back yourself. Call it again with the
        same values after their yes.
        """
        code = CODES[kind]
        record = self._write(call, code, account_id, note=note.strip())
        return Recorded(status=code, reference=record["reference"], kind=kind)  # ty: ignore[invalid-argument-type]


POLICY = """\
- How to pay: only in the Wasla app or on the Wasla website. No bank transfer, cash,
  prepaid card or account number is accepted. Never give one.
- Payment plans and new due dates: the account is overdue, and Wasla does not offer plans
  or reschedule an overdue payment. The full amount is paid in the app or on the website.
- Hardship (illness, disability, losing a job, suspected fraud): the holder contacts Wasla
  support with documents, and support looks at it with them.
- Support: eight hundred, nine two seven, five two, from nine AM to six PM, or email help
  at wasla dot a e. Read the number digit by digit.
"""


def payment_policy() -> Guide:
    """Read the payment policy: how to pay, plans, a new due date, hardship, support.

    Call it only when the holder asks about one of these, and answer from it.
    """
    return Guide(policy=POLICY)
