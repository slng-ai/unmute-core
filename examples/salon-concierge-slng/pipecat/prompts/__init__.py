"""The system prompts, one Markdown file each beside this module.

Each file is read once, at import. A prompt's placeholders are filled from the
call's state where it is used, never here.
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent


def load(name: str) -> str:
    """Read one prompt exactly as the package wrote it.

    Args:
        name: The prompt's file name, without the `.md` suffix.

    Returns:
        The prompt text.
    """
    return (_HERE / f"{name}.md").read_text(encoding="utf-8")


COMPLAINT_SPECIALIST_PROMPT = load("complaint_specialist")
VERIFY_CUSTOMER_TASK_PROMPT = load("tasks/verify_customer")
CONCIERGE_PROMPT = load("concierge")
BOOK_VERIFY_CUSTOMER_TASK_PROMPT = load("tasks/book_verify_customer")
BOOK_MANAGE_BOOKING_TASK_PROMPT = load("tasks/book_manage_booking")
