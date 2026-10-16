"""The agent's spec: the agent defined as data instead of code.

It holds the State as type text, the tasks and groups, which tools the agent has, the
greeting and the turn pace, in the shape the library reads. ``desk.agent`` turns it into
library objects. The tools stay Python: the spec names them, and ``desk.tools`` runs them.
"""

from typing import Any

INSTRUCTIONS = """\
You are the voice agent of Parcel Desk, a parcel delivery company. Today is {{today}}.

Help callers track a parcel, move its delivery to another day, or leave a note for a
person to call them back. Keep each reply to one or two short sentences: everything you
write is read aloud on a phone call. Never invent a parcel, a status or a day.

When the caller is done, say one short goodbye and call end_call in the same reply.
"""

TRACK_PARCEL = """\
Ask for the parcel's tracking id if the caller has not given it, then call find_parcel.
Tell the caller where the parcel is and the day it is due, in one sentence.
"""

CHANGE_DELIVERY = """\
The parcel {{parcel.tracking_id}} is due on {{parcel.delivery_day}}. Ask which day the
caller wants instead, then call reschedule_delivery with that day as YYYY-MM-DD. A day
before tomorrow cannot be booked: say so and ask for another.
"""

CALLBACK_NOTE = """\
Ask what the call back is about, in a few words, then call save_callback_note with it.
"""

SPEC: dict[str, Any] = {
    "agent_id": "parcel-desk",
    "instructions": INSTRUCTIONS,
    "greeting": {"speaks_first": "agent", "text": "Hi, this is Parcel Desk. How can I help?"},
    "turn_pace": "balanced",
    "max_duration_s": 600,
    "variables": [
        {"name": "today", "type": "Date | None", "description": "Today, on the depot clock."},
        {
            "name": "caller_phone",
            "type": "Phone | None",
            "description": "The caller's number, from the call itself.",
        },
        {
            "name": "parcel",
            "type": "Parcel | None",
            "description": "The parcel the caller asked about.",
        },
        {
            "name": "callback_note",
            "type": "str | None",
            "description": "What the caller wants a person to call them back about.",
        },
    ],
    "shapes": [
        {
            "name": "Parcel",
            "description": "One parcel, as the depot knows it.",
            "fields": [
                {"name": "tracking_id", "type": "Id"},
                {
                    "name": "status",
                    "type": 'Literal["in_transit", "out_for_delivery", "delivered", "held"]',
                },
                {"name": "delivery_day", "type": "Date | None"},
                {"name": "town", "type": "str"},
            ],
        }
    ],
    "prefetch": [
        {"clock": "today", "timezone": "Europe/London", "assign": {"today": "date"}},
        {"call_value": "caller", "source": "calling_number", "assign": ["caller_phone"]},
    ],
    "tools": {
        "find_parcel": {"announce": ["Let me look that up.", "One moment."]},
        "reschedule_delivery": {
            "announce": "Let me book that in.",
            "locked": {"tracking_id": "{{parcel.tracking_id}}"},
        },
        "save_callback_note": {"locked": {"call": "{{call.session_id}}"}},
        "end_call": {"effect": "ends_conversation"},
    },
    "tasks": {
        "mode": "record",
        "tasks": [
            {
                "name": "track_parcel",
                "when": "The caller asks where a parcel is.",
                "instructions": TRACK_PARCEL,
                "tools": ["find_parcel"],
                "finish": [{"tool": "find_parcel", "success": {"found": [True]}}],
                "assign": [{"variable": "parcel", "path": ["parcel"]}],
            },
            {
                "name": "change_delivery",
                "instructions": CHANGE_DELIVERY,
                "tools": ["reschedule_delivery"],
                "finish": [
                    {
                        "tool": "reschedule_delivery",
                        "success": {"status": ["rescheduled"]},
                        "confirm": ["Shall I move the delivery to {{input.day}}?"],
                        "say": ["Done, it now arrives on {{result.parcel.delivery_day}}."],
                    }
                ],
                "assign": [{"variable": "parcel", "path": ["parcel"]}],
            },
            {
                "name": "callback_note",
                "instructions": CALLBACK_NOTE,
                "tools": ["save_callback_note"],
                "finish": [
                    {
                        "tool": "save_callback_note",
                        "success": {"saved": [True]},
                        "say": ["Thanks, someone will call you back today. Goodbye."],
                    }
                ],
                "assign": [{"variable": "callback_note", "path": ["note"]}],
            },
        ],
        "groups": [
            {
                "name": "move_delivery",
                "when": "The caller wants a parcel delivered on another day.",
                "steps": ["track_parcel", "change_delivery"],
                "context_scope": "shared",
            },
            {
                "name": "request_callback",
                "when": "The caller wants a person to call them back.",
                "steps": ["callback_note"],
                "context_scope": "shared",
                "then": "end",
            },
        ],
        "task_only_tools": ["find_parcel", "reschedule_delivery", "save_callback_note"],
    },
}
"""Parcel Desk's spec."""
