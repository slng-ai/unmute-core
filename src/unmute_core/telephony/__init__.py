"""Phone calls on any framework: transfers to a person, and the numbers they dial.

A ``Transfer`` says where a caller goes and what happens when nobody answers.
``transfer_tool`` turns it into a tool on the call's ``PhoneLine``, the plug point each
adapter fills for its routes: ``unmute_core.livekit.SipLine`` for LiveKit SIP, and
``TwilioLine``, ``TelnyxLine`` and ``DailySipLine`` in ``unmute_core.pipecat``.
"""

from unmute_core.telephony._numbers import (
    E164,
    E164_PATTERN,
    E164Number,
    check_destination,
    is_e164,
    refer_uri,
    sip_reason,
    sip_uri,
    sip_user,
)
from unmute_core.telephony._transfer import (
    CONNECTED_LINE,
    IN_PROGRESS,
    NO_PHONE_CALLER,
    UNAVAILABLE_LINE,
    VOICEMAIL_MESSAGE,
    NoPhoneCaller,
    OnUnavailable,
    OnVoicemail,
    PhoneLine,
    Transfer,
    TransferError,
    TransferMode,
    transfer_tool,
)

__all__ = [
    "CONNECTED_LINE",
    "E164",
    "E164_PATTERN",
    "IN_PROGRESS",
    "NO_PHONE_CALLER",
    "UNAVAILABLE_LINE",
    "VOICEMAIL_MESSAGE",
    "E164Number",
    "NoPhoneCaller",
    "OnUnavailable",
    "OnVoicemail",
    "PhoneLine",
    "Transfer",
    "TransferError",
    "TransferMode",
    "check_destination",
    "is_e164",
    "refer_uri",
    "sip_reason",
    "sip_uri",
    "sip_user",
    "transfer_tool",
]
