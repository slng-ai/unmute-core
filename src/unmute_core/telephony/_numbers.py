"""Phone numbers and SIP addresses, in the forms each dial or transfer takes.

Ported from unmute@2c78a5a templates/livekit_v1/utils_telephony.py.tmpl:9-47,
templates/pipecat_v1/utils_telephony.py.tmpl:123-141 and telephony_bridge.py.tmpl:45.
"""

import re
from typing import Annotated

from pydantic import AfterValidator

E164_PATTERN = r"\+[1-9][0-9]{1,14}"
"""The text of an E.164 number: a plus, then 2 to 15 digits, the first not 0."""
E164 = re.compile(E164_PATTERN)
"""A phone number in E.164: a plus, then 2 to 15 digits."""
URI_SCHEMES = ("tel:", "sip:", "sips:")
_BUSY = frozenset({486, 600, 603})
_NO_ANSWER = frozenset({408, 480, 487})


def is_e164(number: object) -> bool:
    """Return True for an E.164 number, like ``+34910000000``."""
    return isinstance(number, str) and E164.fullmatch(number) is not None


def _check_e164(number: str) -> str:
    """Return the number as given once it is E.164.

    Raises:
        ValueError: If it is not, naming the form to use.
    """
    if not is_e164(number):
        raise ValueError(f"{number!r} must be an E.164 number, like +34910000000")
    return number


E164Number = Annotated[str, AfterValidator(_check_e164)]
"""A phone number in E.164, like ``+34910000000``, for a field of a Pydantic model."""


def check_destination(destination: str) -> str:
    """Return a transfer destination as given, once it is a number or a URI.

    Raises:
        ValueError: If it is neither an E.164 number nor a ``tel:``, ``sip:`` or ``sips:``
            URI.
    """
    if is_e164(destination) or (
        destination.startswith(URI_SCHEMES) and destination.split(":", 1)[1].strip()
    ):
        return destination
    raise ValueError(
        f"destination {destination!r} must be an E.164 number like +34910000000, or a tel:, "
        "sip: or sips: URI"
    )


def refer_uri(destination: str) -> str:
    """Return the destination as the ``Refer-To`` of a SIP REFER: a bare number gets ``tel:``.

    A carrier that needs its own host in the ``Refer-To``, like Plivo, takes the
    ``sip:+E164@<trunk>`` form as written.
    """
    return destination if destination.startswith(URI_SCHEMES) else f"tel:{destination}"


def sip_user(destination: str) -> str:
    """Return the number or SIP user of a destination, with no scheme and no host.

    LiveKit's dial refuses a full URI: "SipCallTo should be a phone number or SIP user, not
    a full SIP URI". The trunk carries the host.
    """
    if "@" in destination:
        destination = destination.rsplit("@", 1)[0]
    scheme = next((s for s in URI_SCHEMES if destination.startswith(s)), "")
    return destination.removeprefix(scheme)


def sip_uri(destination: str, host: str) -> str:
    """Return the destination as a SIP URI at your trunk's host. A SIP URI stays as written."""
    if destination.startswith(("sip:", "sips:")):
        return destination
    return f"sip:{destination.removeprefix('tel:')}@{host}"


def sip_reason(status_code: int | None) -> str:
    """Return why a SIP leg failed, in words the model can pass on.

    Args:
        status_code: The SIP status, like 486, or None when the carrier gave none.

    Returns:
        ``busy``, ``no answer``, ``sip 503``, or ``failed``.
    """
    if status_code in _BUSY:
        return "busy"
    if status_code in _NO_ANSWER:
        return "no answer"
    return f"sip {status_code}" if status_code else "failed"
