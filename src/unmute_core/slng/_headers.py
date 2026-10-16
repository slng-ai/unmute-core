"""The headers and body field the SLNG Context Router reads on a model request.

The checks match the router's own, so a bad value fails here with a clear message, before a
request is sent.
"""

import logging
import re
from collections.abc import Mapping
from typing import Annotated, TypedDict

from pydantic import (
    AfterValidator,
    BeforeValidator,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.model import SUMMARY_SITE
from unmute_core.state._call_info import CallInfo, TrackingId, no_forbidden

logger = logging.getLogger(__name__)

AGENT_ID_HEADER = "X-Slng-Agent-Id"
SESSION_ID_HEADER = "X-Slng-Session-Id"
LLM_CONFIG_ID_HEADER = "X-Slng-Llm-Config-Id"
PROMPT_SITE_HEADER = "X-Slng-Prompt-Site"
REQUEST_ID_HEADER = "X-Slng-Request-Id"
"""Names one request. The router logs it and sends it back on every answer."""

ID_MAX = 256
PROMPT_SITE = re.compile(r"task:[a-z][a-z0-9_]{0,63}")
SUMMARY_PROMPT_SITE = f"task:{SUMMARY_SITE}"
"""The prompt site a summary goes out at. The router reads only ``task:<name>`` sites, so
no task may be named ``summary``."""
VARIABLE_NAME = re.compile(r"[A-Za-z0-9_]{1,64}")
VARIABLES_MAX = 64
VARIABLE_VALUE_MAX = 4000


RouterId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=ID_MAX),
    AfterValidator(no_forbidden),
]
"""An id as the router reads it: stripped, 1 to 256 characters, no comma, whitespace, braces or bar."""


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class RouterHeaders:
    """Who is asking: the agent, the call, and optionally the router config.

    ``RouterModelCall`` builds it from the call's ``CallInfo``.

    Attributes:
        agent_id: One id per agent. The router groups cache and reports by it.
        session_id: One id per call.
        llm_config_id: The router config to use, when the key does not fix one.
    """

    agent_id: TrackingId
    session_id: TrackingId
    llm_config_id: RouterId | None = None

    def build(self, site: str = "", overlay: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return the headers of one request.

        Args:
            site: ``""`` for the main prompt, ``task:<name>``, or ``summary``. It scopes the
                router cache under the one agent id. A summary goes out at prompt site
                ``task:summary``.
            overlay: Extra headers for this request, like ``traceparent``. They never replace
                the identity headers.

        Returns:
            The overlay, then the agent, session, config and prompt site headers.

        Raises:
            ValueError: If ``site`` is not ``task:`` and a lower snake case name of at most
                64 characters. The router would quietly treat it as the main prompt.
        """
        if site == SUMMARY_SITE:
            site = SUMMARY_PROMPT_SITE
        # a method argument, not a field
        if site and not PROMPT_SITE.fullmatch(site):
            raise ValueError(
                f"prompt site {site!r} must be 'task:<name>' or 'summary', the name lower "
                "snake case, starting with a letter, at most 64 characters"
            )
        identity = {AGENT_ID_HEADER: self.agent_id, SESSION_ID_HEADER: self.session_id}
        if self.llm_config_id:
            identity[LLM_CONFIG_ID_HEADER] = self.llm_config_id
        if site:
            identity[PROMPT_SITE_HEADER] = site
        names = {name.lower() for name in identity}
        kept = {name: value for name, value in (overlay or {}).items() if name.lower() not in names}
        return kept | identity


class TrackingIds(TypedDict):
    """The keyword arguments SLNG's speech plugins take to group usage by agent and call.

    A ``TypedDict``, so ``**tracking_ids(...)`` type-checks against each plugin's own
    parameters.
    """

    external_agent_id: str
    external_session_id: str


def tracking_ids(call: CallInfo) -> TrackingIds:
    """Return the call's agent and session ids as the SLNG speech plugins take them.

    Pass them to SLNG speech-to-text and text-to-speech, so SLNG groups their usage by the
    same agent and call as the model's: ``slng.STT(..., **tracking_ids(state.call))``.

    Args:
        call: What the call carries, like ``state.call``.

    Returns:
        ``external_agent_id`` and ``external_session_id``.
    """
    return TrackingIds(external_agent_id=call.agent_id, external_session_id=call.session_id)


def _at_most_64(variables: object) -> object:
    """Refuse more than 64 names, before any one is read."""
    if isinstance(variables, Mapping) and len(variables) > VARIABLES_MAX:
        raise ValueError(
            f"template_variables holds {len(variables)} names, at most {VARIABLES_MAX}"
        )
    return variables


def _checked_items(variables: dict[str, str]) -> dict[str, str]:
    """Refuse a name that is not 1 to 64 letters, digits or underscores, or a value over 4000."""
    # one pass in order, so the first bad name or value is the one named
    for name, value in variables.items():
        if not VARIABLE_NAME.fullmatch(name):
            raise ValueError(
                f"template variable {name!r} must be 1 to 64 letters, digits or underscores"
            )
        if len(value) > VARIABLE_VALUE_MAX:
            raise ValueError(
                f"template variable {name!r} is {len(value)} characters, at most "
                f"{VARIABLE_VALUE_MAX}"
            )
    return variables


_VARIABLES = TypeAdapter(
    Annotated[dict[str, str], BeforeValidator(_at_most_64), AfterValidator(_checked_items)]
)


def shortened(variables: Mapping[str, str]) -> dict[str, str]:
    """Cut each text value to the router's 4000 characters, with a warning for each one cut.

    The router refuses a longer value, so a long saved value would fail every request. A
    value that is not text stays as it is, for ``router_body`` to refuse with its own error.

    Args:
        variables: Placeholder name to its text.

    Returns:
        The same names, each text value at most 4000 characters.
    """
    cut = {
        name: value[:VARIABLE_VALUE_MAX]
        for name, value in variables.items()
        if isinstance(value, str) and len(value) > VARIABLE_VALUE_MAX
    }
    for name in cut:
        logger.warning(
            "template variable %s is %d characters and is shortened to %d for the router; "
            "a value this long also stops the prompt being cached",
            name,
            len(variables[name]),
            VARIABLE_VALUE_MAX,
        )
    return {**variables, **cut}


def router_body(variables: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Return the ``template_variables`` body field, checked against the router's limits.

    Args:
        variables: Placeholder name to its text.

    Returns:
        ``{"template_variables": {...}}``.

    Raises:
        ValueError: If there are more than 64 names, a name is not 1 to 64 letters, digits
            or underscores, or a value is longer than 4000 characters.
    """
    try:
        return {"template_variables": _VARIABLES.validate_python(variables)}
    except ValidationError as error:
        own = error.errors()[0].get("ctx", {}).get("error")
        raise (own if isinstance(own, ValueError) else error) from None
