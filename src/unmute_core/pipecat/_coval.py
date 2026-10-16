"""Find the Coval simulation that placed a Pipecat call, and how the caller arrived.

Ported from unmute@d70247b internal/generate/templates/pipecat_v1/tracing.py.tmpl:287-367 and
:405-440. Pipecat delivers a call's payload in three shapes, so the lookup tries each.
"""

import logging

from pipecat.runner.types import RunnerArguments

from unmute_core.telemetry import (
    CallOrigin,
    Coval,
    Telemetry,
    coval_simulation_id,
    log_activated,
    log_no_simulation,
)

logger = logging.getLogger(__name__)


def _simulation(runner_args: RunnerArguments, coval: Coval) -> tuple[str, str] | None:
    """Return the simulation ID and where it came from, or None.

    The order is the WebSocket upgrade headers, the SIP headers of a dial-in body, the body
    itself, a carrier's custom parameters, then ``COVAL_SIMULATION_ID``.
    """
    headers = getattr(getattr(runner_args, "websocket", None), "headers", None)
    if found := coval_simulation_id(headers):
        return found, "websocket_header"
    body = getattr(runner_args, "body", None)
    if isinstance(body, dict):
        dialin = body.get("dialin_settings")
        if isinstance(dialin, dict) and (found := coval_simulation_id(dialin.get("sip_headers"))):
            return found, "sip_header"
        if found := coval_simulation_id(body):
            return found, "call_body"
    call_data = getattr(runner_args, "call_data", None)
    nested = (
        call_data.get("body") if isinstance(call_data, dict) else getattr(call_data, "body", None)
    )
    if found := coval_simulation_id(nested if isinstance(nested, dict) else None):
        return found, "carrier_parameter"
    if coval.simulation_id:
        return coval.simulation_id, "environment"
    return None


def _origin(runner_args: RunnerArguments) -> CallOrigin | None:
    """Return how the caller reached the agent, or None when the route cannot be told.

    A dial-in body is a phone call. A WebSocket is a phone call too when the runner detected
    a carrier, and ``websocket`` when it did not. WebRTC and a Daily room are the browser.
    """
    body = getattr(runner_args, "body", None)
    if isinstance(body, dict) and body.get("dialin_settings"):
        return "phone"
    if getattr(runner_args, "websocket", None) is not None:
        carrier = getattr(runner_args, "transport_type", None)
        return "websocket" if carrier in (None, "websocket") else "phone"
    if getattr(runner_args, "webrtc_connection", None) is not None or getattr(
        runner_args, "room_url", None
    ):
        return "browser"
    return None


def correlate_coval(runner_args: RunnerArguments, coval: Coval, telemetry: Telemetry) -> None:
    """Bind the call to its Coval simulation and record how the caller arrived.

    Call it once at the start of the call, with the ``Coval`` sink the call's ``Telemetry``
    was built with. It reads the simulation ID from the runner arguments and activates the
    sink at once, on the calling thread: at call start almost nothing is held, so this is
    quick. A call no simulation names is filed as a Coval conversation when it ends. An ID
    with no ``COVAL_API_KEY`` logs a warning and sends nothing.

    Args:
        runner_args: The runner's arguments for this call, like ``WebSocketRunnerArguments``.
        coval: The call's Coval sink.
        telemetry: The call's telemetry. It gets ``unmute.call.origin``.
    """
    origin = _origin(runner_args)
    telemetry.describe_call(origin=origin)
    if (found := _simulation(runner_args, coval)) is None:
        log_no_simulation(logger, f"{origin or 'unknown-origin'} call")
        return
    simulation_id, method = found
    if not coval.api_key:
        logger.warning("Coval simulation %s found but COVAL_API_KEY is unset", simulation_id)
        return
    coval.activate(simulation_id=simulation_id, method=method)
    log_activated(logger, simulation_id, method, f"{origin or 'unknown-origin'} call")
