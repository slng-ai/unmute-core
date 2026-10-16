"""A phone agent on LiveKit SIP: it answers or dials, and can transfer the caller.

``SipLine`` is the call's phone leg. ``transfer_tool`` gives the model a tool that puts
the caller through to ``TRANSFER_NUMBER``. A job whose metadata names ``phone_number``
dials it first, and hangs up on a voicemail. Every dial and transfer goes out through the
trunk in ``SIP_ADDRESS``, ``SIP_USERNAME`` and ``SIP_PASSWORD``, from ``FROM_NUMBER``.

Run it against a LiveKit server with a SIP trunk and a dispatch rule::

    uv run python phone.py dev
"""

import json
import os

from livekit import api
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai, silero, slng
from unmute_core.livekit import DialError, SipLine, function_tools, sip_call_info
from unmute_core.telephony import Transfer, transfer_tool
from unmute_core.tools import ToolRegistry

PROMPT = (
    "You answer calls for a corner shop. Keep replies short: they are spoken aloud. "
    "When the caller asks for a person, put them through."
)

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Answer or place one call, with a transfer to a person."""
    metadata = json.loads(ctx.job.metadata or "{}")
    slng_host = f"{os.getenv('SLNG_WORLD_PART', 'eu-north')}.api.slng.ai"
    stt = slng.STT(slng_base_url=slng_host, model="soniox/speech-ai:rt-v5")
    session = AgentSession(
        stt=stt,
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=slng.TTS(
            slng_base_url=slng_host,
            model="slng/fish/tts:s2.1-pro",
            voice="8d8aec63a56c460cb7def795bdf754fa",
        ),
        vad=silero.VAD.load(),
    )
    stt.attach_to_session(session)
    trunk = api.SIPOutboundConfig(
        hostname=os.environ["SIP_ADDRESS"],
        auth_username=os.environ["SIP_USERNAME"],
        auth_password=os.environ["SIP_PASSWORD"],
    )
    line = SipLine(session, trunk=trunk, caller_id=os.environ["FROM_NUMBER"])
    person = Transfer(
        "transfer_to_person",
        os.environ["TRANSFER_NUMBER"],
        description="Put the caller through to a person when they ask for one.",
        ring_timeout_s=25,
    )
    agent = Agent(
        instructions=PROMPT,
        tools=function_tools(ToolRegistry([transfer_tool(person, line)])),
    )
    await session.start(agent=agent, room=ctx.room)
    if number := metadata.get("phone_number"):
        try:
            dialled = await line.dial(number)
        except DialError as error:
            ctx.shutdown(f"dial failed: {error.reason}")
            return
        if dialled.ended:
            return
    else:
        participant = await ctx.wait_for_participant()
        print(sip_call_info(participant, metadata))
    session.generate_reply(instructions="Greet the caller and offer to help.")


if __name__ == "__main__":
    cli.run_app(server)
