"""A caller's voice for the audio runs: each caller line spoken by SLNG text to speech.

The audio runs feed this audio to an example's real pipeline, so its voice detector, turn
detector and speech to text hear a person, not typed text. Each line is synthesised once
and kept in ``CONVERSE_VOICE_CACHE`` (a temporary folder by default), as 16 kHz, 16-bit,
mono PCM. It needs ``SLNG_API_KEY``; ``SLNG_WORLD_PART`` picks the region, eu-north by
default.
"""

import asyncio
import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path

from websockets.asyncio.client import connect

SAMPLE_RATE = 16000
CALLER_VOICE = "8634617958764b539e0cdd5eb6c7723e"
"""Fish Audio's "Abel": a voice none of the examples' agents use."""
MODEL = "slng/fish/tts:s2.1-pro"


def cache() -> Path:
    """Return the folder synthesised lines are kept in."""
    folder = Path(
        os.getenv("CONVERSE_VOICE_CACHE") or Path(tempfile.gettempdir()) / "converse-voice"
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


async def spoken(line: str, language: str = "en") -> bytes:
    """Return a caller line as 16 kHz PCM, from the cache or from SLNG."""
    key = hashlib.sha256(f"{MODEL}|{CALLER_VOICE}|{language}|{line}".encode()).hexdigest()[:24]
    path = cache() / f"{key}.pcm"
    if path.exists():
        return path.read_bytes()
    region = os.getenv("SLNG_WORLD_PART", "eu-north")
    url = f"wss://{region}.api.slng.ai/v1/bridges/unmute/tts/{MODEL}"
    audio = bytearray()
    headers = {"Authorization": f"Bearer {os.environ['SLNG_API_KEY']}"}
    async with connect(url, additional_headers=headers) as socket:
        config = {"encoding": "linear16", "sample_rate": SAMPLE_RATE, "language": language}
        await socket.send(json.dumps({"type": "init", "config": config, "voice": CALLER_VOICE}))
        await socket.send(json.dumps({"type": "text", "text": line}))
        await socket.send(json.dumps({"type": "flush"}))
        while True:
            message = await asyncio.wait_for(socket.recv(), 30)
            if isinstance(message, bytes):
                audio.extend(message)
                continue
            data = json.loads(message)
            if isinstance(data.get("audio"), str):
                audio.extend(base64.b64decode(data["audio"]))
            if data.get("type") in ("audio_end", "error"):
                break
    path.write_bytes(bytes(audio))
    return bytes(audio)
