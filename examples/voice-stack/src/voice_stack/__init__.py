"""The EU voice stack the use-case examples share.

Every example describes its agent once, as a framework-neutral ``VoiceApp``. This package
runs that app on LiveKit (``voice_stack.livekit``) or on Pipecat (``voice_stack.pipecat``)
with one set of providers:

- speech to text: Soniox ``rt-v5`` through SLNG, in SLNG's ``eu-north`` region;
- text to speech: Fish Audio ``s2.1-pro`` hosted by SLNG, in the same region;
- the model: Gemini 3.1 Flash Lite on Vertex AI's EU endpoint.

``voice_stack.app`` and ``voice_stack.settings`` import no framework. Each framework module
imports only its own framework, so a project installs one of them.
"""

from voice_stack.app import Handoff, VoiceApp
from voice_stack.settings import StackSettings

__all__ = ["Handoff", "StackSettings", "VoiceApp"]
