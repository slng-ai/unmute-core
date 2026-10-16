"""The stack's settings, read from the environment once per process.

Every name has a default except the two keys, so a project needs only ``SLNG_API_KEY`` and
``GOOGLE_API_KEY`` in its ``.env``.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class StackSettings(BaseSettings):
    """Where speech and the model run, and on which models.

    Attributes:
        slng_api_key: The SLNG key, for speech to text and text to speech.
        slng_world_part: The SLNG region. Requests never leave it. The key must belong to a
            project in this region: another region answers 401.
        google_api_key: A Vertex AI API key, for Gemini.
        vertex_location: The Vertex location: ``eu`` keeps requests in the EU.
        llm_model: The Gemini model every agent answers on.
        stt_model: The SLNG speech-to-text model.
        tts_model: The SLNG text-to-speech model.
    """

    model_config = SettingsConfigDict(extra="ignore")

    slng_api_key: SecretStr
    slng_world_part: str = "eu-north"
    google_api_key: SecretStr
    vertex_location: str = "eu"
    llm_model: str = Field(default="gemini-3.1-flash-lite", min_length=1)
    stt_model: str = "soniox/speech-ai:rt-v5"
    tts_model: str = "slng/fish/tts:s2.1-pro"

    @property
    def slng_host(self) -> str:
        """The SLNG gateway host of the region, like ``eu-north.api.slng.ai``."""
        return f"{self.slng_world_part}.api.slng.ai"
