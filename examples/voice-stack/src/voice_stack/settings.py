"""The stack's settings, read from the environment once per process.

Every name has a default except the two keys, so a project needs only ``SLNG_API_KEY`` and
``GOOGLE_API_KEY`` in its ``.env``.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from unmute_core.model import OpenAIModelCall


class StackSettings(BaseSettings):
    """Where speech and the model run, and on which models.

    Attributes:
        slng_api_key: The SLNG key, for speech to text and text to speech.
        slng_world_part: The SLNG region. Requests never leave it. The key must belong to a
            project in this region: another region answers 401.
        google_api_key: A Vertex AI API key, for Gemini.
        vertex_location: The Vertex location: ``eu`` keeps requests in the EU.
        llm_model: The model every agent answers on: a Gemini model, or a ``gpt-`` model
            on OpenAI.
        thinking_level: How much a Gemini model thinks before it answers. Gemini 3.8 Flash
            refuses ``MINIMAL``: give it ``LOW``.
        openai_api_key: An OpenAI key, for a ``gpt-`` model.
        stt_model: The SLNG speech-to-text model.
        tts_model: The SLNG text-to-speech model.
    """

    model_config = SettingsConfigDict(extra="ignore")

    slng_api_key: SecretStr
    slng_world_part: str = "eu-north"
    google_api_key: SecretStr
    vertex_location: str = "eu"
    llm_model: str = Field(default="gemini-3.1-flash-lite", min_length=1)
    thinking_level: str = "MINIMAL"
    openai_api_key: SecretStr | None = None
    stt_model: str = "soniox/speech-ai:rt-v5"
    tts_model: str = "slng/fish/tts:s2.1-pro"

    @property
    def slng_host(self) -> str:
        """The SLNG gateway host of the region, like ``eu-north.api.slng.ai``."""
        return f"{self.slng_world_part}.api.slng.ai"

    @property
    def on_openai(self) -> bool:
        """The model runs on OpenAI, not on Vertex."""
        return self.llm_model.startswith("gpt-")

    def openai_model(self) -> OpenAIModelCall:
        """Return the OpenAI model call, with no reasoning: tool calls need none on Chat Completions.

        Raises:
            ValueError: If ``OPENAI_API_KEY`` is unset.
        """
        if self.openai_api_key is None:
            raise ValueError(f"LLM_MODEL={self.llm_model} needs OPENAI_API_KEY")
        return OpenAIModelCall(
            model=self.llm_model,
            api_key=self.openai_api_key.get_secret_value(),
            params={"reasoning_effort": "none"},
        )
