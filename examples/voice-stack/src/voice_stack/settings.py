"""The stack's settings, read from the environment once per process.

Every name has a default except the two keys, so a project needs only ``SLNG_API_KEY`` and
``GOOGLE_API_KEY`` in its ``.env``.
"""

from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
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
        openai_api_key: An OpenAI key. A ``gpt-`` model needs it.
        stt_model: The SLNG speech-to-text model.
        tts_model: The SLNG text-to-speech model.
    """

    model_config = SettingsConfigDict(extra="ignore")

    slng_api_key: SecretStr
    slng_world_part: str = "eu-north"
    google_api_key: SecretStr
    vertex_location: str = "eu"
    llm_model: str = Field(default="gemini-3.1-flash-lite", min_length=1)
    thinking_level: Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"] = "MINIMAL"
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

    @model_validator(mode="after")
    def _openai_has_a_key(self) -> Self:
        """Refuse a ``gpt-`` model without an OpenAI key.

        Raises:
            ValueError: If ``LLM_MODEL`` names a ``gpt-`` model and ``OPENAI_API_KEY`` is unset.
        """
        if self.on_openai and self.openai_api_key is None:
            raise ValueError(f"LLM_MODEL={self.llm_model} needs OPENAI_API_KEY")
        return self

    def openai_model(self) -> OpenAIModelCall:
        """Return the OpenAI model call, with no reasoning: Chat Completions takes tools only then."""
        assert self.openai_api_key is not None  # ponytail: _openai_has_a_key, for the type checker
        return OpenAIModelCall(
            model=self.llm_model,
            api_key=self.openai_api_key.get_secret_value(),
            params={"reasoning_effort": "none"},
        )
