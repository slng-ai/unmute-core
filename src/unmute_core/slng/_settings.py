"""The SLNG Context Router's settings, read from ``SLNG_*`` variables.

This module never imports ``openai``, so a startup check can load it without the extra.
"""

import re
from typing import Annotated, ClassVar

from pydantic import (
    AfterValidator,
    Field,
    SecretStr,
    StringConstraints,
    ValidationInfo,
    field_validator,
)
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from unmute_core.settings import UNSET_ANY, EnvSettings, SettingsError

DEFAULT_MODEL = "slng/auto"
WORLD_PART = r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
"""A world part is one DNS label, like ``eu-north`` or ``us``."""


def _one_label(value: str) -> str:
    """Refuse a world part that is not one DNS label. Case was already folded, as in DNS."""
    # a hand check: to_lower runs after a pattern constraint, so the pattern would see the case
    if not re.fullmatch(WORLD_PART, value):
        raise ValueError(f"{value!r} is no world part: write one like eu-north or us")
    return value


WorldPart = Annotated[str, StringConstraints(to_lower=True), AfterValidator(_one_label)]
"""One DNS label, lowercased."""


class SlngConfigError(SettingsError):
    """The router is missing a setting, like its key or URL."""


def router_url(world_part: str) -> str:
    """Return the Context Router URL of a world part, like ``https://eu.context-router.slng.ai/v1``."""
    return f"https://{world_part}.context-router.slng.ai/v1"


class SlngSettings(EnvSettings):
    """Where the Context Router is and how to reach it.

    Attributes:
        api_key: The SLNG key, from ``SLNG_API_KEY``.
        world_part: The world part whose router to use, like ``eu-north``.
        router_url: The router's OpenAI-compatible URL, ending in ``/v1``. It wins over
            ``world_part``. One of the two is needed.
        model: The model name. ``slng/auto`` lets the router config pick and fail over.
        llm_config_id: The stored router config to use, when the key does not fix one.
    """

    model_config = SettingsConfigDict(env_prefix="SLNG_")
    config_error: ClassVar[type[SettingsError]] = SlngConfigError

    api_key: SecretStr = Field(min_length=1, description="The SLNG API key.")
    world_part: WorldPart | None = Field(
        default=None,
        description="The world part whose router to use, like eu-north. Or set SLNG_ROUTER_URL.",
    )
    router_url: str | None = Field(
        default=None,
        min_length=1,
        validate_default=True,
        description="The Context Router URL, like https://eu.context-router.slng.ai/v1. "
        "It wins over SLNG_WORLD_PART.",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        min_length=1,
        description="The model name. slng/auto lets the router config pick and fail over.",
    )
    llm_config_id: str | None = Field(
        default=None,
        min_length=1,
        description="The stored router config to use, when the key does not fix one.",
    )

    @field_validator("router_url")
    @classmethod
    def _url_or_world_part(cls, value: str | None, info: ValidationInfo) -> str | None:
        """Refuse settings with neither a URL nor a world part. A refused world part says so itself.

        A field validator, not a model validator: it must still run when the key is missing, so
        one error names every variable to set.
        """
        if value is None and info.data.get("world_part", "refused") is None:
            raise PydanticCustomError(
                UNSET_ANY, "set one of {names}", {"names": "SLNG_ROUTER_URL or SLNG_WORLD_PART"}
            )
        return value

    @property
    def url(self) -> str:
        """The router URL: ``router_url``, else the world part's."""
        if self.router_url is not None:
            return self.router_url
        assert self.world_part is not None  # ponytail: the validator needs one of the two
        return router_url(self.world_part)


class PassedSlngSettings(SlngSettings):
    """``SlngSettings`` from the values passed to it only. No ``SLNG_*`` variable is read."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Read only the constructor's values."""
        return (init_settings,)
