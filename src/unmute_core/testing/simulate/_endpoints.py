"""The three models of a simulated call: the agent's, the caller's and the judge's.

Each is any OpenAI-compatible chat endpoint: a model name, a base URL and a key. The run
reads them from the command line, then ``SIMULATE_*`` variables, then ``OPENAI_API_KEY``
and ``OPENAI_BASE_URL``. ``SIMULATE_MODEL`` and ``--model`` set all three at once.

The caller and the judge should run on a different model from the agent, so the two
sides of the call do not share one model's habits.
"""

import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from unmute_core.model import OpenAIModelCall
from unmute_core.settings import EnvSettings

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

Role = Literal["agent", "caller", "judge"]
ROLES: tuple[Role, ...] = ("agent", "caller", "judge")
RETRIES = 8
"""How often the caller and the judge retry a request the endpoint refused or dropped."""


class SimulateSettings(EnvSettings):
    """The models a run uses, read from the environment.

    Attributes:
        model: The model of every role that names none of its own.
        base_url: The endpoint of every role that names none.
        api_key: The key of every role that names none.
        agent_model: The agent's model.
        agent_base_url: The agent's endpoint.
        agent_api_key: The agent's key.
        caller_model: The simulated caller's model.
        caller_base_url: The caller's endpoint.
        caller_api_key: The caller's key.
        judge_model: The judge's model.
        judge_base_url: The judge's endpoint.
        judge_api_key: The judge's key.
        openai_base_url: The last endpoint tried: OpenAI's own variable.
        openai_api_key: The last key tried: OpenAI's own variable.
        step_summary: A GitHub Actions job's summary file, where ``--gate`` adds its verdict.
    """

    model: str | None = Field(default=None, validation_alias="SIMULATE_MODEL")
    base_url: str | None = Field(default=None, validation_alias="SIMULATE_BASE_URL")
    api_key: SecretStr | None = Field(default=None, validation_alias="SIMULATE_API_KEY")
    agent_model: str | None = Field(default=None, validation_alias="SIMULATE_AGENT_MODEL")
    agent_base_url: str | None = Field(default=None, validation_alias="SIMULATE_AGENT_BASE_URL")
    agent_api_key: SecretStr | None = Field(default=None, validation_alias="SIMULATE_AGENT_API_KEY")
    caller_model: str | None = Field(default=None, validation_alias="SIMULATE_CALLER_MODEL")
    caller_base_url: str | None = Field(default=None, validation_alias="SIMULATE_CALLER_BASE_URL")
    caller_api_key: SecretStr | None = Field(
        default=None, validation_alias="SIMULATE_CALLER_API_KEY"
    )
    judge_model: str | None = Field(default=None, validation_alias="SIMULATE_JUDGE_MODEL")
    judge_base_url: str | None = Field(default=None, validation_alias="SIMULATE_JUDGE_BASE_URL")
    judge_api_key: SecretStr | None = Field(default=None, validation_alias="SIMULATE_JUDGE_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    step_summary: Path | None = Field(default=None, validation_alias="GITHUB_STEP_SUMMARY")


class Endpoint(BaseModel):
    """One OpenAI-compatible chat endpoint and the model a role runs on.

    Attributes:
        model: The model name.
        base_url: The endpoint, ending in ``/v1`` or the provider's equivalent. None is
            OpenAI.
        api_key: The key.
        params: Body fields sent with every request, like ``{"temperature": 0.7}``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    base_url: str | None = None
    api_key: SecretStr | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    def label(self) -> str:
        """Return the model and its host, for reports. Never the key."""
        host = (self.base_url or "https://api.openai.com/v1").split("//")[-1].split("/")[0]
        return f"{self.model} @ {host}"


def endpoint(
    role: Role,
    settings: SimulateSettings,
    *,
    model: str | None = None,
    base_url: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> Endpoint:
    """Return one role's endpoint: the flags first, then its variables, then the shared ones.

    Raises:
        ValueError: If no model is named for the role.
    """
    picked = model or getattr(settings, f"{role}_model") or settings.model
    if not picked:
        raise ValueError(
            f"name the {role}'s model: --{role}-model, --model, SIMULATE_{role.upper()}_MODEL "
            "or SIMULATE_MODEL"
        )
    return Endpoint(
        model=picked,
        base_url=base_url
        or getattr(settings, f"{role}_base_url")
        or settings.base_url
        or settings.openai_base_url,
        api_key=getattr(settings, f"{role}_api_key") or settings.api_key or settings.openai_api_key,
        params=dict(params or {}),
    )


def agent_model(endpoint: Endpoint) -> OpenAIModelCall:
    """Return the ``ModelCall`` the agent runs on. It sends no SLNG header.

    A slow first chunk, like a thinking model's, gets 60 seconds.
    """
    return OpenAIModelCall(
        endpoint.model,
        base_url=endpoint.base_url,
        api_key=endpoint.api_key.get_secret_value() if endpoint.api_key else None,
        params=endpoint.params,
        timeout=60.0,
    )


def is_rate_limit(error: BaseException) -> bool:
    """Return True for a 429 from the endpoint: our own key's limit, not the agent's fault."""
    return getattr(error, "status_code", None) == 429


def is_upstream(error: BaseException) -> bool:
    """Return True for an error the endpoint caused: a 429, a 5xx, a timeout or a lost link.

    The agent did nothing wrong, so such a call is left out of the pass rate.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    name = type(error).__name__
    return isinstance(error, TimeoutError | ConnectionError) or any(
        word in name for word in ("Timeout", "Connection")
    )


def _client(endpoint: Endpoint) -> "AsyncOpenAI":
    """Return an OpenAI client for the endpoint, which retries nothing itself.

    Raises:
        ImportError: If openai is not installed, naming the extra that brings it.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as error:
        raise ImportError(
            'the simulator needs openai: uv add "unmute-core[slng]"', name="openai"
        ) from error
    return AsyncOpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key.get_secret_value() if endpoint.api_key else None,
        max_retries=0,
        timeout=120.0,
    )


class Chat:
    """A chat endpoint the caller or the judge talks to, retrying what the endpoint refused."""

    def __init__(self, endpoint: Endpoint) -> None:
        """Keep the endpoint. The client is made on the first request."""
        self.endpoint = endpoint
        self._client: AsyncOpenAI | None = None
        self.requests = 0

    async def complete(self, messages: list[dict[str, str]], **params: Any) -> str:
        """Send one chat request and return the reply's text.

        A rate limit, a 5xx, a timeout or a lost link is retried, up to ``RETRIES`` times,
        waiting longer each time.

        Args:
            messages: The chat.
            params: Body fields for this request. They win over the endpoint's.

        Returns:
            The reply's text, or "" when it has none.
        """
        if self._client is None:
            self._client = _client(self.endpoint)
        body = {**self.endpoint.params, **params}
        for attempt in range(RETRIES):
            try:
                response = await self._client.chat.completions.create(  # ty: ignore[no-matching-overload]
                    model=self.endpoint.model, messages=messages, **body
                )
                break
            except Exception as error:
                if attempt == RETRIES - 1 or not is_upstream(error):
                    raise
                logger.info("%s: retrying after %r", self.endpoint.label(), error)
                await asyncio.sleep(min(60, 3 * 2**attempt))
        self.requests += 1
        return (response.choices[0].message.content or "").strip()

    async def aclose(self) -> None:
        """Close the client, if one was made."""
        if self._client is not None:
            await self._client.close()
