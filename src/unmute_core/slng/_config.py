"""The router config a request carries inline, as the ``slng_config`` body field.

The checks match the SLNG Context Router's own, so a bad config fails when it is built, with
a clear message, not at the first request. The router lays the config over the one it
stores, one top-level field at a time. A field left out keeps the stored value.

Every key is a secret: no ``repr`` and no validation error shows it.
"""

import base64
import binascii
import functools
import json
import os
from collections import Counter
from itertools import groupby
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    HttpUrl,
    Secret,
    SecretStr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    Tag,
    WrapSerializer,
    model_validator,
)
from pydantic_core import to_json

from unmute_core.slng._blocked import warn_dropped
from unmute_core.slng._headers import RouterId

CONFIG_MAX_BYTES = 256 * 1024
"""The largest inline config the router takes, as JSON."""
# a hand-written token pattern: the router's own needs lookahead, which Rust regex has not
AUTH_HEADER = r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$"


class _Strict(BaseModel):
    """Refuses unknown fields, as the router does, and never echoes an input in an error."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


def _not_authorization(name: str) -> str:
    """Refuse ``Authorization``: the router sends the key there itself."""
    if name.lower() == "authorization":
        raise ValueError("auth_header must not be authorization: leave it out for a Bearer key")
    return name


REVEAL = {"reveal": True}
"""The serialization context that writes a secret in clear. Without it a secret dumps masked."""


def _reveal_secret(
    value: Any, handler: SerializerFunctionWrapHandler, info: SerializationInfo
) -> Any:
    """Write a secret in clear only when the dump asks for it with ``REVEAL``."""
    if isinstance(info.context, dict) and info.context.get("reveal"):
        return value.get_secret_value()
    return handler(value)


AuthHeader = Annotated[
    str, Field(min_length=1, max_length=64, pattern=AUTH_HEADER), AfterValidator(_not_authorization)
]
Key = Annotated[SecretStr, Field(min_length=1), WrapSerializer(_reveal_secret)]


class OpenAIEndpoint(_Strict):
    """An OpenAI-compatible endpoint: chat completions, or the responses API.

    Attributes:
        provider: ``openai-compat``, the default, or ``openai-responses``.
        url: The endpoint's URL, like ``https://api.openai.com/v1``.
        api_key: Its key.
        model_id: The model name the endpoint knows. Defaults to the entry's ``model``.
        auth_header: The header the key goes in, when not ``Authorization: Bearer``.
    """

    provider: Literal["openai-compat", "openai-responses"] = "openai-compat"
    url: HttpUrl
    api_key: Key
    model_id: str | None = None
    auth_header: AuthHeader | None = None


class AzureEndpoint(_Strict):
    """An Azure OpenAI deployment.

    Attributes:
        url: The resource root, like ``https://my-resource.openai.azure.com``.
        api_key: Its key.
        azure_deployment: The deployment name.
        api_version: The API version, like ``2024-10-21``.
        model_id: Accepted so a stored config validates. The router ignores it:
            azure_deployment names the model.
    """

    provider: Literal["azure"] = "azure"
    url: HttpUrl
    api_key: Key
    azure_deployment: str = Field(min_length=1)
    api_version: str = Field(min_length=1)
    model_id: str | None = None


def read_vertex_key(text: str) -> dict[str, Any]:
    """Read a Google Cloud service-account key, as Unmute does.

    Tried in order: the key's JSON, that JSON base64 encoded, then a path to the key file.

    Args:
        text: The key in one of the three forms, like the value of an env variable.

    Returns:
        The key as a dict.

    Raises:
        ValueError: If the text is none of the three. The message never shows the text.
    """
    # reads three forms and must never echo the secret, so no field constraint fits
    text = text.strip()
    if text.startswith("{"):
        return _key_json(text)
    try:
        decoded = base64.b64decode(text, validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        decoded = ""
    if decoded.lstrip().startswith("{"):
        return _key_json(decoded)
    if os.path.isfile(text):
        try:
            with open(text, encoding="utf-8") as file:
                return _key_json(file.read())
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(
                f"the Vertex key file cannot be read: {type(error).__name__}"
            ) from None
    raise ValueError(
        "the Vertex key must be a Google Cloud service-account key in one of three forms: "
        "the key JSON, that JSON base64 encoded, or a path to the key file"
    )


def _key_json(text: str) -> dict[str, Any]:
    """Parse a key's JSON into a dict.

    Raises:
        ValueError: If it is not a JSON object. The message never shows the text.
    """
    try:
        key = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("the Vertex key is not valid JSON") from None
    if not isinstance(key, dict):
        raise ValueError("the Vertex key must be a JSON object")
    return key


def _read_key(value: object) -> object:
    """Read a key given as text, and refuse one that names no type."""
    if isinstance(value, Secret):
        value = value.get_secret_value()
    key = read_vertex_key(value) if isinstance(value, str) else value
    if not isinstance(key, dict) or not key.get("type"):
        raise ValueError("vertex_credentials needs a type, like service_account")
    return key


def _service_account_fields(key: dict[str, Any]) -> dict[str, Any]:
    """Check that a service-account key holds its private key and client email."""
    if key["type"] == "service_account" and (
        missing := [name for name in ("private_key", "client_email") if not key.get(name)]
    ):
        raise ValueError(f"vertex_credentials of a service account needs {' and '.join(missing)}")
    return key


class VertexEndpoint(_Strict):
    """Gemini and other models on Google Cloud Vertex AI.

    Attributes:
        vertex_credentials: A service-account key: a dict, or text holding its JSON, that
            JSON base64 encoded, or a path to the key file.
        vertex_location: The region, like ``europe-west4`` or ``global``.
        vertex_project: The project. Defaults to the key's.
        model_id: The model name the endpoint knows. Defaults to the entry's model.
    """

    provider: Literal["vertex"] = "vertex"
    vertex_credentials: Annotated[
        Secret[Annotated[dict[str, Any], AfterValidator(_service_account_fields)]],
        BeforeValidator(_read_key),
        WrapSerializer(_reveal_secret),
    ]
    vertex_location: str = Field(min_length=1)
    vertex_project: str | None = None
    model_id: str | None = None


class BedrockEndpoint(_Strict):
    """A model on AWS Bedrock.

    Attributes:
        aws_access_key_id: The access key id.
        aws_secret_access_key: The secret access key.
        aws_region: The region, like ``eu-central-1``.
        model_id: The Bedrock model id.
        aws_session_token: A session token, for temporary credentials.
        aws_bedrock_runtime_endpoint: A runtime URL other than the region's default.
    """

    provider: Literal["bedrock"] = "bedrock"
    aws_access_key_id: Key
    aws_secret_access_key: Key
    aws_region: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    aws_session_token: Key | None = None
    aws_bedrock_runtime_endpoint: HttpUrl | None = None


def _provider(value: object) -> str:
    """Pick the endpoint class by ``provider``. Left out means ``openai-compat``, as on the router."""
    if isinstance(value, dict):
        provider = value.get("provider", "openai-compat")
    else:
        provider = getattr(value, "provider", "")
    return "openai" if provider in ("openai-compat", "openai-responses") else str(provider)


Endpoint = Annotated[
    Annotated[OpenAIEndpoint, Tag("openai")]
    | Annotated[AzureEndpoint, Tag("azure")]
    | Annotated[VertexEndpoint, Tag("vertex")]
    | Annotated[BedrockEndpoint, Tag("bedrock")],
    Discriminator(_provider),
]
"""Where an entry's model runs, picked by ``provider``. With none, an SLNG deployment."""


def _router_keeps(kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Warn once about a default request field the router drops."""
    warn_dropped(kwargs or {}, "a tier entry's kwargs")
    return kwargs


class TierEntry(_Strict):
    """One model in a tier.

    Attributes:
        model: The name a request may pin, like ``gpt-4.1-mini``.
        weight: Its share of the traffic of its priority, 1 to 100. The weights of one
            priority in one tier sum to 100.
        priority: 0 is tried first. A higher priority is a fallback.
        endpoint: Where the model runs. None means an SLNG deployment of ``model``.
        kwargs: Default request fields, like ``{"temperature": 0}``. A request's own win.
            A field the router drops, like ``api_base`` or ``metadata``, is kept with one warning.
    """

    model: str = Field(min_length=1)
    weight: int = Field(ge=1, le=100)
    priority: int = Field(default=0, ge=0, le=100)
    endpoint: Endpoint | None = None
    kwargs: Annotated[dict[str, Any] | None, AfterValidator(_router_keeps)] = None


class RouterConfig(_Strict):
    """The router config a request carries, laid over the stored one field by field.

    Attributes:
        tiers: Tier ``"1"``, ``"2"`` or ``"3"`` to its models. None keeps the stored tiers.
        default_tier: The tier ``slng/auto`` uses when it does not pick one. None keeps the
            stored value.
        cache_enabled: Whether the router may answer from its cache. None keeps the
            stored value.
        template_vars_strict: Whether the router refuses a request that leaves a
            placeholder without a value. None keeps the stored value.
        disable_reasoning: Whether the router asks the model not to reason. None keeps the
            stored value.
        cache_skip_numeric_enabled: Whether the router skips its cache for requests that
            hold numbers. None keeps the stored value.
        slice_assistant_max_chars: The most characters of an earlier assistant reply the
            router keeps in the prompt, 1 to 10000. None keeps the stored value.
        prewarm_cache_enabled: Whether the router fills its cache ahead of requests. None
            keeps the stored value.
        pii_detection_enabled: Whether the router looks for personal data. None keeps the
            stored value.
        pii_regex_packs: The pattern packs the router uses to find personal data. None
            keeps the stored value.
        pii_ner_models: The name-finding models the router uses to find personal data. None
            keeps the stored value.
        slng_flow: A map of agent id to that agent's flow, the steps the router follows for
            that agent. None keeps the stored value.
    """

    tiers: dict[Literal["1", "2", "3"], list[TierEntry]] | None = None
    default_tier: Literal[1, 2, 3] | None = None
    cache_enabled: bool | None = None
    template_vars_strict: bool | None = None
    disable_reasoning: bool | None = None
    cache_skip_numeric_enabled: bool | None = None
    slice_assistant_max_chars: int | None = Field(default=None, ge=1, le=10_000)
    prewarm_cache_enabled: bool | None = None
    pii_detection_enabled: bool | None = None
    pii_regex_packs: tuple[Literal["global", "indian", "european", "us"], ...] | None = None
    pii_ner_models: (
        tuple[Literal["gliner_multilingual", "indic_ner", "gliner_pii_en", "gliner_multi_ar"], ...]
        | None
    ) = None
    slng_flow: dict[RouterId, dict[str, Any]] | None = Field(
        default=None,
        description="Agent id to that agent's flow: the steps the router follows for it.",
    )

    @model_validator(mode="after")
    def _tiers_add_up(self) -> Self:
        """Check each tier by the router's rules.

        Raises:
            ValueError: If a tier lacks a priority-0 entry, a priority's weights do not sum
                to 100, or a model appears twice in one tier.
        """
        for tier, entries in (self.tiers or {}).items():
            if not entries:
                continue
            # cross-entry sums within one tier, which no single field can see
            if all(entry.priority for entry in entries):
                raise ValueError(f"tier {tier} needs an entry with priority 0, tried first")
            by_priority = sorted(entries, key=lambda entry: entry.priority)
            sums = {
                priority: sum(entry.weight for entry in same)
                for priority, same in groupby(by_priority, key=lambda entry: entry.priority)
            }
            for priority, total in sums.items():
                if total != 100:
                    raise ValueError(
                        f"tier {tier} priority {priority}: the weights sum to {total}, not 100"
                    )
            counts = Counter(entry.model for entry in entries)
            if twice := sorted(name for name, count in counts.items() if count > 1):
                raise ValueError(f"tier {tier} names model {twice[0]!r} twice")
        return self

    @classmethod
    def single(cls, model: str, endpoint: Endpoint | None = None) -> Self:
        """Return a config with one model in tier 1, as Unmute sends it.

        Args:
            model: The model name.
            endpoint: Where it runs. None means an SLNG deployment of ``model``.
        """
        return cls(tiers={"1": [TierEntry(model=model, weight=100, endpoint=endpoint)]})

    @functools.cached_property
    def models(self) -> frozenset[str]:
        """Every model name in the tiers, the names a request may pin."""
        return frozenset(
            entry.model for entries in (self.tiers or {}).values() for entry in entries
        )

    def wire(self) -> dict[str, Any]:
        """Return the config as the router reads it: keys in clear, unset fields left out.

        Raises:
            ValueError: If the JSON is over the router's 256 KB.
        """
        body = self.model_dump(mode="json", exclude_none=True, context=REVEAL)
        # the size of the JSON the router receives, which only the finished body shows
        if (size := len(to_json(body))) > CONFIG_MAX_BYTES:
            raise ValueError(f"the router config is {size} bytes, at most {CONFIG_MAX_BYTES}")
        return body
