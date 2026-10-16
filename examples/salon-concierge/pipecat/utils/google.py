"""The Vertex AI model service, with credentials read from the environment."""

from __future__ import annotations

from pipecat.services.google.llm import GoogleLLMService


def _google_vertex_client(api_key: str, location: str):  # noqa: ANN202 - genai.Client is only imported inside
    """Build a GenAI client that reaches Vertex with an API key.

    Args:
        api_key: The Vertex API key.
        location: A Vertex region, or us, eu or global for a multi-region host.

    Returns:
        A google.genai Client pinned to that location's Vertex host.
    """
    from google import genai as _google_genai

    # Google's multi-region host differs from its standard regional host:
    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
    if location in ("us", "eu"):
        host = f"aiplatform.{location}.rep.googleapis.com"
    elif location == "global":
        host = "aiplatform.googleapis.com"
    else:
        host = f"{location}-aiplatform.googleapis.com"
    # Pin the host too: API-key defaults and environment settings must not
    # reroute an explicit location. Collection scope keeps the API-key path
    # project-free even when location is set; it also omits the API version.
    return _google_genai.Client(
        vertexai=True,
        api_key=api_key,
        location=location,
        http_options=_google_genai.types.HttpOptions(
            base_url=f"https://{host}/v1beta1",
            base_url_resource_scope=_google_genai.types.ResourceScope.COLLECTION,
            api_version="v1beta1",
        ),
    )


class _GoogleVertexLLM(GoogleLLMService):
    """The Google LLM service, with its client swapped for a Vertex API-key one."""

    def __init__(self, *, location: str, **kwargs: object) -> None:
        """Remember the Vertex location, then build the service.

        Args:
            location: A Vertex region, or us, eu or global for a multi-region host.
            **kwargs: The service's own constructor arguments.
        """
        self._vertex_location = location
        # The kwargs are the service's own, typed there; naming them here would need Any.
        super().__init__(**kwargs)  # ty: ignore[invalid-argument-type]

    def create_client(self) -> None:
        """Build the Vertex client the service calls through."""
        self._client = _google_vertex_client(self._api_key, self._vertex_location)
