"""Gemini on Vertex AI with an API key, as Pipecat's Google LLM service.

Copied from the ``utils/google.py`` that Unmute emits. The service only reaches Vertex
with Google Cloud credentials. This builds its client with ``GOOGLE_API_KEY`` instead.
"""

from google import genai
from pipecat.services.google.llm import GoogleLLMService


def vertex_client(api_key: str, location: str) -> genai.Client:
    """Return a GenAI client pinned to one location's Vertex host, with an API key.

    Requests ask for Vertex's priority tier, which trims the slow outliers.

    Args:
        api_key: The Vertex API key.
        location: A Vertex region, or ``us``, ``eu`` or ``global`` for a multi-region host.
    """
    if location in ("us", "eu"):
        host = f"aiplatform.{location}.rep.googleapis.com"
    elif location == "global":
        host = "aiplatform.googleapis.com"
    else:
        host = f"{location}-aiplatform.googleapis.com"
    return genai.Client(
        vertexai=True,
        api_key=api_key,
        location=location,
        http_options=genai.types.HttpOptions(
            base_url=f"https://{host}/v1beta1",
            base_url_resource_scope=genai.types.ResourceScope.COLLECTION,
            api_version="v1beta1",
            headers={"X-Vertex-AI-LLM-Shared-Request-Type": "priority"},
        ),
    )


class VertexLLM(GoogleLLMService):
    """Pipecat's Google LLM service, with a Vertex API-key client."""

    def __init__(self, *, location: str, **kwargs: object) -> None:
        """Remember the location, then build the service.

        Args:
            location: The Vertex location.
            **kwargs: The service's own arguments, like ``api_key`` and ``settings``.
        """
        self._vertex_location = location
        super().__init__(**kwargs)  # ty: ignore[invalid-argument-type]

    def create_client(self) -> None:
        """Build the Vertex client the service calls through."""
        self._client = vertex_client(self._api_key, self._vertex_location)
