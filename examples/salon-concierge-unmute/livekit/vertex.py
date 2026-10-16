"""Gemini on Vertex AI with an API key, as LiveKit's Google plugin.

Copied from the ``utils/google.py`` that Unmute emits. The plugin only reaches Vertex with
Google Cloud credentials. This swaps its client for one that uses ``GOOGLE_API_KEY``.
"""

from google import genai
from livekit.plugins import google


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


class VertexLLM(google.LLM):
    """LiveKit's Google LLM plugin, with its client swapped for a Vertex API-key one."""

    def __init__(self, *, api_key: str, location: str, **kwargs: object) -> None:
        """Build the plugin, then replace its client before any request runs.

        Args:
            api_key: The Vertex API key.
            location: The Vertex location.
            **kwargs: The plugin's own arguments, like ``model``.
        """
        super().__init__(api_key=api_key, vertexai=False, **kwargs)  # ty: ignore[invalid-argument-type]
        self._client.close()
        self._client = vertex_client(api_key, location)
