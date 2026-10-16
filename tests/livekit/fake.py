"""A scripted LiveKit LLM and a silent TTS for tests: no network, every request kept."""

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import APIConnectOptions, NotGivenOr, llm, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN


@dataclass
class Reply:
    """One scripted model reply: text, then tool calls.

    ``extra`` is the provider data of its first call, as Gemini 3 signs only that one.
    """

    text: str = ""
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def call(name: str, text: str = "", **arguments: Any) -> Reply:
    """A reply that calls one tool."""
    return Reply(text, [(name, arguments)])


@dataclass
class Request:
    """What the model was sent."""

    chat_ctx: llm.ChatContext
    tools: list[str]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        """The system text of the request."""
        return "\n".join(
            m.text_content or "" for m in self.chat_ctx.messages() if m.role == "system"
        )

    @property
    def texts(self) -> list[tuple[str, str]]:
        """Each user and assistant message, as role and text."""
        return [
            (m.role, m.text_content or "")
            for m in self.chat_ctx.messages()
            if m.role in ("user", "assistant")
        ]

    @property
    def calls(self) -> list[str]:
        """The names of the tool calls in the request."""
        return [i.name for i in self.chat_ctx.items if isinstance(i, llm.FunctionCall)]


class ScriptedLLM(llm.LLM):
    """Answers each request with the next scripted reply. An empty script answers nothing."""

    def __init__(self, *replies: Reply | str, name: str = "unknown") -> None:
        super().__init__()
        self.replies = deque(Reply(r) if isinstance(r, str) else r for r in replies)
        self.requests: list[Request] = []
        self._name = name

    @property
    def model(self) -> str:
        """The model name the spans show."""
        return self._name

    @property
    def provider(self) -> str:
        """Who serves the scripted model."""
        return "scripted"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        names = list(llm.ToolContext(tools or []).function_tools)
        extra = extra_kwargs if isinstance(extra_kwargs, dict) else {}
        self.requests.append(Request(chat_ctx.copy(), names, extra))
        reply = self.replies.popleft() if self.replies else Reply()
        return _Stream(self, reply, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _Stream(llm.LLMStream):
    """Sends one scripted reply as chunks."""

    def __init__(self, model: llm.LLM, reply: Reply, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._reply = reply

    async def _run(self) -> None:
        if self._reply.text:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=utils.shortuuid(),
                    delta=llm.ChoiceDelta(role="assistant", content=self._reply.text),
                )
            )
        for index, (name, arguments) in enumerate(self._reply.calls):
            tool = llm.FunctionToolCall(
                name=name,
                arguments=json.dumps(arguments),
                call_id=utils.shortuuid("call_"),
                extra=self._reply.extra if index == 0 else {},
            )
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=utils.shortuuid(),
                    delta=llm.ChoiceDelta(role="assistant", tool_calls=[tool]),
                )
            )


class Voice(tts.TTS):
    """A TTS that a text-only session never asks to speak. It names a voice for the spans."""

    def __init__(self, voice: str) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1
        )
        self.voice = voice

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        raise NotImplementedError("a text-only session speaks no audio")
