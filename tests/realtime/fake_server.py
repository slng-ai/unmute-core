"""A local fake of OpenAI's Realtime API (GA protocol, websocket), in text only.

LiveKit's own OpenAI realtime engine and Pipecat's ``OpenAIRealtimeLLMService`` both talk to
it with no network. The server keeps the conversation of each connection, and answers every
response with a ``ModelCall``: the instructions, the conversation and the tools go in as one
``ModelRequest``, and its text and tool calls come out as the realtime events a real model
sends. No audio bytes are ever sent. Output text goes out as an audio transcript, as a model
with audio output sends it, unless the session or the response asks for text only.
"""

import asyncio
import contextlib
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from aiohttp import WSMsgType, web

from unmute_core.model import ModelCall, ModelRequest, TextDelta, ToolCall, Usage
from unmute_core.realtime import SAY_LINE

SAY_MARKER = SAY_LINE.partition("{line}")[0]
"""The text that comes before a fixed line in a ``say_line`` instruction."""
THINK_SECONDS = 0.2
"""How long the model takes to answer the caller: long enough for a client to take the turn."""
ITEM_SECONDS = 0.1
"""How long the model takes to write a tool call after the reply's previous item.

A client runs a call as soon as its arguments are done, so it acts before the next call.
"""
ROUTES = ("/v1/realtime", "/realtime")
"""The paths that accept the websocket: LiveKit and Pipecat both end up on the first."""


@dataclass
class _Connection:
    """One client connection: its session settings, its conversation and its responses."""

    ws: web.WebSocketResponse
    session: dict[str, Any] = field(default_factory=dict)
    items: list[dict[str, Any]] = field(default_factory=list)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn: asyncio.Lock = field(default_factory=asyncio.Lock)
    responses: set[asyncio.Task[None]] = field(default_factory=set)

    @property
    def instructions(self) -> str:
        """The session instructions, empty when none were set."""
        return str(self.session.get("instructions") or "")

    @property
    def modality(self) -> str:
        """``audio`` or ``text``: how the session wants its answers."""
        modalities = self.session.get("output_modalities") or self.session.get("modalities")
        return "text" if modalities and "audio" not in modalities else "audio"


class FakeRealtimeServer:
    """A fake Realtime API server on a free local port, answering with a ``ModelCall``.

    Use it as an async context manager. Point LiveKit's ``RealtimeModel`` at ``http_url``
    and Pipecat's ``OpenAIRealtimeLLMService`` at ``ws_url``.

    Attributes:
        requests: Every request the model got, in order. A reply to a ``SAY_LINE``
            instruction is not here: it never reaches the model.
        received: Every event the client sent, as parsed JSON.
        refused: How many ``response.create`` came while a response ran. The server
            refuses them, as OpenAI's does, and the client never gets that reply.
        done: How many ``response.done`` the server has sent: one per response.
        tools_on_lines: How many replies to a ``SAY_LINE`` instruction let the model call
            a tool. A real model may call one instead of saying the line.
    """

    def __init__(self, model: ModelCall) -> None:
        """Keep the model that answers. The server starts in ``__aenter__``."""
        self.model = model
        self.requests: list[ModelRequest] = []
        self.received: list[dict[str, Any]] = []
        self.refused = 0
        self.done = 0
        self.tools_on_lines = 0
        self._runner: web.AppRunner | None = None
        self._port = 0
        self._connections: list[_Connection] = []
        self._connected = asyncio.Event()
        self._active = 0
        self._ids = itertools.count(1)

    @property
    def http_url(self) -> str:
        """The base URL for LiveKit's ``RealtimeModel(base_url=...)``."""
        return f"http://127.0.0.1:{self._port}/v1"

    @property
    def ws_url(self) -> str:
        """The base URL for Pipecat's ``OpenAIRealtimeLLMService(base_url=...)``."""
        return f"ws://127.0.0.1:{self._port}/v1/realtime"

    async def __aenter__(self) -> "FakeRealtimeServer":
        """Start listening on a free port of 127.0.0.1."""
        app = web.Application()
        for route in ROUTES:
            app.router.add_get(route, self._accept)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self._port = site._server.sockets[0].getsockname()[1]  # ty: ignore[unresolved-attribute]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Cancel the open responses, close the connections and stop listening."""
        for conn in self._connections:
            for task in conn.responses:
                task.cancel()
        if self._runner is not None:
            await self._runner.cleanup()

    async def caller_says(self, text: str, *, timeout: float = 5.0) -> None:
        """Play server VAD for one spoken turn on the latest connection, then reply.

        It first waits until the client has sent its session setup and no response runs.
        Sends the speech start and stop, the committed buffer, the user item and its
        transcript. After ``THINK_SECONDS``, as server VAD with ``create_response`` does
        after a real model's pause, it starts a response. The client has taken the turn by
        then, so its interruption does not cut the reply.
        It returns when the events are sent, not when the response is done: use
        ``wait_idle`` for that.
        """
        await self.wait_idle(
            lambda: any(event.get("type") == "session.update" for event in self.received),
            timeout=timeout,
        )
        conn = self._connections[-1]
        item_id = self._next("item")
        previous = conn.items[-1]["id"] if conn.items else None
        await self._send(
            conn, "input_audio_buffer.speech_started", audio_start_ms=0, item_id=item_id
        )
        await self._send(conn, "input_audio_buffer.speech_stopped", audio_end_ms=1, item_id=item_id)
        await self._send(
            conn, "input_audio_buffer.committed", item_id=item_id, previous_item_id=previous
        )
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": [{"type": "input_audio", "transcript": None}],
        }
        conn.items.append(item)
        await self._send(conn, "conversation.item.added", previous_item_id=previous, item=item)
        done = {**item, "content": [{"type": "input_audio", "transcript": text}]}
        conn.items[-1] = done
        await self._send(conn, "conversation.item.done", previous_item_id=previous, item=done)
        await self._send(
            conn,
            "conversation.item.input_audio_transcription.completed",
            item_id=item_id,
            content_index=0,
            transcript=text,
        )
        await asyncio.sleep(THINK_SECONDS)
        self._start_response(conn, {})

    async def wait_idle(
        self, first: Callable[[], bool] = lambda: True, *, timeout: float = 5.0
    ) -> None:
        """Wait until a client has connected, ``first()`` is true and no response runs.

        An idle server is also idle before the client has sent its follow-up to a tool
        call. So ``first`` names what must have happened first, like every scripted reply
        of the turn taken by a request.

        Raises:
            TimeoutError: When no client connects, ``first()`` stays false, or the server
                is still busy, after ``timeout`` seconds.
        """
        async with asyncio.timeout(timeout):
            await self._connected.wait()
            while self._active or not first():
                await asyncio.sleep(0)

    async def wait_for(self, condition: Callable[[], bool], *, timeout: float = 5.0) -> None:
        """Wait until ``condition()`` is true, polling, or raise ``TimeoutError``."""
        async with asyncio.timeout(timeout):
            while not condition():
                await asyncio.sleep(0)

    def _next(self, prefix: str) -> str:
        """Return the next id with this prefix: ``resp_1``, ``item_2``, ``evt_3``."""
        return f"{prefix}_{next(self._ids)}"

    async def _send(self, conn: _Connection, kind: str, **fields: Any) -> None:
        """Send one server event, and ignore a connection the client already closed."""
        event = {"event_id": self._next("evt"), "type": kind, **fields}
        with contextlib.suppress(ConnectionError, RuntimeError):
            async with conn.send_lock:
                await conn.ws.send_str(json.dumps(event))

    async def _accept(self, request: web.Request) -> web.WebSocketResponse:
        """Accept one client websocket and serve it until it closes."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        conn = _Connection(ws=ws, session={"model": request.query.get("model", "fake-realtime")})
        self._connections.append(conn)
        self._connected.set()
        await self._send(conn, "session.created", session=self._session_view(conn))
        async for message in ws:
            if message.type != WSMsgType.TEXT:
                continue
            try:
                event = json.loads(message.data)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                self.received.append(event)
                await self._handle(conn, event)
        for task in conn.responses:
            task.cancel()
        return ws

    def _session_view(self, conn: _Connection) -> dict[str, Any]:
        """The session as a server reports it: no tools or audio settings, which clients skip."""
        return {
            "type": "realtime",
            "id": "sess_fake",
            "object": "realtime.session",
            "model": conn.session.get("model"),
            "instructions": conn.instructions,
            "output_modalities": [conn.modality],
        }

    async def _handle(self, conn: _Connection, event: dict[str, Any]) -> None:
        """React to one client event. An unknown event is ignored."""
        match event.get("type"):
            case "session.update":
                conn.session.update(event.get("session") or {})
                await self._send(conn, "session.updated", session=self._session_view(conn))
            case "conversation.item.create":
                await self._create_item(conn, event)
            case "conversation.item.delete":
                await self._delete_item(conn, event)
            case "conversation.item.truncate":
                await self._send(
                    conn,
                    "conversation.item.truncated",
                    item_id=event.get("item_id"),
                    content_index=event.get("content_index", 0),
                    audio_end_ms=event.get("audio_end_ms", 0),
                )
            case "response.create" if conn.responses:
                self.refused += 1
                await self._send(
                    conn,
                    "error",
                    error={
                        "type": "invalid_request_error",
                        "code": "conversation_already_has_active_response",
                        "message": "Conversation already has an active response in progress.",
                        "event_id": event.get("event_id"),
                    },
                )
            case "response.create":
                self._start_response(conn, event.get("response") or {})
            case "response.cancel":
                for task in list(conn.responses):
                    task.cancel()

    async def _create_item(self, conn: _Connection, event: dict[str, Any]) -> None:
        """Add the client's item to the conversation, after ``previous_item_id`` if given."""
        item = {"object": "realtime.item", "status": "completed", **(event.get("item") or {})}
        item.setdefault("id", self._next("item"))
        ids = [existing["id"] for existing in conn.items]
        anchor = event.get("previous_item_id")
        if anchor == "root":
            position = 0
        elif anchor in ids:
            position = ids.index(anchor) + 1
        else:
            position = len(ids)
        conn.items.insert(position, item)
        previous = conn.items[position - 1]["id"] if position else None
        await self._send(conn, "conversation.item.added", previous_item_id=previous, item=item)
        await self._send(conn, "conversation.item.done", previous_item_id=previous, item=item)

    async def _delete_item(self, conn: _Connection, event: dict[str, Any]) -> None:
        """Drop an item from the conversation and confirm, even when it is unknown."""
        item_id = event.get("item_id")
        conn.items = [item for item in conn.items if item["id"] != item_id]
        await self._send(conn, "conversation.item.deleted", item_id=item_id)

    def _start_response(self, conn: _Connection, params: dict[str, Any]) -> None:
        """Run one response in its own task, after any response still running."""
        self._active += 1
        task = asyncio.create_task(self._respond(conn, params))
        conn.responses.add(task)
        task.add_done_callback(conn.responses.discard)

    async def _respond(self, conn: _Connection, params: dict[str, Any]) -> None:
        """Answer one ``response.create``: created, the output items, then done."""
        response_id = self._next("resp")
        output: list[dict[str, Any]] = []
        try:
            async with conn.turn:
                await self._answer(conn, params, response_id, output)
        except asyncio.CancelledError:
            await self._finish(conn, response_id, params, output, Usage(0, 0), "cancelled")
        finally:
            self._active -= 1

    async def _answer(
        self,
        conn: _Connection,
        params: dict[str, Any],
        response_id: str,
        output: list[dict[str, Any]],
    ) -> None:
        """Get the reply, from the model or from a ``SAY_LINE`` instruction, and send it.

        The instruction comes in the response's instructions, or as the newest message of
        the conversation, as Pipecat sends a greeting: a developer message it turns into a
        user one.
        """
        await self._send(
            conn, "response.created", response=self._response(response_id, params, [], None)
        )
        instructions = params.get("instructions") or conn.instructions
        text: list[str] = []
        calls: list[ToolCall] = []
        usage = Usage(0, 0)
        newest = _chat(conn.items)[-1:]
        told = [str(message.get("content")) for message in newest if message["role"] != "assistant"]
        if said := next((text for text in [instructions, *told] if SAY_MARKER in text), None):
            text.append(said.partition(SAY_MARKER)[2])
            offered = self._request(conn, params, instructions)
            if offered.tools and offered.tool_choice != "none":
                self.tools_on_lines += 1
        else:
            request = self._request(conn, params, instructions)
            self.requests.append(request)
            try:
                async for event in self.model.stream(request):
                    if isinstance(event, TextDelta):
                        text.append(event.text)
                    elif isinstance(event, ToolCall):
                        calls.append(event)
                    elif isinstance(event, Usage):
                        usage = event
            except Exception as error:  # noqa: BLE001
                await self._finish(conn, response_id, params, output, usage, "failed", str(error))
                return
        modality = (params.get("output_modalities") or [conn.modality])[0]
        if "".join(text):
            output.append(await self._send_text(conn, response_id, text, modality, len(output)))
        for call in calls:
            if output:
                await asyncio.sleep(ITEM_SECONDS)
            output.append(await self._send_call(conn, response_id, call, len(output)))
        await self._finish(conn, response_id, params, output, usage, "completed")

    def _request(
        self, conn: _Connection, params: dict[str, Any], instructions: str
    ) -> ModelRequest:
        """Build the model request in OpenAI chat format from the conversation.

        A message that tells the model to say a line is an instruction, not the caller's
        words, so it stays out.
        """
        tools = params.get("tools") or conn.session.get("tools") or []
        choice = params.get("tool_choice") or conn.session.get("tool_choice")
        return ModelRequest(
            messages=[
                {"role": "system", "content": instructions},
                *(
                    message
                    for message in _chat(conn.items)
                    if message["role"] == "assistant"
                    or SAY_MARKER not in str(message.get("content"))
                ),
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters") or {"type": "object"},
                    },
                }
                for tool in tools
                if tool.get("type", "function") == "function"
            ],
            site="",
            tool_choice=choice if choice in ("required", "none") else None,
        )

    async def _send_text(
        self,
        conn: _Connection,
        response_id: str,
        deltas: list[str],
        modality: str,
        index: int,
    ) -> dict[str, Any]:
        """Send an assistant text message: as an audio transcript, or as text."""
        item_id = self._next("item")
        audio = modality == "audio"
        full = "".join(deltas)
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        previous = conn.items[-1]["id"] if conn.items else None
        part_kind, delta_kind = (
            ("output_audio", "response.output_audio_transcript")
            if audio
            else ("output_text", "response.output_text")
        )
        ids = {"response_id": response_id, "item_id": item_id, "output_index": index}
        await self._send(
            conn,
            "response.output_item.added",
            response_id=response_id,
            output_index=index,
            item=item,
        )
        conn.items.append(item)
        await self._send(conn, "conversation.item.added", previous_item_id=previous, item=item)
        empty = {"type": part_kind, "transcript": ""} if audio else {"type": part_kind, "text": ""}
        await self._send(conn, "response.content_part.added", **ids, content_index=0, part=empty)
        for delta in deltas:
            await self._send(conn, f"{delta_kind}.delta", **ids, content_index=0, delta=delta)
        final_key = "transcript" if audio else "text"
        await self._send(conn, f"{delta_kind}.done", **ids, content_index=0, **{final_key: full})
        part = {"type": part_kind, final_key: full}
        await self._send(conn, "response.content_part.done", **ids, content_index=0, part=part)
        item = {**item, "status": "completed", "content": [part]}
        conn.items[-1] = item
        await self._send(
            conn,
            "response.output_item.done",
            response_id=response_id,
            output_index=index,
            item=item,
        )
        await self._send(conn, "conversation.item.done", previous_item_id=previous, item=item)
        return item

    async def _send_call(
        self, conn: _Connection, response_id: str, call: ToolCall, index: int
    ) -> dict[str, Any]:
        """Send one function call item with its arguments."""
        item_id = self._next("item")
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "function_call",
            "status": "in_progress",
            "call_id": call.id,
            "name": call.name,
            "arguments": "",
        }
        previous = conn.items[-1]["id"] if conn.items else None
        await self._send(
            conn,
            "response.output_item.added",
            response_id=response_id,
            output_index=index,
            item=item,
        )
        conn.items.append(item)
        await self._send(conn, "conversation.item.added", previous_item_id=previous, item=item)
        ids = {"response_id": response_id, "item_id": item_id, "output_index": index}
        await self._send(
            conn,
            "response.function_call_arguments.delta",
            **ids,
            call_id=call.id,
            delta=call.arguments,
        )
        await self._send(
            conn,
            "response.function_call_arguments.done",
            **ids,
            call_id=call.id,
            name=call.name,
            arguments=call.arguments,
        )
        item = {**item, "status": "completed", "arguments": call.arguments}
        conn.items[-1] = item
        await self._send(
            conn,
            "response.output_item.done",
            response_id=response_id,
            output_index=index,
            item=item,
        )
        await self._send(conn, "conversation.item.done", previous_item_id=previous, item=item)
        return item

    async def _finish(
        self,
        conn: _Connection,
        response_id: str,
        params: dict[str, Any],
        output: list[dict[str, Any]],
        usage: Usage,
        status: str,
        error: str | None = None,
    ) -> None:
        """Send ``response.done`` with the output items and the usage."""
        response = self._response(response_id, params, output, usage)
        response["status"] = status
        if error is not None:
            response["status_details"] = {
                "type": "failed",
                "error": {"type": "server_error", "code": "model_failed", "message": error},
            }
        await self._send(conn, "response.done", response=response)
        self.done += 1

    def _response(
        self,
        response_id: str,
        params: dict[str, Any],
        output: list[dict[str, Any]],
        usage: Usage | None,
    ) -> dict[str, Any]:
        """The response object as the server reports it, in progress until it is done."""
        total = None
        if usage is not None:
            total = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
                "input_token_details": {
                    "cached_tokens": 0,
                    "text_tokens": usage.input_tokens,
                    "audio_tokens": 0,
                    "image_tokens": 0,
                    "cached_tokens_details": {"text_tokens": 0, "audio_tokens": 0},
                },
                "output_token_details": {
                    "text_tokens": usage.output_tokens,
                    "audio_tokens": 0,
                },
            }
        return {
            "id": response_id,
            "object": "realtime.response",
            "status": "completed" if usage is not None else "in_progress",
            "status_details": None,
            "output": output,
            "metadata": params.get("metadata"),
            "usage": total,
        }


def _chat(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn realtime conversation items into OpenAI chat messages, in order."""
    messages: list[dict[str, Any]] = []
    for item in items:
        match item.get("type"):
            case "message":
                text = "".join(
                    part.get("text") or part.get("transcript") or ""
                    for part in item.get("content") or []
                )
                messages.append({"role": item.get("role", "user"), "content": text})
            case "function_call":
                call = {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item.get("arguments") or ""},
                }
                last = messages[-1] if messages else None
                if last and last["role"] == "assistant" and "tool_calls" in last:
                    last["tool_calls"].append(call)
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            case "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": item.get("output") or "",
                    }
                )
    return messages
