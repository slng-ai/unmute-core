"""One task step as a LiveKit ``AgentTask``.

The core's ``StepRun`` decides everything: the prompt, the tools, what each call does and
when the step ends. ``StepTask`` gives it a LiveKit body: the history it opens on, its
opening line or reply, its tool calls, and its model requests with the empty-reply retry.
"""

import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Sequence
from typing import Any

from livekit import rtc
from livekit.agents import NOT_GIVEN, NotGivenOr, llm, tts
from livekit.agents.voice import Agent, AgentTask, ModelSettings, RunContext, SpeechHandle

from unmute_core.call import AGENT_ENDED
from unmute_core.livekit._calls import announce, end_call_now, raw_tool, say, tool_text
from unmute_core.livekit._finish import FinishChecker
from unmute_core.livekit._gate import HandoffGate
from unmute_core.livekit._history import CallerHistory
from unmute_core.livekit._model import ModelNode
from unmute_core.livekit._replies import Reply, ReplyTracker
from unmute_core.livekit._trace import traced_tts
from unmute_core.tasks import (
    EMPTY_REPLY_ATTEMPTS,
    FINISH_TOOL_NAME,
    GO_BACK_TOOL_NAME,
    AfterReply,
    StepRun,
    ToolStatus,
    empty_reply_note,
)
from unmute_core.tools import ToolSpec

logger = logging.getLogger(__name__)


def _spoken(speech: SpeechHandle) -> list[llm.ChatMessage]:
    """Return the messages of a line that began to play. A line whose voice made no sound has none."""
    return [
        item
        for item in speech.chat_items
        if isinstance(item, llm.ChatMessage) and item.role == "assistant"
    ]


def _played(speech: SpeechHandle) -> str:
    """Return the words of a spoken line that played. LiveKit cuts an interrupted line's text."""
    return " ".join(item.text_content or "" for item in _spoken(speech))


def _heard(speech: SpeechHandle) -> bool:
    """Return True when the caller heard a line to its end.

    LiveKit does not interrupt the speech of a line whose voice failed: the failure ends the
    line's segment, and marks its message interrupted, or adds no message when no sound played
    (voice/agent_activity.py:3930, voice/generation.py:799).
    """
    spoken = _spoken(speech)
    return not speech.interrupted and bool(spoken) and not any(m.interrupted for m in spoken)


def _beside(context: RunContext) -> bool:
    """Return True when the reply that made this call made another call too.

    LiveKit adds every call of a reply to its speech before it runs any of them. The
    speech keeps the calls of earlier tool rounds too, so only calls whose item id starts
    with this reply's id count (voice/generation.py:301).
    """
    own = context.function_call
    reply = own.id.rsplit("/fnc_", 1)[0]
    return any(
        isinstance(item, llm.FunctionCall)
        and item.call_id != own.call_id
        and item.id.rsplit("/fnc_", 1)[0] == reply
        for item in context.speech_handle.chat_items
    )


def _ends_with_tool_result(chat_ctx: llm.ChatContext) -> bool:
    """Return True when a request answers a tool's result: its newest item past prompts."""
    for item in reversed(chat_ctx.items):
        if isinstance(item, llm.ChatMessage) and item.role in ("system", "developer"):
            continue
        if isinstance(item, llm.ChatMessage | llm.FunctionCall | llm.FunctionCallOutput):
            return isinstance(item, llm.FunctionCallOutput)
    return False


def _finish_calls_only(chunk: llm.ChatChunk) -> llm.ChatChunk:
    """Drop every call but finish from a retry that may only finish."""
    assert chunk.delta is not None
    calls = chunk.delta.tool_calls
    allowed = [call for call in calls if call.name == FINISH_TOOL_NAME]
    if len(allowed) == len(calls):
        return chunk
    logger.warning("a retry that may only finish called another tool; it was dropped")
    return chunk.model_copy(
        update={"delta": chunk.delta.model_copy(update={"tool_calls": allowed})}
    )


class StepTask(AgentTask[None]):
    """One step of a task run, as a LiveKit task. Await it once.

    Attributes:
        step: The core's step.
        ending_call: The call the step ended on, once it has.
        handed_to: The agent a handoff of the step gave the call to, once it has.
        handing_off: One of its handoff tools runs now. The owner's ``hand_off`` reads it.
    """

    def __init__(
        self,
        step: StepRun,
        *,
        history: llm.ChatContext,
        models: ModelNode,
        model_llm: NotGivenOr[llm.LLM] = NOT_GIVEN,
        handoffs: Sequence[llm.FunctionTool | llm.RawFunctionTool] = (),
        native_tools: Sequence[llm.Tool | llm.Toolset] = (),
        voice: tts.TTS | None = None,
    ) -> None:
        """Build the step's task. Its history is set as it opens.

        No ``chat_ctx`` goes to LiveKit's constructor, which would drop every call to a
        tool the step does not have (voice/agent.py:91). Old tool records are history.

        Args:
            step: The core's step.
            history: The chat it opens on, from ``step_history``.
            models: Sends its model requests.
            model_llm: The LLM LiveKit holds: the owner's or its task's plugin, or the
                placeholder when a ``ModelCall`` answers. Not given: the session's.
            handoffs: The owner's handoff tools its task offers too.
            native_tools: The owner's framework tools its task offers too, as they are.
            voice: The owner's voice. None speaks with the session's: LiveKit's task does not
                fall back to its owner's.
        """
        self.step = step
        self._native = native_tools
        self.ending_call: llm.FunctionCall | None = None
        self.handed_to: Agent | None = None
        self.handing_off = False
        self._handoffs = handoffs
        self._gate = HandoffGate([tool.id for tool in handoffs], armed=False)
        self._history = history
        self._models = models
        self._replies = ReplyTracker(step)
        self._finishes = FinishChecker(step, models, self)
        super().__init__(
            instructions=step.instructions(),
            tools=self._step_tools(),
            llm=model_llm,
            tts=voice if voice is not None else NOT_GIVEN,
        )

    def _step_tools(self) -> list[llm.Tool | llm.Toolset]:
        """Return the step's own tools, its finish, and its go-back in a group."""
        tools: list[llm.Tool | llm.Toolset] = [
            raw_tool(
                schema["function"],
                self._call_tool,
                keep_running=self.step.tools.keeps_running(schema["function"]["name"]),
            )
            for schema in self.step.tools.schemas()
        ]
        tools.append(raw_tool(self.step.finish_tool(), self._finish))
        if (go_back := self.step.go_back_tool()) is not None:
            tools.append(raw_tool(go_back, self._go_back))
        tools.extend(self._handoff_tool(tool) for tool in self._handoffs)
        tools.extend(self._native)
        return tools

    def _handoff_tool(self, tool: llm.FunctionTool | llm.RawFunctionTool) -> llm.RawFunctionTool:
        """Return one of the owner's handoff tools, as the step offers it.

        The core decides whether the step may hand off now. When it may, the owner's own
        tool runs: its ``hand_off`` reads the step's chat and ends the step once the next
        agent is ready. A tool that raises, or returns no agent, leaves the step running.
        A refusal asks for no reply: the other call of the same reply decides what comes
        next.
        """
        schema = (
            tool.info.raw_schema
            if isinstance(tool, llm.RawFunctionTool)
            else llm.ToolContext([tool]).parse_function_tools("openai", strict=False)[0]["function"]
        )

        async def hand_off(arguments: dict[str, Any], context: RunContext) -> object:
            """End the step on the handoff, or give the model the refusal."""
            refusal = self.step.may_hand_off(alone=not _beside(context))
            if refusal is not None:
                return llm.ToolResult(tool_text(refusal), reply_required=False)
            args, kwargs = llm.utils.prepare_function_arguments(
                fnc=tool, json_arguments=arguments, call_ctx=context
            )
            self.handing_off = True
            try:
                target = await tool(*args, **kwargs)
            finally:
                self.handing_off = False
            if not (self.step.handed_off and isinstance(target, Agent)):
                return target
            self.handed_to = target
            return self._end(context.function_call)

        return raw_tool(schema, hand_off)

    @property
    def said(self) -> str:
        """What the reply that ended the step said, as far as it played.

        A function call's item id is its reply's message id plus ``/fnc_<n>``
        (voice/generation.py:301).
        """
        if self.ending_call is None:
            return ""
        reply_id = self.ending_call.id.rsplit("/fnc_", 1)[0]
        message = self.chat_ctx.get_by_id(reply_id)
        if isinstance(message, llm.ChatMessage):
            return message.text_content or ""
        return ""

    async def on_enter(self) -> None:
        """Open on the step's history, say its line, and reply unless it listens.

        A step with an opening tool runs it first. When its line fills, the call and its
        result join the history, the line is said, and no model request is made.

        Never ``generate_reply(instructions=...)``: LiveKit sends that as a second system
        message (voice/agent_activity.py:3508), and Gemini keeps only the last one. Nothing
        here is awaited, so the opening request runs while the line plays.
        """
        history = self._history
        opening = await self.step.open()
        if opening is not None and (trace := self._models.trace) is not None:
            trace.tool_ran(
                name=opening.tool,
                call_id=opening.call_id,
                arguments=opening.arguments,
                result=tool_text(opening.result),
            )
        line = opening.line if opening is not None else None
        if opening is not None and line is not None:
            history = history.copy()
            history.insert(
                [
                    llm.FunctionCall(
                        call_id=opening.call_id, name=opening.tool, arguments=opening.arguments
                    ),
                    llm.FunctionCallOutput(
                        call_id=opening.call_id,
                        name=opening.tool,
                        output=tool_text(opening.result),
                        is_error=False,
                    ),
                ]
            )
        await self.update_chat_ctx(history, exclude_invalid_function_calls=False)
        self.step.opened(caller_turns=CallerHistory(history).turns)
        self._gate.start_opening(self.chat_ctx)
        if line is not None:
            self._say_reply(line)
        if self.step.announce:
            say(self.session, self.step.announce)
        if not self.step.listens:
            self.session.generate_reply()

    def _end(self, call: llm.FunctionCall) -> llm.ToolResult:
        """Complete the task on the call that ended the step.

        The output says how the step ended, and asks for no reply: the owner replies once
        the run is over.
        """
        self.ending_call = call
        if not self.done():
            self.complete(None)
        status = "handed off" if self.step.handed_off else self.step.status or "went_back"
        return llm.ToolResult(
            json.dumps(ToolStatus(status=status).model_dump()), reply_required=False
        )

    def _say_reply(self, line: str) -> None:
        """Say a line code speaks for the step, and tell the step how much of it played."""
        if (speech := say(self.session, line)) is not None:
            speech.add_done_callback(
                lambda done: self.step.reply_played(line, heard=_heard(done), played=_played(done))
            )

    async def _call_tool(self, arguments: dict[str, Any], context: RunContext) -> object:
        """Run one of the step's own tools. A closing tool that succeeds ends the step.

        A tool that ends the call waits for the words of its reply to play first: closing the
        session cancels a running step and cuts its speech off, where a main agent's speech
        drains.
        """
        call = context.function_call
        result = await self.step.call_tool(call.name, arguments, started=self._started)
        if self.step.ends_call:
            if (hooks := self._models.call) is not None:
                hooks.end_requested(AGENT_ENDED)
            await context.wait_for_playout()
            end_call_now(self.session)
            return llm.ToolResult(tool_text(result), reply_required=False)
        if result is not None and (line := self.step.reply) is not None:
            self._say_reply(line)
            return llm.ToolResult(tool_text(result), reply_required=False)
        if result is not None:
            return tool_text(result)
        trace = self._models.trace
        if trace is not None and self.step.closing_result is not None:
            trace.show_result(call.call_id, tool_text(self.step.closing_result))
        if self.step.done:
            return self._end(call)
        return llm.ToolResult("", reply_required=False)

    def _started(self, tool: ToolSpec) -> Awaitable[None] | None:
        """Say the tool's announce line as it starts."""
        if (line := self.step.announcement(tool)) is not None:
            return announce(self.session, tool, line)
        return None

    async def _finish(self, arguments: dict[str, Any], context: RunContext) -> object:
        """Save the step's values and end it, or give the model the refusal.

        A finish waits for its reply to end, to learn whether the caller heard anything:
        about a step tool's result, or before a refused save. A refusal the core answers
        with no reply asks LiveKit for none.
        """
        call = context.function_call
        unheard = None
        spoke = False
        if (reply := self._replies.reply_of(call.call_id)) and await reply.settled():
            spoke = reply.spoke
            unheard = None if reply.spoke else reply.unheard
        refusal = await self.step.finish(
            arguments, caller_turns=CallerHistory(self.chat_ctx).turns, unheard=unheard, spoke=spoke
        )
        if refusal is None:
            return self._end(call)
        self._finishes.hold_save()
        if unheard is not None:
            self._replies.refuse(call.call_id)
        text = tool_text({"refused": refusal})
        return text if self.step.answer_refusal else llm.ToolResult(text, reply_required=False)

    async def _go_back(self, arguments: dict[str, Any], context: RunContext) -> object:
        """End the step so earlier steps run again, or give the model the refusal."""
        refusal = self.step.go_back(arguments, caller_turns=CallerHistory(self.chat_ctx).turns)
        if refusal is not None:
            return tool_text({"refused": refusal})
        return self._end(context.function_call)

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        """Speak through LiveKit's own TTS node, with a ``tts`` span per piece of speech."""
        return traced_tts(self, self._models.trace, text, model_settings)

    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterator[llm.ChatChunk | str]:
        """Ask the model, again when its reply is empty, at most ``EMPTY_REPLY_ATTEMPTS`` times.

        The step's opening reply leaves out the owner's handoff tools. They come back on the
        first request that answers a newer caller turn. After a step tool's result, a retry
        offers only finish. Each retry adds a different
        note to the prompt. When every reply stays empty, the step says a short line or
        fails (``StepRun.gave_up``).
        """
        answered, unheard = self._replies.unheard(chat_ctx)
        answers_result = _ends_with_tool_result(chat_ctx)
        turns = CallerHistory(chat_ctx).turns
        finish_only = False
        request_tools = self._request_tools(chat_ctx, tools, turns)
        for attempt in range(EMPTY_REPLY_ATTEMPTS):
            prompt = self.step.instructions()
            if attempt:
                prompt += "\n\n" + empty_reply_note(retry=attempt, finish_only=finish_only)
            reply = Reply(unheard)
            ended = False
            cut: list[Exception] = []
            try:
                async for chunk in self._models.stream(
                    self,
                    chat_ctx,
                    request_tools,
                    model_settings,
                    prompt=prompt,
                    site=self.step.site,
                    kind="step",
                    cut=cut,
                ):
                    if isinstance(chunk, llm.ChatChunk) and chunk.delta and finish_only:
                        chunk = _finish_calls_only(chunk)
                    self._replies.read(chunk, reply)
                    yield chunk
                if not cut:
                    ended = True
                    after = self._reply_ended(reply, True, answers_result, turns)
                    if (finish := await self._finishes.after(after, chat_ctx, reply)) is not None:
                        self._replies.read(finish, reply)
                        yield finish
            finally:
                if not ended:
                    self._reply_ended(reply, False, answers_result, turns)
                reply.spoke = reply.spoke or not ended
                reply.done.set()
            if reply.spoke or reply.called:
                self.step.answered()
                self._replies.answered(answered)
                return
            logger.warning(
                "step reply was empty; attempt %d of %d", attempt + 1, EMPTY_REPLY_ATTEMPTS
            )
            if attempt == 0 and answered:
                finish_only = True
                finish = llm.ToolContext(tools).get_function_tool(FINISH_TOOL_NAME)
                request_tools = [finish] if finish else tools
        line = self.step.gave_up(finish_only=finish_only)
        if line is None:
            self.complete(None)
            return
        yield line

    def _request_tools(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], turns: int
    ) -> list[llm.Tool]:
        """Return the tools the first request offers: no go-back too early, no held handoff."""
        offered = (
            tools
            if self.step.offers_go_back(caller_turns=turns)
            else [tool for tool in tools if tool.id != GO_BACK_TOOL_NAME]
        )
        return self._gate.offered(chat_ctx, offered)

    def _reply_ended(
        self, reply: Reply, completed: bool, answers_result: bool, turns: int
    ) -> AfterReply | None:
        """Tell the step a reply ended, and return what it sends after it."""
        return self.step.reply_ended(
            spoke=reply.spoke,
            called=reply.called,
            completed=completed,
            answers_result=answers_result,
            caller_turns=turns,
        )
