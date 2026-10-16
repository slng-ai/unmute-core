"""The finish checks of a task step: the model's call on whether the step may end."""

import asyncio
import json
import logging

from livekit.agents import AgentSession, llm, utils
from livekit.agents.voice import Agent, ModelSettings, RunContext
from pydantic import SkipValidation
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.livekit._calls import raw_tool
from unmute_core.livekit._history import CallerHistory
from unmute_core.livekit._model import ModelNode
from unmute_core.livekit._replies import Reply
from unmute_core.tasks import (
    FINISH_TOOL_NAME,
    AfterReply,
    FinishCheck,
    StepRun,
    decided_finish,
)
from unmute_core.tools import ToolError

logger = logging.getLogger(__name__)


async def _never_run(arguments: dict[str, object], context: RunContext) -> object:
    """Stand in for a finish check's tools, which the check only reads, never runs."""
    raise ToolError("a finish check's tools never run")


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class FinishDecision:
    """The finish a finish check chose, as the step sends it.

    Attributes:
        call_id: The id of the finish call.
        arguments: What the finish saves.
        extra: Provider data the check's model sent with the call, like a Gemini thought
            signature.
    """

    call_id: str
    arguments: SkipValidation[dict[str, object]]
    extra: SkipValidation[dict[str, object]]


class FinishChecker:
    """Runs a step's finish checks, and holds a refused save for the call's end."""

    def __init__(self, step: StepRun, models: ModelNode, owner: Agent) -> None:
        """Keep what a check asks with.

        Args:
            step: The core's step.
            models: Sends the check's request.
            owner: The task asking: the agent of each request, and the source of the
                step's chat and session.
        """
        self._step = step
        self._models = models
        self._owner = owner

    async def after(
        self, after: AfterReply | None, chat_ctx: llm.ChatContext, reply: Reply
    ) -> llm.ChatChunk | None:
        """Return the finish call a reply ends on, sent by the runtime, or None.

        A held finish goes out as it is. A finish check runs its own request first, and
        its finish keeps the provider data the check's model sent with it.
        """
        if after is None:
            return None
        arguments = after.finish
        call_id = f"unmute_finish_{utils.shortuuid()}"
        extra: dict[str, object] = {}
        if after.check is not None:
            checked = await self.check(chat_ctx, "".join(reply.text), after.check)
            if checked is None:
                return None
            call_id, arguments, extra = checked.call_id, checked.arguments, checked.extra
        if arguments is None:
            return None
        call = llm.FunctionToolCall(
            name=FINISH_TOOL_NAME, arguments=json.dumps(arguments), call_id=call_id, extra=extra
        )
        return llm.ChatChunk(id=call_id, delta=llm.ChoiceDelta(role="assistant", tool_calls=[call]))

    async def check(
        self,
        chat_ctx: llm.ChatContext,
        said: str,
        check: FinishCheck,
        *,
        session: AgentSession | None = None,
    ) -> FinishDecision | None:
        """Run a finish check: the step's chat and the reply just said, finish or not_yet.

        Args:
            chat_ctx: The step's chat.
            said: The reply just said, or ``""``.
            check: The check.
            session: The call's session, once the step no longer runs.

        Returns:
            The finish the check chose, or None for ``not_yet``, a failure or a check that
            ran out of time.
        """
        step = self._step
        ctx = chat_ctx.copy()
        if said.strip():
            ctx.add_message(role="assistant", content=said)
        tools: list[llm.Tool] = [raw_tool(schema, _never_run) for schema in check.tools]
        calls: list[llm.FunctionToolCall] = []

        async def ask() -> None:
            """Stream the check and keep the calls it makes."""
            async for chunk in self._models.stream(
                self._owner,
                ctx,
                tools,
                ModelSettings(tool_choice="required"),
                prompt=step.instructions() + "\n\n" + check.instruction,
                site=step.site,
                session=session,
                kind="finish_check",
            ):
                if isinstance(chunk, llm.ChatChunk) and chunk.delta is not None:
                    calls.extend(chunk.delta.tool_calls)

        try:
            await asyncio.wait_for(ask(), timeout=check.timeout_s)
        except Exception:
            logger.warning(
                "finish check of %s failed; the step goes on", step.task.name, exc_info=True
            )
            return None
        arguments = decided_finish((call.name, call.arguments) for call in calls)
        if arguments is None:
            return None
        finish = next(call for call in calls if call.name == FINISH_TOOL_NAME)
        logger.info("finish check of %s: the runtime sends its finish", step.task.name)
        return FinishDecision(finish.call_id, arguments, dict(finish.extra or {}))

    def hold_save(self) -> None:
        """Keep the save the step just refused before the caller spoke, for the call's end.

        The call's end comes after LiveKit has stopped every agent, so the decision runs on
        the session itself. It reads the step's chat and what the session heard after. Both
        are told apart by message id: a turn that began before the hold but joined the
        history after it is still news.
        """
        session = self._owner.session
        heard = CallerHistory(session.history).turns
        before = {message.id for message in session.history.messages()}

        async def decide(check: FinishCheck) -> dict[str, object] | None:
            """Ask the step about its held save, on its chat and what the call heard since."""
            ctx = self._owner.chat_ctx.copy()
            known = before | {item.id for item in ctx.items}
            ctx.insert(
                [message for message in session.history.messages() if message.id not in known]
            )
            checked = await self.check(ctx, "", check, session=session)
            return None if checked is None else checked.arguments

        self._step.hold_save(
            caller_spoke_since=lambda: CallerHistory(session.history).turns > heard, decide=decide
        )
