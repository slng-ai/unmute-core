"""Every text a task shows the model, in the two modes.

``status`` texts are Unmute's (unmute@4a738ad internal/generate/task_prompt.go,
livekit_v1_build.go delegateReturnFinality, templates/*/agents.py.tmpl, ir/variables.go).
``record`` texts began as a copy of Unmute's and grew with the record, the hand-back and
going back. Both were measured on live calls: change one only with a live check.

A step's prompt keeps its ``{{name}}`` placeholders. The model call fills them, natively
or through the router, so the model sees the same prompt either way.
"""

from collections.abc import Mapping, Sequence
from dataclasses import field
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.tasks._spec import UNSERVED_FIELD, FinishRule, TaskInput, TaskSpec
from unmute_core.tools._lines import success_word


class _Result(BaseModel):
    """What a tool result is: immutable, and it refuses a key it does not know."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Refused(_Result):
    """The result a refused call reads: one sentence that says why and what to do.

    Attributes:
        refused: The sentence.
    """

    refused: str


class ToolStatus(_Result):
    """The result of a tool that says how the call stands.

    Attributes:
        status: The word, like ``completed`` or ``not_run_yet``.
    """

    status: str


class ConfirmAsked(ToolStatus):
    """What the model reads once code asked the caller to confirm a call before it runs.

    Attributes:
        asked: The line code asked.
        summary: What the model does next.
    """

    asked: str
    summary: str


class ToolDeclaration(_Result):
    """A tool as a model reads it: name, description and parameters.

    Dumped, its parameters are a copy at every depth, so a caller that edits one
    declaration cannot change another.

    Attributes:
        name: The tool's name.
        description: What the model reads.
        parameters: The JSON Schema of its arguments.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the declaration as the dict a model call takes."""
        return self.model_dump()


def refused(text: str) -> dict[str, str]:
    """Return the result a refusal reads, as a fresh dict."""
    return Refused(refused=text).model_dump()


def empty_parameters() -> dict[str, Any]:
    """Return the parameters of a tool that takes none, as a fresh dict."""
    return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


INPUTS_HEADING = "Your inputs:"
INPUT_NOT_AVAILABLE = "not available yet"
STEP_BOUNDARY = (
    "This step starts now. The caller's last reply answered the step before this one, not "
    "anything in this step: nothing here has been asked or agreed yet. Open with this "
    "step's own first move. If the caller already said something this step needs, read it "
    "back and ask them to confirm it: nothing this step saves is kept until they answer in it."
)
"""The last sentence is there because a caller who gave a later step's answer early had it
refused on that step's opening, saved again, and refused again until they hung up."""
BEGIN_STEP = "Begin this step. Work from what the caller has already said."
WAIT_FOR_CALLER = "Wait for the caller."
SPOKEN_FORM = (
    "Everything you write is read aloud on a phone call, so write it the way you would say "
    "it: plain sentences, with no lists, bullets, bold or headings. Read details back in "
    'one natural sentence, like "a haircut on Tuesday, October 6th, at 5 PM".'
)

UNSERVED_DESCRIPTION = (
    "Leave empty unless the caller asked for something this step cannot serve. Then put "
    "that request here in one short plain sentence, in the caller's own terms, so the "
    "agent that owns this step can take it."
)
UNSERVED_DESCRIPTION_RECORD = (
    UNSERVED_DESCRIPTION + " Pass the values the caller already confirmed in this step in "
    "the same call: they are saved before the hand-back."
)
FINISH_DESCRIPTION = "Save this step's values and finish, or return unserved without saving."
FINISH_DESCRIPTION_RECORD = (
    "Save this step's values and finish, or hand the caller back with unserved_request, "
    "passing any of this step's values they already confirmed so they are saved first. It "
    "ends the step at once, so say anything the caller should hear in the same reply."
)

_UNSERVED_OWNER_RULE = (
    "A completed status means the step finished. Read any saved values through your own "
    "prompt references. An unserved status means the step could not help. Ask the caller "
    "what they need, then use your tools or a handoff."
)
_RETURN_FINALITY = (
    " When this flow finishes it returns a status. Continue with the caller. Do not run "
    "this flow again for the same request. " + _UNSERVED_OWNER_RULE
)
_TERMINAL_OWNER_RULE = (
    " The caller's words right above the status are what the step answered; it served the "
    "part it owns, so act only on what remains."
)
_RECORD_OWNER_RULE = (
    "The result lists each step that ran: what it saved and confirmed, and, for a step "
    "that handed the caller back, what it could not serve, in the caller's words. The "
    "caller has already heard what a step said, so do not repeat it; a step's not_heard is "
    "a tool's result the caller has not heard, so tell them what it means for them. Reply "
    "only with what still needs saying, and speak of what the flow did in the steps' own "
    "terms: what a step saved is recorded, and booked, sent or paid only where a step said "
    "so. A completed status means the flow finished. An unserved status means a step "
    "handed the caller back without doing that request: serve it with your own tools or "
    "another task that covers it, or ask the caller. Until one of them has done it, it is "
    "still undone, and that is what the caller hears from you. To change or undo something "
    "a task saved, start the task that saved it or one that makes that change; a step that "
    "can start on its own is also a task you can start."
)
_FAILED_OWNER_RULE = (
    " A failed status means the step broke before it finished and saved nothing further. "
    "Tell the caller it did not go through, then use your tools or a handoff."
)
_RECORD_TERMINAL_OWNER_RULE = (
    " The caller's words right above the result are what the step answered; it served the "
    "part it owns, so act only on what remains."
)
_RECORD_RETURN_FINALITY = (
    " When this flow finishes, the call returns what it did. Continue with the caller. Do "
    "not run this flow again for the same request. "
)
_FINISH_ENDS_STEP = (
    " Saving ends this step at once, so say any confirmation, what a tool's result means "
    "for the caller, or what happens next in the same reply as that call."
)

REFUSED_RERUN = refused(
    "This already ran and finished, and the caller has not spoken since, so its result "
    "is the answer to what they asked. Reply to them from the values in your prompt. "
    "Run it again only once they have asked for something new."
)
REFUSED_RERUN_HANDED_BACK = refused(
    "This handed the caller's request back and the caller has not spoken since, so "
    "running it again would hand it back again. Serve the request in its result with "
    "your tools or another task, or ask the caller."
)
REFUSED_BUSY = refused(
    "Another task was already running, so this one did not start. Once that one "
    "returns, run this only if the caller still needs it."
)

REFUSED_HANDOFF_BESIDE = refused(
    "Not handed over: this reply also called another tool, which runs first. Once it "
    "returns, hand the caller over in a reply of its own if they still need it."
)
REFUSED_HANDOFF_ENDED = refused(
    "This step already ended. Hand the caller over from where you are now."
)

SAVE_BEFORE_CALLER = (
    "Not saved: the caller has not said anything in this step yet. Tell them what you have "
    "and ask them to confirm it, or ask for what you still need; save once they answer."
)
UNSERVED_ON_REENTRY = (
    "Not recorded: this step runs again because the caller went back to change something, "
    "and they have not asked for anything new since. Do this step's own work; call finish "
    f"with {UNSERVED_FIELD} only when the caller asks for something this step cannot serve."
)
GO_BACK_DESCRIPTION = (
    "Call this when the caller wants to change something an earlier step of this flow "
    "already handled, for example correcting their email or a date they gave. Name the "
    "steps to go back to, in the order the caller asked. Those steps run again, then this "
    "step continues. The earlier steps and what each handles: "
)
GO_BACK_TOO_EARLY = refused("Only go back when the caller asks to change an earlier answer.")
GO_BACK_AFTER_ACTION = refused("This step's action already ran. Call finish to record it first.")
REENTRY_REDO = (
    "The caller came back to this step to change what it saved. Redo this step with what "
    "they say now; this is this step's own work, not a request for something else."
)
REENTRY_RESUME = (
    "The caller went back to change an earlier step, and that is done. Carry on with this "
    "step; the change they asked for is not a request for something else."
)
SERVED_HEADING = "Done earlier in this call:"
SERVED_RULE = (
    "What the caller asked of these steps is done, and the caller has been told: do not "
    "mention it, offer to do it again, or hand it back. Anything the caller asks for now, "
    "including a change to one of these, is a new request."
)
TERMINAL_SAVE_REFUSED = (
    ". Call finish with the values this tool returned; recover a missing one with a read "
    "tool of this step, or finish with unserved_request saying the action happened and its "
    "details could not be saved."
)
NOT_YET_TOOL_NAME = "not_yet"
FINISH_CHECK_INSTRUCTION = (
    "Check only; nothing you write here is said to the caller. If your last reply asked the "
    "caller to confirm or answer anything this step needs, call not_yet: their answer has "
    "not come. Otherwise: has the caller agreed to everything this step saves, in what they "
    "said in this step? If yes, call finish with exactly those values. If anything is still "
    "missing, unclear or not yet agreed, call not_yet."
)
"""The narrow question a step's reply with words and no call gets, once the caller has spoken
in the step. A weak model says "booked" and never calls finish, and the caller hangs up on a
step that never ended."""
NOT_YET_SCHEMA: dict[str, Any] = ToolDeclaration(
    name=NOT_YET_TOOL_NAME,
    description="Nothing to save yet: the step goes on.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
).to_dict()
HELD_SAVE_INSTRUCTION = (
    "Check only; the call has ended and nothing you write here is said to anyone. Earlier "
    "in this step you tried to save these values before the caller had said anything in "
    "it, and they were held: {values}. Read what the caller said after that. If they "
    "agreed, or said nothing against them (a thanks or goodbye), call finish with those "
    "values, with any correction the caller gave. If they rejected or doubted them, or "
    "what they said leaves the values unclear, call not_yet."
)
"""The question a held save's step is asked at the call's end, when the caller spoke after
the hold."""
FINISH_CHECK_SECONDS = 8.0
"""How long a finish check may take. Past it, the step goes on."""

EMPTY_REPLY_ATTEMPTS = 3
"""How many times a step asks the model before it gives up on an empty reply."""
EMPTY_REPLY_GAVE_UP = "Sorry, I couldn't complete that. Please try again."
EMPTY_REPLY_GAVE_UP_AFTER_TOOL = (
    "I couldn't finish after the last tool result. Please ask me to check the current state "
    "before trying again."
)


def empty_reply_note(*, retry: int, finish_only: bool) -> str:
    """What a step's prompt ends on when the model asks again after an empty reply.

    Each retry reads a different note, because the Responses API reuses an earlier answer
    when the request has not changed. "Never ask again" is the part that matters: without
    it the model restarts the step and asks again for what the caller just said.

    Args:
        retry: 1 for the first retry, 2 for the second.
        finish_only: The empty reply followed a step tool's result, so the retry offers
            only finish.
    """
    if finish_only:
        if retry == 1:
            return (
                "The response after the tool result was empty. Use the tool result already "
                "in context. Do not repeat that operation or call another operation. Produce "
                "the task's next valid response now, using what the caller has already told "
                "you rather than asking again; call finish only if the task is complete."
            )
        return (
            "The finish retry was also empty. This is the second retry. Use the existing tool "
            "result without repeating any operation. Produce the task's next valid response "
            "now; call finish only if the task is complete."
        )
    if retry == 1:
        return (
            "The previous response was empty. Follow the current task instructions and "
            "produce its next valid response now. The caller's turns are already in context: "
            "use what they have already told you, and never ask again for something they have "
            "answered."
        )
    return (
        "The prior retry was also empty. This is the second retry. Follow the current task "
        "instructions and produce its next non-empty valid response now, using what the "
        "caller has already said rather than asking again."
    )


def not_yet_tool() -> dict[str, Any]:
    """Return the ``not_yet`` tool a finish check offers, as a fresh dict."""
    return ToolDeclaration.model_validate(NOT_YET_SCHEMA).to_dict()


def arguments_refused(delegate: str, problem: str) -> dict[str, str]:
    """What a delegate call reads when a value it passed does not fit."""
    return refused(
        f"{delegate} did not start: {problem} Pass that value as the caller said "
        f"it, or leave it out, and call {delegate} again."
    )


def opening_ran(tool: str, arguments: str, result: str) -> str:
    """What a step reads when its opening tool ran and no line was said from it."""
    return (
        f"`{tool}` already ran as this step opened, with {arguments}, and returned: {result}. "
        "Answer from that, and do not call it again for the same values."
    )


def confirm_asked(tool: str, line: str) -> dict[str, str]:
    """What the model reads when code asked the caller to confirm a call before it runs."""
    return ConfirmAsked(
        status="not_run_yet",
        asked=line,
        summary=f"{tool} has not run. Code asked the caller: {line!r} Wait for their "
        f"answer. On a yes, call {tool} again with the same input. If they change something, "
        "call it with the new input, and code asks again.",
    ).model_dump()


def confirm_unfilled(tool: str, missing: Sequence[str]) -> dict[str, str]:
    """What the model reads when a confirm line cannot be filled from the call's input."""
    return refused(
        f"{tool} did not run: before it runs, code asks the caller about "
        f"{_join_words(missing) or 'its input'}. Get that from the caller, then call it again."
    )


def already_succeeded(tool: str) -> dict[str, str]:
    """What a second closing tool reads once one has succeeded in the step."""
    return refused(
        f"{tool} already succeeded in this step. Call finish to record it; a new "
        "action is a new request from the caller."
    )


def still_running(running: str, tool: str) -> dict[str, str]:
    """What a closing tool reads while another closing tool of the step still runs."""
    return refused(
        f"{running} is still running in this step, so {tool} did not run. Wait "
        "for its result; a new action is a new request from the caller."
    )


def finish_while_running(tool: str) -> str:
    """What a finish reads when the closing tool it waited for still runs."""
    return (
        f"Not recorded: {tool} is still running. Wait for its result, then call finish. If "
        "the caller cannot wait, call finish with unserved_request."
    )


def finish_after_failure(tool: str) -> str:
    """What a finish sent before a closing tool's result reads when that tool failed."""
    return (
        f"Not recorded: {tool} did not succeed, and this finish was sent before its result. "
        "Read that result, then continue with the caller."
    )


def finish_without_closing(tools: Sequence[str], values: Sequence[str]) -> str:
    """What a finish reads that saves new values while no closing tool of its step succeeded."""
    names = _join_words([f"`{tool}`" for tool in tools])
    saved = ", ".join(values[:-1]) + " and " + values[-1] if len(values) > 1 else values[0]
    return (
        f"Not recorded: {saved} come from {names}, which has not succeeded in "
        f"this step, so nothing is done yet. Call it, or call finish with {UNSERVED_FIELD} "
        "if the caller wants something this step cannot do."
    )


def finish_before_heard(tool: str) -> str:
    """What a silent finish after a step tool's result reads, once per step."""
    return (
        f"Not recorded: the caller has not heard what {tool} returned. Tell them what it "
        "means for them, and call finish in that same reply."
    )


def finish_refusal(message: str, *, record: bool, repeated: bool = False) -> str:
    """What a finish whose values do not fit reads.

    Args:
        message: What was wrong, like ``status: Input should be 'open' or 'closed'``.
        record: The mode is ``record``.
        repeated: The finish sent the very values just refused.
    """
    if not record:
        return f"Not recorded: {message}. Ask again, then call finish with a value that fits."
    if repeated:
        return (
            f"Not recorded: {message}. These are the values that were just refused, so don't "
            "call finish with them again: ask the caller, in your reply, for what is missing."
        )
    return (
        f"Not recorded: {message}. Convert what the caller already told you into that format "
        "and call finish again; ask the caller only for something they haven't said."
    )


def _join_words(words: Sequence[str]) -> str:
    """Join as a sentence reads a choice: "a", "a or b", "a, b or c"."""
    if len(words) < 2:
        return "".join(words)
    return ", ".join(words[:-1]) + " or " + words[-1]


def _success_text(rules: Sequence[FinishRule]) -> str:
    """What a tool must return to end the step, as ``success_matches`` reads it."""
    where = [
        " and ".join(
            f"`{field}` is " + _join_words([f"`{success_word(value)}`" for value in values])
            for field, values in rule.success.items()
        )
        for rule in rules
    ]
    return "a result where " + ", or where ".join(where)


def _finish_call(
    call: str, result_names: Sequence[str], *, record: bool, next_step: str | None
) -> str:
    """The first paragraph of the tail: the call that ends the step, and what follows it."""
    tail = f"\n\nWhen this step is complete, call {call}"
    if result_names:
        tail += " with: " + ", ".join(result_names)
    tail += "."
    if record:
        tail += _FINISH_ENDS_STEP
        if next_step:
            tail += (
                f" The call goes on after this step: the {next_step.replace('_', ' ')} step "
                "comes next. Close with what this step did and keep the goodbye for the end "
                "of the call."
            )
    return tail


def _ends_on_text(call: str, ends_on: Mapping[str, Sequence[FinishRule]], *, record: bool) -> str:
    """The paragraphs about the tools that end the step by themselves."""
    if record:
        returns = [f"`{name}` returns {_success_text(ends_on[name])}" for name in sorted(ends_on)]
        tail = (
            f"\n\nWhen {_join_words(returns)}, this step ends by itself and saves it; do "
            f"not call {call} after it. Any other result leaves this step open: tell the "
            f"caller what it means for them, carry on with the step, and call {call} "
            "yourself once it is complete. "
        )
    else:
        quoted = [f"`{name}`" for name in sorted(ends_on)]
        tail = (
            f"\n\nWhen {_join_words(quoted)} returns a successful result, this step ends by "
            f"itself and saves it; do not call {call} after it. "
        )
    tail += (
        f"{call} is for a request this step cannot serve, for values you already hold, "
        "and for recording a result whose save was refused."
    )
    if asked := sorted(name for name, rules in ends_on.items() if rules[0].confirm):
        quoted = _join_words([f"`{name}`" for name in asked])
        tail += (
            f"\n\nCode asks the caller to confirm before {quoted} runs, so never ask that "
            "yourself: call it as soon as the caller has given what it needs. It answers "
            "that it has not run yet. When the caller says yes, call it again with the same "
            "input."
        )
    if told := sorted(name for name, rules in ends_on.items() if any(r.say for r in rules)):
        quoted = _join_words([f"`{name}`" for name in told])
        tail += (
            f" When {quoted} succeeds, code tells the caller. Never say yourself that it "
            "is done: until it has succeeded, it is not."
        )
    return tail


def _unserved_text(call: str, *, record: bool) -> str:
    """The last paragraph of the tail: the way out for a request the step cannot serve."""
    tail = (
        f"\n\n`{UNSERVED_FIELD}` is for a request this step cannot serve. Do this step's own "
        "work first, and never use it to skip that work: the caller's original reason for "
        "being here is not an unserved request. If a handoff here covers what they want, "
        "call that handoff instead. Only when no tool and no handoff here can serve what the "
        f"caller is asking, call {call} with their request in `{UNSERVED_FIELD}`, in their own "
        "words, rather than refusing or explaining what you cannot do here. "
    )
    if record:
        tail += (
            "If the caller confirmed this step's values before asking, pass those values in "
            "the same call: they are saved before the caller is handed back. "
        )
    return tail + "The agent that owns this step reads that status and takes the caller from there."


def finish_contract(
    finish_name: str,
    result_names: Sequence[str],
    *,
    record: bool,
    ends_on: Mapping[str, Sequence[FinishRule]] | None = None,
    go_back: str | None = None,
    next_step: str | None = None,
) -> str:
    """The tail on every step's prompt: the call that ends the step, and the way out.

    Args:
        finish_name: The finish tool's name.
        result_names: Its value parameters.
        record: The mode is ``record``.
        ends_on: Each tool that ends the step by itself, with its rules.
        go_back: The go-back tool's name, when the step has one. ``record`` only.
        next_step: The step that comes next in the run. ``record`` only.

    Returns:
        The tail, starting with a blank line.
    """
    call = f"`{finish_name}`"
    tail = _finish_call(call, result_names, record=record, next_step=next_step)
    if ends_on:
        tail += _ends_on_text(call, ends_on, record=record)
    if record and go_back:
        tail += (
            "\n\nIf the caller wants to change something an earlier step of this flow saved, "
            f"call `{go_back}`: only that step can change it, so call it before you tell them "
            "it is changed."
        )
    tail += _unserved_text(call, record=record)
    return tail + ("\n\n" + SPOKEN_FORM if record else "")


def literal(text: str) -> str:
    """Break ``{{`` in text that is not a template, so the model call never fills it."""
    return text.replace("{{", "{ {")


@original_errors
@dataclass(eq=False, config=VALUE)
class CallStartResults:
    """What each tool that runs at call start returned, by its label.

    A prefetch lookup records its result as JSON: ``{"ok": true}`` for a tool that hides it,
    ``{"error": ...}`` when it failed. A step's ``tool`` input shows it.
    """

    _texts: dict[str, str] = field(init=False, default_factory=dict)

    def record(self, label: str, text: str) -> None:
        """Keep what a call-start tool gave the main agent."""
        self._texts[label] = text

    def describe(self, label: str) -> str:
        """Return what the main agent read for the tool, or that it is not available yet."""
        return self._texts.get(label, INPUT_NOT_AVAILABLE)


def render_inputs(inputs: Sequence[TaskInput], call_start: CallStartResults | None) -> str:
    """The inputs section of a step's prompt, or "" when it takes none.

    A ``memory`` input is a placeholder, filled with the State value as the prompt is sent.
    A ``tool`` input is its call-start text, each further line indented under it.

    Args:
        inputs: The step's inputs, from ``TaskPlan.inputs_for``.
        call_start: The call-start results, or None before any is recorded.

    Returns:
        The section, or "".
    """
    if not inputs:
        return ""
    lines = [INPUTS_HEADING]
    for row in inputs:
        if row.kind == "tool":
            text = call_start.describe(row.name) if call_start is not None else INPUT_NOT_AVAILABLE
            first, *rest = literal(text).split("\n")
            lines.append(f"- {row.name}, from the call start: {first}")
            lines.extend(f"  {line}" if line else "" for line in rest)
        elif row.kind == "memory":
            lines.append(f"- {row.name}: {{{{{row.name}}}}}")
        else:
            lines.append(f"- {row.name}: {literal(row.value or '')}")
    return "\n".join(lines)


def render_instructions(
    task: TaskSpec,
    *,
    record: bool,
    finish_name: str,
    result_names: Sequence[str],
    ends_on: Mapping[str, Sequence[FinishRule]] | None = None,
    go_back: str | None = None,
    next_step: str | None = None,
    inputs: Sequence[TaskInput] = (),
    call_start: CallStartResults | None = None,
    served: str = "",
    opening: str = "",
    contract: str | None = None,
) -> str:
    """A step's prompt: its instructions, inputs, earlier runs, the finish tail, the opening.

    A step with no inputs and nothing served before it renders exactly as without them.

    Args:
        task: The task.
        record: The mode is ``record``.
        finish_name: The finish tool's name.
        result_names: Its value parameters.
        ends_on: Each tool that ends the step by itself, with its rules.
        go_back: The go-back tool's name, when the step has one.
        next_step: The step that comes next in the run.
        inputs: The step's inputs.
        call_start: The call-start results.
        served: The section of what earlier runs did (``ServedRequests.section``).
        opening: Why the step opens as it does, last.
        contract: The finish tail, when the caller already built it with ``finish_contract``
            for these arguments. None builds it here.

    Returns:
        The prompt, with its ``{{name}}`` placeholders still in it.
    """
    text = task.instructions
    if section := render_inputs(inputs, call_start):
        text += "\n\n" + section
    if served:
        text += "\n\n" + served
    text += (
        contract
        if contract is not None
        else finish_contract(
            finish_name,
            result_names,
            record=record,
            ends_on=ends_on,
            go_back=go_back,
            next_step=next_step,
        )
    )
    return text + ("\n\n" + opening if opening else "")


_ENDS_CALL = (
    " This flow does not return to you: when it finishes the call ends. Only a flow that "
    "hands the caller back, or fails, returns."
)
"""Unmute's sentence for ``then: end`` (unmute@868ae23 internal/generate/
livekit_v1_build.go:1127), with the two cases that still return."""


def _transfers_to(target: str) -> str:
    """Return Unmute's sentence for ``then: transfer``, with the two cases that still return.

    The wording is Unmute's (unmute@d70247b internal/generate/livekit_v1_build.go:1125).
    """
    return (
        " This flow does not return to you: when it finishes the caller is handed to the "
        f"{target}. Only a flow that hands the caller back, or fails, returns."
    )


def delegate_description(
    when: str | None,
    *,
    record: bool,
    carries_turn: bool,
    ends_call: bool = False,
    transfers_to: str | None = None,
) -> str:
    """What the main agent reads for the tool that starts a task or group.

    Args:
        when: The ``when`` sentence, or None.
        record: The mode is ``record``.
        carries_turn: A step of it can end on its own tool, so it hands the owner the
            caller's turn it answered.
        ends_call: The group ends the call when it completes (``then: end``).
        transfers_to: The agent the group hands the call to when it completes
            (``then: transfer``), or None.
    """
    lead = (when or "").strip() or "Run this flow."
    final = _ENDS_CALL if ends_call else _transfers_to(transfers_to) if transfers_to else None
    if record:
        finality = final + " " if final else _RECORD_RETURN_FINALITY
        text = lead + finality + _RECORD_OWNER_RULE + _FAILED_OWNER_RULE
        return text + (_RECORD_TERMINAL_OWNER_RULE if carries_turn else "")
    finality = final + " " + _UNSERVED_OWNER_RULE if final else _RETURN_FINALITY
    return lead + (_TERMINAL_OWNER_RULE if carries_turn else "") + finality


def via_description(description: str, *, record: bool, carries_turn: bool) -> str:
    """What the main agent reads for a task's tool whose success enters the task.

    Args:
        description: The tool's own description.
        record: The mode is ``record``.
        carries_turn: The task can end on its own tool.
    """
    lead = description.strip()
    lead = lead + " " if lead else ""
    if record:
        text = lead + (
            "When this succeeds, the caller moves into a step that continues from its "
            "result, and the call returns what that step did. Continue with the caller. "
        )
        text += _RECORD_OWNER_RULE + _FAILED_OWNER_RULE
        return text + (_RECORD_TERMINAL_OWNER_RULE if carries_turn else "")
    text = lead + (
        "When this succeeds, the caller moves into a step that continues from its result, "
        "and the call returns a status. Continue with the caller. "
    )
    return text + _UNSERVED_OWNER_RULE + (_TERMINAL_OWNER_RULE if carries_turn else "")
