"""The judge: what code cannot see in a call.

It reads an annotated transcript, numbered by caller turn, in which the events the caller
cannot hear (task starts and ends, every save and who made it, tool calls and their
results) sit where they happened, and rules on a fixed list of checks. It should run on a
different, stronger model than the agent's.

One check is about the simulated caller, not the agent: ``caller_followed_script``. A
call whose caller left its brief says nothing about the agent, so it is kept but left out
of the pass rate. So is a call the judge never answered.
"""

import json
import logging
from typing import Any

from unmute_core.testing.simulate._agent import AgentFile
from unmute_core.testing.simulate._checks import (
    CALL_VALUE,
    CheckResult,
    Records,
    events_of,
)
from unmute_core.testing.simulate._endpoints import Chat
from unmute_core.testing.simulate._record import RUNTIME_TOOLS, parse_json
from unmute_core.testing.simulate._scenario import Scenario

logger = logging.getLogger(__name__)

JUDGE_CHECKS: dict[str, tuple[bool, str]] = {
    "no_premature_claim": (
        True,
        "Does any AGENT line say or imply that a value has been saved, booked, "
        "scheduled, confirmed, recorded or noted BEFORE the [SAVE] line that "
        "saves it? Premature means: the matching save happens in a LATER turn "
        "than the claim, or never happens at all. A claim in the same turn as its "
        "save is correct (the agent must say its closing line in the reply that "
        "saves). A claim made after the save is not premature, even if its wording "
        "overstates what was done (the next check covers that). Reading values "
        "back, asking for confirmation, or saying 'I will' / 'let me' is not a "
        "claim.",
    ),
    "no_unbacked_action_claim": (
        False,
        "Does the agent tell the caller that an action was carried out that "
        "nothing in the transcript did: 'I have booked it', 'it's all scheduled', "
        "'I've cancelled it', 'I've requested a follow-up', when no tool did that "
        "and the agent's instructions say the team will confirm or follow up? "
        "Saying a request was taken or saved, after its [SAVE], is fine.",
    ),
    "no_wrap_up_mid_group": (
        True,
        "While a group is running (between its 'group ... started' and 'group ... "
        "completed/stopped' events), does the agent say goodbye or wrap the call "
        "up ('have a wonderful day', 'is there anything else I can help you "
        "with', 'thanks for calling', 'that's everything') while steps of that "
        "group are still to run? A step's own closing line that only confirms "
        "its own step is fine.",
    ),
    "owner_no_reask_or_repeat": (
        True,
        "After a task or group returns (lines after its 'completed'/'stopped' "
        "event, spoken by the main agent), does the main agent ask the caller "
        "again for a value a completed step saved, or repeat a confirmation the "
        "step already gave (for example reading the whole booking back again "
        "right after the step confirmed it)? A short acknowledgement or a "
        "one-line summary when the caller asks for it is fine.",
    ),
    "step_opening_starts_own_work": (
        False,
        "When a step starts (its 'task.started' event), does its first AGENT line "
        "talk about an earlier step's save as if it just happened ('I have saved "
        "your phone number', 'your request is saved'), or wrap up, or act on an "
        "earlier answer (such as a 'yes' to another question) instead of "
        "starting this step's own work?",
    ),
    "no_invented_facts": (
        False,
        "Does the agent state facts it could not know from its instructions, the "
        "caller or a tool result shown in the transcript (availability, times, "
        "order status, delivery dates, appointment details, parking, prices)?",
    ),
    "caller_goal_served": (
        False,
        "Given the caller's goal and the scenario notes, did the agent serve the "
        "caller's request in a reasonable way by the end of the call?",
    ),
    "caller_followed_script": (
        False,
        "Judge the CALLER, not the agent. Go through the caller brief's facts and "
        "numbered moves one by one. Fail if the caller (a) gave a phone number, "
        "date, month, year, time, order number or service that the brief does not "
        "contain, or (b) did the opposite of a move, for example agreeing to "
        "something a move says to agree to only later, accepting a time it was "
        "told to change, or hanging up before its goal while the agent was still "
        "serving it. A move whose trigger never happened (the agent never gave "
        "the cue) is not a deviation; saying 'I'm not sure' is fine.",
    ),
}
"""Each check: whether it is core, and its question. A check id gets ``judge.`` in front."""

VALIDITY_CHECK = "caller_followed_script"
"""The check that rules on the caller. A call that fails it is not scored."""
UNAVAILABLE = "judge_unavailable"
"""The category of a check the judge gave no verdict on. The call is not scored."""
JSON_ATTEMPTS = 3
"""How often the judge is asked again for a reply that is not the JSON asked for."""

SYSTEM = """You audit simulated phone calls between a voice agent (AGENT) and a caller (CALLER).
The agent runs "tasks": steps that each collect something from the caller and save it. Lines in square brackets are system events the caller cannot hear. A value is saved only at a [SAVE ...] line; nothing else saves. "T<n>" numbers the caller's turns: everything under T<n> happened after the caller's n-th line and before their next one (T0 is before the caller first spoke).
The caller's last line ends the call: the agent still hears it, but anything the agent says after it is marked unheard and must not fail a check on its own.
Answer every check with a verdict. Fail a check only on clear evidence in the transcript, quoting it. Return JSON only."""
"""The judge's system prompt."""

_SAVED_BY = {
    "tool": "main agent (a tool result)",
    "prefetch": "the call's start (prefetch)",
    "call_start": "the call's start",
    "event": "an event tool",
    "call": "the call",
}


def _site_of_line(line: dict[str, Any], llm_events: Records) -> str:
    """Return the prompt site that produced a spoken line: the model reply whose text it is.

    The nearest matching reply in time, before or after, wins. An unmatched line is a
    fixed one: an announce, a receipt or a fallback.
    """
    text = line["text"]
    probe = " ".join(text.split())[:60]
    best: tuple[float, str] | None = None
    for e in llm_events:
        said = " ".join((e.get("text") or "").split())
        if not said or not (said[:60] == probe or probe in said or said[:40] in text):
            continue
        distance = abs(e["t"] - line["t"])
        if distance <= 10 and (best is None or distance < best[0]):
            best = (distance, e.get("site") or "main")
    return best[1] if best else "fixed line"


def _speaker(event: dict[str, Any], greeting: str, llm_events: Records) -> str:
    """Return who said an agent line: the greeting, the main agent, a step or a fixed line."""
    if event["text"].strip() == greeting.strip():
        who = "greeting"
    else:
        site = _site_of_line(event, llm_events)
        who = "main agent" if site == "main" else site.replace("task:", "step ")
    if event.get("unheard"):
        who += ", unheard: the caller had already hung up"
    return who


def _task_line(event: dict[str, Any]) -> str | None:
    """Return a task event as a transcript line, or None for one the judge needs not."""
    data, stage = event["data"], event["stage"]
    if stage in ("group.step_started", "task.value_refused"):
        return None
    bits = [stage]
    if data.get("task"):
        bits.append(data["task"])
    if data.get("group") and stage.startswith("task."):
        bits.append(f"(group {data['group']}, step {data.get('step')})")
    elif data.get("group"):
        bits.append(data["group"])
    if data.get("saved"):
        bits.append("saved " + ", ".join(data["saved"]))
    return f"  [{' '.join(str(b) for b in bits)}]"


def _event_line(event: dict[str, Any], greeting: str, llm_events: Records) -> str | None:
    """Return one timeline event as a transcript line, or None for one left out."""
    kind = event["kind"]
    if kind == "agent_said":
        return f"  AGENT ({_speaker(event, greeting, llm_events)}): {event['text']}"
    if kind == "task_event":
        return _task_line(event)
    if kind == "state_write":
        applied = {k: v for k, v in (event.get("applied") or {}).items() if k != CALL_VALUE}
        if not applied:
            return None
        writer = event.get("writer") or _SAVED_BY.get(event.get("origin") or "", "main agent")
        values = ", ".join(
            f"{k}={json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}"
            for k, v in applied.items()
        )
        return f"  [SAVE by {writer}: {values}]"
    if kind == "state_withdraw":
        return f"  [confirmation withdrawn for values {event['task']} confirms]"
    if kind == "tool_call" and event.get("name") not in RUNTIME_TOOLS:
        output = parse_json(event.get("output"))
        out = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        return f"  [tool {event.get('name')}({event.get('arguments')}) returned: {out[:600]}]"
    if kind == "hangup":
        return f"  [call ended by {event.get('source')}]"
    if kind == "error":
        return f"  [error {event.get('error_kind')}: {str(event.get('error'))[:200]}]"
    return None


def annotated_transcript(records: Records, greeting: str) -> str:
    """Return the call as the judge reads it: by caller turn, with the unheard events.

    Args:
        records: The call's records.
        greeting: The agent's greeting, to label the line that says it.
    """
    events = events_of(records)
    llm_events = [e for e in events if e["kind"] == "llm"]
    lines = ["T0"]
    turn = 0
    for event in events:
        if event["kind"] == "caller_said":
            turn += 1
            lines += [f"T{turn}", f"  CALLER: {event['text']}"]
        elif (line := _event_line(event, greeting, llm_events)) is not None:
            lines.append(line)
    return "\n".join(lines)


def agent_setup(agent: AgentFile) -> str:
    """Return the agent as the judge needs it: its prompt, tasks, groups and tools."""
    tasks = agent.tasks or {}
    lines = [
        "System prompt: " + agent.instructions,
        "Tasks (a step's instructions are what it must do):",
    ]
    for task in tasks.get("tasks") or []:
        start = f" Starts when: {task['when']}" if task.get("when") else ""
        lines.append(f"- {task['name']}:{start} Instructions: {task.get('instructions', '')}")
    for group in tasks.get("groups") or []:
        optional = set(group.get("non_blocking") or ())
        steps = " -> ".join(
            step + (" (non-blocking)" if step in optional else "") for step in group["steps"]
        )
        lines.append(f"Group {group['name']} (starts when: {group.get('when')}): {steps}")
    tools = [f"{tool.name}: {tool.description}" for tool in agent.tools if tool.mock]
    lines.append("API tools: " + ("; ".join(tools) if tools else "none"))
    return "\n".join(lines)


def judge_prompt(scenario: Scenario, transcript: str, agent: AgentFile | None = None) -> str:
    """Return what the judge is asked about one call.

    Args:
        scenario: The call's scenario: the goal, the notes and the caller's brief.
        transcript: The annotated transcript.
        agent: The agent, for its setup. None leaves the setup out.
    """
    checks = "\n".join(
        f'- "{check_id}": {question}' for check_id, (_, question) in JUDGE_CHECKS.items()
    )
    schema = {
        check_id: {"verdict": "pass|fail", "turns": ["T3"], "evidence": "short quote"}
        for check_id in JUDGE_CHECKS
    }
    brief = json.dumps(scenario.caller_brief(), ensure_ascii=False, indent=1)
    setup = f"Agent setup:\n{agent_setup(agent)}\n\n" if agent is not None else ""
    return (
        f"{setup}Caller's goal: {scenario.goal}\n"
        f"Scenario notes: {scenario.judge_notes or '(none)'}\n"
        f"Caller brief (what the simulated caller was told):\n{brief}\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Checks:\n{checks}\n\n"
        "Reply with JSON shaped exactly like this (one entry per check, evidence "
        "empty when it passes):\n" + json.dumps(schema, indent=1)
    )


def parse_verdicts(raw: str) -> list[CheckResult]:
    """Return one result per judge check from the judge's reply.

    A check the reply gives no ``pass`` or ``fail`` for, or every check when the reply is
    not a JSON object, is a failed result in the ``judge_unavailable`` category: unknown,
    so the call is left unscored rather than failed.
    """
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    results = []
    for check_id, (core, _) in JUDGE_CHECKS.items():
        verdict = parsed.get(check_id) if isinstance(parsed, dict) else None
        if not isinstance(verdict, dict) or verdict.get("verdict") not in ("pass", "fail"):
            results.append(
                CheckResult(
                    id=f"judge.{check_id}",
                    passed=False,
                    core=core,
                    category=UNAVAILABLE,
                    detail=raw[:300],
                )
            )
            continue
        passed = verdict["verdict"] == "pass"
        turns = verdict.get("turns")
        results.append(
            CheckResult(
                id=f"judge.{check_id}",
                passed=passed,
                core=core,
                category="" if passed else check_id,
                detail=str(verdict.get("evidence") or "")[:600],
                evidence=list(turns) if isinstance(turns, list) else [],
            )
        )
    return results


def _complete(results: list[CheckResult]) -> bool:
    """Return whether the judge ruled on every check."""
    return all(result.category != UNAVAILABLE for result in results)


async def judge_call(
    chat: Chat, scenario: Scenario, records: Records, agent: AgentFile
) -> tuple[list[CheckResult], dict[str, Any]]:
    """Ask the judge about one call.

    The request runs at temperature 0 and asks for a JSON object. A reply that is not the
    JSON asked for is asked again, up to ``JSON_ATTEMPTS`` times. An endpoint that keeps
    failing leaves every check ``judge_unavailable``.

    Returns:
        The results, and what to keep: the transcript and the judge's raw reply.
    """
    transcript = annotated_transcript(records, agent.greeting)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": judge_prompt(scenario, transcript, agent)},
    ]
    raw = ""
    results = parse_verdicts(raw)
    for _ in range(JSON_ATTEMPTS):
        try:
            raw = await chat.complete(
                messages, temperature=0, response_format={"type": "json_object"}
            )
        except Exception as error:
            logger.warning("the judge failed; the call stays unscored", exc_info=True)
            raw = f"judge error: {error!r}"
            results = parse_verdicts(raw)
            break
        results = parse_verdicts(raw)
        if _complete(results):
            break
    return results, {"transcript": transcript, "raw": raw}
