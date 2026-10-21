"""The simulated caller: a separate model that hears only what the agent says.

It plays a scenario's persona with scripted facts and scripted moves, answers in short
spoken turns, and ends the call with ``HANGUP``. It runs on its own endpoint, so it does
not share the agent model's habits.
"""

from typing import Any

from unmute_core.testing.simulate._endpoints import Chat

HANGUP = "[HANGUP]"
"""What the caller writes at the end of the reply that hangs up."""

RULES = f"""You are role-playing a person who has phoned a business. Everything you write is spoken aloud on a phone line, so:
- Reply with only what you say: no stage directions, no quotes, no narration, no emojis, no markdown.
- Keep turns short and natural (usually one or two sentences), like a real caller.
- Answer what the agent asks. Do not volunteer facts the agent has not asked for unless your moves tell you to.
- Use ONLY the facts listed below. Never invent a different phone number, date, month, year, time, order number or service. If the agent asks for a detail of the booking or order that is not in your facts, say you are not sure rather than making one up. Only for small personal details with no fact (for example your name, if not listed) give a plausible short answer.
- Say numbers the way a person says them on the phone (for example "six one two, three four five, six seven eight"), unless a fact gives the exact wording to use.
- If the agent reads something back correctly, confirm it plainly ("yes, that's right"). If it reads something back wrong, correct it.
- Follow your moves in order, each at the moment it describes. A move is never skipped because the agent did not prompt it.
- When your goal is done (or the agent has clearly finished with you) end the call: say a short goodbye and put {HANGUP} at the very end of that same reply. Never write {HANGUP} before your goal is done unless a move says to hang up.
- A part of your goal is done only once the agent has dealt with it in its own words: something you mentioned in passing (a reminder you want, a detail you added) is not done until the agent has asked about it or confirmed it. And the agent confirming one part (the booking, the number) is not the end of the call: it may still have questions. Unless a move says to hang up, wait until the agent wraps up (says goodbye, or asks whether there is anything else) before you write {HANGUP}.
- Never hang up while the agent is waiting for an answer from you: if its last words ask you something (even after saying goodbye), answer it and do not write {HANGUP} in that reply; hang up on a later turn, once nothing is left to answer.
- If the agent says nothing you can respond to, say something short like "Hello? Are you there?"."""
"""The caller's standing rules, ahead of its persona, goal, facts and moves."""


def caller_prompt(brief: dict[str, Any]) -> str:
    """Return the caller's system prompt for a scenario's brief."""
    facts = "\n".join(f"- {key}: {value}" for key, value in brief["facts"].items())
    moves = "\n".join(f"{index}. {move}" for index, move in enumerate(brief["moves"], 1))
    return (
        RULES
        + f"\n\nWho you are: {brief['persona']}\n"
        + f"\nYour goal: {brief['goal']}\n"
        + f"\nFacts you know:\n{facts or '- (none)'}\n"
        + (f"\nYour moves, in order:\n{moves}\n" if moves else "")
    )


class SimCaller:
    """One simulated caller: its chat so far, and the endpoint it answers on."""

    def __init__(self, chat: Chat, brief: dict[str, Any]) -> None:
        """Start the caller's chat with its rules and brief."""
        self.chat = chat
        self.messages: list[dict[str, str]] = [{"role": "system", "content": caller_prompt(brief)}]

    async def reply(self, agent_said: str) -> str:
        """Return what the caller says to the agent's last lines."""
        self.messages.append(
            {"role": "user", "content": agent_said.strip() or "(the agent says nothing)"}
        )
        text = await self.chat.complete(self.messages)
        self.messages.append({"role": "assistant", "content": text})
        return text


def hangs_up(text: str) -> tuple[str, bool]:
    """Split a caller reply into what they say and whether they hang up with it."""
    if HANGUP not in text:
        return text.strip(), False
    return text.replace(HANGUP, "").strip(), True
