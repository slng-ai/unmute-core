"""Run simulated calls into a run directory, check and judge them, and write the summary.

Each call is ``<out>/calls/<scenario>/<nn>.jsonl``: the call record, its timeline, every
mock request, the judge's transcript and reply, and the checks. ``<out>/run.json`` says
how the run ran: the framework and each role's model, never a key. A call file with its
checks is done, so a rerun into the same directory resumes where it stopped. A call the
endpoint or the simulator broke has no verdict on the agent, and runs again.
"""

import asyncio
import datetime as dt
import importlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from unmute_core.testing.simulate._agent import AgentCall, AgentFile, build
from unmute_core.testing.simulate._call import TextCall, run_call
from unmute_core.testing.simulate._caller import SimCaller
from unmute_core.testing.simulate._checks import CheckResult, Records, call_of, run_checks
from unmute_core.testing.simulate._endpoints import Chat, Endpoint, agent_model
from unmute_core.testing.simulate._gate import attempts
from unmute_core.testing.simulate._judge import UNAVAILABLE, VALIDITY_CHECK, judge_call
from unmute_core.testing.simulate._record import ENVIRONMENT, HARNESS, Recorder
from unmute_core.testing.simulate._report import read_records, write_summary
from unmute_core.testing.simulate._scenario import Scenario, load_scenario, scenario_paths

Framework = Literal["livekit", "pipecat"]
DRIVERS: dict[Framework, str] = {"livekit": "LiveKitCall", "pipecat": "PipecatCall"}
"""Each framework's driver, in ``unmute_core.<framework>._simulate``."""


class RunOptions(BaseModel):
    """How a run runs.

    Attributes:
        framework: The framework the agent runs on.
        scenarios: Bundled scenario ids, ``tag:<tag>`` or scenario files. Empty is all.
        calls: Calls per scenario.
        concurrency: Calls at once.
        out: The run directory.
        agent: The agent's model. Only a run that makes calls needs it.
        caller: The simulated caller's model. Only a run that makes calls needs it.
        judge: The judge's model, or None to run without the judge.
        retry_failed: How often a failed or broken call runs again. The last attempt counts.
        label: A name for the run, in the summary's title.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    framework: Framework
    scenarios: list[str] = []
    calls: int = 1
    concurrency: int = 4
    out: Path
    agent: Endpoint | None = None
    caller: Endpoint | None = None
    judge: Endpoint | None = None
    retry_failed: int = 0
    label: str | None = None

    def models(self) -> dict[str, str]:
        """Return each role's model and host, for reports. Never a key."""
        roles = {"agent": self.agent, "caller": self.caller, "judge": self.judge}
        return {role: point.label() for role, point in roles.items() if point is not None}


def call_path(out: Path, scenario: str, index: int) -> Path:
    """Return the file of one call."""
    return out / "calls" / scenario / f"{index:02d}.jsonl"


def _checks_of(path: Path) -> dict[str, Any] | None:
    """Return a call file's checks, or None for a call with none yet."""
    if not path.exists():
        return None
    return next((r for r in read_records(path) if r.get("type") == "checks"), None)


def done(path: Path) -> bool:
    """Return whether a call has its verdict. One the endpoint or the simulator broke has not."""
    checks = _checks_of(path)
    return checks is not None and checks.get("left_out") not in (ENVIRONMENT, HARNESS)


def needs_retry(path: Path) -> bool:
    """Return whether ``--retry-failed`` runs a call again: no verdict, failed, or broken."""
    checks = _checks_of(path)
    return checks is None or not done(path) or not checks.get("passed")


def set_aside(path: Path) -> None:
    """Keep a failed attempt beside the call, as ``<nn>.attempt<k>.json``.

    The ``.json`` suffix keeps it out of the summary. A call the simulator broke left no
    file, so a marker stands in, and a pass on a retry is still reported as flaky.
    """
    target = path.with_name(f"{path.stem}.attempt{len(attempts(path)) + 1}.json")
    if path.exists():
        path.rename(target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"type": "attempt", "crashed": True}) + "\n")


def left_out(call: dict[str, Any], results: Sequence[CheckResult]) -> str | None:
    """Return why a call is not scored, or None for a scored call.

    ``environment``: the agent's endpoint failed, after retries. ``harness``: the simulator
    broke. ``judge``: the judge gave no verdict. ``caller``: the simulated caller left its
    brief. Each says nothing about the agent.
    """
    kinds = [str(error.get("kind", "")) for error in call.get("errors") or []]
    if any(kind.startswith(ENVIRONMENT) for kind in kinds):
        return ENVIRONMENT
    if any(kind.startswith(HARNESS) for kind in kinds):
        return HARNESS
    if any(result.category == UNAVAILABLE for result in results):
        return "judge"
    if any(r.id == f"judge.{VALIDITY_CHECK}" and not r.passed for r in results):
        return "caller"
    return None


def _write(path: Path, records: Records) -> None:
    """Write a call file whole, so a reader never sees half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as handle:
        handle.writelines(
            json.dumps(record, ensure_ascii=False, default=str) + "\n" for record in records
        )
    tmp.replace(path)


async def verdict(
    records: Records,
    scenario: Scenario,
    agent: AgentFile,
    call: AgentCall,
    *,
    judge: Chat | None,
    models: dict[str, str],
    keep_judge: bool = False,
) -> Records:
    """Return a call's records with its checks and, with a judge, the judge's verdicts.

    Args:
        records: The call's records. Earlier checks and verdicts are replaced.
        scenario: Its scenario.
        agent: Its agent.
        call: A call built for the agent, for the names in its task plan.
        judge: The judge, or None.
        models: Each role's model, kept with the checks.
        keep_judge: Keep the judge's earlier verdicts instead of asking again.
    """
    kept = [r for r in records if r.get("type") not in ("checks", "judge")]
    results = run_checks(
        kept,
        scenario,
        delegates=call.delegates(),
        providers=call.providers(),
        via_tools=call.via_tools(),
        task_values=call.task_values(),
    )
    judged: dict[str, Any] = {}
    if keep_judge:
        old = next((r for r in records if r.get("type") == "checks"), {})
        results += [
            CheckResult.model_validate(row)
            for row in old.get("results") or []
            if str(row.get("id", "")).startswith("judge.")
        ]
        judged = next((r for r in records if r.get("type") == "judge"), {})
    elif judge is not None:
        verdicts, judged = await judge_call(judge, scenario, kept, agent)
        results += verdicts
        judged = {"type": "judge", **judged}
    reason = left_out(call_of(kept), results)
    earlier = next((r for r in records if r.get("type") == "checks"), {})
    checks = {
        "type": "checks",
        "scenario": scenario.id,
        "passed": all(r.passed for r in results if r.core),
        "valid": reason is None,
        "left_out": reason,
        "results": [r.model_dump() for r in results],
        "expect_kinds": {name: next(iter(m), "") for name, m in scenario.expect.items()},
        "models": {**earlier.get("models", {}), **models},
    }
    return [*kept, *([judged] if judged else []), checks]


class Runner:
    """Runs the planned calls of one run directory."""

    def __init__(self, options: RunOptions) -> None:
        """Load the scenarios and open the caller's and the judge's endpoints."""
        self.options = options
        self.scenarios = [load_scenario(path) for path in scenario_paths(options.scenarios)]
        self.caller = Chat(options.caller) if options.caller is not None else None
        self.judge = Chat(options.judge) if options.judge is not None else None
        module = importlib.import_module(f"unmute_core.{options.framework}._simulate")
        self.driver: Callable[[AgentCall, Recorder], TextCall] = getattr(
            module, DRIVERS[options.framework]
        )
        self._limit = asyncio.Semaphore(options.concurrency)

    def planned(self) -> list[tuple[str, int]]:
        """Return each planned call: its scenario and index."""
        return [(s.id, i) for s, _ in self.scenarios for i in range(self.options.calls)]

    def _write_meta(self) -> None:
        """Write ``run.json``: how the run ran."""
        out = self.options.out
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "framework": self.options.framework,
            "models": self.options.models(),
            "label": self.options.label,
            "started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "scenarios": [s.id for s, _ in self.scenarios],
            "calls": self.options.calls,
        }
        (out / "run.json").write_text(json.dumps(meta, indent=1))

    async def one(self, scenario: Scenario, agent: AgentFile, index: int) -> None:
        """Run one call, check and judge it, and write its file."""
        assert self.options.agent is not None, "a run that makes calls needs the agent's model"
        assert self.caller is not None, "a run that makes calls needs the caller's model"
        async with self._limit:
            call = build(agent, agent_model(self.options.agent), line=scenario.line)
            recorder = Recorder(call.tool_names - call.via_tools().keys())
            recorder.attach(call.hooks)
            recorder.watch(call.state)
            driver = self.driver(call, recorder)
            caller = SimCaller(self.caller, scenario.caller_brief())
            records = await run_call(driver, caller, scenario, call, recorder)
        records = await verdict(
            records, scenario, agent, call, judge=self.judge, models=self.options.models()
        )
        _write(call_path(self.options.out, scenario.id, index), records)
        checks = records[-1]
        outcome = (
            f"left out ({checks['left_out']})"
            if checks["left_out"]
            else "pass"
            if checks["passed"]
            else "FAIL: "
            + ", ".join(r["id"] for r in checks["results"] if r["core"] and not r["passed"])
        )
        record = records[0]
        print(
            f"{scenario.id} #{index:02d}: {outcome} "
            f"({record['ended_by']} after {record['turns']} turns)",
            flush=True,
        )

    async def run(self) -> list[tuple[str, int]]:
        """Run every planned call that has no verdict yet, retry the failed ones, summarize.

        Returns:
            The planned calls.
        """
        self._write_meta()
        out = self.options.out
        todo = [
            (s, a, i)
            for s, a in self.scenarios
            for i in range(self.options.calls)
            if not done(call_path(out, s.id, i))
        ]
        for scenario, _, index in todo:
            for stale in attempts(call_path(out, scenario.id, index)):
                stale.unlink()
        try:
            await self._all(todo)
            for attempt in range(1, self.options.retry_failed + 1):
                retry = [
                    (s, a, i)
                    for s, a in self.scenarios
                    for i in range(self.options.calls)
                    if needs_retry(call_path(out, s.id, i))
                ]
                if not retry:
                    break
                print(f"running {len(retry)} failed call(s) again (retry {attempt})", flush=True)
                for scenario, _, index in retry:
                    set_aside(call_path(out, scenario.id, index))
                await self._all(retry)
        finally:
            await self.close()
        write_summary(out)
        return self.planned()

    async def _all(self, calls: list[tuple[Scenario, AgentFile, int]]) -> None:
        """Run calls, ``concurrency`` at a time."""
        await asyncio.gather(*(self.one(s, a, i) for s, a, i in calls))

    def run_plan(self) -> list[tuple[str, int]]:
        """Return the calls the run in ``out`` planned, of the scenarios this runner has.

        A run with no ``run.json`` yet is planned as this runner plans it.
        """
        meta_path = self.options.out / "run.json"
        if not meta_path.exists():
            return self.planned()
        meta = json.loads(meta_path.read_text())
        named = {s.id for s, _ in self.scenarios}
        return [(s, i) for s in meta["scenarios"] if s in named for i in range(meta["calls"])]

    async def rejudge(self, *, keep_judge: bool) -> list[tuple[str, int]]:
        """Check the calls already run again, and judge them again unless ``keep_judge``.

        The calls are the ones the run planned, so ``--calls`` is not needed again. A call
        with no file yet is left alone.

        Returns:
            The planned calls.
        """
        out = self.options.out
        planned = self.run_plan()
        jobs = []
        for scenario, agent in self.scenarios:
            call = build(agent, None, line=scenario.line)
            for index in (i for s, i in planned if s == scenario.id):
                path = call_path(out, scenario.id, index)
                if path.exists():
                    jobs.append(self._rejudge_one(path, scenario, agent, call, keep_judge))
        try:
            await asyncio.gather(*jobs)
        finally:
            await self.close()
        write_summary(out)
        return planned

    async def _rejudge_one(
        self, path: Path, scenario: Scenario, agent: AgentFile, call: AgentCall, keep: bool
    ) -> None:
        """Check one call again, and judge it again unless ``keep``."""
        async with self._limit:
            records = await verdict(
                read_records(path),
                scenario,
                agent,
                call,
                judge=self.judge,
                models=self.options.models(),
                keep_judge=keep,
            )
        _write(path, records)

    async def close(self) -> None:
        """Close the caller's and the judge's clients."""
        if self.caller is not None:
            await self.caller.aclose()
        if self.judge is not None:
            await self.judge.aclose()
