"""``python -m unmute_core.testing simulate|report|parity|scenarios``: the call simulator.

See ``unmute_core.testing.simulate`` for the method.
"""

import argparse
import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from unmute_core.testing.simulate._agent import AGENTS, describe, load_agent
from unmute_core.testing.simulate._endpoints import ROLES, Endpoint, SimulateSettings, endpoint
from unmute_core.testing.simulate._gate import (
    DEFAULT_MIN_PASS,
    DEFAULT_MIN_SCORED,
    evaluate,
    write,
)
from unmute_core.testing.simulate._parity import parity
from unmute_core.testing.simulate._report import compare, write_summary
from unmute_core.testing.simulate._run import Runner, RunOptions
from unmute_core.testing.simulate._scenario import SCENARIOS

COMMANDS = ("simulate", "report", "parity", "scenarios")
"""The commands this module runs."""
CALLER_PARAMS = {"temperature": 0.7}
"""What the caller's requests send unless ``--caller-params`` says otherwise."""


def _json(text: str) -> dict[str, Any]:
    """Return a JSON object given on the command line."""
    value = json.loads(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("give a JSON object, like '{\"temperature\": 0}'")
    return value


def _parser() -> argparse.ArgumentParser:
    """Return the command line."""
    parser = argparse.ArgumentParser(prog="python -m unmute_core.testing")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("simulate", help="run simulated calls, check and judge them")
    run.add_argument("framework", choices=("livekit", "pipecat"))
    run.add_argument(
        "scenarios", nargs="*", help="bundled ids, tag:<tag> or scenario files; none is all"
    )
    run.add_argument("--calls", type=int, default=1, help="calls per scenario (1)")
    run.add_argument("--concurrency", type=int, default=4, help="calls at once (4)")
    run.add_argument("--out", type=Path, help="run directory (simulate-runs/<framework>)")
    run.add_argument("--label", help="a name for the run, in the summary")
    run.add_argument("--model", help="the model of every role that names none")
    run.add_argument("--base-url", help="the endpoint of every role that names none")
    for role in ROLES:
        run.add_argument(f"--{role}-model", help=f"the {role}'s model")
        run.add_argument(f"--{role}-base-url", help=f"the {role}'s OpenAI-compatible endpoint")
        run.add_argument(
            f"--{role}-params", type=_json, help=f"JSON body fields for the {role}'s requests"
        )
    run.add_argument("--no-judge", action="store_true", help="checks only, no judge")
    again = run.add_mutually_exclusive_group()
    again.add_argument(
        "--rejudge", action="store_true", help="check and judge the calls in --out again"
    )
    again.add_argument(
        "--recheck",
        action="store_true",
        help="run the checks on the calls in --out again, keeping the judge's verdicts",
    )
    run.add_argument(
        "--retry-failed",
        type=int,
        default=0,
        metavar="N",
        help="run a failed or broken call again, up to N times; the last attempt counts",
    )
    run.add_argument("--gate", action="store_true", help="exit 1 when the run fails the gate")
    run.add_argument("--gate-min-pass", type=float, default=DEFAULT_MIN_PASS)
    run.add_argument("--gate-min-scored", type=float, default=DEFAULT_MIN_SCORED)
    report = commands.add_parser("report", help="write and print a run's summary")
    report.add_argument("out", type=Path)
    report.add_argument("--compare", type=Path, help="another run, side by side")
    report.add_argument("--scenarios", nargs="*", help="only these scenarios, with --compare")
    pair = commands.add_parser("parity", help="two runs of the same scenarios, call by call")
    pair.add_argument("a", type=Path)
    pair.add_argument("b", type=Path)
    pair.add_argument("--out", type=Path, help="write the Markdown here")
    pair.add_argument("--top", type=int, default=10, help="divergent pairs shown in full (10)")
    commands.add_parser("scenarios", help="list the bundled scenarios and agents")
    return parser


def _options(args: argparse.Namespace) -> RunOptions:
    """Return the run's options from the command line and the environment."""
    settings = SimulateSettings()
    needed = {"judge"} if args.rejudge else set() if args.recheck else {"agent", "caller", "judge"}
    if args.no_judge:
        needed.discard("judge")
    points: dict[str, Endpoint | None] = {}
    for role in ROLES:
        params = getattr(args, f"{role}_params")
        if role == "caller":
            params = {**CALLER_PARAMS, **(params or {})}
        try:
            points[role] = endpoint(
                role,
                settings,
                model=getattr(args, f"{role}_model") or args.model,
                base_url=getattr(args, f"{role}_base_url") or args.base_url,
                params=params,
            )
        except ValueError:
            if role in needed:
                raise
            points[role] = None
    return RunOptions(
        framework=args.framework,
        scenarios=args.scenarios,
        calls=args.calls,
        concurrency=args.concurrency,
        out=args.out or Path("simulate-runs") / args.framework,
        agent=points["agent"],
        caller=points["caller"],
        judge=points["judge"] if "judge" in needed else None,
        retry_failed=args.retry_failed,
        label=args.label,
    )


def _simulate(args: argparse.Namespace) -> int:
    """Run, rejudge or recheck, print the summary, and return the exit code."""
    options = _options(args)
    runner = Runner(options)
    if args.rejudge or args.recheck:
        planned = asyncio.run(runner.rejudge(keep_judge=args.recheck))
    else:
        planned = asyncio.run(runner.run())
    print((options.out / "summary.md").read_text())
    if not args.gate:
        return 0
    verdict = evaluate(
        options.out,
        planned=planned,
        min_pass=args.gate_min_pass,
        min_scored=args.gate_min_scored,
    )
    step_summary = SimulateSettings().step_summary
    print(write(options.out, verdict, framework=options.framework, step_summary=step_summary))
    return 0 if verdict.passed else 1


def _scenarios() -> None:
    """Print the bundled scenarios and agents."""
    print("Scenarios (python -m unmute_core.testing simulate <framework> <id>):")
    for path in sorted(SCENARIOS.glob("*.json")):
        raw = json.loads(path.read_text())
        tags = ", ".join(raw.get("tags") or ())
        print(f"  {path.stem:40} agent {raw['agent']:22} {tags}")
    print("\nAgents:")
    for path in sorted(AGENTS.glob("*.json")):
        print(f"  {describe(load_agent(path.stem))}")


def main(argv: list[str]) -> int:
    """Run one command and return its exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("livekit.agents").setLevel(logging.ERROR)
    if getattr(args, "framework", None) == "pipecat":
        # Pipecat logs through loguru, at DEBUG by default: every prompt, every frame.
        logger = importlib.import_module("loguru").logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    if args.command == "simulate":
        return _simulate(args)
    if args.command == "report":
        if args.compare is not None:
            scenarios = set(args.scenarios) if args.scenarios else None
            print(compare(args.out, args.compare, scenarios=scenarios))
        else:
            write_summary(args.out)
            print((args.out / "summary.md").read_text())
        return 0
    if args.command == "parity":
        text = parity(args.a, args.b, top=args.top)
        if args.out is not None:
            args.out.write_text(text)
        print(text)
        return 0
    _scenarios()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
