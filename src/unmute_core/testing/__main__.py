"""``python -m unmute_core.testing``: run a project's agent through scripted calls.

``simulate``, ``report``, ``parity`` and ``scenarios`` run the call simulator instead:
see ``unmute_core.testing.simulate``.
"""

import argparse
import asyncio
import importlib
import os
import sys

from unmute_core.testing._scenario import ConverseSettings, exit_code, load

SIMULATOR_COMMANDS = ("simulate", "report", "parity", "scenarios")
"""The commands the call simulator runs."""


def _parser() -> argparse.ArgumentParser:
    """Return the command line: what to run, on which framework, through which module."""
    parser = argparse.ArgumentParser(prog="python -m unmute_core.testing")
    parser.add_argument("command", choices=("converse", "smoke"))
    parser.add_argument("framework", choices=("livekit", "pipecat"))
    parser.add_argument("module", help="the project's module, like agent or bot")
    parser.add_argument("scenarios", nargs="*", help="scenario files, for converse")
    parser.add_argument("--live", action="store_true", help="smoke on the real model")
    parser.add_argument(
        "--patch",
        action="append",
        default=[],
        metavar="PREFIX",
        help="also patch the modules whose name starts with PREFIX, like voice_stack",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the command, and exit 1 when a scenario failed or the agent did not answer."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in SIMULATOR_COMMANDS:
        simulator = importlib.import_module("unmute_core.testing.simulate._cli")
        raise SystemExit(simulator.main(argv))
    args = _parser().parse_intermixed_args(argv)
    sys.path.insert(0, os.getcwd())
    runner = importlib.import_module(f"unmute_core.{args.framework}.testing")
    if args.command == "smoke":
        reply = runner.smoke(args.module, live=args.live, patch=args.patch)
        print(f"ok: {args.module} answered: {reply[:80]!r}")
        return
    if not args.scenarios:
        raise SystemExit("converse needs at least one scenario file")
    module = importlib.import_module(args.module)
    results = asyncio.run(
        runner.converse(module, load(args.scenarios), ConverseSettings(), patch=args.patch)
    )
    exit_code(results)


if __name__ == "__main__":
    main()
