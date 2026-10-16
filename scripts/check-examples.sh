#!/usr/bin/env bash
# Check that example projects install, import unmute-core, and answer a caller.
# Usage: scripts/check-examples.sh                         # every project
#        scripts/check-examples.sh examples/salon-concierge/livekit
#        scripts/check-examples.sh --live examples/minimal/livekit
# For each project: install from its lock, import every module with an empty
# environment and no network, then call unmute_core.require() with this repo's version.
# A hand-written project then runs one text turn through each of its agents, with a
# scripted model and no network (tests/examples/smoke_*.py).
# --live skips the import check and runs the same turn with the real model and the keys
# in your environment. A project whose keys are unset is skipped.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
version="$(cd "$repo" && uv version --short)"
python_version="$(cat "$repo/.python-version")"
live=""
if [[ "${1:-}" == "--live" ]]; then
  live="--live"
  shift
fi
if [[ $# -eq 0 ]]; then
  set -- "$repo"/examples/*/livekit "$repo"/examples/*/pipecat
fi

# The agents a smoke turn runs, per hand-written project. The compiled projects do not
# import unmute-core yet, so a turn through them would test the compiler, not the library.
# realtime.py and phone.py need a realtime model or a phone line: tests/live covers them.
smoke_modules() {
  case "$1" in
    */minimal/*) echo greeting handoff hooks idle router tasks telemetry tools ;;
    */parcel-desk/livekit | */salon-concierge-unmute/livekit) echo agent ;;
    */parcel-desk/pipecat | */salon-concierge-unmute/pipecat) echo bot ;;
    */kestrel/livekit | */wasla/livekit | */nosh/livekit | */faro/livekit) echo agent ;;
    */kestrel/pipecat | */wasla/pipecat | */nosh/pipecat | */faro/pipecat) echo bot ;;
  esac
}

# The values a dispatch must carry for a project's smoke call. A call without them never starts.
start_values() {
  case "$1" in
    */wasla/*) echo '{"account_id": "WA-1001"}' ;;
    */nosh/*) echo '{"customer_name": "Sam"}' ;;
  esac
}

live_keys() {
  case "$1" in
    */minimal/*) echo OPENAI_API_KEY SLNG_API_KEY LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY UNMUTE_AGENT_ID ;;
    */parcel-desk/*) echo SLNG_API_KEY ;;
    */salon-concierge-unmute/*) echo GOOGLE_API_KEY OPENAI_API_KEY SLNG_API_KEY ;;
    */kestrel/* | */wasla/* | */nosh/* | */faro/*) echo GOOGLE_API_KEY SLNG_API_KEY ;;
  esac
}

import_check() {
  (cd "$1" && env -i .venv/bin/python -I - "$version" <<'PY'
import importlib
import pathlib
import socket
import sys


def refuse_network(*args: object, **kwargs: object) -> None:
    """Fail any network call made while the modules import."""
    raise OSError("the import check runs without network")


socket.socket.connect = refuse_network
socket.getaddrinfo = refuse_network
sys.path.insert(0, ".")

# Pipecat's bot.py checks its keys and builds the knowledge index at import, on
# purpose, so a container without keys never reports ready. It may raise, but
# never ImportError.
ENTRY_POINTS = {"bot"}

failures = []
for path in sorted(pathlib.Path().rglob("*.py")):
    if ".venv" in path.parts:
        continue
    parts = path.with_suffix("").parts
    module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    try:
        importlib.import_module(module)
    except ImportError as error:
        failures.append(f"{module}: {error!r}")
    except Exception as error:
        if module not in ENTRY_POINTS:
            failures.append(f"{module}: {error!r}")
        else:
            print(f"{module} refused to start without keys, as designed: {error!r}")

if failures:
    sys.exit("modules that do not import:\n  " + "\n  ".join(failures))

import unmute_core

unmute_core.require(sys.argv[1])
print(f"ok: every module imports, unmute-core {unmute_core.__version__}")
PY
  )
}

for project in "$@"; do
  echo "== $project"
  uv sync --quiet --locked --project "$project" --python "$python_version"
  [[ -z "$live" ]] && import_check "$project"
  modules="$(smoke_modules "$project")"
  if [[ -z "$modules" ]]; then
    echo "no smoke turn: a compiled project"
    continue
  fi
  framework="$(basename "$project")"
  if [[ -n "$live" ]]; then
    missing=""
    for key in $(live_keys "$project"); do
      [[ -z "${!key:-}" ]] && missing="$missing $key"
    done
    if [[ -n "$missing" ]]; then
      echo "skipped: set$missing to run it live"
      continue
    fi
  fi
  start=()
  values="$(start_values "$project")"
  [[ -n "$values" ]] && start=("UNMUTE_CALL_START=$values")
  for module in $modules; do
    if [[ -n "$live" ]]; then
      (cd "$project" && env "${start[@]}" .venv/bin/python -I "$repo/tests/examples/smoke_$framework.py" "$module" --live)
    else
      (cd "$project" && env -i "${start[@]}" .venv/bin/python -I "$repo/tests/examples/smoke_$framework.py" "$module")
    fi
  done
done
