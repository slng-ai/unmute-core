#!/usr/bin/env bash
# Run every scenario of a ported use case as a whole call, on its real model.
# Usage: scripts/converse-examples.sh                        # every use case, both frameworks
#        scripts/converse-examples.sh examples/kestrel/livekit
#        scripts/converse-examples.sh --audio examples/nosh/pipecat
# Each project installs from its lock, then python -m unmute_core.testing converse runs
# ../scenarios/*.json through the project's own entrypoint and prints each call's
# transcript, the tools called and the model requests per turn. --audio runs the whole
# voice pipeline, with a caller whose lines are spoken by SLNG text to speech.
# Needs GOOGLE_API_KEY and SLNG_API_KEY. Exits non-zero when a scenario fails.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
python_version="$(cat "$repo/.python-version")"
if [[ "${1:-}" == "--audio" ]]; then
  export CONVERSE_AUDIO=1
  shift
fi
if [[ $# -eq 0 ]]; then
  set -- "$repo"/examples/{kestrel,wasla,nosh,faro}/{livekit,pipecat}
fi
export FARO_DB="${FARO_DB:-:memory:}"

status=0
for project in "$@"; do
  echo "== $project"
  uv sync --quiet --locked --project "$project" --python "$python_version"
  case "$project" in
    */livekit) framework=livekit module=agent ;;
    */pipecat) framework=pipecat module=bot ;;
  esac
  (cd "$project" && .venv/bin/python -I -m unmute_core.testing converse "$framework" "$module" \
    --patch voice_stack ../scenarios/*.json) || status=1
done
exit "$status"
