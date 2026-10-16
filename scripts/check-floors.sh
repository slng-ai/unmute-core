#!/usr/bin/env bash
# Run the tests with every dependency at the lowest version pyproject.toml allows.
# Usage: scripts/check-floors.sh [pytest arguments]
# An app that pins an older version than our lock still gets a library that works.
# Builds the wheel once, then runs the tests against it in two fresh venvs:
#   1. every extra. livekit-agents needs a newer openai and OpenTelemetry than our own
#      floors, so those two follow its floors and every other dependency is at ours.
#   2. every extra but livekit, every dependency at our floor. This is the app on Pipecat,
#      or on the core alone, with the oldest openai and OpenTelemetry we allow.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
python_version="$(cat "$repo/.python-version")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv run --quiet --no-project --python "$python_version" python - "$repo/pyproject.toml" > "$work/floors.txt" <<'PY'
import re
import sys
import tomllib

project = tomllib.load(open(sys.argv[1], "rb"))["project"]
extras = project["optional-dependencies"].values()
for dependency in project["dependencies"] + [d for extra in extras for d in extra]:
    found = re.match(r"([A-Za-z0-9_.-]+)(?:\[[^]]*\])?\s*>=\s*([0-9][0-9.]*)", dependency)
    if found is None:
        sys.exit(f"no floor in {dependency!r}: write it as name>=version")
    print(f"{found[1]}=={found[2]}")
PY
grep -v -E '^(openai|opentelemetry-[a-z-]+)==' "$work/floors.txt" > "$work/floors-livekit.txt"

uv build --quiet --wheel --out-dir "$work/dist" "$repo"
wheel="$(ls "$work"/dist/*.whl)"

run() {
  local name="$1" extras="$2" floors="$3"
  shift 3
  echo "== $name: $(tr '\n' ' ' < "$floors")"
  uv venv --quiet --python "$python_version" "$work/$name"
  VIRTUAL_ENV="$work/$name" uv pip install --quiet --override "$floors" \
    "unmute-core[$extras] @ file://$wheel" "pytest>=8" "pytest-asyncio>=1"
  (cd "$repo" && "$work/$name/bin/python" -m pytest -q -p no:cacheprovider \
    --ignore=tests/test_docs.py "$@")
}

run all-extras "livekit,pipecat,slng,knowledge" "$work/floors-livekit.txt" "$@"
run no-livekit "pipecat,slng,knowledge" "$work/floors.txt" "$@"
