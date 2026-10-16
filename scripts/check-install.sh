#!/usr/bin/env bash
# Install one unmute-core version in a fresh project, import every adapter and check the version.
# Usage: scripts/check-install.sh VERSION [WHEEL | INDEX_URL]
#   no source   unmute-core from PyPI
#   WHEEL       unmute-core from that wheel file
#   INDEX_URL   unmute-core from that index, like TestPyPI. Its dependencies still come from PyPI.
# A new upload takes a while to show in an index, so an index install retries for 5 minutes.
set -euo pipefail

version="$1"
source="${2:-}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat > "$work/pyproject.toml" <<EOF
[project]
name = "check-install"
version = "0"
requires-python = ">=3.12"
dependencies = ["unmute-core[livekit,pipecat,slng]==$version"]
EOF

if [[ "$source" == *.whl ]]; then
  printf '\n[tool.uv.sources]\nunmute-core = { path = "%s" }\n' "$(cd "$(dirname "$source")" && pwd)/$(basename "$source")" >> "$work/pyproject.toml"
elif [ -n "$source" ]; then
  cat >> "$work/pyproject.toml" <<EOF

[[tool.uv.index]]
name = "release"
url = "$source"
explicit = true

[tool.uv.sources]
unmute-core = { index = "release" }
EOF
fi

for attempt in 1 2 3 4 5 6 7 8 9 10; do
  uv sync --quiet --refresh-package unmute-core --project "$work" && break
  if [ "$attempt" = 10 ]; then exit 1; fi
  echo "unmute-core $version does not install yet. Retry $attempt of 9 in 30 s."
  sleep 30
done

uv run --no-sync --project "$work" python - "$version" <<'PY'
import re
import sys

import unmute_core
import unmute_core.livekit
import unmute_core.pipecat
import unmute_core.slng

version = sys.argv[1]
assert unmute_core.__version__ == version, f"installed {unmute_core.__version__}, expected {version}"
unmute_core.require(re.match(r"\d+\.\d+\.\d+", version)[0])
print(f"unmute-core {version} installs, imports both adapters and passes require()")
PY
