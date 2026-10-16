#!/usr/bin/env bash
# Compile the example projects in examples/ with Unmute at the SHA in examples/UNMUTE_SHA.
# Usage: scripts/compile-examples.sh          # build Unmute at that SHA and recompile all examples
#        scripts/compile-examples.sh --bump   # first set the SHA to Unmute's current HEAD
# Needs git, go, uv, rsync and patch, and network access to clone Unmute.
# The output replaces each project, so `git diff examples/` afterwards shows what the
# hand edits add on top of what Unmute emits today. examples/README.md explains the rule.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
unmute_url="https://github.com/slng-ai/unmute"
examples="$repo/examples"

if [[ "${1:-}" == "--bump" ]]; then
  git ls-remote "$unmute_url" HEAD | cut -f1 >"$examples/UNMUTE_SHA"
fi
sha="$(cat "$examples/UNMUTE_SHA")"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Read from the public tree at a fixed SHA, never from a working checkout:
# a checkout can hold .env files and builds with keys in them.
mkdir -p "$work/unmute" "$work/bin" "$work/packages"
git clone --quiet --filter=blob:none --no-checkout "$unmute_url" "$work/clone"
git -C "$work/clone" archive "$sha" | tar -x -C "$work/unmute"
(cd "$work/unmute" && CGO_ENABLED=0 go build -o "$work/bin/unmute" .)

# compile formats with the ruff on PATH, so pin the one the emitted pyproject pins.
ruff_pin="$(grep -o 'ruff==[0-9.]*' "$work/unmute/internal/generate/pyproject.go" | head -1)"
printf '#!/bin/sh\nexec uvx %s "$@"\n' "$ruff_pin" >"$work/bin/ruff"
chmod +x "$work/bin/ruff"
export PATH="$work/bin:$PATH"

cp "$work/unmute/LICENSE" "$examples/LICENSE-unmute"
cp -R "$work/unmute/examples/salon-concierge" "$work/packages/"
cp -R "$work/unmute/examples/salon-concierge-single-prompt" "$work/packages/"
cp -R "$work/unmute/examples/salon-concierge" "$work/packages/salon-concierge-slng"
patch --quiet -p1 -d "$work/packages/salon-concierge-slng" <"$examples/salon-concierge-slng.patch"

# rsync keeps excluded files on the destination, so a local .env survives a recompile.
copy() { rsync -a --delete --exclude .env --exclude dev.log --exclude __pycache__ --exclude .ruff_cache --exclude .venv "$@"; }

for package in "$work"/packages/*/; do
  name="$(basename "$package")"
  unmute compile "$package" >/dev/null
  mkdir -p "$examples/$name"
  copy --exclude /build "$package" "$examples/$name/package/"
  copy "$package/build/livekit/" "$examples/$name/livekit/"
  copy "$package/build/pipecat/" "$examples/$name/pipecat/"
  echo "compiled $name"
done
echo "Unmute ${sha:0:7}: run scripts/check-examples.sh, then review git diff examples/"
