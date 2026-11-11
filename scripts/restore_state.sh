#!/usr/bin/env bash
# Pull the last saved run state from the `state` branch (if any) into ./state.json
set -uo pipefail
if git ls-remote --exit-code --heads origin state >/dev/null 2>&1; then
  git fetch -q --depth 1 origin state
  git show FETCH_HEAD:state.json > state.json 2>/dev/null || rm -f state.json
fi
[ -f state.json ] && echo "restored state.json" || echo "no saved state: fresh run"
exit 0
