#!/usr/bin/env bash
# Pull the last published bot state from the `state` branch into $STATE_DIR and remember which
# commit it came from. publish_state.sh uses that commit as a compare-and-swap lease.
set -uo pipefail
STATE_DIR="${STATE_DIR:-state}"
mkdir -p "$STATE_DIR"
: > "$STATE_DIR/.remote_sha"
if git ls-remote --exit-code --heads origin state >/dev/null 2>&1; then
  git fetch -q --depth 1 origin state
  git rev-parse FETCH_HEAD > "$STATE_DIR/.remote_sha"
  git show FETCH_HEAD:state.json > "$STATE_DIR/state.json" 2>/dev/null || rm -f "$STATE_DIR/state.json"
fi
if [ -f "$STATE_DIR/state.json" ]; then echo "restored state from $(cat "$STATE_DIR/.remote_sha")"; else echo "no saved state: fresh start"; fi
exit 0
