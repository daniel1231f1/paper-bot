#!/usr/bin/env bash
# Start a NEW experiment: archive the current published state to archive/state-<timestamp>, then
# clear the local copy so the bot starts fresh with whatever config.yaml says (new locked hash).
set -euo pipefail
STATE_DIR="${STATE_DIR:-state}"
sha=$(cat "$STATE_DIR/.remote_sha" 2>/dev/null || true)
if [ -n "$sha" ]; then
  git push -q origin "${sha}:refs/heads/archive/state-$(date -u +%Y%m%d-%H%M%S)"
  echo "archived previous state ($sha)"
fi
rm -f "$STATE_DIR/state.json" "$STATE_DIR/dashboard.json"
echo "experiment reset: the 10-day clock restarts with the first bar inside the trading window"
