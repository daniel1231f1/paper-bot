#!/usr/bin/env bash
# Publish state.json + dashboard.json + the dashboard page to the single-commit `state` branch
# (served by GitHub Pages). The push is a compare-and-swap: it only succeeds if the branch still
# points at the commit this run restored/last published. If another runner advanced it, exit 3 so
# the bot stops instead of overwriting newer state (no double trading from overlapping runs).
set -uo pipefail
STATE_DIR="${STATE_DIR:-state}"
[ -f "$STATE_DIR/state.json" ] && [ -f "$STATE_DIR/dashboard.json" ] || { echo "nothing to publish"; exit 0; }
remote="${PUBLISH_REMOTE:-https://x-access-token:${GH_TOKEN:-}@github.com/${GITHUB_REPOSITORY:-}.git}"
src=$(pwd)
expected=$(cat "$STATE_DIR/.remote_sha" 2>/dev/null || true)
pub=$(mktemp -d)
trap 'rm -rf "$pub"' EXIT
cp "$STATE_DIR/state.json" "$STATE_DIR/dashboard.json" "$pub/"
cp dashboard/index.html "$pub/index.html"
touch "$pub/.nojekyll"
cd "$pub" || exit 1
git init -q -b state
git config user.name "paper-bot"
git config user.email "paper-bot@users.noreply.github.com"
git add -A
git commit -q -m "state $(date -u +%FT%TZ)"
new_sha=$(git rev-parse HEAD)
out=$(git push -q "--force-with-lease=state:${expected}" "$remote" state 2>&1)
code=$?
cd "$src" || exit 1
if [ $code -eq 0 ]; then
  echo "$new_sha" > "$STATE_DIR/.remote_sha"
  echo "published $new_sha"
  exit 0
fi
if echo "$out" | grep -qiE "stale info|rejected|fetch first"; then
  echo "state branch moved since this run restored it: refusing to overwrite" >&2
  exit 3
fi
echo "publish failed (network?): ${out//${GH_TOKEN:-__none__}/***}" >&2
exit 1
