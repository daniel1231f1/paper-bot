#!/usr/bin/env bash
# Force-push state.json, snapshot.json and the dashboard to the single-commit `state` branch (no history growth).
set -euo pipefail
[ -f state.json ] || exit 0
src=$(pwd)
remote="${PUBLISH_REMOTE:-https://x-access-token:${GH_TOKEN:-}@github.com/${GITHUB_REPOSITORY:-}.git}"
pub=$(mktemp -d)
cp state.json "$pub/"
[ -f snapshot.json ] && cp snapshot.json "$pub/" || true
cp static/index.html "$pub/index.html"
touch "$pub/.nojekyll"
cd "$pub"
git init -q -b state
git config user.name "paper-bot"
git config user.email "paper-bot@users.noreply.github.com"
git add -A
git commit -q -m "state $(date -u +%FT%TZ)"
git push -q --force "$remote" state
echo "published state to branch 'state'"
cd "$src"; rm -rf "$pub"
