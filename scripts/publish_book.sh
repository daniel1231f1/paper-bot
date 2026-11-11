#!/usr/bin/env bash
# Force-push book.csv + report.txt to the single-commit `book` branch.
set -uo pipefail
[ -f book.csv ] || exit 0
src=$(pwd)
remote="${PUBLISH_REMOTE:-https://x-access-token:${GH_TOKEN:-}@github.com/${GITHUB_REPOSITORY:-}.git}"
${PYTHON:-python} research/analyze_book.py book.csv > report.txt 2>&1 || true
pub=$(mktemp -d)
cp book.csv report.txt "$pub/"
cd "$pub"
git init -q -b book
git config user.name "paper-bot"; git config user.email "paper-bot@users.noreply.github.com"
git add -A && git commit -q -m "book $(date -u +%FT%TZ)"
git push -q --force "$remote" book && echo "published book"
cd "$src"; rm -rf "$pub"
