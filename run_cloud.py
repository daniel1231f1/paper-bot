"""Headless runner for GitHub Actions: same LiveBot as the local server, no web UI.
Loops for RUN_SECONDS (a job is capped at 6h), writing snapshot.json for the static dashboard."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from bot import Coinbase, LiveBot
from settings import load

SNAPSHOT = Path(os.getenv("SNAPSHOT_FILE") or Path(__file__).with_name("snapshot.json"))
PUBLISH_EVERY = 300  # seconds between pushes of state to the `state` branch


def write_snapshot(bot: LiveBot) -> dict:
    snap = bot.snapshot()
    tmp = SNAPSHOT.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap))
    tmp.replace(SNAPSHOT)
    return snap


def publish() -> None:
    if os.getenv("PUBLISH") == "1":
        subprocess.run(["bash", "scripts/publish_state.sh"], check=False)  # never let a push hiccup kill the bot


def main() -> None:
    load_dotenv()
    cfg, run_days = load()
    budget = float(os.getenv("RUN_SECONDS", "20400"))
    bot = LiveBot(cfg, Coinbase(), run_days)
    for attempt in range(1, 6):
        try:
            bot.seed()
            break
        except Exception as e:
            print(f"seed attempt {attempt} failed: {e}", flush=True)
            time.sleep(15)
    else:
        sys.exit("could not fetch market data")

    end = time.time() + budget
    last_snap = last_pub = 0.0
    while time.time() < end:
        if bot.snapshot()["finished"]:
            print("run complete", flush=True)
            break
        bot.tick()
        now = time.time()
        if now - last_snap >= 60:
            snap = write_snapshot(bot)
            last_snap = now
            print(f"{time.strftime('%H:%M:%S')} px={snap['price']:.2f} eq={snap['equity']:.4f} "
                  f"p_up={snap['p_up']} pos={snap['position']} trades={snap['stats']['trades']} err={snap['error']}", flush=True)
        if now - last_pub >= PUBLISH_EVERY:
            bot._save()
            publish()
            last_pub = now
        time.sleep(cfg.poll_seconds)
    bot._save()
    write_snapshot(bot)


if __name__ == "__main__":
    main()
