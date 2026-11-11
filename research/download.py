"""Bulk-download candle history into the local cache (research only)."""
from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.config import Config  # noqa: E402
from bot.data import MarketData, make_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
START_MS = int(os.getenv("START_MS", "1759276800000"))  # default 2025-10-01T00:00:00Z


def pull(symbol: str) -> str:
    cfg = Config()
    client = make_client(cfg.market.exchange)
    client.rateLimit = 250  # ms between calls per client; 2 workers => ~8 req/s, under the 10 req/s public cap
    md = MarketData(cfg.market.exchange, cfg.market.timeframe_minutes, Path(".cache/candles"), client=client)
    df = md.history(symbol, START_MS, int(time.time() * 1000))
    return f"{symbol}: {len(df)} bars, {int(df['synthetic'].sum())} synthetic"


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2"))) as pool:
        for line in pool.map(pull, Config().market.universe):
            print(line, flush=True)
