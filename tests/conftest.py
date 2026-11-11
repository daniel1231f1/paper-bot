"""Shared fixtures: deterministic synthetic candles, a fake exchange client, IST time helpers."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.config import Config  # noqa: E402

BAR = 300_000
IST = ZoneInfo("Asia/Kolkata")


def ist(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> int:
    """Epoch ms of a wall-clock time in India."""
    return int(datetime(y, m, d, hh, mm, tzinfo=IST).timestamp() * 1000)


def make_candles(n: int, start_ts: int, seed: int = 0, price: float = 100.0, sigma: float = 0.002,
                 volume: float = 5_000.0) -> pd.DataFrame:
    """Random-walk OHLCV on a gap-free 5m grid."""
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    open_ = np.concatenate([[price], close[:-1]])
    spread = np.abs(rng.normal(0, sigma / 2, n)) * close
    return pd.DataFrame({"ts": start_ts + BAR * np.arange(n, dtype="int64"), "open": open_,
                         "high": np.maximum(open_, close) + spread, "low": np.minimum(open_, close) - spread,
                         "close": close, "volume": volume * (1 + rng.random(n))})


class FakeClient:
    """Stands in for a ccxt exchange. Serves bars with ts <= now (the newest one is still forming)."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames, self.now, self.calls, self.fail = frames, 0, 0, 0

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None):
        import ccxt
        self.calls += 1
        if self.fail > 0:
            self.fail -= 1
            raise ccxt.NetworkError("simulated outage")
        df = self.frames[symbol]
        lo = since or 0
        hi = lo + (limit or 300) * BAR
        m = (df["ts"] >= lo) & (df["ts"] < hi) & (df["ts"] <= self.now)
        return df.loc[m, ["ts", "open", "high", "low", "close", "volume"]].values.tolist()


def make_config(**sections: dict) -> Config:
    """Small, fast config: two symbols, no volume impact (exact cost arithmetic), short lookbacks."""
    base: dict = {
        "market": {"universe": ["AAA/USD", "BBB/USD"], "qty_step": {"AAA/USD": 0.001, "BBB/USD": 0.001}},
        "costs": {"impact_coef": 0.0},
        "strategy": {"atr_pct_bars": 288},
        "ops": {"warmup_bars": 700},
    }
    for name, values in sections.items():
        base.setdefault(name, {}).update(values)
    return Config.from_dict(base)


@pytest.fixture
def cfg() -> Config:
    return make_config()
