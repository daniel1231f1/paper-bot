"""Market data: free public candles through ccxt, with retry/backoff, an on-disk cache,
closed-bar filtering and gap normalisation. No API keys, no private endpoints.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("bot.data")

COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
PAGE = 300  # Coinbase Exchange hard limit per candles request


class DataUnavailable(RuntimeError):
    """The exchange could not be reached after all retries. Callers must skip trading safely."""


class CandleClient(Protocol):
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int | None = None,
                    limit: int | None = None) -> list[list[float]]: ...


def make_client(exchange_id: str) -> CandleClient:
    import ccxt  # imported lazily so unit tests never need the network stack

    if exchange_id not in ccxt.exchanges:
        raise ValueError(f"unknown ccxt exchange: {exchange_id}")
    return getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 15_000})


def _retryable(exc: Exception) -> bool:
    try:
        import ccxt
    except ImportError:  # pragma: no cover
        return True
    if isinstance(exc, (ccxt.BadSymbol, ccxt.AuthenticationError, ccxt.NotSupported)):
        return False
    return isinstance(exc, (ccxt.NetworkError, ccxt.ExchangeError, OSError, TimeoutError))


class MarketData:
    """Fetches and caches closed candles on a fixed time grid."""

    def __init__(self, exchange_id: str, timeframe_minutes: int, cache_dir: Path | None = None,
                 client: CandleClient | None = None, max_retries: int = 4, retry_base_seconds: float = 1.5,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.exchange_id = exchange_id
        self.tf_min = timeframe_minutes
        self.bar_ms = timeframe_minutes * 60_000
        self.timeframe = f"{timeframe_minutes}m"
        self.cache_dir = cache_dir
        self._client = client
        self.max_retries = max_retries
        self.retry_base = retry_base_seconds
        self._sleep = sleep
        self._mem: dict[str, pd.DataFrame] = {}

    @property
    def client(self) -> CandleClient:
        if self._client is None:
            self._client = make_client(self.exchange_id)
        return self._client

    # ---- raw fetch with backoff ---------------------------------------
    def _fetch_page(self, symbol: str, since: int, limit: int) -> list[list[float]]:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.client.fetch_ohlcv(symbol, self.timeframe, since=since, limit=limit)
            except Exception as exc:  # noqa: BLE001 - classified below
                if not _retryable(exc):
                    raise
                last = exc
                delay = self.retry_base * (2 ** attempt) * (1 + random.random() * 0.25)  # exp backoff + jitter
                log.warning("fetch failed (%s %s attempt %d/%d): %s; retrying in %.1fs", symbol, self.timeframe,
                            attempt + 1, self.max_retries, type(exc).__name__, delay)
                self._sleep(delay)
        raise DataUnavailable(f"{self.exchange_id} {symbol}: {type(last).__name__}: {last}")

    def fetch_range(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Raw candles with start_ms <= ts < end_ms (may contain gaps). O(n/300) requests."""
        rows: list[list[float]] = []
        since = start_ms - start_ms % self.bar_ms
        while since < end_ms:
            limit = int(min(PAGE, max(1, -(-(end_ms - since) // self.bar_ms))))
            page = self._fetch_page(symbol, since, limit)
            rows.extend(r for r in page if start_ms <= r[0] < end_ms)
            since += limit * self.bar_ms
        df = pd.DataFrame(rows, columns=COLUMNS)
        if df.empty:
            return df.astype({"ts": "int64"})
        df["ts"] = df["ts"].astype("int64")
        return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    # ---- cache ---------------------------------------------------------
    def _cache_path(self, symbol: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / self.exchange_id / f"{symbol.replace('/', '-')}_{self.timeframe}.csv.gz"

    def _load_cache(self, symbol: str) -> pd.DataFrame:
        if symbol in self._mem:
            return self._mem[symbol]
        path = self._cache_path(symbol)
        if path is not None and path.exists():
            try:
                df = pd.read_csv(path)
                if list(df.columns) == COLUMNS:
                    df["ts"] = df["ts"].astype("int64")
                    return df
            except (OSError, ValueError) as exc:
                log.warning("ignoring unreadable candle cache %s: %s", path, exc)
        return pd.DataFrame(columns=COLUMNS).astype({"ts": "int64"})

    def _save_cache(self, symbol: str, df: pd.DataFrame) -> None:
        self._mem[symbol] = df
        path = self._cache_path(symbol)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.gz")
        df.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(path)  # atomic

    def remember(self, symbol: str, df: pd.DataFrame) -> None:
        """Keep raw candles in memory only (used by the live loop between polls)."""
        self._mem[symbol] = df

    def flush(self) -> None:
        """Write every in-memory candle frame to the disk cache (called at the end of a run)."""
        for symbol, df in list(self._mem.items()):
            self._save_cache(symbol, df)

    # ---- public API ------------------------------------------------------
    def history(self, symbol: str, start_ms: int, now_ms: int, persist: bool = True) -> pd.DataFrame:
        """Closed candles on a gap-free grid for start_ms <= ts, bar close <= now_ms.

        Only missing bars are requested (incremental). The last 3 cached bars are always
        re-fetched because exchanges may revise the newest candle for a few seconds.
        """
        last_closed = (now_ms // self.bar_ms) * self.bar_ms - self.bar_ms  # open time of newest closed bar
        start_ms -= start_ms % self.bar_ms
        cached = self._load_cache(symbol)
        cached = cached[cached["ts"] <= last_closed]
        parts = [cached]
        if cached.empty or cached["ts"].iloc[0] > start_ms:
            head_end = int(cached["ts"].iloc[0]) if not cached.empty else last_closed + self.bar_ms
            parts.append(self.fetch_range(symbol, start_ms, head_end))
        if not cached.empty:
            tail_start = max(start_ms, int(cached["ts"].iloc[-1]) - 2 * self.bar_ms)
            if tail_start <= last_closed:
                parts.append(self.fetch_range(symbol, tail_start, last_closed + self.bar_ms))
        raw = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) \
            else cached
        raw = raw.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
        if persist:
            self._save_cache(symbol, raw)
        else:
            self.remember(symbol, raw)
        return normalize(raw[raw["ts"] >= start_ms], self.bar_ms, end_ts=None)


def normalize(raw: pd.DataFrame, bar_ms: int, end_ts: int | None = None) -> pd.DataFrame:
    """Put candles on a complete grid. Missing bars (no trades on the venue) become synthetic:
    flat at the previous close with zero volume, flagged so the broker refuses to fill on them.
    """
    if raw.empty:
        out = raw.copy()
        out["synthetic"] = pd.Series(dtype=bool)
        return out
    first, last = int(raw["ts"].iloc[0]), int(end_ts if end_ts is not None else raw["ts"].iloc[-1])
    grid = np.arange(first, last + bar_ms, bar_ms, dtype="int64")
    out = raw.set_index("ts").reindex(grid)
    synthetic = out["close"].isna().to_numpy()
    out["close"] = out["close"].ffill()
    for col in ("open", "high", "low"):
        out[col] = out[col].where(~synthetic, out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    out["synthetic"] = synthetic
    out.index.name = "ts"
    return out.reset_index()


def fetch_many(md: MarketData, symbols: Sequence[str], start_ms: int, now_ms: int,
               persist: bool = True) -> dict[str, pd.DataFrame]:
    """History for every symbol. Raises DataUnavailable if any symbol cannot be fetched."""
    return {s: md.history(s, start_ms, now_ms, persist=persist) for s in symbols}
