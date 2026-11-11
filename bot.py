"""Prediction-driven PAPER trading bot on real Coinbase market data. No exchange client with order access exists in this code."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import requests

BASE_URL = "https://api.exchange.coinbase.com"  # public market data, no API key
STATE_FILE = Path(os.getenv("STATE_FILE") or Path(__file__).with_name("state.json"))


# ---------- data layer ----------
class Coinbase:
    MAX_CANDLES = 300

    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "paper-bot/2.0"

    def _get(self, path: str, **params):
        r = self.s.get(f"{BASE_URL}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def price(self, product: str) -> float:
        return float(self._get(f"/products/{product}/ticker")["price"])

    def candles(self, product: str, granularity: int, count: int) -> list[tuple[int, float]]:
        """[(bucket_start_unix, close)] oldest first, paginated (300/page). granularity: 60/300/900/3600/21600/86400."""
        out: dict[int, float] = {}
        end = int(time.time())
        while len(out) < count:
            n = min(self.MAX_CANDLES, count - len(out))
            start = end - n * granularity
            rows = self._get(f"/products/{product}/candles", granularity=granularity,
                             start=_iso(start), end=_iso(end))
            if not rows:
                break
            for ts, _lo, _hi, _op, close, _vol in rows:
                out[int(ts)] = float(close)
            end = start
        return sorted(out.items())


def _iso(ts: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


# ---------- config ----------
@dataclass(frozen=True)
class Config:
    product: str = "BTC-USD"
    capital: float = 100.0
    candle_seconds: int = 900          # 15-min candles
    horizon: int = 4                   # predict direction 4 candles (1h) ahead
    fast: int = 9
    slow: int = 21                     # must be <= 30
    fee_rate: float = 0.001            # per side
    slippage: float = 0.0005           # per side, adverse fill
    entry_p: float = 0.60              # min P(up) to buy
    exit_p: float = 0.45               # sell when P(up) falls below
    min_vol_mult: float = 1.0          # expected 1h move (volatility) must cover round-trip cost * this
    stop_pct: float = 0.012            # hard stop from entry
    trail_activate: float = 0.006      # trailing stop arms after this gain
    trail_pct: float = 0.005
    cooldown: int = 2                  # candles to sit out after an exit
    train_candles: int = 2000
    min_train: int = 300
    retrain_every: int = 4
    poll_seconds: int = 15

    @property
    def round_trip_cost(self) -> float:
        return 2 * (self.fee_rate + self.slippage)


# ---------- model ----------
def features(c: np.ndarray, t: int, fast: int, slow: int) -> np.ndarray:
    """Feature row for candle t using only data up to t. Needs t >= 30."""
    ret = lambda k: math.log(c[t] / c[t - k])
    w = c[t - 19 : t + 1]
    sd = float(w.std()) or 1e-9
    steps = np.diff(np.log(c[t - 20 : t + 1]))
    d = np.diff(c[t - 14 : t + 1])
    gain, loss = d[d > 0].sum(), -d[d < 0].sum()
    rsi = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    sf, ss = c[t - fast + 1 : t + 1].mean(), c[t - slow + 1 : t + 1].mean()
    return np.array([ret(1), ret(3), ret(6), ret(12), ret(24), (rsi - 50) / 50,
                     (c[t] - w.mean()) / sd, float(steps.std()), sf / ss - 1])


class LogReg:
    """L2-regularised logistic regression, plain gradient descent. O(epochs * n * d)."""

    def __init__(self, lr: float = 0.2, epochs: int = 300, l2: float = 0.05) -> None:
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w = self.mu = self.sd = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        if len(y) < 100 or y.min() == y.max():
            return False
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.clip((X - mu) / sd, -5, 5)
        w, b = np.zeros(Z.shape[1]), 0.0
        for _ in range(self.epochs):
            g = 1 / (1 + np.exp(-(Z @ w + b))) - y
            w -= self.lr * (Z.T @ g / len(y) + self.l2 * w)
            b -= self.lr * g.mean()
        self.w, self.b, self.mu, self.sd = w, b, mu, sd
        return True

    def proba(self, x: np.ndarray) -> float:
        z = float(np.clip((x - self.mu) / self.sd, -5, 5) @ self.w + self.b)
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


# ---------- trading engine (shared by live run and backtest) ----------
@dataclass
class Trade:
    ts: float
    side: str
    price: float
    qty: float
    fee: float
    pnl: float | None
    reason: str
    p: float | None


class Engine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.min_hist = max(30, cfg.slow)
        self.cash = cfg.capital
        self.pos: dict | None = None  # qty, entry, cost, peak
        self.trades: list[Trade] = []
        self.ts: list[int] = []
        self.closes: list[float] = []
        self.feat: list[np.ndarray | None] = []
        self.model = LogReg()
        self.ready = False
        self.since_fit = 0
        self.cooldown = 0
        self.pending: list[dict] = []
        self.resolved = self.hits = self.strong = self.strong_hits = 0
        self.last_p: float | None = None
        self.entries_enabled = True

    @property
    def last_ts(self) -> int:
        return self.ts[-1] if self.ts else 0

    def equity(self, price: float) -> float:
        return self.cash + (self.pos["qty"] * price if self.pos else 0.0)

    # --- history / learning ---
    def warm(self, rows: list[tuple[int, float]]) -> None:
        for ts, c in rows:
            self._append(ts, c)
        self._fit()
        if self.ready and self.feat and self.feat[-1] is not None:
            self.last_p = self.model.proba(self.feat[-1])  # show a prediction immediately; not graded

    def _append(self, ts: int, close: float) -> int:
        self.ts.append(ts)
        self.closes.append(close)
        t = len(self.closes) - 1
        self.feat.append(features(np.asarray(self.closes[-60:]), min(t, 59), self.cfg.fast, self.cfg.slow)
                         if t >= self.min_hist else None)
        return t

    def _fit(self) -> None:
        h, n = self.cfg.horizon, len(self.closes)
        lo = max(self.min_hist, n - h - self.cfg.train_candles)
        idx = [t for t in range(lo, n - h) if self.feat[t] is not None]
        if len(idx) < 100:
            return
        c = np.asarray(self.closes)
        X = np.array([self.feat[t] for t in idx])
        y = np.array([c[t + h] > c[t] for t in idx], dtype=float)
        self.ready = self.model.fit(X, y)
        self.since_fit = 0

    def _resolve(self, t: int) -> None:
        keep = []
        for pr in self.pending:
            if t - pr["t"] >= self.cfg.horizon:
                up = self.closes[t] > pr["price"]
                self.resolved += 1
                self.hits += int(up == (pr["p"] >= 0.5))
                if pr["p"] >= self.cfg.entry_p:
                    self.strong += 1
                    self.strong_hits += int(up)
            else:
                keep.append(pr)
        self.pending = keep

    # --- events ---
    def on_candle(self, ts: int, close: float) -> None:
        t = self._append(ts, close)
        self._resolve(t)
        self.cooldown = max(0, self.cooldown - 1)
        self.since_fit += 1
        if not self.ready or self.since_fit >= self.cfg.retrain_every:
            self._fit()
        if not self.ready or self.feat[t] is None:
            return
        p = self.model.proba(self.feat[t])
        self.last_p = p
        self.pending.append({"t": t, "p": p, "price": close})
        cfg = self.cfg
        if self.pos is None:
            trend_ok = np.mean(self.closes[-cfg.fast:]) >= np.mean(self.closes[-cfg.slow:])
            steps = np.diff(np.log(self.closes[-21:]))
            vol_ok = float(steps.std()) * math.sqrt(cfg.horizon) >= cfg.round_trip_cost * cfg.min_vol_mult
            if self.entries_enabled and self.cooldown == 0 and p >= cfg.entry_p and trend_ok and vol_ok:
                self._buy(ts + cfg.candle_seconds, close, p, f"P(up)={p:.0%}, trend up")
        elif p <= cfg.exit_p:
            self._sell(ts + cfg.candle_seconds, close, p, f"P(up) fell to {p:.0%}")

    def on_tick(self, ts: float, price: float) -> None:
        if not self.pos:
            return
        cfg, pos = self.cfg, self.pos
        pos["peak"] = max(pos["peak"], price)
        if price <= pos["entry"] * (1 - cfg.stop_pct):
            self._sell(ts, price, None, f"stop loss -{cfg.stop_pct:.1%}")
        elif pos["peak"] >= pos["entry"] * (1 + cfg.trail_activate) and price <= pos["peak"] * (1 - cfg.trail_pct):
            self._sell(ts, price, None, f"trailing stop -{cfg.trail_pct:.1%} from peak")

    # --- paper execution (all-in, long only; fees + adverse slippage) ---
    def _buy(self, ts: float, price: float, p: float | None, reason: str) -> None:
        fill = price * (1 + self.cfg.slippage)
        fee = self.cash * self.cfg.fee_rate
        qty = (self.cash - fee) / fill
        self.pos = {"qty": qty, "entry": fill, "cost": self.cash, "peak": fill}
        self.trades.append(Trade(ts, "buy", fill, qty, fee, None, reason, p))
        self.cash = 0.0

    def _sell(self, ts: float, price: float, p: float | None, reason: str) -> None:
        pos = self.pos
        fill = price * (1 - self.cfg.slippage)
        gross = pos["qty"] * fill
        fee = gross * self.cfg.fee_rate
        self.cash = gross - fee
        self.trades.append(Trade(ts, "sell", fill, pos["qty"], fee, self.cash - pos["cost"], reason, p))
        self.pos = None
        self.cooldown = self.cfg.cooldown

    # --- reporting / persistence ---
    def stats(self) -> dict:
        sells = [t for t in self.trades if t.side == "sell"]
        wins = [t for t in sells if t.pnl > 0]
        return {
            "trades": len(self.trades), "round_trips": len(sells),
            "win_rate": len(wins) / len(sells) if sells else None,
            "fees_paid": sum(t.fee for t in self.trades),
            "pred_resolved": self.resolved,
            "pred_accuracy": self.hits / self.resolved if self.resolved else None,
            "strong_n": self.strong,
            "strong_precision": self.strong_hits / self.strong if self.strong else None,
        }

    def dump(self) -> dict:
        return {"cash": self.cash, "pos": self.pos, "cooldown": self.cooldown,
                "trades": [asdict(t) for t in self.trades], "resolved": self.resolved, "hits": self.hits,
                "strong": self.strong, "strong_hits": self.strong_hits}

    def load(self, d: dict) -> None:
        self.cash, self.pos, self.cooldown = d["cash"], d["pos"], d["cooldown"]
        self.trades = [Trade(**t) for t in d["trades"]]
        self.resolved, self.hits, self.strong, self.strong_hits = d["resolved"], d["hits"], d["strong"], d["strong_hits"]


def max_drawdown(equity: list[float]) -> float:
    peak, dd = 0.0, 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            dd = max(dd, (peak - e) / peak)
    return dd


# ---------- walk-forward backtest (same Engine; model only ever sees the past) ----------
def run_backtest(cfg: Config, rows: list[tuple[int, float]]) -> dict:
    cfg = replace(cfg, retrain_every=16)  # speed
    eng = Engine(cfg)
    warm = cfg.min_train
    eng.warm(rows[:warm])
    curve = []
    for ts, c in rows[warm:]:
        eng.on_tick(ts, c)
        eng.on_candle(ts, c)
        curve.append({"t": ts, "equity": eng.equity(c), "price": c})
    base = rows[warm][1] if len(rows) > warm else rows[-1][1]
    last = rows[-1][1]
    final = eng.equity(last)
    return {
        "final_equity": final, "return_pct": (final / cfg.capital - 1) * 100,
        "buy_hold_equity": cfg.capital * last / base, "buy_hold_pct": (last / base - 1) * 100,
        "max_drawdown_pct": max_drawdown([p["equity"] for p in curve]) * 100,
        "stats": eng.stats(), "trades": [asdict(t) for t in eng.trades], "curve": curve,
    }


# ---------- live runner ----------
class LiveBot:
    def __init__(self, cfg: Config, api: Coinbase, run_days: float) -> None:
        self.cfg, self.api, self.run_days = cfg, api, run_days
        self.eng = Engine(cfg)
        self.lock = threading.RLock()
        self.samples: list[list[float]] = []  # [ts, equity, price]
        self.run_started = time.time()
        self.start_price: float | None = None
        self.paused = False
        self.price: float | None = None
        self.error: str | None = None
        self.next_check = 0.0
        self._last_save = 0.0
        self._stop = threading.Event()
        self._saved = self._read_state()

    def _read_state(self) -> dict | None:
        try:
            d = json.loads(STATE_FILE.read_text())
        except (OSError, ValueError):
            return None
        if d.get("product") != self.cfg.product or d.get("capital") != self.cfg.capital:
            return None  # different setup => fresh run
        return d

    def _save(self) -> None:
        d = {"product": self.cfg.product, "capital": self.cfg.capital, "run_started": self.run_started,
             "start_price": self.start_price, "paused": self.paused, "samples": self.samples[-12000:],
             "engine": self.eng.dump()}
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        tmp.replace(STATE_FILE)  # atomic: a crash mid-write can't corrupt the run
        self._last_save = time.time()

    def seed(self) -> None:
        g = self.cfg.candle_seconds
        bucket = int(time.time() // g) * g
        rows = [r for r in self.api.candles(self.cfg.product, g, self.cfg.train_candles + 60) if r[0] < bucket]
        px = self.api.price(self.cfg.product)
        with self.lock:
            self.eng.warm(rows)
            self.price = px
            if self._saved:
                d = self._saved
                self.eng.load(d["engine"])
                self.samples, self.run_started = d["samples"], d["run_started"]
                self.start_price, self.paused = d["start_price"], d["paused"]
            else:
                self.start_price = px
            self.eng.entries_enabled = not self.paused

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.is_set() and not self.eng.closes:
            try:
                self.seed()
                self.error = None
            except Exception as e:
                self.error = f"startup data fetch failed, retrying: {e}"
                self._stop.wait(10)
        while not self._stop.is_set():
            self.tick()
            self.next_check = time.time() + self.cfg.poll_seconds
            self._stop.wait(self.cfg.poll_seconds)

    def tick(self) -> None:
        cfg, g = self.cfg, self.cfg.candle_seconds
        try:
            px = self.api.price(cfg.product)
            now = time.time()
            bucket = int(now // g) * g
            new: list[tuple[int, float]] = []
            if bucket - g > self.eng.last_ts:  # a candle has closed: take its true close from the exchange
                new = [r for r in self.api.candles(cfg.product, g, 6) if self.eng.last_ts < r[0] < bucket]
        except Exception as e:
            self.error = f"fetch failed (will retry): {e}"
            return
        with self.lock:
            self.error = None
            self.price = px
            self.eng.on_tick(now, px)
            for ts, c in new:
                self.eng.on_candle(ts, c)
            if not self.samples or now - self.samples[-1][0] >= 60:
                self.samples.append([now, self.eng.equity(px), px])
            if new or now - self._last_save >= 60:
                self._save()

    def set_paused(self, v: bool) -> None:
        with self.lock:
            self.paused = v
            self.eng.entries_enabled = not v
            self._save()

    def snapshot(self) -> dict:
        with self.lock:
            eng, px = self.eng, self.price or 0.0
            eq = eng.equity(px)
            step = max(1, len(self.samples) // 500)
            curve = [{"t": s[0], "e": s[1], "p": s[2]} for s in self.samples[::step]]
            hold = self.cfg.capital * px / self.start_price if self.start_price else self.cfg.capital
            elapsed = time.time() - self.run_started
            p = eng.last_p
            return {
                "product": self.cfg.product, "price": px, "paused": self.paused, "error": self.error,
                "capital": self.cfg.capital, "equity": eq, "buy_hold_equity": hold,
                "max_drawdown_pct": max_drawdown([s[1] for s in self.samples] + [eq]) * 100,
                "position": "LONG" if eng.pos else "FLAT", "cash": eng.cash, "qty": eng.pos["qty"] if eng.pos else 0.0,
                "position_entry": eng.pos["entry"] if eng.pos else None,
                "p_up": p, "horizon_min": self.cfg.horizon * self.cfg.candle_seconds // 60,
                "model_ready": eng.ready, "entry_p": self.cfg.entry_p, "exit_p": self.cfg.exit_p,
                "elapsed_days": elapsed / 86400, "run_days": self.run_days,
                "candles": len(eng.closes), "candle_minutes": self.cfg.candle_seconds // 60,
                "next_check": self.next_check, "stats": eng.stats(), "curve": curve,
                "trades": [asdict(t) for t in eng.trades][-100:],
                "round_trip_cost_pct": self.cfg.round_trip_cost * 100,
                "finished": elapsed / 86400 >= self.run_days, "updated": time.time(),
            }
