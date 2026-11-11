"""Event engine: one closed bar at a time, identical code for backtest and live.

Bar processing order (bar with open time `ts`, decision time = ts + bar):
  1. orders queued at the previous close fill at THIS bar's open (never the signal bar)
  2. exits: experiment end / kill switch / session close / time stop at the open, then stop & target intrabar
  3. mark-to-market at the close, update drawdown, daily-loss halt and kill switch
  4. evaluate new candidates at the close -> queue orders for the NEXT bar's open

Idempotency: a bar whose ts <= last_bar_ts is ignored, so replaying candles after a
crashed/overlapping run can never double-trade. Staleness: when `fresh` is False
(catch-up of old bars) exits and marks are still applied but no entry is queued.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .broker import LONG, Fill, barrier_hit, fill_entry, fill_exit_limit, fill_exit_market
from .clock import SessionClock
from .config import Config
from .risk import RiskState, position_size

MAX_DECISIONS = 300


@dataclass(frozen=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    synthetic: bool = False


@dataclass(frozen=True)
class Candidate:
    symbol: str
    side: int
    signal: str
    regime: str
    stop_dist: float
    tp_dist: float
    prob: float | None = None  # calibrated P(target before stop); None = model not ready
    ev_r: float | None = None  # expected value after costs, in R
    skip_reason: str = ""  # veto decided before the model (e.g. costs dominate)


@dataclass
class PendingOrder:
    symbol: str
    side: int
    qty: float
    decision_ts: int
    fill_ts: int
    stop_dist: float
    tp_dist: float
    signal: str
    regime: str
    prob: float | None
    ev_r: float | None
    flatten_ts: int
    ref_price: float


@dataclass
class Position:
    symbol: str
    side: int
    qty: float
    entry_ts: int
    entry_price: float
    stop: float
    target: float
    stop_dist: float
    flatten_ts: int
    max_hold_ts: int
    signal: str
    regime: str
    prob: float | None
    ev_r: float | None
    entry_fee: float
    entry_slip: float
    risk_usd: float
    partial: bool = False


@dataclass
class Trade:
    symbol: str
    side: int
    signal: str
    regime: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    qty: float
    fees: float
    slippage: float
    pnl: float
    r: float
    prob: float | None
    ev_r: float | None
    exit_reason: str
    bars_held: int
    partial: bool = False


@dataclass
class Engine:
    cfg: Config
    clock: SessionClock
    enforce_clock: bool = True  # live: stop for good at the end of experiment day N. Backtest: off.
    cash: float = 0.0
    equity: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    pending: dict[str, PendingOrder] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    risk: RiskState = field(default_factory=lambda: RiskState(0.0))
    last_bar_ts: int = 0
    start_ts: int | None = None
    end_ts: int | None = None
    frozen: bool = False
    frozen_ts: int | None = None
    last_close: dict[str, float] = field(default_factory=dict)
    bench_units: dict[str, float] = field(default_factory=dict)
    bench: float = 0.0
    curve: list[list[float]] = field(default_factory=list)  # [close_ts, equity, benchmark]
    decisions: list[dict[str, Any]] = field(default_factory=list)
    skip_counts: dict[str, int] = field(default_factory=dict)
    record_curve: bool = True

    def __post_init__(self) -> None:
        if self.cash == 0.0 and not self.trades and self.start_ts is None:
            cap = self.cfg.experiment.starting_capital
            self.cash = self.equity = self.bench = cap
            self.risk = RiskState(peak_equity=cap)

    # ------------------------------------------------------------------ lifecycle
    def begin(self, ts: int, closes: Mapping[str, float]) -> None:
        """Start the experiment clock and buy the equal-weight buy-and-hold benchmark (with costs)."""
        self.start_ts = ts
        if self.enforce_clock:
            self.end_ts = self.clock.experiment_end(ts, self.cfg.experiment.days)
        cap, n = self.cfg.experiment.starting_capital, len(closes)
        self.bench_units = {s: (cap / n) / (px * (1 + self.cfg.costs.side_cost)) for s, px in closes.items()}

    # ------------------------------------------------------------------ main step
    def on_bar(self, ts: int, bars: Mapping[str, Bar], candidates: Sequence[Candidate] = (),
               fresh: bool = True) -> None:
        if self.frozen or ts <= self.last_bar_ts:
            return  # idempotent: never process a bar twice, never trade after the freeze
        cfg, bar_ms = self.cfg, self.cfg.bar_ms
        close_ts = ts + bar_ms
        if self.start_ts is None:
            self.begin(close_ts, {s: b.close for s, b in bars.items()})  # clock starts at the first decision time
        day_id = self.clock.local_date(ts).toordinal()
        if day_id != self.risk.day_id:
            self.risk.start_day(day_id, self.equity)
        ending = self.end_ts is not None and ts >= self.end_ts
        flatten_all = ending or self.risk.killed

        # 1. fills at this bar's open ---------------------------------------------
        for sym, od in list(self.pending.items()):
            del self.pending[sym]
            bar = bars.get(sym)
            if flatten_all or od.fill_ts != ts or bar is None:
                self._log(close_ts, od.symbol, od.signal, od.regime, od.side, od.prob, od.ev_r, "cancelled",
                          "order cancelled: bot halted or fill bar missed")
                continue
            self._fill_entry(ts, od, bar)

        # 2. exits ------------------------------------------------------------------
        for sym, pos in list(self.positions.items()):
            bar = bars.get(sym)
            if bar is None:
                continue  # no data for this symbol this bar: keep the position, mark at last known close
            reason = ("experiment end" if ending else "kill switch" if self.risk.killed
                      else "session close" if ts >= pos.flatten_ts
                      else "time stop" if ts >= pos.max_hold_ts else "")
            if reason:
                self._close(pos, fill_exit_market(pos.side, bar.open, pos.qty, bar.volume, cfg.costs), ts, reason)
                continue
            hit = barrier_hit(pos.side, bar.open, bar.high, bar.low, pos.stop, pos.target)
            if hit is None:
                continue
            kind, ref = hit
            fill = fill_exit_limit(ref, pos.qty, cfg.costs) if kind == "target" \
                else fill_exit_market(pos.side, ref, pos.qty, bar.volume, cfg.costs)
            self._close(pos, fill, close_ts, kind)

        # 3. mark to market -----------------------------------------------------------
        for sym, bar in bars.items():
            self.last_close[sym] = bar.close
        self.equity = self.cash + sum(p.side * p.qty * self.last_close[p.symbol] for p in self.positions.values())
        self.risk.mark(self.equity, close_ts, cfg.risk)
        if self.bench_units:
            self.bench = sum(u * self.last_close.get(s, 0.0) for s, u in self.bench_units.items())
        if self.record_curve:
            self.curve.append([close_ts, self.equity, self.bench])
        self.last_bar_ts = ts
        if ending:
            self.frozen, self.frozen_ts = True, ts
            return

        # 4. new entries (queued for the next bar's open) --------------------------------
        for cand in sorted(candidates, key=lambda c: -(c.ev_r if c.ev_r is not None else -1e9)):
            self._consider(cand, close_ts, bars, fresh)

    # ------------------------------------------------------------------ entries
    def _consider(self, c: Candidate, decision_ts: int, bars: Mapping[str, Bar], fresh: bool) -> None:
        cfg = self.cfg
        why = c.skip_reason
        if not why and not self.clock.entries_allowed(decision_ts):
            why = "outside the entry window"
        if not why and self.end_ts is not None and decision_ts >= self.end_ts:
            why = "experiment finished"
        if not why and not fresh:
            why = "stale signal: bar closed too long ago to act on"
        if not why:
            why = self.risk.entry_block(c.symbol, decision_ts)
        if not why and c.prob is None:
            why = "model not ready: not enough resolved training samples"
        if not why and c.prob < cfg.model.min_prob:
            why = f"model probability {c.prob:.0%} is below the {cfg.model.min_prob:.0%} threshold"
        if not why and (c.ev_r is None or c.ev_r < cfg.model.min_ev_r):
            ev = float("nan") if c.ev_r is None else c.ev_r
            why = f"expected value after costs {ev:+.2f}R is below {cfg.model.min_ev_r:+.2f}R"
        if not why and (c.symbol in self.positions or c.symbol in self.pending):
            why = "already in this symbol (no averaging down)"
        if not why and len(self.positions) + len(self.pending) >= cfg.risk.max_positions:
            why = f"max concurrent positions ({cfg.risk.max_positions}) reached"
        qty = 0.0
        price = bars[c.symbol].close
        if not why:
            reserved = sum(o.qty * o.ref_price * (1 + cfg.costs.side_cost) for o in self.pending.values())
            qty, why = position_size(self.equity, self.cash - reserved, price, c.stop_dist, c.symbol,
                                     cfg.risk, cfg.market, cfg.costs)
        if why:
            self._log(decision_ts, c.symbol, c.signal, c.regime, c.side, c.prob, c.ev_r, "skip", why)
            return
        day_close = self.clock.session_close(self.clock.local_date(decision_ts))
        self.pending[c.symbol] = PendingOrder(c.symbol, c.side, qty, decision_ts, decision_ts, c.stop_dist, c.tp_dist,
                                              c.signal, c.regime, c.prob, c.ev_r, day_close, price)
        self._log(decision_ts, c.symbol, c.signal, c.regime, c.side, c.prob, c.ev_r, "enter",
                  f"order queued for next bar open: {qty:g} units, risk {cfg.risk.risk_per_trade:.1%} of equity max")

    def _fill_entry(self, ts: int, od: PendingOrder, bar: Bar) -> None:
        cfg = self.cfg
        affordable = self.cash / (bar.open * (1 + cfg.costs.side_cost + 0.002)) if od.side == LONG else od.qty
        fill = fill_entry(od.side, bar.open, min(od.qty, affordable), bar.volume, bar.synthetic, cfg.costs,
                          cfg.market.min_notional, cfg.market.step(od.symbol))
        if not fill.ok:
            self._log(ts, od.symbol, od.signal, od.regime, od.side, od.prob, od.ev_r, "rejected", fill.reason)
            return
        self.cash -= od.side * fill.qty * fill.price + fill.fee
        self.positions[od.symbol] = Position(
            symbol=od.symbol, side=od.side, qty=fill.qty, entry_ts=ts, entry_price=fill.price,
            stop=bar.open - od.side * od.stop_dist, target=bar.open + od.side * od.tp_dist, stop_dist=od.stop_dist,
            flatten_ts=od.flatten_ts, max_hold_ts=ts + cfg.strategy.max_hold_bars * cfg.bar_ms, signal=od.signal,
            regime=od.regime, prob=od.prob, ev_r=od.ev_r, entry_fee=fill.fee, entry_slip=fill.slippage_cost,
            risk_usd=fill.qty * od.stop_dist, partial=fill.status == "partial")
        self.risk.risk_opened_today += fill.qty * od.stop_dist
        if fill.status == "partial":
            self._log(ts, od.symbol, od.signal, od.regime, od.side, od.prob, od.ev_r, "partial", fill.reason)

    # ------------------------------------------------------------------ exits
    def _close(self, pos: Position, fill: Fill, exit_ts: int, reason: str) -> None:
        del self.positions[pos.symbol]
        self.cash += pos.side * pos.qty * fill.price - fill.fee
        fees = pos.entry_fee + fill.fee
        pnl = pos.side * (fill.price - pos.entry_price) * pos.qty - fees
        self.trades.append(Trade(
            symbol=pos.symbol, side=pos.side, signal=pos.signal, regime=pos.regime, entry_ts=pos.entry_ts,
            exit_ts=exit_ts, entry_price=pos.entry_price, exit_price=fill.price, qty=pos.qty, fees=fees,
            slippage=pos.entry_slip + fill.slippage_cost, pnl=pnl, r=pnl / pos.risk_usd if pos.risk_usd else 0.0,
            prob=pos.prob, ev_r=pos.ev_r, exit_reason=reason,
            bars_held=max(1, round((exit_ts - pos.entry_ts) / self.cfg.bar_ms)), partial=pos.partial))
        self.risk.on_trade_closed(pnl, pos.symbol, exit_ts, self.cfg.bar_ms, self.cfg.risk)

    def _log(self, ts: int, symbol: str, signal: str, regime: str, side: int, prob: float | None,
             ev_r: float | None, action: str, reason: str) -> None:
        key = reason.split(":")[0] if action != "enter" else "entered"
        if "probability" in reason:
            key = "model probability below threshold"
        elif "expected value" in reason:
            key = "expected value below threshold"
        self.skip_counts[key] = self.skip_counts.get(key, 0) + 1
        self.decisions.append({"ts": ts, "symbol": symbol, "signal": signal, "regime": regime, "side": side,
                               "prob": prob, "ev_r": ev_r, "action": action, "reason": reason})
        if len(self.decisions) > MAX_DECISIONS:
            del self.decisions[: len(self.decisions) - MAX_DECISIONS]

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash, "equity": self.equity, "last_bar_ts": self.last_bar_ts, "start_ts": self.start_ts,
            "end_ts": self.end_ts, "frozen": self.frozen, "frozen_ts": self.frozen_ts,
            "positions": [asdict(p) for p in self.positions.values()],
            "pending": [asdict(o) for o in self.pending.values()],
            "trades": [asdict(t) for t in self.trades], "risk": asdict(self.risk),
            "last_close": self.last_close, "bench_units": self.bench_units, "bench": self.bench,
            "curve": self.curve, "decisions": self.decisions, "skip_counts": self.skip_counts,
        }

    @classmethod
    def from_dict(cls, cfg: Config, clock: SessionClock, d: Mapping[str, Any], enforce_clock: bool = True) -> "Engine":
        eng = cls(cfg, clock, enforce_clock=enforce_clock)
        eng.cash, eng.equity, eng.last_bar_ts = float(d["cash"]), float(d["equity"]), int(d["last_bar_ts"])
        eng.start_ts, eng.end_ts = d["start_ts"], d["end_ts"]
        eng.frozen, eng.frozen_ts = bool(d["frozen"]), d["frozen_ts"]
        eng.positions = {p["symbol"]: Position(**p) for p in d["positions"]}
        eng.pending = {o["symbol"]: PendingOrder(**o) for o in d["pending"]}
        eng.trades = [Trade(**t) for t in d["trades"]]
        eng.risk = RiskState(**d["risk"])
        eng.last_close, eng.bench_units, eng.bench = dict(d["last_close"]), dict(d["bench_units"]), float(d["bench"])
        eng.curve, eng.decisions = [list(p) for p in d["curve"]], list(d["decisions"])
        eng.skip_counts = dict(d["skip_counts"])
        return eng
