"""Engine behaviour: next-bar fills, agreement with the label simulator, session flatten,
idempotency, kill switch, daily halt, cooldown, and the day-10 freeze."""
from __future__ import annotations

import numpy as np
import pytest

from bot.broker import LONG, simulate_trade
from bot.clock import SessionClock
from bot.engine import Bar, Candidate, Engine, Position
from conftest import BAR, ist, make_config

SYM = "AAA/USD"
T0 = ist(2026, 1, 5, 10, 0)


def flat(px: float, vol: float = 10_000.0) -> dict[str, Bar]:
    return {SYM: Bar(px, px, px, px, vol)}


def cand(prob: float = 0.9, ev: float = 0.5, stop: float = 1.0, tp: float = 2.0, sym: str = SYM) -> Candidate:
    return Candidate(sym, LONG, "breakout", "TREND_UP", stop, tp, prob, ev)


def engine(**sections) -> Engine:
    cfg = make_config(**sections)
    return Engine(cfg, SessionClock.from_cfg(cfg.schedule), enforce_clock=False)


def test_order_fills_at_next_bar_open_never_on_the_signal_bar():
    eng = engine()
    eng.on_bar(T0, flat(100.0), [cand()])
    assert not eng.positions and SYM in eng.pending  # nothing filled on the signal bar
    eng.on_bar(T0 + BAR, {SYM: Bar(101.0, 101.4, 100.9, 101.2, 10_000)})
    pos = eng.positions[SYM]
    assert pos.entry_ts == T0 + BAR
    assert pos.entry_price == pytest.approx(101.0 * 1.0005)  # next bar's OPEN plus spread + slippage
    assert pos.stop == pytest.approx(100.0) and pos.target == pytest.approx(103.0)  # anchored on that open


def test_position_size_respects_risk_and_cap():
    eng = engine()
    eng.on_bar(T0, flat(100.0), [cand()])
    eng.on_bar(T0 + BAR, flat(100.0))
    pos = eng.positions[SYM]
    assert pos.qty * 100.0 <= 0.33 * 100.0 + 1e-9  # max 33% of equity in one position
    worst_loss = pos.qty * (pos.stop_dist + 100.0 * eng.cfg.costs.round_trip)
    assert worst_loss <= 0.01 * 100.0 + 1e-9  # <= 1% of equity at the stop, costs included


def test_engine_matches_label_simulator_for_stop_target_and_time_exits():
    rng = np.random.default_rng(3)
    for trial in range(40):
        n = 60
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
        open_ = np.concatenate([[100.0], close[:-1]])
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
        eng = engine(strategy={"max_hold_bars": 30})
        for i in range(n):
            bars = {SYM: Bar(open_[i], high[i], low[i], close[i], 1e6)}
            eng.on_bar(T0 + i * BAR, bars, [cand(stop=1.5, tp=3.0)] if i == 0 else [])
        expect = simulate_trade(open_, high, low, 1, LONG, 1.5, 3.0, flat_idx=31, costs=eng.cfg.costs)
        assert expect is not None and len(eng.trades) == 1, trial
        t = eng.trades[0]
        assert t.exit_reason == {"time": "time stop"}.get(expect[3], expect[3])
        assert t.r == pytest.approx(expect[1], rel=1e-6, abs=1e-9)


def test_no_averaging_down_and_one_position_per_symbol():
    eng = engine()
    eng.on_bar(T0, flat(100.0), [cand()])
    eng.on_bar(T0 + BAR, flat(100.0), [cand()])  # second signal while the first order just filled
    eng.on_bar(T0 + 2 * BAR, flat(99.5), [cand()])  # price lower: must not add
    assert len(eng.positions) == 1 and not eng.pending
    qty = eng.positions[SYM].qty
    eng.on_bar(T0 + 3 * BAR, flat(99.4), [cand()])
    assert eng.positions[SYM].qty == qty
    assert any("no averaging down" in d["reason"] for d in eng.decisions)


def test_max_concurrent_positions():
    cfg = make_config(market={"universe": ["A/USD", "B/USD", "C/USD", "D/USD"], "qty_step": {}},
                      risk={"max_positions": 2, "max_position_pct": 0.3})
    eng = Engine(cfg, SessionClock.from_cfg(cfg.schedule), enforce_clock=False)
    bars = {s: Bar(100, 100, 100, 100, 1e6) for s in cfg.market.universe}
    eng.on_bar(T0, bars, [cand(sym=s) for s in cfg.market.universe])
    eng.on_bar(T0 + BAR, bars)
    assert len(eng.positions) == 2
    assert sum("max concurrent" in d["reason"] for d in eng.decisions) == 2


def test_model_gate_blocks_low_probability_low_ev_and_missing_model():
    eng = engine()
    eng.on_bar(T0, flat(100.0), [cand(prob=0.2)])
    eng.on_bar(T0 + BAR, flat(100.0), [cand(prob=0.9, ev=-0.2)])
    eng.on_bar(T0 + 2 * BAR, flat(100.0), [Candidate(SYM, LONG, "breakout", "TREND_UP", 1.0, 2.0, None, None)])
    eng.on_bar(T0 + 3 * BAR, flat(100.0))
    assert not eng.positions and not eng.trades
    reasons = " | ".join(d["reason"] for d in eng.decisions)
    assert "probability" in reasons and "expected value" in reasons and "model not ready" in reasons


def test_stale_bar_never_opens_a_trade_but_still_manages_exits():
    eng = engine()
    eng.on_bar(T0, flat(100.0), [cand()])
    eng.on_bar(T0 + BAR, flat(100.0))
    assert SYM in eng.positions
    # catch-up bars (fresh=False): the stop still fires, new signals are ignored
    eng.on_bar(T0 + 2 * BAR, {SYM: Bar(100, 100, 98.5, 99, 1e4)}, [cand()], fresh=False)
    assert not eng.positions and eng.trades[-1].exit_reason == "stop"
    eng.on_bar(T0 + 9 * BAR, flat(99.0), [cand()], fresh=False)
    assert not eng.pending and any("stale" in d["reason"] for d in eng.decisions)


def test_processing_the_same_bar_twice_is_a_no_op():
    eng = engine()
    eng.on_bar(T0, flat(100.0), [cand()])
    eng.on_bar(T0 + BAR, flat(100.0))
    snap = eng.to_dict()
    eng.on_bar(T0 + BAR, flat(100.0), [cand()])  # replay of an already-processed bar
    eng.on_bar(T0, flat(50.0), [cand()])  # and of an older one
    assert eng.to_dict() == snap


def test_no_entries_outside_window_and_flatten_at_session_close():
    eng = engine()
    late = ist(2026, 1, 5, 21, 50)
    eng.on_bar(late, flat(100.0), [cand(stop=5, tp=10)])  # decision 21:55 -> allowed
    eng.on_bar(late + BAR, flat(100.0), [cand(stop=5, tp=10)])  # decision 22:00 -> entry cutoff
    assert SYM in eng.positions
    ts = late + 2 * BAR
    while ts < ist(2026, 1, 5, 23, 0):
        eng.on_bar(ts, flat(100.0))
        ts += BAR
    assert SYM in eng.positions  # still open at 22:55
    eng.on_bar(ist(2026, 1, 5, 23, 0), flat(100.0))  # first bar at/after 23:00 IST: flat at its open
    assert not eng.positions and eng.trades[-1].exit_reason == "session close"
    eng.on_bar(ist(2026, 1, 6, 2, 0), flat(100.0), [cand()])  # 02:00 IST: asleep
    assert not eng.pending and "outside the entry window" in eng.decisions[-1]["reason"]


def test_kill_switch_flattens_and_stops_for_good():
    eng = engine()
    eng.on_bar(T0, flat(100.0))
    eng.cash = 0.0  # put the whole account into one oversized position to force a deep drawdown
    eng.positions[SYM] = Position(SYM, LONG, 1.0, T0, 100.0, stop=10.0, target=500.0, stop_dist=90.0,
                                  flatten_ts=T0 + 10**9, max_hold_ts=T0 + 10**9, signal="breakout", regime="TREND_UP",
                                  prob=0.9, ev_r=0.5, entry_fee=0.0, entry_slip=0.0, risk_usd=90.0)
    eng.on_bar(T0 + BAR, flat(84.0))  # -16% from the peak
    assert eng.risk.killed and SYM in eng.positions  # tripped at the close; exit happens on the next open
    eng.on_bar(T0 + 2 * BAR, flat(84.0), [cand()])
    assert not eng.positions and eng.trades[-1].exit_reason == "kill switch"
    for k in range(3, 400):
        eng.on_bar(T0 + k * BAR, flat(84.0), [cand()])
    assert not eng.positions and not eng.pending and len(eng.trades) == 1
    assert "kill switch" in eng.decisions[-1]["reason"]


def test_kill_switch_not_tripped_just_below_threshold():
    eng = engine()
    eng.on_bar(T0, flat(100.0))
    eng.risk.mark(85.5, T0, eng.cfg.risk)
    assert not eng.risk.killed
    eng.risk.mark(85.0, T0, eng.cfg.risk)
    assert eng.risk.killed


def test_daily_loss_limit_halts_until_next_session():
    eng = engine()
    eng.on_bar(T0, flat(100.0))
    eng.cash = 96.9  # -3.1% on the day
    eng.on_bar(T0 + BAR, flat(100.0), [cand()])
    assert eng.risk.daily_halt and not eng.pending
    assert "daily loss limit" in eng.decisions[-1]["reason"]
    eng.on_bar(ist(2026, 1, 6, 10, 0), flat(100.0), [cand()])  # next IST day: halt cleared
    assert not eng.risk.daily_halt and SYM in eng.pending


def test_cooldown_after_consecutive_losses():
    eng = engine(risk={"loss_streak": 2, "loss_streak_cooldown_bars": 10, "symbol_cooldown_bars": 0})
    ts = T0
    for _ in range(2):  # two stop-outs in a row
        eng.on_bar(ts, flat(100.0), [cand()])
        eng.on_bar(ts + BAR, {SYM: Bar(100, 100, 98.0, 99, 1e4)})
        ts += 2 * BAR
    assert len(eng.trades) == 2 and all(t.pnl < 0 for t in eng.trades)
    eng.on_bar(ts, flat(100.0), [cand()])
    assert not eng.pending and "cooling down" in eng.decisions[-1]["reason"]
    eng.on_bar(ts + 11 * BAR, flat(100.0), [cand()])
    assert SYM in eng.pending


def test_day_ten_freeze_flattens_and_ignores_everything_after():
    cfg = make_config(experiment={"days": 2})
    clock = SessionClock.from_cfg(cfg.schedule)
    eng = Engine(cfg, clock, enforce_clock=True)
    start = ist(2026, 1, 5, 20, 0)
    end = ist(2026, 1, 6, 23, 0)
    ts = start
    while ts < end:
        eng.on_bar(ts, flat(100.0), [cand(stop=15, tp=30)])  # wide levels: positions stay open
        ts += BAR
    assert eng.end_ts == end and not eng.frozen
    assert eng.trades and eng.trades[0].exit_reason == "session close"  # day 1 was flattened at 23:00
    assert SYM in eng.positions  # a position is still open going into the final bar
    eng.on_bar(end, flat(100.0), [cand()])
    assert eng.frozen and eng.frozen_ts == end and not eng.positions and not eng.pending
    assert eng.trades[-1].exit_reason == "experiment end"
    snap = eng.to_dict()
    for k in range(1, 50):
        eng.on_bar(end + k * BAR, flat(150.0), [cand()])
    assert eng.to_dict() == snap  # frozen: later bars change nothing


def test_state_round_trips_through_json():
    import json
    eng = engine()
    eng.on_bar(T0, flat(100.0), [cand()])
    eng.on_bar(T0 + BAR, flat(100.0), [cand(prob=0.1)])
    blob = json.loads(json.dumps(eng.to_dict()))
    again = Engine.from_dict(eng.cfg, eng.clock, blob, enforce_clock=False)
    assert again.to_dict() == eng.to_dict()
    eng.on_bar(T0 + 2 * BAR, {SYM: Bar(100, 103, 99.5, 102, 1e4)})
    again.on_bar(T0 + 2 * BAR, {SYM: Bar(100, 103, 99.5, 102, 1e4)})
    assert again.to_dict() == eng.to_dict() and again.trades[-1].exit_reason == "target"
