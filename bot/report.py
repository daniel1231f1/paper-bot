"""Builds dashboard.json (the only thing the static dashboard reads) and the final report."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .clock import SessionClock
from .config import Config
from .engine import Engine
from .metrics import _clean, breakdown, daily_equity, equity_stats, trade_stats, trades_frame, verdict

REGIME_TEXT = {
    "WARMUP": "indicators are still warming up",
    "DEAD": "volatility is near the bottom of its 7-day range: moves are too small to cover costs",
    "SHOCK": "volatility is extreme: standing aside",
    "TREND_UP": "hourly uptrend",
    "TREND_DOWN": "hourly downtrend",
    "RANGE": "range-bound market",
    "CHOP": "no clean trend or range: sitting out chop",
}


def status_of(eng: Engine, clock: SessionClock, now_ms: int, data_stale: bool, data_error: str | None) -> dict[str, Any]:
    tz_label = "IST" if str(clock.tz) == "Asia/Kolkata" else str(clock.tz)
    nxt = clock.next_open(now_ms)
    open_txt = f"{clock.start.strftime('%H:%M')} {tz_label}"
    if eng.frozen:
        return {"code": "FROZEN", "label": "Experiment complete", "detail": "Day-10 freeze: no further trading.", "next_open_ts": None}
    if eng.risk.killed:
        return {"code": "KILLED", "label": "Halted", "next_open_ts": None,
                "detail": "Kill switch tripped: total drawdown limit reached. The bot is flat and will not trade again."}
    in_window = clock.in_window(now_ms)
    if eng.start_ts is None:
        return {"code": "ARMED", "label": f"Armed, starts {open_txt}", "next_open_ts": nxt,
                "detail": "The 10-day clock starts at the next session open so that day 1 is a full trading day."}
    if not in_window:
        return {"code": "SLEEPING", "label": f"Sleeping until {open_txt}", "next_open_ts": nxt,
                "detail": "Outside the 07:00-23:00 session. Positions are flat; nothing trades until the next open."}
    if data_stale or data_error:
        return {"code": "DATA_STALE", "label": "Data stale, not trading", "next_open_ts": None,
                "detail": data_error or "Newest candle is too old. Open positions keep their stops; no new entries."}
    if eng.risk.daily_halt:
        return {"code": "HALTED_DAY", "label": "Halted for today", "next_open_ts": nxt,
                "detail": "Daily loss limit hit. No new entries until the next session; open positions keep their stops."}
    return {"code": "ACTIVE", "label": "Active", "detail": "Scanning every closed 5-minute bar.", "next_open_ts": None}


def _downsample(curve: list[list[float]], limit: int) -> list[list[float]]:
    if len(curve) <= limit:
        return curve
    idx = np.unique(np.linspace(0, len(curve) - 1, limit).round().astype(int))
    return [curve[i] for i in idx]


def performance(eng: Engine, cfg: Config, clock: SessionClock) -> dict[str, Any]:
    tf = trades_frame(eng.trades)
    cap = cfg.experiment.starting_capital
    perf: dict[str, Any] = {**trade_stats(tf, cfg.market.timeframe_minutes), **equity_stats(eng.curve, clock, cap, 1)}
    perf["max_drawdown"] = _clean(max(eng.risk.max_drawdown, perf.get("max_drawdown") or 0.0))
    perf["benchmark"] = equity_stats(eng.curve, clock, cap, 2)
    if len(tf):
        hours = pd.Series(clock.local_minutes(tf["entry_ts"].to_numpy()) // 60, index=tf.index)
        perf["by_hour"] = breakdown(tf, hours)
        perf["by_symbol"] = breakdown(tf, "symbol")
        perf["by_signal"] = breakdown(tf, "signal")
        perf["by_regime"] = breakdown(tf, "regime")
        perf["by_exit"] = breakdown(tf, "exit_reason")
    else:
        perf.update({"by_hour": [], "by_symbol": [], "by_signal": [], "by_regime": [], "by_exit": []})
    daily = daily_equity(eng.curve, clock, 1)
    prev, rows = cap, []
    for day, value in daily.items():
        rows.append({"date": pd.Timestamp(int(day), unit="D").strftime("%Y-%m-%d"), "equity": _clean(value),
                     "pnl": _clean(value - prev)})
        prev = value
    perf["daily"] = rows
    return perf


def build_view(eng: Engine, cfg: Config, clock: SessionClock, now_ms: int, *, data_stale: bool = False,
               data_error: str | None = None, why: Mapping[str, Any] | None = None,
               model: Mapping[str, Any] | None = None, meta: Mapping[str, Any] | None = None,
               final_report: Mapping[str, Any] | None = None, shadow: Engine | None = None) -> dict[str, Any]:
    cap = cfg.experiment.starting_capital
    meta = dict(meta or {})
    status = status_of(eng, clock, now_ms, data_stale, data_error)
    day = 0 if eng.start_ts is None else min(cfg.experiment.days, clock.day_number(eng.start_ts, min(now_ms, (eng.end_ts or now_ms) - 1)))
    progress = 0.0
    if eng.start_ts is not None and eng.end_ts is not None:
        progress = float(np.clip((now_ms - eng.start_ts) / max(1, eng.end_ts - eng.start_ts), 0, 1))
    if eng.frozen:
        day, progress = cfg.experiment.days, 1.0

    positions = []
    for p in eng.positions.values():
        last = eng.last_close.get(p.symbol, p.entry_price)
        upnl = p.side * (last - p.entry_price) * p.qty - p.entry_fee
        positions.append({**asdict(p), "last": last, "upnl": _clean(upnl), "notional": _clean(p.qty * last),
                          "upnl_pct": _clean(p.side * (last / p.entry_price - 1)),
                          "r_now": _clean(upnl / p.risk_usd) if p.risk_usd else None,
                          "open_risk": _clean(max(0.0, p.side * (last - p.stop)) * p.qty)})
    unrealized = sum(p["upnl"] or 0.0 for p in positions)
    realized = sum(t.pnl for t in eng.trades)
    open_risk = sum(p["open_risk"] or 0.0 for p in positions)
    dd = eng.risk.drawdown(eng.equity)
    daily_loss = eng.risk.daily_loss(eng.equity)
    perf = performance(eng, cfg, clock)

    return {
        "generated_ts": now_ms,
        "heartbeat_ts": now_ms,
        "status": status,
        "experiment": {
            "name": cfg.experiment.name, "days": cfg.experiment.days, "day": day, "progress": progress,
            "start_ts": eng.start_ts, "end_ts": eng.end_ts, "frozen": eng.frozen, "frozen_ts": eng.frozen_ts,
            "config_hash": cfg.hash, "config_file_hash": meta.get("config_file_hash"),
            "config_dirty": bool(meta.get("config_file_hash")) and meta.get("config_file_hash") != cfg.hash,
            "code_sha": meta.get("code_sha"), "code_sha_start": meta.get("code_sha_start"),
            "timezone": cfg.schedule.timezone, "window": [cfg.schedule.active_start, cfg.schedule.active_end],
            "paper_only": True,
        },
        "account": {
            "capital": cap, "equity": _clean(eng.equity), "cash": _clean(eng.cash),
            "total_return": _clean(eng.equity / cap - 1), "realized": _clean(realized), "unrealized": _clean(unrealized),
            "today_pnl": _clean(eng.equity - eng.risk.day_start_equity) if eng.risk.day_start_equity else 0.0,
            "today_pnl_pct": _clean(eng.equity / eng.risk.day_start_equity - 1) if eng.risk.day_start_equity else 0.0,
            "bench_equity": _clean(eng.bench), "bench_return": _clean(eng.bench / cap - 1),
            "last_prices": {s: _clean(v) for s, v in eng.last_close.items()},
        },
        "curve": _downsample(eng.curve, cfg.ops.equity_curve_points),
        "positions": positions,
        "pending": [asdict(o) for o in eng.pending.values()],
        "trades": [asdict(t) for t in eng.trades],
        "performance": perf,
        "why": dict(why or {}),
        "decisions": eng.decisions[-60:],
        "skip_counts": dict(sorted(eng.skip_counts.items(), key=lambda kv: -kv[1])),
        "risk": {
            "risk_per_trade": cfg.risk.risk_per_trade, "max_positions": cfg.risk.max_positions,
            "positions_used": len(eng.positions) + len(eng.pending),
            "open_risk": _clean(open_risk), "open_risk_pct": _clean(open_risk / eng.equity) if eng.equity else None,
            "risk_opened_today": _clean(eng.risk.risk_opened_today),
            "risk_opened_today_pct": _clean(eng.risk.risk_opened_today / eng.risk.day_start_equity) if eng.risk.day_start_equity else 0.0,
            "daily_loss": _clean(daily_loss), "daily_loss_limit": cfg.risk.daily_loss_limit,
            "daily_remaining": _clean(max(0.0, cfg.risk.daily_loss_limit - daily_loss)), "daily_halt": eng.risk.daily_halt,
            "drawdown": _clean(dd), "max_drawdown": _clean(eng.risk.max_drawdown), "kill_switch": cfg.risk.kill_switch_drawdown,
            "kill_distance": _clean(max(0.0, cfg.risk.kill_switch_drawdown - dd)),
            "kill_equity": _clean(eng.risk.peak_equity * (1 - cfg.risk.kill_switch_drawdown)),
            "peak_equity": _clean(eng.risk.peak_equity), "killed": eng.risk.killed, "killed_ts": eng.risk.killed_ts,
            "loss_streak": eng.risk.loss_streak, "cooldown_until": eng.risk.cooldown_until,
        },
        "verdict": dict(final_report["verdict"]) if final_report else
        verdict(trades_frame(eng.trades), eng.risk.max_drawdown, cfg.verdict, final=False),
        "final_report": dict(final_report) if final_report else None,
        "model": dict(model or {}),
        "data": {"exchange": cfg.market.exchange, "last_bar_ts": eng.last_bar_ts or None, "stale": data_stale,
                 "error": data_error, "timeframe_minutes": cfg.market.timeframe_minutes},
        "config": {"universe": list(cfg.market.universe), "costs": asdict(cfg.costs), "risk": asdict(cfg.risk),
                   "model": {"kind": cfg.model.kind, "min_prob": cfg.model.min_prob, "min_ev_r": cfg.model.min_ev_r,
                             "train_days": cfg.model.train_days},
                   "strategy": {"signals": list(cfg.strategy.enabled_signals), "allow_short": cfg.strategy.allow_short,
                                "stop_atr": cfg.strategy.stop_atr, "target_atr": cfg.strategy.target_atr,
                                "max_hold_bars": cfg.strategy.max_hold_bars},
                   "verdict": asdict(cfg.verdict)},
        "backtest": meta.get("backtest"),
        "shadow": _shadow_view(shadow, cfg) if shadow is not None else None,
    }


def _shadow_view(sh: Engine, cfg: Config) -> dict[str, Any]:
    """The same rules with the model gate removed. Reference only: the verdict never uses it."""
    cap = cfg.experiment.starting_capital
    stats = trade_stats(trades_frame(sh.trades), cfg.market.timeframe_minutes)
    return {"equity": _clean(sh.equity), "total_return": _clean(sh.equity / cap - 1), "trades": stats["trades"],
            "win_rate": stats["win_rate"], "profit_factor": stats["profit_factor"], "expectancy": stats["expectancy"],
            "max_drawdown": _clean(sh.risk.max_drawdown), "killed": sh.risk.killed, "open_positions": len(sh.positions),
            "curve": [[p[0], p[1]] for p in _downsample(sh.curve, cfg.ops.equity_curve_points)]}


def final_report(eng: Engine, cfg: Config, clock: SessionClock, now_ms: int) -> dict[str, Any]:
    """Frozen once, at the end of experiment day N."""
    tf = trades_frame(eng.trades)
    perf = performance(eng, cfg, clock)
    return {"generated_ts": now_ms, "verdict": verdict(tf, eng.risk.max_drawdown, cfg.verdict, final=True),
            "performance": {k: v for k, v in perf.items() if not k.startswith("by_") and k != "daily"},
            "equity": _clean(eng.equity), "benchmark_equity": _clean(eng.bench), "config_hash": cfg.hash,
            "killed": eng.risk.killed}
