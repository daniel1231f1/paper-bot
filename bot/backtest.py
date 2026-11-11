"""Walk-forward backtest on the same Engine the live bot uses."""
from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .clock import SessionClock
from .config import Config
from .engine import Bar, Candidate, Engine
from .features import add_market_context, compute_features
from .metrics import breakdown, equity_stats, trade_stats, trades_frame
from .model import build_dataset
from .signals import generate_candidates

log = logging.getLogger("bot.backtest")


def align(raw: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Trim every symbol to the common [start, end] range so all frames share one grid."""
    start = max(int(df["ts"].iloc[0]) for df in raw.values())
    end = min(int(df["ts"].iloc[-1]) for df in raw.values())
    return {s: df[(df["ts"] >= start) & (df["ts"] <= end)].reset_index(drop=True) for s, df in raw.items()}


def build_candidates(cfg: Config, clock: SessionClock, bars: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """features -> market context -> regime + signal candidates, per symbol."""
    feats = {s: compute_features(df, cfg.strategy, clock, cfg.bar_ms) for s, df in bars.items()}
    leader = cfg.market.universe[0] if cfg.market.universe[0] in feats else next(iter(feats))
    add_market_context(feats, leader)
    return {s: generate_candidates(f, cfg.strategy, cfg.costs) for s, f in feats.items()}


def prepare(cfg: Config, clock: SessionClock, raw: Mapping[str, pd.DataFrame]) -> tuple[dict, dict, pd.DataFrame]:
    bars = align(raw)
    cands = build_candidates(cfg, clock, bars)
    return bars, cands, build_dataset(cands, bars, cfg, clock, with_labels=True)


def candidates_by_ts(scored: pd.DataFrame, use_model: bool = True) -> dict[int, list[Candidate]]:
    """Group scored dataset rows into engine candidates keyed by signal-bar ts."""
    out: dict[int, list[Candidate]] = {}
    for row in scored.itertuples(index=False):
        if use_model:
            prob = None if np.isnan(row.prob) else float(row.prob)
            ev = None if np.isnan(row.ev_r) else float(row.ev_r)
        else:
            prob, ev = 1.0, 9.0  # rules-only baseline: every setup passes the gate
        out.setdefault(int(row.ts), []).append(Candidate(row.symbol, int(row.side), row.signal, row.regime,
                                                         float(row.stop_dist), float(row.tp_dist), prob, ev))
    return out


def run_engine(cfg: Config, clock: SessionClock, bars: Mapping[str, pd.DataFrame], cands: Mapping[int, list[Candidate]],
               start_ts: int, end_ts: int) -> Engine:
    """Replay bars in [start_ts, end_ts) through a fresh engine. O(bars * symbols)."""
    eng = Engine(cfg, clock, enforce_clock=False)
    syms = list(bars)
    ts_all = bars[syms[0]]["ts"].to_numpy()
    i0, i1 = int(np.searchsorted(ts_all, start_ts)), int(np.searchsorted(ts_all, end_ts))
    cols = {s: [bars[s][c].to_numpy() for c in ("open", "high", "low", "close", "volume", "synthetic")] for s in syms}
    for i in range(i0, i1):
        ts = int(ts_all[i])
        frame = {s: Bar(float(c[0][i]), float(c[1][i]), float(c[2][i]), float(c[3][i]), float(c[4][i]), bool(c[5][i]))
                 for s, c in cols.items()}
        eng.on_bar(ts, frame, cands.get(ts, ()))
    return eng


def summarize(eng: Engine, cfg: Config, clock: SessionClock) -> dict[str, Any]:
    tf = trades_frame(eng.trades)
    cap = cfg.experiment.starting_capital
    out: dict[str, Any] = {"strategy": {**equity_stats(eng.curve, clock, cap, 1), **trade_stats(tf, cfg.market.timeframe_minutes)},
                           "buy_hold": equity_stats(eng.curve, clock, cap, 2)}
    if len(tf):
        hours = pd.Series(clock.local_minutes(tf["entry_ts"].to_numpy()) // 60, index=tf.index)
        out["by_signal"] = breakdown(tf, "signal")
        out["by_regime"] = breakdown(tf, "regime")
        out["by_symbol"] = breakdown(tf, "symbol")
        out["by_hour"] = breakdown(tf, hours)
        out["by_exit"] = breakdown(tf, "exit_reason")
    out["skips"] = dict(sorted(eng.skip_counts.items(), key=lambda kv: -kv[1]))
    return out
