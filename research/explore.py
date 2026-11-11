"""DEV-only exploration: where (if anywhere) do raw setups have gross edge larger than costs?
Never touches data on/after the holdout boundary."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.backtest import prepare  # noqa: E402
from bot.clock import SessionClock  # noqa: E402
from bot.config import Config  # noqa: E402
from research.run import DEV, apply_overrides, load_cached, ms  # noqa: E402

if __name__ == "__main__":
    cfg = apply_overrides(Config(), sys.argv[1:])
    clock = SessionClock.from_cfg(cfg.schedule)
    raw = load_cached(cfg)
    hold = ms(DEV[1])
    raw = {s: df[df["ts"] < hold].reset_index(drop=True) for s, df in raw.items()}  # holdout physically removed
    cfg = replace(cfg, market=replace(cfg.market, universe=tuple(raw)))
    bars, cands, ds = prepare(cfg, clock, raw)
    ds = ds[ds["label"].notna()].copy()
    ds["month"] = pd.to_datetime(ds["decision_ts"], unit="ms").dt.strftime("%Y-%m")
    ds["cost_r"] = ds["close"] * cfg.costs.round_trip / ds["stop_dist"]
    ds["gross_r"] = ds["net_r"] + ds["cost_r"]
    ds["hour"] = clock.local_minutes(ds["decision_ts"].to_numpy()) // 60
    pd.set_option("display.width", 220)
    agg = dict(n=("label", "size"), hit=("label", "mean"), gross_r=("gross_r", "mean"), net_r=("net_r", "mean"),
               cost_r=("cost_r", "mean"))
    print(f"overrides={sys.argv[1:]}  DEV-only candidates={len(ds)}")
    print(ds.groupby(["signal", "side"]).agg(**agg).round(3))
    print(ds.groupby(["signal", "side", "month"]).agg(n=("label", "size"), net_r=("net_r", "mean")).round(2).unstack("month").to_string())
    print(ds.groupby(["hour"]).agg(**agg).round(3).T.to_string())
