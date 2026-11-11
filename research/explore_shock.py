"""DEV-only probe of one a-priori hypothesis: liquidation-style flushes (fast drop on heavy volume)
overshoot and bounce. Long after a flush / short after a blow-off, both directions reported."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.backtest import align  # noqa: E402
from bot.broker import simulate_trade  # noqa: E402
from bot.clock import SessionClock  # noqa: E402
from bot.config import Config  # noqa: E402
from bot.features import compute_features  # noqa: E402
from research.run import DEV, load_cached, ms  # noqa: E402

cfg = Config()
clock = SessionClock.from_cfg(cfg.schedule)
hold = ms(DEV[1])
bars = align({s: df[df["ts"] < hold].reset_index(drop=True) for s, df in load_cached(cfg).items()})
rows = []
for sym, df in bars.items():
    f = compute_features(df, cfg.strategy, clock, cfg.bar_ms)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ts = df["ts"].to_numpy()
    unit = f["atr_h"].to_numpy()
    move = (c - np.roll(c, 12)) / unit  # 1h move in hourly-ATR units
    vol6 = f["vol_ratio_6"].to_numpy()
    minutes = clock.local_minutes(ts + cfg.bar_ms)
    in_win = (minutes >= clock.start_minute) & (minutes < clock.end_minute - 60)
    for k_move in (2.0, 3.0, 4.0):
        for vmin in (1.5, 3.0):
            for side, cond in ((1, (move <= -k_move) & (c > o) & (c > np.roll(c, 1))),
                               (-1, (move >= k_move) & (c < o) & (c < np.roll(c, 1)))):
                idx = np.flatnonzero(cond & (vol6 >= vmin) & in_win & np.isfinite(unit))
                last = -10**9
                for i in idx:
                    if i - last < 12 or i + 1 >= len(o):  # one event per hour per symbol
                        continue
                    last = i
                    day_close = ts[i] + cfg.bar_ms + (clock.end_minute - minutes[i]) * 60_000
                    flat = min(int(np.searchsorted(ts, day_close)), i + 1 + 48)
                    for stop_m, tp_m in ((1.0, 1.5), (1.5, 1.5), (1.5, 3.0)):
                        res = simulate_trade(o, h, l, i + 1, side, stop_m * unit[i], tp_m * unit[i], flat, cfg.costs)
                        if res:
                            rows.append((sym, k_move, vmin, side, f"{stop_m}/{tp_m}", pd.Timestamp(ts[i], unit="ms").strftime("%Y-%m"),
                                         res[0], res[1], c[i] * cfg.costs.round_trip / (stop_m * unit[i])))
d = pd.DataFrame(rows, columns=["sym", "move", "vol", "side", "geom", "month", "label", "net_r", "cost_r"])
d["gross_r"] = d["net_r"] + d["cost_r"]
pd.set_option("display.width", 200)
print(d.groupby(["move", "vol", "side", "geom"]).agg(n=("label", "size"), hit=("label", "mean"), gross_r=("gross_r", "mean"),
                                                      net_r=("net_r", "mean")).round(3).to_string())
