"""Regime classifier + signal ensemble.

The regime decides which signal family may propose a trade:
  TREND_UP / TREND_DOWN -> trend_pullback, breakout     RANGE -> mean_reversion
  CHOP, DEAD (too quiet to cover costs), SHOCK (extreme volatility) -> stand aside
Signals only *propose* candidates; the ML gate and the risk manager decide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostCfg, StrategyCfg

REGIMES = ("WARMUP", "DEAD", "SHOCK", "TREND_UP", "TREND_DOWN", "RANGE", "CHOP")
SIGNALS = ("trend_pullback", "breakout", "mean_reversion")


def classify_regime(f: pd.DataFrame, cfg: StrategyCfg) -> np.ndarray:
    """Per-bar regime label. Volatility extremes take priority over trend state."""
    c = f["close"]
    trending = f["adx_h"] >= cfg.adx_trend_h
    up = trending & (f["ema_fast_h"] > f["ema_slow_h"]) & (c > f["ema_slow_h"])
    down = trending & (f["ema_fast_h"] < f["ema_slow_h"]) & (c < f["ema_slow_h"])
    ranging = f["adx_h"] <= cfg.adx_range_h
    warm = f[["atr_rank", "adx_h", "ema_slow_h", "atr_h", "ema_slow_dev"]].isna().any(axis=1)
    regime = np.select(
        [warm, f["atr_rank"] < cfg.atr_pct_dead, f["atr_rank"] > cfg.atr_pct_shock, up, down, ranging],
        ["WARMUP", "DEAD", "SHOCK", "TREND_UP", "TREND_DOWN", "RANGE"], default="CHOP")
    return regime.astype(object)


def generate_candidates(f: pd.DataFrame, cfg: StrategyCfg, costs: CostCfg) -> pd.DataFrame:
    """Adds regime / signal / side / stop_dist / tp_dist columns (and model one-hots) to a copy of `f`.

    A row with signal == "" has no setup. `skip_reason` explains a setup that was vetoed
    before the model (e.g. stop too tight relative to costs).
    """
    out = f.copy()
    regime = classify_regime(f, cfg)
    out["regime"] = regime
    c, o = f["close"], f["open"]
    bull_bar = c > o
    bear_bar = c < o

    # -- trend pullback: dip (RSI reset) inside an hourly uptrend, then reclaim of the fast EMA
    reclaim_up = (c > f["ema_fast"]) & (f["prev_close"] <= f["prev_ema_fast"])
    pull_long = (regime == "TREND_UP") & (f["rsi_min6"] <= cfg.pullback_rsi_max) & reclaim_up & bull_bar
    lose_down = (c < f["ema_fast"]) & (f["prev_close"] >= f["prev_ema_fast"])
    rsi_max6 = 100 - cfg.pullback_rsi_max
    pull_short = (regime == "TREND_DOWN") & (f["rsi"].rolling(6).max() >= rsi_max6) & lose_down & bear_bar

    # -- breakout: fresh close through the Donchian channel on expanding volume
    vol_ok = f["vol_ratio"] >= cfg.breakout_vol_mult
    brk_long = (regime == "TREND_UP") & (c > f["donch_high"]) & (f["prev_close"] <= f["donch_high"]) & vol_ok
    brk_short = (regime == "TREND_DOWN") & (c < f["donch_low"]) & (f["prev_close"] >= f["donch_low"]) & vol_ok

    # -- mean reversion: stretched below band + oversold inside a range, first up-close
    mr_long = (regime == "RANGE") & (f["bbz_min3"] <= cfg.mr_bb_z) & (f["rsi_min3"] <= cfg.mr_rsi_max) \
        & bull_bar & (c > f["prev_close"]) & (f["vwap_dev"] < 0)
    mr_short = (regime == "RANGE") & (f["bb_z"].rolling(3).max() >= -cfg.mr_bb_z) \
        & (f["rsi"].rolling(3).max() >= 100 - cfg.mr_rsi_max) & bear_bar & (c < f["prev_close"]) & (f["vwap_dev"] > 0)

    enabled = set(cfg.enabled_signals)
    longs = {"breakout": brk_long, "trend_pullback": pull_long, "mean_reversion": mr_long}
    shorts = {"breakout": brk_short, "trend_pullback": pull_short, "mean_reversion": mr_short}
    signal = np.full(len(f), "", dtype=object)
    side = np.zeros(len(f), dtype=np.int8)
    for name in ("mean_reversion", "trend_pullback", "breakout"):  # later entries win ties: breakout highest
        if name not in enabled:
            continue
        m = longs[name].to_numpy(dtype=bool, na_value=False)
        signal[m], side[m] = name, 1
        if cfg.allow_short:
            ms = shorts[name].to_numpy(dtype=bool, na_value=False) & ~m
            signal[ms], side[ms] = name, -1

    is_mr = signal == "mean_reversion"
    unit = f["atr_h"].to_numpy()
    stop = np.where(is_mr, cfg.mr_stop_atr, cfg.stop_atr) * unit
    tp = np.where(is_mr, cfg.mr_target_atr, cfg.target_atr) * unit
    stop_pct = stop / c.to_numpy()

    skip = np.full(len(f), "", dtype=object)
    has = signal != ""
    too_tight = has & ~(stop_pct >= cfg.min_stop_cost_mult * costs.round_trip)  # also catches NaN
    skip[too_tight] = "stop distance too small relative to trading costs"
    synth = has & f["synthetic"].to_numpy(dtype=bool)
    skip[synth] = "no trades printed on this bar"

    out["signal"], out["side"], out["skip_reason"] = signal, side, skip
    out["stop_dist"], out["tp_dist"], out["stop_pct"] = stop, tp, stop_pct
    for name in SIGNALS:
        out[f"sig_{name}"] = (signal == name).astype(float)
    return out
