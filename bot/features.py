"""Causal feature pipeline.

Every value in row i is computed from bars <= i only (rolling / exponentially-weighted
windows that look backwards). tests/test_no_lookahead.py proves this by truncation:
features computed on df[:i+1] must equal row i of features computed on the full frame.

Hourly (higher-timeframe) values are only exposed once the hour has fully closed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clock import SessionClock
from .config import StrategyCfg

HOUR_MS = 3_600_000

# Columns fed to the ML model. Scale-free by construction (ratios, z-scores, ATR units).
MODEL_FEATURES: tuple[str, ...] = (
    "ret_3", "ret_12", "ret_48", "ret_288", "rsi", "bb_z", "vwap_dev", "adx", "atr_rank", "atr_rel",
    "ema_fast_dev", "ema_mid_dev", "ema_slow_dev", "donch_pos", "vol_ratio", "vol_ratio_6", "body_atr",
    "trend_h", "adx_h", "slope_h", "rsi_h", "tod_sin", "tod_cos", "mkt_ret_48", "mkt_trend_h", "mkt_atr_rank",
    "stop_pct", "sig_trend_pullback", "sig_breakout", "sig_mean_reversion",
)


def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up, down = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    up, down = high.diff(), -low.diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = wilder(true_range(high, low, close), n).replace(0, np.nan)
    pdi, mdi = 100 * wilder(plus, n) / tr, 100 * wilder(minus, n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx.fillna(0.0), n)


def hourly_frame(df: pd.DataFrame, cfg: StrategyCfg) -> pd.DataFrame:
    """Indicators on hourly bars, indexed by hour id (ts // 1h)."""
    hour = df["ts"] // HOUR_MS
    g = df.groupby(hour, sort=True)
    h = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last()})
    out = pd.DataFrame(index=h.index)
    out["ema_fast_h"] = ema(h["close"], cfg.ema_fast_h)
    out["ema_slow_h"] = ema(h["close"], cfg.ema_slow_h)
    out["atr_h"] = atr(h["high"], h["low"], h["close"], cfg.atr_period_h)
    out["adx_h"] = adx(h["high"], h["low"], h["close"], cfg.adx_period_h)
    out["rsi_h"] = rsi(h["close"], cfg.rsi_period)
    out["slope_h"] = (out["ema_slow_h"] - out["ema_slow_h"].shift(3)) / out["atr_h"]
    return out


def compute_features(df: pd.DataFrame, cfg: StrategyCfg, clock: SessionClock, bar_ms: int) -> pd.DataFrame:
    """Feature frame aligned 1:1 with `df` (normalised candles for one symbol). O(n)."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    f = pd.DataFrame({"ts": df["ts"].to_numpy()}, index=df.index)
    f["close"] = c
    f["atr"] = atr(h, l, c, cfg.atr_period)
    f["atr_rel"] = f["atr"] / c
    # percentile of current volatility within its own trailing distribution (volatility regime)
    f["atr_rank"] = f["atr_rel"].rolling(cfg.atr_pct_bars, min_periods=cfg.atr_pct_bars // 2).rank(pct=True)

    # ---- higher timeframe, exposed only after the hour closes --------------------
    hf = hourly_frame(df, cfg)
    hour = (df["ts"] // HOUR_MS).to_numpy()
    last_in_hour = ((df["ts"].to_numpy() + bar_ms) % HOUR_MS) == 0
    avail = np.where(last_in_hour, hour, hour - 1)
    hmap = hf.reindex(avail)
    for col in hf.columns:
        f[col] = hmap[col].to_numpy()
    unit = f["atr_h"]  # risk unit: one hourly ATR, in price

    e_fast, e_mid, e_slow = ema(c, cfg.ema_fast), ema(c, cfg.ema_mid), ema(c, cfg.ema_slow)
    f["ema_fast"], f["ema_mid"] = e_fast, e_mid
    f["ema_fast_dev"] = (c - e_fast) / unit
    f["ema_mid_dev"] = (c - e_mid) / unit
    f["ema_slow_dev"] = (c - e_slow) / unit
    f["rsi"] = rsi(c, cfg.rsi_period)
    mid = c.rolling(cfg.bb_period).mean()
    sd = c.rolling(cfg.bb_period).std(ddof=0)
    f["bb_z"] = (c - mid) / sd.replace(0, np.nan)
    pv = (c * v).rolling(cfg.vwap_bars, min_periods=cfg.vwap_bars // 4).sum()
    vv = v.rolling(cfg.vwap_bars, min_periods=cfg.vwap_bars // 4).sum()
    f["vwap_dev"] = (c - pv / vv.replace(0, np.nan)) / unit
    f["adx"] = adx(h, l, c, cfg.adx_period)
    f["donch_high"] = h.shift(1).rolling(cfg.donchian_bars).max()  # excludes the current bar
    f["donch_low"] = l.shift(1).rolling(cfg.donchian_bars).min()
    f["donch_pos"] = (c - f["donch_low"]) / (f["donch_high"] - f["donch_low"]).replace(0, np.nan)
    vavg = v.rolling(cfg.vol_avg_bars, min_periods=cfg.vol_avg_bars // 4).mean().replace(0, np.nan)
    f["vol_ratio"] = v / vavg
    f["vol_ratio_6"] = v.rolling(6).mean() / vavg
    f["body_atr"] = (c - o) / f["atr"].replace(0, np.nan)
    logc = np.log(c)
    for k in (3, 12, 48, 288):
        f[f"ret_{k}"] = logc - logc.shift(k)
    f["trend_h"] = (f["ema_fast_h"] - f["ema_slow_h"]) / unit
    f["prev_close"], f["prev_high"], f["open"] = c.shift(1), h.shift(1), o
    f["prev_ema_fast"] = e_fast.shift(1)
    f["rsi_min6"] = f["rsi"].rolling(6).min()
    f["bbz_min3"] = f["bb_z"].rolling(3).min()
    f["rsi_min3"] = f["rsi"].rolling(3).min()

    # decision time = bar close; time-of-day in the session timezone
    minutes = clock.local_minutes(df["ts"].to_numpy() + bar_ms)
    f["tod_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    f["tod_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    f["synthetic"] = df["synthetic"].to_numpy() if "synthetic" in df else False
    return f


def add_market_context(feats: dict[str, pd.DataFrame], leader: str) -> None:
    """Join the market leader's state (default BTC) onto every symbol by timestamp. In place."""
    lead = feats[leader].set_index("ts")[["ret_48", "trend_h", "atr_rank"]]
    lead.columns = ["mkt_ret_48", "mkt_trend_h", "mkt_atr_rank"]
    for f in feats.values():
        ctx = lead.reindex(f["ts"].to_numpy())
        for col in lead.columns:
            f[col] = ctx[col].to_numpy()
