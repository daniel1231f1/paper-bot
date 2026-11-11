"""Performance statistics, bootstrap confidence ranges and the pre-registered verdict."""
from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .clock import SessionClock
from .config import VerdictCfg

ANNUAL_DAYS = 365  # crypto trades every day


def _clean(x: float | None) -> float | None:
    """JSON-safe float (NaN/inf -> None)."""
    if x is None or isinstance(x, str):
        return x
    x = float(x)
    return x if math.isfinite(x) else None


def trades_frame(trades: Iterable[Any]) -> pd.DataFrame:
    rows = [asdict(t) if is_dataclass(t) else dict(t) for t in trades]
    cols = ["symbol", "side", "signal", "regime", "entry_ts", "exit_ts", "entry_price", "exit_price", "qty", "fees",
            "slippage", "pnl", "r", "prob", "ev_r", "exit_reason", "bars_held", "partial"]
    return pd.DataFrame(rows, columns=cols)


def profit_factor(pnl: np.ndarray) -> float:
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def trade_stats(tf: pd.DataFrame, bar_minutes: int = 5) -> dict[str, Any]:
    n = len(tf)
    if n == 0:
        return {"trades": 0, "win_rate": None, "profit_factor": None, "expectancy": None, "expectancy_r": None,
                "avg_win": None, "avg_loss": None, "net_pnl": 0.0, "fees": 0.0, "slippage": 0.0,
                "avg_hold_min": None, "best": None, "worst": None}
    pnl = tf["pnl"].to_numpy(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return {
        "trades": n, "win_rate": _clean(len(wins) / n), "profit_factor": _clean(profit_factor(pnl)),
        "expectancy": _clean(pnl.mean()), "expectancy_r": _clean(tf["r"].mean()),
        "avg_win": _clean(wins.mean()) if len(wins) else None,
        "avg_loss": _clean(losses.mean()) if len(losses) else None,
        "net_pnl": _clean(pnl.sum()), "fees": _clean(tf["fees"].sum()), "slippage": _clean(tf["slippage"].sum()),
        "avg_hold_min": _clean(tf["bars_held"].mean() * bar_minutes), "best": _clean(pnl.max()),
        "worst": _clean(pnl.min()),
    }


def breakdown(tf: pd.DataFrame, key: pd.Series | str) -> list[dict[str, Any]]:
    if len(tf) == 0:
        return []
    out = []
    for k, g in tf.groupby(key, sort=True):
        pnl = g["pnl"].to_numpy(float)
        out.append({"key": k if isinstance(k, str) else int(k), "trades": len(g), "pnl": _clean(pnl.sum()),
                    "win_rate": _clean((pnl > 0).mean()), "expectancy_r": _clean(g["r"].mean()),
                    "profit_factor": _clean(profit_factor(pnl))})
    return out


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(1 - equity / peak))


def daily_equity(curve: Sequence[Sequence[float]], clock: SessionClock, col: int = 1) -> pd.Series:
    """Last value of each local calendar day."""
    arr = np.asarray(curve, dtype=float)
    if arr.size == 0:
        return pd.Series(dtype=float)
    days = clock.local_day_ordinal(arr[:, 0].astype("int64") - 1)  # a bar closing at midnight belongs to the old day
    return pd.Series(arr[:, col]).groupby(days).last()


def equity_stats(curve: Sequence[Sequence[float]], clock: SessionClock, capital: float, col: int = 1) -> dict[str, Any]:
    arr = np.asarray(curve, dtype=float)
    if arr.size == 0:
        return {"total_return": 0.0, "sharpe": None, "sortino": None, "max_drawdown": 0.0, "days": 0}
    daily = daily_equity(curve, clock, col)
    rets = daily.pct_change().dropna().to_numpy()
    if len(daily) >= 1:
        rets = np.concatenate([[daily.iloc[0] / capital - 1], rets])
    sharpe = sortino = None
    if len(rets) >= 3 and rets.std(ddof=1) > 0:
        sharpe = rets.mean() / rets.std(ddof=1) * math.sqrt(ANNUAL_DAYS)
        downside = math.sqrt(np.mean(np.minimum(rets, 0) ** 2))
        sortino = rets.mean() / downside * math.sqrt(ANNUAL_DAYS) if downside > 0 else None
    return {"total_return": _clean(arr[-1, col] / capital - 1), "sharpe": _clean(sharpe), "sortino": _clean(sortino),
            "max_drawdown": _clean(max_drawdown(np.concatenate([[capital], arr[:, col]]))), "days": int(len(daily))}


def bootstrap_ci(values: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int, conf: float,
                 seed: int = 11) -> tuple[float | None, float | None]:
    """Percentile bootstrap over trades (i.i.d. resampling; ignores serial dependence). O(n_boot * n)."""
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = np.array([stat(values[i]) for i in idx], dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return None, None
    lo, hi = np.quantile(samples, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(lo), float(hi)


def verdict(tf: pd.DataFrame, max_dd: float, cfg: VerdictCfg, final: bool, seed: int = 11) -> dict[str, Any]:
    """PASS / MARGINAL / FAIL against the criteria fixed before the experiment started."""
    pnl = tf["pnl"].to_numpy(float) if len(tf) else np.array([])
    n = len(pnl)
    expectancy = float(pnl.mean()) if n else 0.0
    pf = profit_factor(pnl) if n else float("nan")
    ex_best = float(pnl.sum() - pnl.max()) if n else 0.0
    checks = [
        {"id": "expectancy", "label": "Positive expectancy after costs", "target": "> $0 per trade",
         "value": _clean(expectancy), "pass": n > 0 and expectancy > 0},
        {"id": "profit_factor", "label": "Profit factor", "target": f"> {cfg.min_profit_factor:g}",
         "value": _clean(pf), "pass": n > 0 and (pf > cfg.min_profit_factor)},
        {"id": "max_drawdown", "label": "Max drawdown", "target": f"< {cfg.max_drawdown:.0%}",
         "value": _clean(max_dd), "pass": max_dd < cfg.max_drawdown},
        {"id": "trades", "label": "Sample size", "target": f">= {cfg.min_trades} trades", "value": n,
         "pass": n >= cfg.min_trades},
        {"id": "lucky", "label": "Not driven by one lucky trade", "target": "net P&L > $0 without the best trade",
         "value": _clean(ex_best), "pass": n > 1 and ex_best > 0},
    ]
    by_id = {c["id"]: c["pass"] for c in checks}
    if all(by_id.values()):
        result = "PASS"
    elif not by_id["expectancy"] or not by_id["max_drawdown"]:
        result = "FAIL"
    else:
        result = "MARGINAL"
    exp_ci = bootstrap_ci(pnl, lambda x: float(x.mean()), cfg.bootstrap_samples, cfg.confidence, seed)
    pf_ci = bootstrap_ci(pnl, profit_factor, cfg.bootstrap_samples, cfg.confidence, seed + 1)
    p_pos = None
    if n >= 2:
        rng = np.random.default_rng(seed + 2)
        means = pnl[rng.integers(0, n, size=(cfg.bootstrap_samples, n))].mean(axis=1)
        p_pos = float((means > 0).mean())
    return {"result": result, "final": final, "checks": checks, "trades": n,
            "expectancy_ci": [_clean(exp_ci[0]), _clean(exp_ci[1])], "profit_factor_ci": [_clean(pf_ci[0]), _clean(pf_ci[1])],
            "prob_expectancy_positive": _clean(p_pos), "confidence": cfg.confidence,
            "explanation": _explain(result, n, expectancy, pf, max_dd, exp_ci, p_pos, cfg, final)}


def _usd(x: float) -> str:
    return f"{'-' if x < 0 else '+'}${abs(x):.3f}"


def _explain(result: str, n: int, expectancy: float, pf: float, max_dd: float,
             exp_ci: tuple[float | None, float | None], p_pos: float | None, cfg: VerdictCfg, final: bool) -> str:
    if n == 0:
        base = ("The bot closed no trades. That is a legitimate outcome of a strategy that sits out when its model "
                "sees no edge after costs, but it means there is no evidence of an edge either.")
        return base if final else "No closed trades yet."
    parts = [f"{n} closed trades averaged {_usd(expectancy)} each after fees, spread and slippage"
             + (f" (profit factor {pf:.2f})." if math.isfinite(pf) else ".")]
    if exp_ci[0] is not None:
        lo, hi = exp_ci
        parts.append(f"Resampling those trades puts the plausible per-trade average between {_usd(lo)} and {_usd(hi)} "
                     f"({cfg.confidence:.0%} range)"
                     + (f"; it was positive in {p_pos:.0%} of resamples." if p_pos is not None else "."))
        if lo <= 0 <= hi:
            parts.append("That range includes zero, so the result is statistically indistinguishable from no edge.")
    parts.append(f"Worst peak-to-trough drawdown was {max_dd:.1%}.")
    if result == "PASS":
        parts.append("Every pre-registered criterion was met.")
    elif result == "MARGINAL":
        parts.append("The account did not lose money per trade and stayed inside the drawdown limit, but at least one "
                     "pre-registered criterion was missed, so this is not evidence of a dependable edge.")
    else:
        parts.append("The strategy lost money per trade after costs or breached the drawdown limit: no edge shown.")
    parts.append("Ten days is a small sample: one market regime, few independent bets. Treat this as a sanity check, "
                 "not proof, and never as a forecast of live results.")
    return " ".join(parts)
