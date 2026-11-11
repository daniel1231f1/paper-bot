"""Research driver: walk-forward backtest on cached candles. Never fetches; run research/download.py first.

Protocol (fixed before looking at results):
  DEV      2025-04-01 .. 2026-07-15   iterate here (rounds 1-4 used 2025-12-20.. before history was extended)
  HOLDOUT  2026-07-15 .. data end     look once, at the end, report whatever it says
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.backtest import candidates_by_ts, prepare, run_engine, summarize  # noqa: E402
from bot.clock import SessionClock  # noqa: E402
from bot.config import Config  # noqa: E402
from bot.data import MarketData, normalize  # noqa: E402
from bot.model import walk_forward  # noqa: E402

DEV = (pd.Timestamp("2025-04-01", tz="UTC"), pd.Timestamp("2026-07-15", tz="UTC"))
HOLD = (pd.Timestamp("2026-07-15", tz="UTC"), pd.Timestamp("2027-01-01", tz="UTC"))


def ms(t: pd.Timestamp) -> int:
    return int(t.timestamp() * 1000)


def load_cached(cfg: Config) -> dict[str, pd.DataFrame]:
    md = MarketData(cfg.market.exchange, cfg.market.timeframe_minutes, Path(".cache/candles"))
    out = {}
    for s in cfg.market.universe:
        raw = md._load_cache(s)
        if len(raw):
            out[s] = normalize(raw, cfg.bar_ms)
    return out


def apply_overrides(cfg: Config, sets: list[str]) -> Config:
    d = cfg.to_dict()
    for item in sets:
        key, _, val = item.partition("=")
        sec, name = key.split(".")
        d[sec][name] = json.loads(val)
    return Config.from_dict(d)


def oos_diagnostics(scored: pd.DataFrame, lo: int, hi: int) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    s = scored[(scored["decision_ts"] >= lo) & (scored["decision_ts"] < hi) & scored["prob"].notna() & scored["label"].notna()]
    if len(s) < 50 or s["label"].nunique() < 2:
        return {"n": len(s)}
    y, p = s["label"].to_numpy(int), s["prob"].to_numpy(float)
    bins = pd.Series(pd.cut(p, [0, .1, .15, .2, .25, .3, .35, .4, .45, .5, .6, 1.0]).astype(str), index=s.index, name="bin")
    rel = s.groupby(bins).agg(n=("label", "size"), pred=("prob", "mean"), hit=("label", "mean"),
                                             net_r=("net_r", "mean"))
    return {"n": len(s), "base_rate": float(y.mean()), "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "brier_base": float(brier_score_loss(y, np.full(len(y), y.mean()))),
            "reliability": rel.round(3).reset_index().to_dict("records")}


def fmt(summary: dict) -> str:
    s, b = summary["strategy"], summary["buy_hold"]
    f = lambda v, p=2: "n/a" if v is None else f"{v:.{p}f}"
    lines = [f"  strategy : ret {s['total_return']*100:+.2f}%  sharpe {f(s['sharpe'])}  sortino {f(s['sortino'])}  "
             f"maxDD {s['max_drawdown']*100:.2f}%  trades {s['trades']}  win {f((s['win_rate'] or 0)*100,1)}%  "
             f"PF {f(s['profit_factor'])}  exp ${f(s['expectancy'],4)} / {f(s['expectancy_r'],3)}R  "
             f"fees ${f(s['fees'])}  hold {f(s['avg_hold_min'],0)}m",
             f"  buy&hold : ret {b['total_return']*100:+.2f}%  sharpe {f(b['sharpe'])}  maxDD {b['max_drawdown']*100:.2f}%"]
    for key in ("by_signal", "by_regime", "by_exit", "by_symbol"):
        if key in summary:
            lines.append(f"  {key}: " + " | ".join(
                f"{r['key']} n={r['trades']} pnl={r['pnl']:+.2f} win={r['win_rate']*100:.0f}% R={r['expectancy_r']:+.3f}"
                for r in summary[key]))
    return "\n".join(lines)


def export_summary(report: dict, cfg: Config, periods: list, data_end: int) -> dict:
    """Compact, always-truthful summary for the dashboard: numbers straight from the run, wording derived from them."""
    labels = {"DEV": "Development", "HOLDOUT": "Holdout (never used for tuning)"}
    rows, notes = [], []
    for name, (a, b) in periods:
        gated, rules = report[name]["ensemble + ML gate"], report[name]["rules only (no ML gate)"]
        end = min(pd.Timestamp(data_end, unit="ms", tz="UTC"), b)
        rows.append({"name": f"{labels[name]} · {a:%b %Y} to {end:%b %Y}", "strategy": gated["strategy"],
                     "rules_only": rules["strategy"], "buy_hold": gated["buy_hold"], "model": report[name]["model"]})
        s = gated["strategy"]
        beat = (s["expectancy"] or 0) > 0
        notes.append(f"{labels[name].split(' ')[0]}: {s['trades']} trades, {s['total_return']*100:+.2f}% "
                     f"({'positive' if beat else 'did not beat costs'}; buy-and-hold {gated['buy_hold']['total_return']*100:+.2f}%).")
    return {"config_hash": cfg.hash, "generated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "periods": rows,
            "summary": "Walk-forward backtest of the exact configuration running here: the model is retrained daily and "
                       "only ever scores bars that come after its training data. " + " ".join(notes),
            "caveat": "A backtest is a simulation with modelled fees, spread and slippage. It is not a forecast and "
                      "guarantees nothing about live or future results."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], help="override, e.g. strategy.stop_atr=2.0")
    ap.add_argument("--holdout", action="store_true", help="also evaluate the untouched holdout period")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--export", default="", help="write a dashboard-ready summary JSON to this path")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = Config.load(args.config) if Path(args.config).exists() else Config()
    cfg = apply_overrides(cfg, args.set)
    clock = SessionClock.from_cfg(cfg.schedule)
    t0 = time.time()
    raw = load_cached(cfg)
    cfg = replace(cfg, market=replace(cfg.market, universe=tuple(raw)))
    bars, cands, ds = prepare(cfg, clock, raw)
    first = next(iter(bars.values()))
    data_start, data_end = int(first["ts"].iloc[0]), int(first["ts"].iloc[-1])
    print(f"[{args.tag}] symbols={list(bars)} bars={len(first)} "
          f"{pd.Timestamp(data_start, unit='ms')} .. {pd.Timestamp(data_end, unit='ms')}  cfg={cfg.hash[:12]}")
    res = ds[ds["label"].notna() & (ds["decision_ts"] >= ms(DEV[0])) & (ds["decision_ts"] < ms(DEV[1]))]
    print(f"candidates: {len(ds)} total, {len(res)} resolved in DEV   prep {time.time()-t0:.0f}s")
    for (sig, side), g in res.groupby(["signal", "side"]):
        print(f"  DEV raw {sig:15s} side={side:+d} n={len(g):6d} P(target first)={g['label'].mean():.3f} "
              f"mean net R={g['net_r'].mean():+.3f} stop%={g['stop_pct'].mean()*100:.2f}")
    scored = walk_forward(ds, cfg, clock, data_start, data_end)
    print(f"walk-forward done {time.time()-t0:.0f}s")
    periods = [("DEV", DEV)] + ([("HOLDOUT", HOLD)] if args.holdout else [])
    report = {"config_hash": cfg.hash, "overrides": args.set, "symbols": list(bars)}
    for name, (a, b) in periods:
        lo, hi = ms(a), min(ms(b), data_end)
        diag = oos_diagnostics(scored, lo, hi)
        print(f"\n=== {name} {a.date()} .. {pd.Timestamp(hi, unit='ms').date()} ===")
        if "auc" in diag:
            print(f"  model OOS: n={diag['n']} base={diag['base_rate']:.3f} AUC={diag['auc']:.3f} "
                  f"Brier={diag['brier']:.4f} (base-rate {diag['brier_base']:.4f})")
            for r in diag["reliability"]:
                print(f"    p in {r['bin']:14s} n={r['n']:5d} predicted={r['pred']:.3f} realised={r['hit']:.3f} net R={r['net_r']:+.3f}")
        report[name] = {"model": diag}
        for label, use_model in (("rules only (no ML gate)", False), ("ensemble + ML gate", True)):
            eng = run_engine(cfg, clock, bars, candidates_by_ts(scored, use_model), lo, hi)
            summ = summarize(eng, cfg, clock)
            print(f" {label}\n{fmt(summ)}")
            if use_model:
                print("  skips:", {k: v for k, v in list(summ['skips'].items())[:6]})
            report[name][label] = summ
    if args.export:
        Path(args.export).write_text(json.dumps(export_summary(report, cfg, periods, data_end), indent=1))
    out = Path("research/out"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\ntotal {time.time()-t0:.0f}s -> research/out/{args.tag}.json")


if __name__ == "__main__":
    main()
