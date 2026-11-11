"""Paper broker simulator. There is no real order path anywhere in this project.

Fill rules (deliberately conservative):
  * market orders fill at the NEXT bar's open, moved against us by half-spread + slippage
    + a volume-impact term; a taker fee is charged on every fill
  * entries are capped at a share of the fill bar's volume (partial fill, remainder cancelled)
    and rejected when the bar printed no trades or the fill would be below the exchange minimum
  * stops are stop-market: they fill at min(open, stop) for longs (gaps hurt), then pay slippage
  * targets are resting limits: they need price to trade *through* the level and still pay the taker fee
  * if stop and target are both inside one bar, the stop is assumed to have been hit first
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import CostCfg

LONG, SHORT = 1, -1


@dataclass(frozen=True)
class Fill:
    status: str  # "filled" | "partial" | "rejected"
    qty: float
    price: float
    fee: float
    slippage_cost: float  # money lost versus the reference price (spread + slippage + impact)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "rejected"


REJECT = Fill("rejected", 0.0, 0.0, 0.0, 0.0)


def floor_step(qty: float, step: float) -> float:
    """Round a quantity down to the exchange lot size (float-safe)."""
    if qty <= 0:
        return 0.0
    return math.floor(qty / step + 1e-9) * step


def market_price(direction: int, ref_price: float, notional: float, bar_dollar_volume: float, costs: CostCfg) -> float:
    """Adverse fill price for a market order. direction: +1 buy, -1 sell."""
    impact = costs.impact_coef * notional / bar_dollar_volume if bar_dollar_volume > 0 else 0.0
    adverse = costs.half_spread + costs.slippage + min(impact, 0.01)
    return ref_price * (1 + direction * adverse)


def fill_entry(direction: int, ref_price: float, qty: float, bar_volume: float, synthetic: bool, costs: CostCfg,
               min_notional: float, step: float) -> Fill:
    """Market entry at the bar open, with partial fill / rejection on thin volume."""
    if synthetic or bar_volume <= 0:
        return Fill("rejected", 0.0, 0.0, 0.0, 0.0, "no liquidity: bar printed no trades")
    allowed = floor_step(min(qty, costs.max_participation * bar_volume), step)
    if allowed * ref_price < min_notional:
        why = "volume too thin for the minimum order" if allowed < floor_step(qty, step) else "below exchange minimum"
        return Fill("rejected", 0.0, 0.0, 0.0, 0.0, why)
    price = market_price(direction, ref_price, allowed * ref_price, bar_volume * ref_price, costs)
    status = "partial" if allowed < floor_step(qty, step) - 1e-12 else "filled"
    return Fill(status, allowed, price, allowed * price * costs.taker_fee, abs(price - ref_price) * allowed,
                "partial fill: capped by bar volume" if status == "partial" else "")


def fill_exit_market(side: int, ref_price: float, qty: float, bar_volume: float, costs: CostCfg) -> Fill:
    """Market exit (stop, time stop, session flatten). Always completes; thin volume costs extra impact."""
    dollar_vol = bar_volume * ref_price
    price = market_price(-side, ref_price, qty * ref_price, dollar_vol if dollar_vol > 0 else qty * ref_price * 10, costs)
    return Fill("filled", qty, price, qty * price * costs.taker_fee, abs(price - ref_price) * qty)


def fill_exit_limit(price: float, qty: float, costs: CostCfg) -> Fill:
    """Take-profit limit: fills at its price, no slippage, taker fee charged to stay conservative."""
    return Fill("filled", qty, price, qty * price * costs.taker_fee, 0.0)


def barrier_hit(side: int, bar_open: float, high: float, low: float, stop: float, target: float) -> tuple[str, float] | None:
    """Did this bar hit the stop or the target? Returns (reason, reference price). Stop wins ties."""
    if side == LONG:
        if low <= stop:
            return "stop", min(bar_open, stop)
        if high > target:
            return "target", target
    else:
        if high >= stop:
            return "stop", max(bar_open, stop)
        if low < target:
            return "target", target
    return None


def tp_net_r(price: float, stop_dist: float, tp_dist: float, costs: CostCfg) -> float:
    """Net R-multiple of a trade that reaches its target, after fees/spread/slippage."""
    entry = price * (1 + costs.half_spread + costs.slippage)
    exit_ = price + tp_dist
    return (exit_ - entry - (entry + exit_) * costs.taker_fee) / stop_dist


def simulate_trade(o: np.ndarray, h: np.ndarray, l: np.ndarray, i_entry: int, side: int, stop_dist: float,
                   tp_dist: float, flat_idx: int, costs: CostCfg) -> tuple[int, float, int, str] | None:
    """Outcome of one trade entered at the open of bar `i_entry` (labels for the ML layer).

    Uses the same rules as the event engine (tests assert they agree). Returns
    (label, net_R, exit_bar_index, reason) or None if the outcome is not known yet.
    label = 1 only when the target is hit before the stop and before the time/session exit at `flat_idx`.
    """
    n = len(o)
    if i_entry >= n or flat_idx <= i_entry:
        return None
    ref = float(o[i_entry])
    adverse = costs.half_spread + costs.slippage
    entry = ref * (1 + side * adverse)
    stop, target = ref - side * stop_dist, ref + side * tp_dist
    end = min(flat_idx, n)
    seg_h, seg_l = h[i_entry:end], l[i_entry:end]
    if side == LONG:
        stop_hits, tp_hits = seg_l <= stop, seg_h > target
    else:
        stop_hits, tp_hits = seg_h >= stop, seg_l < target
    k_stop = int(np.argmax(stop_hits)) if stop_hits.any() else -1
    k_tp = int(np.argmax(tp_hits)) if tp_hits.any() else -1
    if k_stop >= 0 and (k_tp < 0 or k_stop <= k_tp):
        k = i_entry + k_stop
        ref_exit = min(float(o[k]), stop) if side == LONG else max(float(o[k]), stop)
        exit_price, label, reason = ref_exit * (1 - side * adverse), 0, "stop"
    elif k_tp >= 0:
        k = i_entry + k_tp
        exit_price, label, reason = target, 1, "target"
    else:
        if flat_idx >= n:
            return None  # still open at the end of the data
        k = flat_idx
        exit_price, label, reason = float(o[k]) * (1 - side * adverse), 0, "time"
    fees = (entry + exit_price) * costs.taker_fee
    net_r = (side * (exit_price - entry) - fees) / stop_dist
    return label, float(net_r), k, reason
