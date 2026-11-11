"""Fee / spread / slippage arithmetic, partial fills, rejections and barrier rules."""
from __future__ import annotations

import numpy as np
import pytest

from bot.broker import (LONG, SHORT, barrier_hit, fill_entry, fill_exit_limit, fill_exit_market, floor_step,
                        market_price, simulate_trade, tp_net_r)
from bot.config import CostCfg

COSTS = CostCfg(taker_fee=0.001, half_spread=0.0002, slippage=0.0003, impact_coef=0.0, max_participation=0.02)


def test_market_buy_pays_spread_and_slippage_plus_fee():
    f = fill_entry(LONG, 100.0, 0.5, bar_volume=10_000, synthetic=False, costs=COSTS, min_notional=5, step=0.001)
    assert f.status == "filled" and f.qty == pytest.approx(0.5)
    assert f.price == pytest.approx(100.0 * 1.0005)  # 2 bps half-spread + 3 bps slippage, against us
    assert f.fee == pytest.approx(0.5 * 100.05 * 0.001)
    assert f.slippage_cost == pytest.approx(0.5 * 0.05)


def test_round_trip_at_unchanged_price_loses_exactly_the_modelled_costs():
    qty = 1.0
    buy = fill_entry(LONG, 100.0, qty, 10_000, False, COSTS, 5, 0.001)
    sell = fill_exit_market(LONG, 100.0, qty, 10_000, COSTS)
    pnl = (sell.price - buy.price) * qty - buy.fee - sell.fee
    expected = -(100.0 * 0.0005 * 2) - (100.05 + 99.95) * 0.001
    assert pnl == pytest.approx(expected)
    assert pnl / 100.0 == pytest.approx(-COSTS.round_trip, rel=1e-3)  # ~0.30% of notional


def test_sell_side_slippage_is_adverse_for_shorts_too():
    assert market_price(-1, 100.0, 10, 1e9, COSTS) == pytest.approx(99.95)
    assert fill_exit_market(SHORT, 100.0, 1.0, 10_000, COSTS).price == pytest.approx(100.05)  # buying back costs more


def test_volume_impact_increases_slippage():
    costs = CostCfg(impact_coef=0.1)
    small = market_price(1, 100.0, 100.0, 1_000_000.0, costs)
    large = market_price(1, 100.0, 100_000.0, 1_000_000.0, costs)
    assert large > small > 100.0


def test_partial_fill_when_order_exceeds_participation_cap():
    f = fill_entry(LONG, 100.0, 5.0, bar_volume=100.0, synthetic=False, costs=COSTS, min_notional=5, step=0.001)
    assert f.status == "partial"
    assert f.qty == pytest.approx(2.0)  # 2% of 100 units


def test_rejected_when_bar_has_no_trades_or_volume_too_thin():
    assert not fill_entry(LONG, 100.0, 1.0, 0.0, False, COSTS, 5, 0.001).ok
    assert not fill_entry(LONG, 100.0, 1.0, 500.0, True, COSTS, 5, 0.001).ok  # synthetic bar
    thin = fill_entry(LONG, 100.0, 1.0, 1.0, False, COSTS, 5, 0.001)  # cap = 0.02 units = $2 < $5 minimum
    assert not thin.ok and "thin" in thin.reason


def test_rejected_below_exchange_minimum():
    f = fill_entry(LONG, 100.0, 0.04, 10_000, False, COSTS, min_notional=5, step=0.001)
    assert not f.ok and "minimum" in f.reason


def test_floor_step_never_rounds_up():
    assert floor_step(0.123456, 0.001) == pytest.approx(0.123)
    assert floor_step(0.3, 0.1) == pytest.approx(0.3)  # float-safe
    assert floor_step(-1, 0.1) == 0.0


def test_stop_and_target_in_same_bar_assumes_stop_first():
    assert barrier_hit(LONG, 100, 103, 97, stop=98, target=102) == ("stop", 98)


def test_gap_through_stop_fills_at_the_open_not_the_stop():
    assert barrier_hit(LONG, 95, 96, 94, stop=98, target=102) == ("stop", 95)
    assert barrier_hit(SHORT, 105, 106, 104, stop=102, target=98) == ("stop", 105)


def test_target_needs_price_to_trade_through_the_level():
    assert barrier_hit(LONG, 100, 102.0, 99.5, stop=98, target=102) is None  # touch only: resting limit not filled
    assert barrier_hit(LONG, 100, 102.01, 99.5, stop=98, target=102) == ("target", 102)


def test_take_profit_limit_has_no_slippage_but_pays_fee():
    f = fill_exit_limit(102.0, 2.0, COSTS)
    assert f.price == 102.0 and f.slippage_cost == 0.0 and f.fee == pytest.approx(2 * 102 * 0.001)


def test_tp_net_r_is_below_gross_reward_ratio():
    r = tp_net_r(100.0, stop_dist=1.0, tp_dist=2.0, costs=COSTS)
    assert r == pytest.approx((102 - 100.05 - (100.05 + 102) * 0.001) / 1.0)
    assert r < 2.0


def test_simulate_trade_outcomes():
    o = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    flat = np.full(5, 100.0)
    # target hit on bar 2
    h = np.array([100.0, 100.5, 102.5, 100.0, 100.0])
    label, r, k, why = simulate_trade(o, h, flat, 1, LONG, 1.0, 2.0, flat_idx=4, costs=COSTS)
    assert (label, k, why) == (1, 2, "target") and r == pytest.approx(tp_net_r(100.0, 1.0, 2.0, COSTS))
    # stop hit on bar 1
    low = np.array([100.0, 98.9, 100.0, 100.0, 100.0])
    label, r, k, why = simulate_trade(o, flat, low, 1, LONG, 1.0, 2.0, flat_idx=4, costs=COSTS)
    assert (label, k, why) == (0, 1, "stop") and r < -1.0  # worse than -1R once costs are included
    # neither: time exit at the open of flat_idx, costs only
    label, r, k, why = simulate_trade(o, flat, flat, 1, LONG, 1.0, 2.0, flat_idx=4, costs=COSTS)
    assert (label, k, why) == (0, 4, "time") and r == pytest.approx(-100 * COSTS.round_trip, rel=1e-2)
    # outcome unknown (data ends before the exit bar): no label
    assert simulate_trade(o, flat, flat, 1, LONG, 1.0, 2.0, flat_idx=9, costs=COSTS) is None
