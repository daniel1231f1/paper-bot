"""Position sizing for a $100 account: risk cap, position cap, cash, lot size, exchange minimum."""
from __future__ import annotations

import pytest

from bot.config import CostCfg, MarketCfg, RiskCfg
from bot.risk import RiskState, position_size

COSTS = CostCfg(impact_coef=0.0)
MARKET = MarketCfg(universe=("BTC/USD", "DOGE/USD"), qty_step={"BTC/USD": 0.00001, "DOGE/USD": 1.0}, min_notional=5.0)
RISK = RiskCfg()


def test_risk_based_size_includes_costs():
    # wide stop: risk is the binding constraint. loss/unit = 10 + 100 * 0.3% = 10.3 -> $1 risk => 0.0970 units
    qty, why = position_size(100.0, 100.0, 100.0, 10.0, "BTC/USD", RISK, MARKET, COSTS)
    assert why == "" and qty == pytest.approx(0.09708, abs=1e-5)
    assert qty * (10.0 + 100.0 * COSTS.round_trip) <= 1.0 + 1e-9


def test_position_cap_binds_for_tight_stops():
    qty, _ = position_size(100.0, 100.0, 100.0, 0.5, "BTC/USD", RISK, MARKET, COSTS)
    assert qty * 100.0 == pytest.approx(33.0, abs=0.01)  # 33% of equity, well below the 1% risk size


def test_cash_limits_size_and_reports_reason():
    qty, _ = position_size(100.0, 20.0, 100.0, 0.5, "BTC/USD", RISK, MARKET, COSTS)
    assert qty * 100.0 * (1 + COSTS.side_cost) <= 20.0
    qty, why = position_size(100.0, 3.0, 100.0, 0.5, "BTC/USD", RISK, MARKET, COSTS)
    assert qty == 0.0 and "cash" in why


def test_fractional_quantity_floors_to_lot_size():
    qty, _ = position_size(100.0, 100.0, 73_500.0, 800.0, "BTC/USD", RISK, MARKET, COSTS)
    assert qty == pytest.approx(0.00044)  # $33 / 73,500 = 0.000449 -> floored to the 0.00001 lot
    qty, _ = position_size(100.0, 100.0, 0.1737, 0.004, "DOGE/USD", RISK, MARKET, COSTS)
    assert qty == int(qty) and qty == 189  # whole DOGE only


def test_below_exchange_minimum_is_refused():
    # 1% risk with a very wide stop gives a tiny order: 1 / (40 + 0.3) * 100 = $2.48 < $5
    qty, why = position_size(100.0, 100.0, 100.0, 40.0, "BTC/USD", RISK, MARKET, COSTS)
    assert qty == 0.0 and "minimum" in why


def test_invalid_inputs():
    assert position_size(0.0, 0.0, 100.0, 1.0, "BTC/USD", RISK, MARKET, COSTS)[0] == 0.0
    assert position_size(100.0, 100.0, 100.0, 0.0, "BTC/USD", RISK, MARKET, COSTS)[0] == 0.0


def test_risk_state_drawdown_tracking():
    rs = RiskState(peak_equity=100.0)
    rs.start_day(1, 100.0)
    rs.mark(104.0, 1, RISK)
    rs.mark(98.8, 2, RISK)
    assert rs.peak_equity == 104.0 and rs.max_drawdown == pytest.approx(0.05)
    assert not rs.daily_halt  # -1.2% on the day
    rs.mark(96.9, 3, RISK)
    assert rs.daily_halt and not rs.killed
    rs.mark(88.4, 4, RISK)  # -15% from the 104 peak
    assert rs.killed and rs.killed_ts == 4
