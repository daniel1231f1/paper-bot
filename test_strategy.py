import numpy as np

from bot import Config, Engine, LogReg, features, max_drawdown, run_backtest


def test_features_shape_and_finite():
    c = np.cumsum(np.random.default_rng(0).normal(0, 1, 100)) + 500
    f = features(c, 99, 9, 21)
    assert f.shape == (9,) and np.isfinite(f).all()


def test_logreg_learns_separable():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] > 0).astype(float)
    m = LogReg()
    assert m.fit(X, y)
    assert m.proba(np.array([2.0, 0, 0])) > 0.8 > 0.2 > m.proba(np.array([-2.0, 0, 0]))


def test_round_trip_costs_money_on_flat_price():
    e = Engine(Config(capital=100))
    e._buy(0, 100.0, 0.7, "t")
    e._sell(1, 100.0, None, "t")
    assert e.pos is None and e.cash < 100 and abs(e.cash - 100 * (1 - 0.003)) < 0.05


def test_stop_loss_triggers():
    e = Engine(Config(capital=100, stop_pct=0.01))
    e._buy(0, 100.0, 0.7, "t")
    e.on_tick(1, 98.0)
    assert e.pos is None and e.trades[-1].reason.startswith("stop loss") and e.trades[-1].pnl < 0


def test_trailing_stop_locks_gain():
    e = Engine(Config(capital=100))
    e._buy(0, 100.0, 0.7, "t")
    e.on_tick(1, 102.0)  # arms trail
    e.on_tick(2, 101.0)  # >0.5% off peak
    assert e.pos is None and e.trades[-1].pnl > 0


def test_entries_disabled_blocks_buys():
    e = Engine(Config(capital=100))
    e.entries_enabled = False
    rng = np.random.default_rng(2)
    prices = np.cumsum(rng.normal(0, 1, 700)) + 1000
    e.warm([(i * 900, float(p)) for i, p in enumerate(prices[:400])])
    for i in range(400, 700):
        e.on_candle(i * 900, float(prices[i]))
    assert not e.trades


def test_backtest_runs_no_crash_and_sane():
    rng = np.random.default_rng(3)
    prices = 1000 * np.exp(np.cumsum(rng.normal(0, 0.004, 900)))
    r = run_backtest(Config(capital=100), [(i * 900, float(p)) for i, p in enumerate(prices)])
    assert r["final_equity"] > 0 and len(r["curve"]) == 600


def test_max_drawdown():
    assert abs(max_drawdown([100, 120, 90, 110]) - 0.25) < 1e-9
