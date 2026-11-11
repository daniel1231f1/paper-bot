"""Config locking, atomic persistence, data layer resilience, metrics and verdict rules."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bot import state as state_mod
from bot.config import Config, ConfigError, VerdictCfg
from bot.data import DataUnavailable, MarketData, normalize
from bot.metrics import bootstrap_ci, max_drawdown, profit_factor, verdict
from bot.state import Store, atomic_write_json, sanitize
from conftest import BAR, FakeClient, ist, make_candles


# ---------------------------------------------------------------- config
def test_hash_is_stable_and_sensitive_to_trading_parameters():
    a, b = Config(), Config()
    assert a.hash == b.hash and len(a.hash) == 64
    changed = Config.from_dict({"risk": {"risk_per_trade": 0.02}})
    assert changed.hash != a.hash


def test_ops_settings_are_not_part_of_the_lock():
    assert Config.from_dict({"ops": {"poll_seconds": 5}}).hash == Config().hash


def test_unknown_keys_and_bad_values_are_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"risk": {"risk_per_trad": 0.01}})
    with pytest.raises(ConfigError):
        Config.from_dict({"risk": {"risk_per_trade": 0.05}})  # above the 2% cap
    with pytest.raises(ConfigError):
        Config.from_dict({"risk": {"max_positions": 5}})
    with pytest.raises(ConfigError):
        Config.from_dict({"experiment": {"starting_capital": "100"}})


def test_config_round_trips_through_its_locked_snapshot():
    cfg = Config.from_dict({"strategy": {"allow_short": True}, "market": {"qty_step": {"BTC/USD": 1e-5}}})
    again = Config.from_dict({**cfg.locked_dict(), "ops": cfg.to_dict()["ops"]})
    assert again == cfg and again.hash == cfg.hash


def test_repo_config_file_loads():
    cfg = Config.load("config.yaml")
    assert cfg.experiment.starting_capital == 100.0 and cfg.experiment.days == 10
    assert cfg.schedule.timezone == "Asia/Kolkata" and (cfg.schedule.active_start, cfg.schedule.active_end) == ("07:00", "23:00")
    assert cfg.risk.kill_switch_drawdown == 0.15 and cfg.risk.risk_per_trade <= 0.02


# ---------------------------------------------------------------- persistence
def test_atomic_write_survives_a_crash_mid_write(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"v": 1})

    def boom(*_a, **_k):
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(state_mod.json, "dump", boom)
    with pytest.raises(RuntimeError):
        atomic_write_json(path, {"v": 2})
    monkeypatch.undo()
    assert json.loads(path.read_text()) == {"v": 1}  # old state intact
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]  # no temp litter


def test_sanitize_makes_strict_json():
    out = sanitize({"a": float("nan"), "b": np.float64(1.5), "c": [np.int64(2), float("inf")], 3: True})
    assert out == {"a": None, "b": 1.5, "c": [2, None], "3": True}
    json.dumps(out, allow_nan=False)


def test_store_ignores_state_from_an_older_schema(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"product": "BTC-USD", "capital": 100}))  # v1 bot format
    assert Store(tmp_path).load() is None


def test_store_refuses_corrupt_state(tmp_path):
    (tmp_path / "state.json").write_text("{ not json")
    with pytest.raises(state_mod.StateError):
        Store(tmp_path).load()


# ---------------------------------------------------------------- data layer
def market(tmp_path=None, n=900):
    start = ist(2026, 1, 1, 0, 0)
    client = FakeClient({"AAA/USD": make_candles(n, start, seed=2)})
    naps: list[float] = []
    md = MarketData("fake", 5, tmp_path, client=client, max_retries=3, retry_base_seconds=0.5, sleep=naps.append)
    return md, client, start, naps


def test_only_closed_bars_are_returned():
    md, client, start, _ = market()
    now = start + 100 * BAR + 120_000  # 2 minutes into bar #100
    client.now = now
    df = md.history("AAA/USD", start, now)
    assert int(df["ts"].iloc[-1]) == start + 99 * BAR  # bar #100 is still forming: excluded


def test_retry_with_backoff_then_success():
    md, client, start, naps = market()
    client.now = now = start + 50 * BAR
    client.fail = 2
    df = md.history("AAA/USD", start, now)
    assert len(df) == 50 and len(naps) == 2 and naps[1] > naps[0]  # exponential backoff


def test_outage_raises_data_unavailable_instead_of_returning_stale_data():
    md, client, start, naps = market()
    client.now = now = start + 50 * BAR
    client.fail = 99
    with pytest.raises(DataUnavailable):
        md.history("AAA/USD", start, now)
    assert len(naps) == 3


def test_incremental_fetch_and_cache_round_trip(tmp_path):
    md, client, start, _ = market(tmp_path)
    client.now = now = start + 700 * BAR
    first = md.history("AAA/USD", start, now)
    calls = client.calls
    client.now = now = now + 2 * BAR
    second = md.history("AAA/USD", start, now)
    assert client.calls - calls == 1 and len(second) == len(first) + 2  # one small request for the tail
    md2 = MarketData("fake", 5, tmp_path, client=client)
    again = md2.history("AAA/USD", start, now)
    pd.testing.assert_frame_equal(again, second)


def test_missing_bars_become_flagged_synthetic_bars():
    raw = make_candles(20, 0, seed=3).drop(index=[7, 8]).reset_index(drop=True)
    out = normalize(raw, BAR)
    assert len(out) == 20 and out["synthetic"].sum() == 2
    gap = out[out["synthetic"]]
    prev_close = out["close"].iloc[6]
    assert (gap[["open", "high", "low", "close"]] == prev_close).all().all() and (gap["volume"] == 0).all()


# ---------------------------------------------------------------- metrics / verdict
def trades(pnls):
    return pd.DataFrame({"pnl": pnls, "r": [p / 1.0 for p in pnls]})


def test_profit_factor_and_drawdown():
    assert profit_factor(np.array([2.0, -1.0, 1.0, -1.0])) == pytest.approx(1.5)
    assert max_drawdown(np.array([100, 110, 99, 120.0])) == pytest.approx(0.1)


def test_verdict_pass_requires_every_criterion():
    rng = np.random.default_rng(0)
    good = list(rng.normal(0.3, 0.5, 60))
    v = verdict(trades(good), 0.05, VerdictCfg(bootstrap_samples=500), final=True)
    assert v["result"] == "PASS" and all(c["pass"] for c in v["checks"])
    assert v["expectancy_ci"][0] > 0 and "small sample" in v["explanation"]


def test_verdict_marginal_when_sample_is_too_small_or_one_trade_carries_it():
    few = verdict(trades([0.5, 0.4, -0.2, 0.3]), 0.02, VerdictCfg(bootstrap_samples=300), final=True)
    assert few["result"] == "MARGINAL"
    lucky = verdict(trades([-0.1] * 35 + [9.0]), 0.04, VerdictCfg(bootstrap_samples=300), final=True)
    assert lucky["result"] == "MARGINAL"
    assert not next(c for c in lucky["checks"] if c["id"] == "lucky")["pass"]


def test_verdict_fail_on_negative_expectancy_or_drawdown_breach():
    assert verdict(trades([-0.2] * 40), 0.05, VerdictCfg(bootstrap_samples=300), final=True)["result"] == "FAIL"
    assert verdict(trades([0.3] * 40), 0.16, VerdictCfg(bootstrap_samples=300), final=True)["result"] == "FAIL"
    assert verdict(trades([]), 0.0, VerdictCfg(bootstrap_samples=300), final=True)["result"] == "FAIL"


def test_bootstrap_is_deterministic():
    x = np.random.default_rng(1).normal(0, 1, 50)
    assert bootstrap_ci(x, lambda v: float(v.mean()), 400, 0.9) == bootstrap_ci(x, lambda v: float(v.mean()), 400, 0.9)
