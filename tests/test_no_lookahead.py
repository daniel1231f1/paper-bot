"""No-look-ahead guarantees for features, signals, hourly context, labels and walk-forward training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.clock import DAY_MS, SessionClock
from bot.data import normalize
from bot.features import MODEL_FEATURES, compute_features
from bot.model import score, train, walk_forward
from bot.signals import generate_candidates
from conftest import BAR, ist, make_candles, make_config

CFG = make_config()
CLOCK = SessionClock.from_cfg(CFG.schedule)
START = ist(2026, 1, 1, 0, 0)


def pipeline(df: pd.DataFrame) -> pd.DataFrame:
    return generate_candidates(compute_features(df, CFG.strategy, CLOCK, CFG.bar_ms), CFG.strategy, CFG.costs)


def frame(n: int = 1500, seed: int = 1) -> pd.DataFrame:
    return normalize(make_candles(n, START, seed=seed, sigma=0.004), BAR)


def assert_rows_equal(a: pd.Series, b: pd.Series) -> None:
    for col in a.index:
        x, y = a[col], b[col]
        if isinstance(x, (float, np.floating)):
            assert (np.isnan(x) and np.isnan(y)) or x == pytest.approx(y, rel=1e-9, abs=1e-12), col
        else:
            assert x == y, col


@pytest.mark.parametrize("cut", [650, 911, 1200, 1499])
def test_features_at_bar_i_do_not_depend_on_later_bars(cut: int):
    df = frame()
    full = pipeline(df)
    truncated = pipeline(df.iloc[: cut + 1].reset_index(drop=True))
    assert_rows_equal(truncated.iloc[-1], full.iloc[cut])


def test_mutating_the_future_leaves_the_past_unchanged():
    df = frame()
    cut = 1000
    before = pipeline(df).iloc[: cut + 1]
    future = df.copy()
    future.loc[cut + 1:, ["open", "high", "low", "close"]] *= 3.0
    future.loc[cut + 1:, "volume"] *= 50.0
    after = pipeline(future).iloc[: cut + 1]
    pd.testing.assert_frame_equal(before, after)


def test_all_model_features_exist_and_some_signals_fire():
    out = pipeline(frame(3000, seed=4))
    missing = [c for c in MODEL_FEATURES if c not in out.columns and not c.startswith("mkt_")]
    assert not missing
    assert (out["signal"] != "").sum() > 0


def test_hourly_context_only_updates_after_the_hour_closes():
    df = frame()
    ts = df["ts"].to_numpy()
    # choose a bar 20 minutes into an hour, well after warm-up
    i = int(np.flatnonzero((ts % 3_600_000 == 20 * 60_000) & (np.arange(len(ts)) > 1000))[0])
    hour_start = i - 4
    last_in_hour = hour_start + 11
    base = compute_features(df, CFG.strategy, CLOCK, CFG.bar_ms)
    shocked = df.copy()
    shocked.loc[hour_start:i, ["open", "high", "low", "close"]] *= 1.05  # the still-forming hour moves 5%
    feat = compute_features(shocked, CFG.strategy, CLOCK, CFG.bar_ms)
    for col in ("ema_fast_h", "ema_slow_h", "atr_h", "adx_h"):
        assert feat[col].iloc[i] == pytest.approx(base[col].iloc[i]), col  # mid-hour: incomplete hour invisible
    # once the hour has closed its (higher) range is visible
    assert feat["atr_h"].iloc[last_in_hour] > base["atr_h"].iloc[last_in_hour]


def synthetic_dataset(n: int = 1500, seed: int = 5) -> pd.DataFrame:
    """Candidate table with a weak real relationship between one feature and the label."""
    rng = np.random.default_rng(seed)
    start = ist(2026, 1, 1, 7, 0)
    decision = np.sort(start + rng.integers(0, 60 * DAY_MS // BAR, n) * BAR)
    X = rng.normal(size=(n, len(MODEL_FEATURES)))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X[:, 0] - 1.0)))).astype(float)
    ds = pd.DataFrame(X, columns=list(MODEL_FEATURES))
    ds["decision_ts"], ds["ts"] = decision, decision - BAR
    ds["exit_ts"] = decision + rng.integers(1, 60, n) * BAR  # outcome known up to 5h later
    ds["label"], ds["net_r"] = y, np.where(y == 1, 1.7, -0.9)
    ds["signal"], ds["regime"], ds["symbol"], ds["side"] = "breakout", "TREND_UP", "AAA/USD", 1
    ds["close"], ds["stop_dist"], ds["tp_dist"] = 100.0, 1.0, 2.0
    return ds


def test_training_ignores_everything_at_or_after_the_cutoff():
    ds = synthetic_dataset()
    cutoff = ist(2026, 2, 10, 7, 0)
    m1 = train(ds, cutoff, CFG)
    assert m1 is not None
    poisoned = ds.copy()
    late = (poisoned["decision_ts"] >= cutoff) | (poisoned["exit_ts"] > cutoff)  # unresolved or future
    assert late.sum() > 50
    poisoned.loc[late, "label"] = 1 - poisoned.loc[late, "label"]
    poisoned.loc[late, list(MODEL_FEATURES)] = 99.0
    m2 = train(poisoned, cutoff, CFG)
    probe = ds.iloc[:200]
    np.testing.assert_allclose(score(m1, probe, CFG)[0], score(m2, probe, CFG)[0])


def test_purge_excludes_trades_still_open_at_the_cutoff():
    ds = synthetic_dataset()
    cutoff = ist(2026, 2, 10, 7, 0)
    m = train(ds, cutoff, CFG)
    eligible = ds[(ds["decision_ts"] < cutoff) & (ds["decision_ts"] >= cutoff - CFG.model.train_days * DAY_MS)]
    resolved = eligible[eligible["exit_ts"] <= cutoff]
    assert len(resolved) < len(eligible)  # some decisions were made before the cutoff but resolved after it
    assert m.n_fit + m.n_calib <= len(resolved)


def test_model_needs_enough_resolved_samples():
    ds = synthetic_dataset(n=120)
    assert train(ds, ist(2026, 3, 5, 7, 0), CFG) is None


def test_walk_forward_predictions_are_out_of_sample(monkeypatch):
    import sys
    monkeypatch.setattr(sys.modules[__name__], "CFG", make_config(model={"kind": "logreg"}))  # fast estimator
    ds = synthetic_dataset()
    start, end = int(ds["decision_ts"].min()), int(ds["decision_ts"].max()) + BAR
    scored = walk_forward(ds, CFG, CLOCK, start, end)
    assert scored["prob"].notna().sum() > 200 and scored["prob"].iloc[:50].isna().all()  # nothing before the first fit
    # changing labels of a later day cannot change predictions made for an earlier day
    pivot = ist(2026, 2, 15, 7, 0)
    tampered = ds.copy()
    tampered.loc[tampered["decision_ts"] >= pivot, "label"] = 1.0
    rescored = walk_forward(tampered, CFG, CLOCK, start, end)
    early = (ds["decision_ts"] < pivot).to_numpy()
    np.testing.assert_allclose(scored.loc[early, "prob"].to_numpy(), rescored.loc[early, "prob"].to_numpy(), equal_nan=True)
    # and the model found the real signal without leakage: OOS ranking is better than chance
    ok = scored["prob"].notna()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(ds.loc[ok, "label"], scored.loc[ok, "prob"]) > 0.6
