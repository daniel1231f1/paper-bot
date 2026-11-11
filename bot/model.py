"""ML gate (meta-labelling): P(target is hit before the stop / time exit) for each candidate.

Leakage controls:
  * features are causal (see features.py) and taken at the signal bar's close
  * a sample may be used for training at cutoff T only if its trade outcome was fully
    known by T (purging) -- the label needs future bars, so unresolved trades are excluded
  * calibration uses the most recent slice of the window, again purged against the fit slice
  * evaluation is walk-forward only: a model trained with cutoff T predicts decisions >= T.
    No shuffling, no K-fold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .broker import simulate_trade, tp_net_r
from .clock import DAY_MS, SessionClock
from .config import Config
from .features import MODEL_FEATURES

log = logging.getLogger("bot.model")

DATASET_META = ["symbol", "ts", "decision_ts", "side", "signal", "regime", "close", "stop_dist", "tp_dist"]


def build_dataset(cands: Mapping[str, pd.DataFrame], bars: Mapping[str, pd.DataFrame], cfg: Config,
                  clock: SessionClock, with_labels: bool = True) -> pd.DataFrame:
    """One row per tradeable candidate (setup present, no pre-model veto, inside the entry window).

    With labels: adds label / net_r / exit_ts by simulating each trade forward with broker rules.
    Rows whose outcome is not known yet get label = NaN and exit_ts = +inf.
    """
    bar_ms, frames = cfg.bar_ms, []
    cutoff_min = clock.end_minute - clock.entry_cutoff_minutes
    for sym, cf in cands.items():
        decision_ts = cf["ts"].to_numpy() + bar_ms
        minutes = clock.local_minutes(decision_ts)
        in_window = (minutes >= clock.start_minute) & (minutes < cutoff_min)
        mask = (cf["signal"].to_numpy() != "") & (cf["skip_reason"].to_numpy() == "") & in_window
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        part = cf.iloc[idx][["ts", "side", "signal", "regime", "close", "stop_dist", "tp_dist", *MODEL_FEATURES]].copy()
        part.insert(0, "symbol", sym)
        part.insert(2, "decision_ts", decision_ts[idx])
        part["bar_idx"] = idx
        if with_labels:
            b = bars[sym]
            o, h, l = b["open"].to_numpy(float), b["high"].to_numpy(float), b["low"].to_numpy(float)
            ts = b["ts"].to_numpy()
            # first bar at/after the session close of the decision's local day -> flatten there
            day_close = (decision_ts[idx] - (minutes[idx] - clock.end_minute) * 60_000)
            flat = np.searchsorted(ts, day_close, side="left")
            labels, rs, exits = np.full(len(idx), np.nan), np.full(len(idx), np.nan), np.full(len(idx), np.inf)
            for k, i in enumerate(idx):
                res = simulate_trade(o, h, l, int(i) + 1, int(part["side"].iat[k]), float(part["stop_dist"].iat[k]),
                                     float(part["tp_dist"].iat[k]),
                                     int(min(flat[k], i + 1 + cfg.strategy.max_hold_bars)), cfg.costs)
                if res is not None:
                    labels[k], rs[k] = res[0], res[1]
                    exits[k] = ts[res[2]] + (bar_ms if res[3] != "time" else 0)
            part["label"], part["net_r"], part["exit_ts"] = labels, rs, exits
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=[*DATASET_META, *MODEL_FEATURES, "bar_idx", "label", "net_r", "exit_ts"])
    return pd.concat(frames, ignore_index=True).sort_values(["decision_ts", "symbol"]).reset_index(drop=True)


@dataclass
class TrainedModel:
    cutoff_ts: int
    kind: str
    n_fit: int
    n_calib: int
    base_rate: float
    estimator: Any
    calibrator: Any
    loss_r: dict[str, float] = field(default_factory=dict)  # mean net R of non-winning trades, per signal
    diag: dict[str, float] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.estimator.predict_proba(X)[:, 1]
        return self.calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]

    def expected_r(self, prob: float, signal: str, price: float, stop_dist: float, tp_dist: float, cfg: Config) -> float:
        """EV in R after costs: winners earn the (net) target, non-winners lose what they lost historically."""
        win = tp_net_r(price, stop_dist, tp_dist, cfg.costs)
        loss = self.loss_r.get(signal, self.loss_r["all"])
        return prob * win + (1 - prob) * loss


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _make_estimator(cfg: Config) -> Any:
    if cfg.model.kind == "gbt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        # shallow + heavily regularised: a few thousand noisy samples cannot support more
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
                                              l2_regularization=1.0, early_stopping=False,
                                              random_state=cfg.model.seed)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=500))


def train(ds: pd.DataFrame, cutoff_ts: int, cfg: Config) -> TrainedModel | None:
    """Fit + calibrate on candidates decided in [cutoff - train_days, cutoff) whose outcome resolved by cutoff."""
    from sklearn.linear_model import LogisticRegression

    lo = cutoff_ts - cfg.model.train_days * DAY_MS
    win = ds[(ds["decision_ts"] >= lo) & (ds["decision_ts"] < cutoff_ts) & (ds["exit_ts"] <= cutoff_ts)]
    win = win[win["label"].notna()].sort_values("decision_ts")
    n = len(win)
    if n < cfg.model.min_samples:
        return None
    n_cal = max(30, int(n * cfg.model.calib_fraction))
    cal = win.iloc[n - n_cal:]
    fit = win.iloc[: n - n_cal]
    fit = fit[fit["exit_ts"] <= cal["decision_ts"].iloc[0]]  # purge: no fit label may overlap the calibration slice
    y_fit, y_cal = fit["label"].to_numpy(int), cal["label"].to_numpy(int)
    if len(fit) < cfg.model.min_samples // 2 or len(set(y_fit)) < 2 or len(set(y_cal)) < 2:
        return None
    feats = list(MODEL_FEATURES)
    est = _make_estimator(cfg).fit(fit[feats].to_numpy(float), y_fit)
    raw_cal = est.predict_proba(cal[feats].to_numpy(float))[:, 1]
    calibrator = LogisticRegression(C=1e3, max_iter=500).fit(_logit(raw_cal).reshape(-1, 1), y_cal)  # Platt scaling
    losers = win[win["label"] == 0]
    loss_r = {"all": float(losers["net_r"].mean()) if len(losers) else -1.0}
    for sig, grp in losers.groupby("signal"):
        if len(grp) >= 40:
            loss_r[str(sig)] = float(grp["net_r"].mean())
    return TrainedModel(cutoff_ts=cutoff_ts, kind=cfg.model.kind, n_fit=len(fit), n_calib=n_cal,
                        base_rate=float(win["label"].mean()), estimator=est, calibrator=calibrator, loss_r=loss_r,
                        diag={"n": float(n), "base_rate": float(win["label"].mean()),
                              "mean_net_r": float(win["net_r"].mean())})


def score(model: TrainedModel | None, rows: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(prob, ev_r) for candidate rows; NaN when the model is not ready."""
    n = len(rows)
    if model is None or n == 0:
        return np.full(n, np.nan), np.full(n, np.nan)
    prob = model.predict(rows[list(MODEL_FEATURES)].to_numpy(float))
    ev = np.array([model.expected_r(float(p), str(s), float(c), float(sd), float(td), cfg)
                   for p, s, c, sd, td in zip(prob, rows["signal"], rows["close"], rows["stop_dist"], rows["tp_dist"])])
    return prob, ev


def walk_forward(ds: pd.DataFrame, cfg: Config, clock: SessionClock, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Out-of-sample predictions for every candidate: each session is scored by a model whose
    training cutoff is that session's open (retrain cadence = model.retrain_hours)."""
    out = ds.copy()
    out["prob"], out["ev_r"] = np.nan, np.nan
    step = cfg.model.retrain_hours * 3_600_000
    cutoff = clock.next_open(start_ts - 1)
    fits = 0
    while cutoff < end_ts:
        mask = ((out["decision_ts"] >= cutoff) & (out["decision_ts"] < cutoff + step)).to_numpy()
        if mask.any():
            model = train(ds, cutoff, cfg)
            fits += model is not None
            p, ev = score(model, out.loc[mask], cfg)
            out.loc[mask, "prob"], out.loc[mask, "ev_r"] = p, ev
        cutoff += step
    log.info("walk-forward: %d model fits, %d/%d candidates scored", fits, int(out["prob"].notna().sum()), len(out))
    return out
