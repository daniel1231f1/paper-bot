"""Honest test: can order book + trade flow predict the mid price a few seconds ahead? Chronological split, no shuffling."""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
d = np.loadtxt(sys.argv[1], delimiter=",", ndmin=2)
t, bb, ba, bs, as_, d1, d3, d10, buy, sell, ntr = d.T
mid = (bb + ba) / 2; n = len(mid); print("rows", n, "minutes", round((t[-1]-t[0])/60, 1))
flow = buy - sell; vol = buy + sell
roll = lambda x, k: np.convolve(x, np.ones(k), "full")[:n]                      # trailing sum, past only
def ret(k): r = np.zeros(n); r[k:] = np.log(mid[k:] / mid[:-k]); return r
F = np.column_stack([(bs - as_) / (bs + as_ + 1e-12), d1, d3, d10,
                     *[roll(flow, k) / (roll(vol, k) + 1e-9) for k in (1, 5, 15, 30)],
                     ret(1) * 1e4, ret(5) * 1e4, ret(15) * 1e4, ret(60) * 1e4, (ba - bb) / mid * 1e4])
cut = int(n * 0.6)
for h in (5, 10, 30, 60):
    dm = np.zeros(n); dm[:-h] = mid[h:] - mid[:-h]
    ok = np.arange(n) < n - h; ok[:60] = False
    tr = ok & (np.arange(n) < cut - h) & (dm != 0); te = ok & (np.arange(n) >= cut)
    sc = StandardScaler().fit(F[tr]); m = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(F[tr]), dm[tr] > 0)
    p = m.predict_proba(sc.transform(F[te]))[:, 1]; y = dm[te]; mv = y != 0
    acc = ((p[mv] > .5) == (y[mv] > 0)).mean(); conf = mv & (np.abs(p - .5) > .1)
    cacc = ((p[conf] > .5) == (y[conf] > 0)).mean() if conf.sum() else float("nan")
    edge = np.abs(y[mv]).mean() / mid.mean() * 1e4
    print(f"h={h:3d}s  test n={mv.sum():5d} (moved {mv.mean():.0%})  acc={acc:.1%}  up-rate={(y[mv]>0).mean():.1%} guess-majority={max((y[mv]>0).mean(),1-(y[mv]>0).mean()):.1%}  confident(|p-.5|>.1): n={conf.sum()} acc={cacc:.1%}  avg move={edge:.2f} bps (fees ~30 bps)")
