# Backtest report

**Bottom line: the strategy does not have a demonstrated edge after costs.** Out of sample, the
model-gated strategy lost a little (−0.89% over 15.5 months, 20 trades) and then found nothing
worth trading for the entire 4-month holdout (0 trades). The same rules without the model gate lost
about 14.5% in *both* periods before the kill switch stopped them. Nothing below is a forecast.

## Setup

| | |
|---|---|
| Data | Coinbase Exchange public 5-minute candles, 2025-01-01 → 2026-11-11, 195,644 bars × 7 symbols |
| Universe | BTC, ETH, SOL, XRP, DOGE, LINK, AVAX (USD pairs) |
| Capital / session | $100, trades only 07:00–23:00 IST, flat by 23:00 |
| Costs per side | 0.10% taker fee + 0.02% half-spread + 0.03% slippage (+ volume impact) = **0.30% round trip** |
| Fills | next bar's open; stops gap through; targets need a trade-through; stop wins ties |
| Engine | the same `bot/engine.py` the live bot runs |
| Model | logistic regression + Platt calibration, 120-day rolling window, retrained daily, purged labels |
| Validation | walk-forward only: every prediction comes from a model trained strictly on earlier, already-resolved trades |

Periods were fixed before looking at results:

* **Development** 2025-04-01 → 2026-07-15: all iteration happened here.
* **Holdout** 2026-07-15 → 2026-11-11: evaluated once, for the final configuration only.

## Final configuration (config hash `eab6ad7b726a`)

| Period | Variant | Return | Sharpe | Sortino | Max DD | Trades | Win rate | Profit factor | Expectancy / trade |
|---|---|---|---|---|---|---|---|---|---|
| Development | **Strategy (ensemble + ML gate)** | **−0.89%** | −0.15 | −0.35 | 5.18% | 20 | 35.0% | 0.85 | −$0.044 (−0.30R) |
| Development | Rules only, no ML gate | −14.79% ¹ | −1.17 | −1.45 | 15.16% | 106 | 34.9% | 0.58 | −$0.140 (−0.26R) |
| Development | Buy & hold, equal weight | −37.36% | −0.26 | −0.36 | 70.30% | – | – | – | – |
| Holdout | **Strategy (ensemble + ML gate)** | **0.00%** | n/a | n/a | 0.00% | **0** | – | – | – |
| Holdout | Rules only, no ML gate | −14.45% ¹ | −2.90 | −3.51 | 15.16% | 223 | 38.6% | 0.70 | −$0.065 (−0.21R) |
| Holdout | Buy & hold, equal weight | +9.20% | 0.78 | 1.18 | 29.30% | – | – | – | – |

¹ The −15% kill switch tripped and stopped trading; the loss is capped by the risk manager, not by the strategy.

The strategy "beat" buy-and-hold in the development period only by staying out of a −37% bear
market, and it lagged buy-and-hold in the holdout by doing nothing while the basket rose 9%. Neither
is evidence of skill.

### By regime and signal

| Period | Variant | Bucket | Trades | Net P&L | Win rate | Avg R |
|---|---|---|---|---|---|---|
| Development | Strategy | TREND_UP | 10 | +$2.11 | 50% | +0.18 |
| Development | Strategy | RANGE | 10 | −$3.00 | 20% | −0.79 |
| Development | Strategy | breakout | 8 | +$2.26 | 50% | +0.21 |
| Development | Strategy | mean reversion | 10 | −$3.00 | 20% | −0.79 |
| Development | Strategy | trend pullback | 2 | −$0.15 | 50% | +0.06 |
| Development | Rules only | TREND_UP | 82 | −$13.03 | 34% | −0.29 |
| Development | Rules only | RANGE | 24 | −$1.76 | 38% | −0.16 |
| Holdout | Rules only | TREND_UP | 155 | −$12.09 | 34% | −0.25 |
| Holdout | Rules only | RANGE | 68 | −$2.36 | 50% | −0.12 |

With 2–10 trades per bucket the strategy rows are noise; they are shown for completeness, not as findings.

### Model quality (out of sample)

| Period | Candidates scored | Base rate | AUC | Brier | Brier of always predicting the base rate |
|---|---|---|---|---|---|
| Development | 8,136 | 13.2% | 0.535 | 0.1181 | 0.1149 |
| Holdout | 1,949 | 18.0% | 0.528 | 0.1518 | 0.1473 |

An AUC of 0.53 is distinguishable from a coin flip on this many samples, but it is far too weak to
overcome a 0.30% round trip. The Brier score is slightly *worse* than a constant prediction, and the
model's high-probability calls were overconfident (predicted 48–54%, realised 7–36% in development).
Only 13–18% of setups reach their target at all; the gate asks for a calibrated 45% and +0.10R of
expected value after costs, and the model almost never claims that.

## Why it fails: costs

Round-trip cost is about 0.21R of a typical 1.5-ATR stop. Before costs the raw long setups average
roughly −0.10R to 0.00R, i.e. no gross edge to speak of, so costs are essentially the whole loss.

Cost sensitivity for the final configuration (informational only, not used for selection):

| Round-trip cost | Dev strategy | Dev rules only | Holdout strategy | Holdout rules only |
|---|---|---|---|---|
| 0.30% (base case) | −0.89% (20 trades) | −14.79% | 0.00% (0 trades) | −14.45% |
| 0.15% | +1.85% (16 trades, PF 1.49) | −14.40% | −0.68% (1 trade) | −7.75% |
| 0.00% | +1.13% (16 trades, PF 1.29) | −10.76% | −0.62% (1 trade) | +8.51% (buy & hold +9.20%) |

Even with free trading the rules lose in the development period and merely track the market in the
holdout. Lower fees would not rescue this strategy.

## Iteration log (everything that was tried)

Rounds 1–4 used the development window 2025-12-20 → 2026-07-15 (history then started in Oct 2025);
round 5 and the final run used the extended window above. The holdout was never consulted for any choice.

| Round | Change | Result on development data |
|---|---|---|
| 1 | Baseline: long-only, gradient-boosted trees, 60-day window | Strategy −5.30% (27 trades, PF 0.42); rules only −14.80%; model AUC 0.46 (worse than chance) |
| 2a | Allow shorts, GBT 60d | −6.67% (24 trades, PF 0.23), AUC 0.52 |
| 2b | Allow shorts, logistic regression 120d | −1.88% (12 trades, PF 0.48), AUC 0.53 |
| 2c | Allow shorts, GBT 120d, closer targets (2.0 ATR) | −4.59% (14 trades, PF 0.25), AUC 0.53 |
| 2d | Allow shorts, logreg 120d, closer targets | −8.11% (68 trades, PF 0.56), AUC 0.53 |
| 3 | Wider stops (2.5 ATR), hold to session close, raw-edge study | Best cell: short breakouts +0.03R net in a −38% market (beta, not alpha); every long setup negative |
| 4 | New hypothesis: fade liquidation-style flushes | All 36 parameter cells negative after costs (best −0.09R) |
| 5 | Final grid on 15.5 months: {long, long/short} × {GBT 60d, logreg 120d, GBT 120d} | Long logreg 120d −0.89% (best); long GBT 120d −1.16%; L/S logreg −1.21%; L/S GBT 120d −1.75%; L/S GBT 60d −4.65%; long GBT 60d −5.92% |

The final configuration is the least-bad row of round 5: long-only (realistic for a $100 spot
account), logistic regression, 120-day window. "Least bad" is the honest description. I stopped
iterating here because further search over the same 15 months would be curve-fitting.

## What to expect from the 10-day live paper test

* Most likely the gate keeps the bot flat most or all of the time (it traded about 1.3 times per
  month in development and never in the holdout). A flat 10 days ends in **FAIL** on the
  pre-registered criteria (no positive expectancy shown, fewer than 30 trades). That would be the
  system working as designed, and it would agree with this backtest.
* The dashboard's "shadow" line shows what the same rules would do without the gate. The backtest
  says to expect it to lose.

## Limitations

* One venue's candles; fills are simulated from 5-minute bars, not an order book. Intrabar order of
  stop vs target is unknowable from bars, so the stop is assumed first.
* Fees are modelled at 0.10% taker. Retail tiers on Coinbase or Kraken are 4–12× higher; at those
  rates every variant is far worse.
* 22 months is two market regimes (a rally into Oct 2025, then a deep bear market). Results in other
  regimes are unknown.
* The bootstrap in the live verdict resamples trades as if independent; crypto trades cluster.
* Rounds 1–2 printed aggregate raw-setup statistics that included holdout dates (no model or
  strategy results). It did not influence any choice, but strictly the holdout was not 100% blind.
