from __future__ import annotations

import os

from bot import Config

ALLOWED = {"BTC-USD", "ETH-USD", "SOL-USD"}  # whitelist: product id is placed in an upstream URL path


def load() -> tuple[Config, float]:
    product = os.getenv("COIN", "BTC-USD")
    if product not in ALLOWED:
        raise SystemExit(f"COIN must be one of {sorted(ALLOWED)}")
    cfg = Config(
        product=product,
        capital=float(os.getenv("START_CAPITAL", "100")),
        fee_rate=float(os.getenv("FEE_RATE", "0.001")),
        slippage=float(os.getenv("SLIPPAGE", "0.0005")),
        entry_p=float(os.getenv("ENTRY_P", "0.60")),
        stop_pct=float(os.getenv("STOP_PCT", "0.012")),
    )
    if cfg.capital <= 0 or not 0.5 < cfg.entry_p < 1:
        raise SystemExit("START_CAPITAL must be > 0 and ENTRY_P in (0.5, 1)")
    return cfg, float(os.getenv("RUN_DAYS", "5"))
