"""Risk manager: position sizing and the account-level circuit breakers."""
from __future__ import annotations

from dataclasses import dataclass, field

from .broker import floor_step
from .config import CostCfg, MarketCfg, RiskCfg


def position_size(equity: float, cash: float, price: float, stop_dist: float, symbol: str, risk: RiskCfg,
                  market: MarketCfg, costs: CostCfg) -> tuple[float, str]:
    """Quantity to buy and, when it is zero, the reason.

    Sized so that a full stop-out *including round-trip costs* loses at most
    risk_per_trade of equity, then capped by the per-position limit and available cash.
    """
    if equity <= 0 or price <= 0 or stop_dist <= 0:
        return 0.0, "invalid sizing inputs"
    loss_per_unit = stop_dist + price * costs.round_trip
    by_risk = equity * risk.risk_per_trade / loss_per_unit
    by_cap = equity * risk.max_position_pct / price
    by_cash = cash / (price * (1 + costs.side_cost + 0.002))  # keep a small buffer for fees and slippage
    qty = floor_step(min(by_risk, by_cap, by_cash), market.step(symbol))
    if qty * price < market.min_notional:
        if by_cash < min(by_risk, by_cap):
            return 0.0, "not enough free cash"
        return 0.0, f"order below exchange minimum (${market.min_notional:g})"
    return qty, ""


@dataclass
class RiskState:
    peak_equity: float
    day_id: int = -1
    day_start_equity: float = 0.0
    daily_halt: bool = False
    killed: bool = False
    killed_ts: int | None = None
    loss_streak: int = 0
    cooldown_until: int = 0  # ts (ms) before which no entries are allowed after a losing streak
    symbol_cooldown: dict[str, int] = field(default_factory=dict)
    risk_opened_today: float = 0.0  # sum of initial risk ($) of trades opened this session
    max_drawdown: float = 0.0

    def start_day(self, day_id: int, equity: float) -> None:
        self.day_id, self.day_start_equity = day_id, equity
        self.daily_halt, self.risk_opened_today = False, 0.0

    def mark(self, equity: float, ts: int, cfg: RiskCfg) -> None:
        """Update drawdown state after a bar closes. Trips the daily halt and the kill switch."""
        self.peak_equity = max(self.peak_equity, equity)
        dd = 1 - equity / self.peak_equity if self.peak_equity > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, dd)
        if not self.killed and dd >= cfg.kill_switch_drawdown - 1e-12:  # tolerate float rounding at the limit
            self.killed, self.killed_ts = True, ts
        if self.day_start_equity > 0 and 1 - equity / self.day_start_equity >= cfg.daily_loss_limit - 1e-12:
            self.daily_halt = True

    def on_trade_closed(self, pnl: float, symbol: str, ts: int, bar_ms: int, cfg: RiskCfg) -> None:
        self.symbol_cooldown[symbol] = ts + cfg.symbol_cooldown_bars * bar_ms
        if pnl < 0:
            self.loss_streak += 1
            if self.loss_streak >= cfg.loss_streak:
                self.cooldown_until = ts + cfg.loss_streak_cooldown_bars * bar_ms
                self.loss_streak = 0
        else:
            self.loss_streak = 0

    def entry_block(self, symbol: str, ts: int) -> str:
        """Reason new entries are blocked right now, or '' if allowed."""
        if self.killed:
            return "kill switch tripped: drawdown limit reached"
        if self.daily_halt:
            return "daily loss limit hit: halted until next session"
        if ts < self.cooldown_until:
            return "cooling down after consecutive losses"
        if ts < self.symbol_cooldown.get(symbol, 0):
            return "symbol cooldown after last exit"
        return ""

    def daily_loss(self, equity: float) -> float:
        return max(0.0, 1 - equity / self.day_start_equity) if self.day_start_equity > 0 else 0.0

    def drawdown(self, equity: float) -> float:
        return max(0.0, 1 - equity / self.peak_equity) if self.peak_equity > 0 else 0.0
