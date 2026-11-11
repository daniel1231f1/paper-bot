"""Session clock: every schedule decision goes through here.

All inputs/outputs are UTC epoch milliseconds; the timezone conversion happens via zoneinfo,
so there is no hand-rolled offset arithmetic. IST has no daylight saving, but any tz works.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import ScheduleCfg

DAY_MS = 86_400_000


def _parse_hhmm(text: str) -> time:
    hh, mm = text.split(":")
    return time(int(hh), int(mm))


@dataclass(frozen=True)
class SessionClock:
    tz: ZoneInfo
    start: time
    end: time
    entry_cutoff_minutes: int

    @classmethod
    def from_cfg(cls, cfg: ScheduleCfg) -> "SessionClock":
        return cls(ZoneInfo(cfg.timezone), _parse_hhmm(cfg.active_start), _parse_hhmm(cfg.active_end),
                   cfg.entry_cutoff_minutes)

    # ---- scalar helpers ----------------------------------------------
    def local(self, ts_ms: int) -> datetime:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(self.tz)

    def local_date(self, ts_ms: int) -> date:
        return self.local(ts_ms).date()

    def _at(self, day: date, t: time) -> int:
        return int(datetime.combine(day, t, tzinfo=self.tz).timestamp() * 1000)

    def session_open(self, day: date) -> int:
        return self._at(day, self.start)

    def session_close(self, day: date) -> int:
        return self._at(day, self.end)

    def in_window(self, ts_ms: int) -> bool:
        """True for start <= local time < end."""
        return self.start <= self.local(ts_ms).time() < self.end

    def entries_allowed(self, ts_ms: int) -> bool:
        """A decision made at ts_ms may open a trade (inside the window, before the entry cutoff)."""
        if not self.in_window(ts_ms):
            return False
        close = self.session_close(self.local_date(ts_ms))
        return ts_ms < close - self.entry_cutoff_minutes * 60_000

    def next_open(self, ts_ms: int) -> int:
        """Start of the next session strictly after ts_ms (today's if it has not opened yet)."""
        day = self.local_date(ts_ms)
        opening = self.session_open(day)
        return opening if ts_ms < opening else self.session_open(day + timedelta(days=1))

    def day_number(self, start_ts_ms: int, ts_ms: int) -> int:
        """1-based experiment day by local calendar date (day 1 = the date the experiment started)."""
        return (self.local_date(ts_ms) - self.local_date(start_ts_ms)).days + 1

    def experiment_end(self, start_ts_ms: int, days: int) -> int:
        """Close of the session on experiment day `days`."""
        return self.session_close(self.local_date(start_ts_ms) + timedelta(days=days - 1))

    # ---- vectorised helpers (used by the feature pipeline) -------------
    def _local_index(self, ts_ms: np.ndarray) -> pd.DatetimeIndex:
        return pd.to_datetime(np.asarray(ts_ms, dtype="int64"), unit="ms", utc=True).tz_convert(self.tz)

    def local_minutes(self, ts_ms: np.ndarray) -> np.ndarray:
        idx = self._local_index(ts_ms)
        return np.asarray(idx.hour * 60 + idx.minute, dtype=np.int64)

    def local_day_ordinal(self, ts_ms: np.ndarray) -> np.ndarray:
        """Days since 1970-01-01 of the local calendar date (monotonic id for 'which session day')."""
        naive = self._local_index(ts_ms).tz_localize(None).as_unit("ms")
        return np.asarray(naive.asi8 // DAY_MS, dtype=np.int64)

    @property
    def start_minute(self) -> int:
        return self.start.hour * 60 + self.start.minute

    @property
    def end_minute(self) -> int:
        return self.end.hour * 60 + self.end.minute
