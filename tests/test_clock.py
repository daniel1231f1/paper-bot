"""IST schedule logic: 07:00-23:00 Asia/Kolkata, converted from UTC, no daylight saving."""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np

from bot.clock import SessionClock
from bot.config import ScheduleCfg
from conftest import ist

CLOCK = SessionClock.from_cfg(ScheduleCfg())


def utc(y, m, d, hh, mm) -> int:
    return int(datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp() * 1000)


def test_window_boundaries_in_utc():
    # 07:00 IST == 01:30 UTC ; 23:00 IST == 17:30 UTC
    assert not CLOCK.in_window(utc(2026, 11, 12, 1, 29))
    assert CLOCK.in_window(utc(2026, 11, 12, 1, 30))
    assert CLOCK.in_window(utc(2026, 11, 12, 17, 29))
    assert not CLOCK.in_window(utc(2026, 11, 12, 17, 30))  # end is exclusive
    assert not CLOCK.in_window(utc(2026, 11, 12, 20, 0))  # 01:30 IST next day


def test_utc_conversion_matches_ist_wall_clock():
    assert ist(2026, 11, 12, 7, 0) == utc(2026, 11, 12, 1, 30)
    assert CLOCK.local(utc(2026, 11, 11, 18, 31)).date() == date(2026, 11, 12)  # 00:01 IST is already tomorrow


def test_no_daylight_saving_shift():
    for month in (1, 3, 6, 10, 11):  # months where DST zones change their offset
        assert CLOCK.in_window(utc(2026, month, 15, 1, 30))
        assert not CLOCK.in_window(utc(2026, month, 15, 1, 29))
        assert CLOCK.local(utc(2026, month, 15, 0, 0)).utcoffset().total_seconds() == 5.5 * 3600


def test_entry_cutoff_before_close():
    assert CLOCK.entries_allowed(ist(2026, 11, 12, 21, 59))
    assert not CLOCK.entries_allowed(ist(2026, 11, 12, 22, 0))  # last hour: manage positions only
    assert not CLOCK.entries_allowed(ist(2026, 11, 12, 6, 59))
    assert CLOCK.entries_allowed(ist(2026, 11, 12, 7, 0))


def test_next_open():
    assert CLOCK.next_open(ist(2026, 11, 12, 3, 0)) == ist(2026, 11, 12, 7, 0)
    assert CLOCK.next_open(ist(2026, 11, 12, 7, 0)) == ist(2026, 11, 13, 7, 0)
    assert CLOCK.next_open(ist(2026, 11, 12, 23, 30)) == ist(2026, 11, 13, 7, 0)


def test_day_counter_uses_ist_calendar_days():
    start = ist(2026, 11, 11, 14, 0)
    assert CLOCK.day_number(start, start) == 1
    assert CLOCK.day_number(start, ist(2026, 11, 11, 23, 59)) == 1
    assert CLOCK.day_number(start, ist(2026, 11, 12, 0, 0)) == 2
    assert CLOCK.day_number(start, ist(2026, 11, 20, 22, 0)) == 10


def test_experiment_ends_at_close_of_day_ten():
    start = ist(2026, 11, 11, 14, 0)
    assert CLOCK.experiment_end(start, 10) == ist(2026, 11, 20, 23, 0)


def test_vectorised_helpers_agree_with_scalar():
    ts = np.array([ist(2026, 11, 12, 6, 55), ist(2026, 11, 12, 7, 0), ist(2026, 11, 12, 23, 0), ist(2026, 11, 13, 0, 5)])
    assert list(CLOCK.local_minutes(ts)) == [415, 420, 1380, 5]
    days = CLOCK.local_day_ordinal(ts)
    assert days[0] == days[1] == days[2] and days[3] == days[0] + 1
