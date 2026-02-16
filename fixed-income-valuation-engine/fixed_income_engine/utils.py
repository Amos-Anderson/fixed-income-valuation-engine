from __future__ import annotations

import pandas as pd
from typing import List, Tuple
from functools import lru_cache


def yearfrac(start: pd.Timestamp, end: pd.Timestamp, convention: str) -> float:
    """
    Year fraction between two dates under a day count convention.

    Supported:
    - ACT/365, ACT/365F
    - ACT/360
    - 30/360, 30/360US (US bond basis)
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    convention = convention.upper().replace(" ", "")
    if end < start:
        raise ValueError(f"end < start: {start=} {end=}")

    if convention in ("ACT/365", "ACT/365F"):
        return (end - start).days / 365.0

    if convention == "ACT/360":
        return (end - start).days / 360.0

    if convention in ("30/360", "30/360US"):
        y1, m1, d1 = start.year, start.month, start.day
        y2, m2, d2 = end.year, end.month, end.day

        # 30/360 US convention
        if d1 == 31:
            d1 = 30
        if d2 == 31 and d1 == 30:
            d2 = 30

        return ((y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)) / 360.0

    raise ValueError(f"Unsupported day count convention: {convention}")


def semiannual_coupon_dates(val_date: pd.Timestamp, maturity: pd.Timestamp) -> List[pd.Timestamp]:
    """
    Generate semiannual coupon payment dates strictly AFTER val_date, ending at maturity.

    Avoids accidentally including a coupon date very close to val_date due to naive backward stepping.
    """
    val_date = pd.Timestamp(val_date)
    maturity = pd.Timestamp(maturity)

    if maturity <= val_date:
        raise ValueError("Maturity must be after valuation date.")

    d = maturity
    while d > val_date:
        d = d - pd.DateOffset(months=6)

    prev_coupon = d  # <= val_date
    next_coupon = prev_coupon + pd.DateOffset(months=6)

    dates: List[pd.Timestamp] = []
    d = next_coupon
    while d < maturity:
        dates.append(d)
        d = d + pd.DateOffset(months=6)

    dates.append(maturity)
    return [x for x in dates if x > val_date]


def settlement_date(val_date: pd.Timestamp, lag_days: int = 2) -> pd.Timestamp:
    """
    Simplified settlement date: valuation date + lag_days (calendar days).
    In production: use business-day calendars + holiday schedules.
    """
    return pd.Timestamp(val_date) + pd.Timedelta(days=lag_days)


def previous_coupon_date(settle: pd.Timestamp, maturity: pd.Timestamp, freq: int = 2) -> pd.Timestamp:
    """
    Most recent coupon date on or before settlement, schedule anchored at maturity.
    Pragmatic approximation when issue date is unknown.
    """
    if freq <= 0:
        raise ValueError("freq must be positive")

    months = int(12 / freq)
    d = pd.Timestamp(maturity)
    settle = pd.Timestamp(settle)

    while d > settle:
        d = d - pd.DateOffset(months=months)
    return d


def next_coupon_date(settle: pd.Timestamp, maturity: pd.Timestamp, freq: int = 2) -> pd.Timestamp:
    """Next coupon date strictly after settlement."""
    prev = previous_coupon_date(settle, maturity, freq)
    months = int(12 / freq)
    return prev + pd.DateOffset(months=months)


def coupon_schedule_after_settlement(
    settle: pd.Timestamp,
    maturity: pd.Timestamp,
    freq: int = 2,
) -> List[pd.Timestamp]:
    """All coupon payment dates strictly after settlement, ending at maturity."""
    settle = pd.Timestamp(settle)
    maturity = pd.Timestamp(maturity)

    if settle >= maturity:
        return []

    months = int(12 / freq)
    d = next_coupon_date(settle, maturity, freq)

    dates: List[pd.Timestamp] = []
    while d < maturity:
        dates.append(d)
        d = d + pd.DateOffset(months=months)

    dates.append(maturity)
    return dates


def accrued_interest(
    settle: pd.Timestamp,
    maturity: pd.Timestamp,
    coupon_rate: float,
    face: float,
    freq: int,
    day_count: str,
) -> float:
    """
    Accrued interest in currency units (not per 100).
    """
    settle = pd.Timestamp(settle)
    maturity = pd.Timestamp(maturity)

    if settle >= maturity:
        return 0.0

    t_prev = previous_coupon_date(settle, maturity, freq)
    t_next = next_coupon_date(settle, maturity, freq)

    accrual_num = yearfrac(t_prev, settle, day_count)
    accrual_den = yearfrac(t_prev, t_next, day_count)

    if accrual_den <= 0:
        raise ValueError("Invalid coupon period length from schedule/daycount.")

    coupon_per_period = face * (coupon_rate / freq)
    return coupon_per_period * (accrual_num / accrual_den)


@lru_cache(maxsize=100_000)
def cached_schedule(settle: pd.Timestamp, maturity: pd.Timestamp, freq: int) -> Tuple[pd.Timestamp, ...]:
    """Cache schedules by (settle, maturity, freq)."""
    dates = coupon_schedule_after_settlement(pd.Timestamp(settle), pd.Timestamp(maturity), int(freq))
    return tuple(dates)
