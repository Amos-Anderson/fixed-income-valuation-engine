from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List

from .curves import ZeroCurve
from .utils import (
    settlement_date,
    coupon_schedule_after_settlement,
    accrued_interest,
    yearfrac,
)


@dataclass(frozen=True)
class Bond:
    bond_id: str
    maturity: pd.Timestamp
    coupon_rate: float
    freq: int = 2
    day_count: str = "30/360"
    face: float = 100.0


@dataclass(frozen=True)
class CorporateBond(Bond):
    spread: float = 0.0  # decimal, e.g. 0.015 = 150bp


def price_bond_dirty_clean(
    curve: ZeroCurve,
    bond: Bond,
    val_date: pd.Timestamp,
    settle: Optional[pd.Timestamp] = None,
) -> Tuple[float, float, float]:
    """
    Returns (dirty_per_100, clean_per_100, accrued_per_100).
    Settlement-adjusted discounting: D_s(T)=D(T)/D(s).
    """
    if settle is None:
        settle = settlement_date(val_date, lag_days=2)

    settle = pd.Timestamp(settle)
    maturity = pd.Timestamp(bond.maturity)

    if settle >= maturity:
        raise ValueError(f"Bond {bond.bond_id} settles on/after maturity.")

    if bond.freq not in (1, 2, 4):
        raise NotImplementedError("Supported frequencies: 1, 2, 4.")

    pay_dates = coupon_schedule_after_settlement(settle, maturity, bond.freq)
    if len(pay_dates) == 0:
        raise ValueError("No remaining cashflows after settlement.")

    coupon_cf = bond.face * (bond.coupon_rate / bond.freq)
    cfs = np.full(len(pay_dates), coupon_cf, dtype=float)
    cfs[-1] += bond.face

    df_pay = curve.df(pay_dates)
    df_settle = float(curve.df([settle])[0])
    dfs = df_pay / df_settle

    pv = float(np.sum(cfs * dfs))
    dirty = 100.0 * pv / bond.face

    ai_amt = accrued_interest(settle, maturity, bond.coupon_rate, bond.face, bond.freq, bond.day_count)
    ai = 100.0 * ai_amt / bond.face

    clean = dirty - ai
    return dirty, clean, ai


def price_corporate_bond_dirty_clean(
    curve: ZeroCurve,
    bond: CorporateBond,
    val_date: pd.Timestamp,
    settle: Optional[pd.Timestamp] = None,
) -> Tuple[float, float, float]:
    """
    Corporate bond pricing with constant spread applied to ALL cashflows:
      D_corp(t) = D(t)/D(settle) * exp(-spread * tau(settle,t))
    """
    if settle is None:
        settle = settlement_date(val_date, lag_days=2)

    settle = pd.Timestamp(settle)
    if settle >= bond.maturity:
        raise ValueError(f"{bond.bond_id}: matured at settlement.")

    pay_dates = coupon_schedule_after_settlement(settle, bond.maturity, bond.freq)
    coupon_cf = bond.face * (bond.coupon_rate / bond.freq)

    cfs = np.full(len(pay_dates), coupon_cf, dtype=float)
    cfs[-1] += bond.face

    df_pay = curve.df(pay_dates)
    df_settle = float(curve.df([settle])[0])
    df_settle_adj = df_pay / df_settle

    taus_settle = np.array([yearfrac(settle, d, curve.zero_day_count) for d in pay_dates], dtype=float)
    spread_adj = np.exp(-bond.spread * taus_settle)

    pv = float(np.sum(cfs * df_settle_adj * spread_adj))
    dirty = 100.0 * pv / bond.face

    ai_amt = accrued_interest(settle, bond.maturity, bond.coupon_rate, bond.face, bond.freq, bond.day_count)
    ai = 100.0 * ai_amt / bond.face
    clean = dirty - ai

    return dirty, clean, ai


class BondPricer:
    def __init__(self, curve: ZeroCurve):
        self.curve = curve

    def validate(self, bond: Bond, settle: pd.Timestamp) -> None:
        if settle >= bond.maturity:
            raise ValueError(f"{bond.bond_id}: matured at settlement.")
        if bond.freq not in (1, 2, 4):
            raise NotImplementedError("Supported frequencies: 1, 2, 4.")
        if not (-0.01 <= bond.coupon_rate <= 0.25):
            raise ValueError(f"{bond.bond_id}: coupon out of plausible range.")

        dates = coupon_schedule_after_settlement(settle, bond.maturity, bond.freq)
        if any(dates[i] >= dates[i + 1] for i in range(len(dates) - 1)):
            raise ValueError(f"{bond.bond_id}: non-increasing schedule.")

    def price(self, bond: Bond, val_date: pd.Timestamp, settle: Optional[pd.Timestamp] = None):
        if settle is None:
            settle = settlement_date(val_date, 2)

        settle = pd.Timestamp(settle)
        self.validate(bond, settle)

        return price_bond_dirty_clean(self.curve, bond, val_date, settle)
