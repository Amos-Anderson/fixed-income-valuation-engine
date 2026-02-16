from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

from .curves import ZeroCurve
from .bonds import Bond
from .utils import cached_schedule, accrued_interest


def qc_flags_for_bond(bond: Bond, settle: pd.Timestamp) -> List[str]:
    flags: List[str] = []

    if pd.Timestamp(settle) >= pd.Timestamp(bond.maturity):
        flags.append("MATURED")

    if bond.freq not in (1, 2, 4):
        flags.append("BAD_FREQ")

    if bond.coupon_rate < -0.01 or bond.coupon_rate > 0.25:
        flags.append("BAD_COUPON")

    return flags


def build_cashflow_table(portfolio: pd.DataFrame, settle: pd.Timestamp) -> pd.DataFrame:
    rows = []
    settle = pd.Timestamp(settle)

    for _, r in portfolio.iterrows():
        bond_id = str(r["bond_id"])
        maturity = pd.Timestamp(r["maturity"])
        c = float(r["coupon_rate"])
        freq = int(r["freq"])
        face = float(r["face"])
        dc = str(r.get("day_count", "30/360"))

        if settle >= maturity:
            continue

        pay_dates = cached_schedule(settle, maturity, freq)
        if len(pay_dates) == 0:
            continue

        coupon_cf = face * (c / freq)
        for i, d in enumerate(pay_dates):
            cf = coupon_cf
            if i == len(pay_dates) - 1:
                cf += face
            rows.append((bond_id, maturity, c, freq, dc, face, pd.Timestamp(d), cf))

    return pd.DataFrame(
        rows,
        columns=["bond_id", "maturity", "coupon_rate", "freq", "day_count", "face", "pay_date", "cashflow"],
    )


def price_portfolio_vectorized(curve: ZeroCurve, portfolio: pd.DataFrame, val_date: pd.Timestamp, settle: pd.Timestamp) -> pd.DataFrame:
    settle = pd.Timestamp(settle)

    cf = build_cashflow_table(portfolio, settle)
    if cf.empty:
        raise ValueError("Cashflow table is empty. Check portfolio or settlement/maturity logic.")

    unique_dates = sorted(cf["pay_date"].unique())
    df_pay = curve.df(unique_dates)
    df_map = dict(zip(unique_dates, df_pay))

    df_settle = float(curve.df([settle])[0])

    cf["df_pay"] = cf["pay_date"].map(df_map).astype(float)
    cf["df_settle_adj"] = cf["df_pay"] / df_settle
    cf["pv_cf"] = cf["cashflow"] * cf["df_settle_adj"]

    pv_by_bond = cf.groupby("bond_id", as_index=False)["pv_cf"].sum().rename(columns={"pv_cf": "pv"})

    static_cols = ["bond_id", "maturity", "coupon_rate", "freq", "day_count", "face"]
    out = portfolio[static_cols].merge(pv_by_bond, on="bond_id", how="left")

    out["dirty"] = 100.0 * out["pv"] / out["face"]

    accrued_list = []
    flag_list = []
    for _, r in out.iterrows():
        bond = Bond(
            bond_id=str(r["bond_id"]),
            maturity=pd.Timestamp(r["maturity"]),
            coupon_rate=float(r["coupon_rate"]),
            freq=int(r["freq"]),
            day_count=str(r["day_count"]),
            face=float(r["face"]),
        )

        flags = qc_flags_for_bond(bond, settle)
        if "MATURED" in flags:
            accrued_list.append(np.nan)
        else:
            ai_amt = accrued_interest(settle, bond.maturity, bond.coupon_rate, bond.face, bond.freq, bond.day_count)
            accrued_list.append(100.0 * ai_amt / bond.face)

        flag_list.append("|".join(flags) if flags else "")

    out["accrued"] = accrued_list
    out["clean"] = out["dirty"] - out["accrued"]
    out["flags"] = flag_list

    return out


# ---------- Corporate Portfolio (spread column) ----------

def build_cashflow_table_corp(portfolio: pd.DataFrame, settle: pd.Timestamp) -> pd.DataFrame:
    rows = []
    settle = pd.Timestamp(settle)

    for _, r in portfolio.iterrows():
        bond_id = str(r["bond_id"])
        maturity = pd.Timestamp(r["maturity"])
        c = float(r["coupon_rate"])
        freq = int(r["freq"])
        face = float(r["face"])
        dc = str(r.get("day_count", "30/360"))
        spr = float(r["spread"])

        if settle >= maturity:
            continue

        pay_dates = cached_schedule(settle, maturity, freq)
        if len(pay_dates) == 0:
            continue

        coupon_cf = face * (c / freq)
        for i, d in enumerate(pay_dates):
            cf = coupon_cf
            if i == len(pay_dates) - 1:
                cf += face
            rows.append((bond_id, maturity, c, freq, dc, face, spr, pd.Timestamp(d), cf))

    return pd.DataFrame(
        rows,
        columns=["bond_id", "maturity", "coupon_rate", "freq", "day_count", "face", "spread", "pay_date", "cashflow"],
    )


def price_corporate_portfolio_vectorized(
    curve: ZeroCurve,
    portfolio: pd.DataFrame,
    val_date: pd.Timestamp,
    settle: pd.Timestamp,
) -> pd.DataFrame:
    from .utils import yearfrac  # local import to keep module boundaries clean

    settle = pd.Timestamp(settle)
    cf = build_cashflow_table_corp(portfolio, settle)
    if cf.empty:
        raise ValueError("Corporate cashflow table is empty.")

    unique_dates = sorted(cf["pay_date"].unique())
    df_pay = curve.df(unique_dates)
    df_map = dict(zip(unique_dates, df_pay))

    df_settle = float(curve.df([settle])[0])

    cf["df_pay"] = cf["pay_date"].map(df_map).astype(float)
    cf["df_settle_adj"] = cf["df_pay"] / df_settle
    cf["tau_settle"] = cf["pay_date"].apply(lambda d: yearfrac(settle, d, curve.zero_day_count)).astype(float)
    cf["spread_adj"] = np.exp(-cf["spread"].astype(float) * cf["tau_settle"])
    cf["pv_cf"] = cf["cashflow"] * cf["df_settle_adj"] * cf["spread_adj"]

    pv_by_bond = cf.groupby("bond_id", as_index=False)["pv_cf"].sum().rename(columns={"pv_cf": "pv"})

    static_cols = ["bond_id", "maturity", "coupon_rate", "freq", "day_count", "face", "spread"]
    out = portfolio[static_cols].merge(pv_by_bond, on="bond_id", how="left")
    out["dirty"] = 100.0 * out["pv"] / out["face"]

    accrued = []
    flags = []
    for _, r in out.iterrows():
        bond = Bond(
            bond_id=str(r["bond_id"]),
            maturity=pd.Timestamp(r["maturity"]),
            coupon_rate=float(r["coupon_rate"]),
            freq=int(r["freq"]),
            day_count=str(r["day_count"]),
            face=float(r["face"]),
        )

        f = qc_flags_for_bond(bond, settle)
        flags.append("|".join(f) if f else "")

        if "MATURED" in f:
            accrued.append(np.nan)
        else:
            ai_amt = accrued_interest(settle, bond.maturity, bond.coupon_rate, bond.face, bond.freq, bond.day_count)
            accrued.append(100.0 * ai_amt / bond.face)

    out["accrued"] = accrued
    out["clean"] = out["dirty"] - out["accrued"]
    out["flags"] = flags

    return out

def make_sample_portfolio(
    n: int = 20,
    val_date: pd.Timestamp | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Create a synthetic fixed-rate bond portfolio for demo/testing.

    - Maturities: integer years from val_date (1Y..10Y), snapped to Feb-15 cycle.
    - Coupons: uniform in [2%, 8%]
    - Frequency: semiannual
    - Day count: mostly 30/360 with some ACT/365
    - Face: 100

    Notes:
    - This is NOT "market realistic issuance". It exists purely to exercise the engine.
    - In production you would ingest real security master data + schedules.
    """
    if val_date is None:
        val_date = pd.Timestamp.today().normalize()
    val_date = pd.Timestamp(val_date)

    rng = np.random.default_rng(seed)

    years = rng.integers(1, 11, size=n)  # 1..10 years
    mats = [val_date + pd.DateOffset(years=int(y)) for y in years]
    mats = [pd.Timestamp(m).replace(month=2, day=15) for m in mats]  # align to Feb 15

    coupons = rng.uniform(0.02, 0.08, size=n)
    freqs = np.full(n, 2)  # semiannual baseline
    dcs = rng.choice(["30/360", "ACT/365"], size=n, p=[0.8, 0.2])

    return pd.DataFrame({
        "bond_id": [f"BOND_{i:03d}" for i in range(n)],
        "maturity": mats,
        "coupon_rate": coupons,
        "freq": freqs,
        "day_count": dcs,
        "face": 100.0,
    })


def add_random_spreads(
    portfolio: pd.DataFrame,
    seed: int = 11,
    min_bp: float = 30.0,
    max_bp: float = 250.0,
) -> pd.DataFrame:
    """
    Add a synthetic constant spread column (decimal) to a portfolio DataFrame.

    Default: IG-ish spreads between 30bp and 250bp.
    """
    rng = np.random.default_rng(seed)
    out = portfolio.copy()
    out["spread"] = rng.uniform(min_bp, max_bp, size=len(out)) / 10000.0
    return out

