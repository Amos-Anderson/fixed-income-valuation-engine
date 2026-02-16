from __future__ import annotations

import numpy as np
import pandas as pd

from .curves import ZeroCurve, shocked_curve_parallel, curve_from_shifted_zeros, parallel_shift_bp
from .portfolio import price_portfolio_vectorized, price_corporate_portfolio_vectorized


def compute_portfolio_dv01(curve: ZeroCurve, portfolio: pd.DataFrame, val_date: pd.Timestamp, settle: pd.Timestamp) -> pd.DataFrame:
    base = price_portfolio_vectorized(curve, portfolio, val_date, settle)

    shocked = shocked_curve_parallel(curve, shift_bp=1.0)
    shocked_prices = price_portfolio_vectorized(shocked, portfolio, val_date, settle)

    out = base[["bond_id", "dirty"]].merge(
        shocked_prices[["bond_id", "dirty"]],
        on="bond_id",
        suffixes=("_base", "_up1bp"),
    )

    out["dv01"] = out["dirty_up1bp"] - out["dirty_base"]
    return out


def compute_portfolio_convexity(curve: ZeroCurve, portfolio: pd.DataFrame, val_date: pd.Timestamp, settle: pd.Timestamp) -> pd.DataFrame:
    base = price_portfolio_vectorized(curve, portfolio, val_date, settle)
    up = shocked_curve_parallel(curve, shift_bp=1.0)
    down = shocked_curve_parallel(curve, shift_bp=-1.0)

    price_up = price_portfolio_vectorized(up, portfolio, val_date, settle)
    price_down = price_portfolio_vectorized(down, portfolio, val_date, settle)

    out = base[["bond_id", "dirty"]].merge(
        price_up[["bond_id", "dirty"]], on="bond_id", suffixes=("_base", "_up")
    ).merge(
        price_down[["bond_id", "dirty"]], on="bond_id"
    )
    out = out.rename(columns={"dirty": "dirty_down"})

    h = 0.0001
    out["convexity"] = (out["dirty_up"] + out["dirty_down"] - 2 * out["dirty_base"]) / (out["dirty_base"] * h**2)
    return out[["bond_id", "convexity"]]


def duration_from_dv01(dv01_df: pd.DataFrame) -> pd.DataFrame:
    df = dv01_df.copy()
    df["mod_duration"] = -df["dv01"] / (df["dirty_base"] * 0.0001)
    return df[["bond_id", "mod_duration"]]


# ---- KRD hat basis reconciliation ----

def hat_basis(taus: np.ndarray, k: int):
    n = len(taus)
    if not (0 <= k < n):
        raise ValueError("k out of range")

    def b(tau: float) -> float:
        if k == 0:
            if tau <= taus[0]:
                return 1.0
            if tau >= taus[1]:
                return 0.0
            return (taus[1] - tau) / (taus[1] - taus[0])

        if k == n - 1:
            if tau <= taus[n - 2]:
                return 0.0
            if tau >= taus[n - 1]:
                return 1.0
            return (tau - taus[n - 2]) / (taus[n - 1] - taus[n - 2])

        if tau <= taus[k - 1] or tau >= taus[k + 1]:
            return 0.0
        if tau <= taus[k]:
            return (tau - taus[k - 1]) / (taus[k] - taus[k - 1])
        return (taus[k + 1] - tau) / (taus[k + 1] - taus[k])

    return b


def key_rate_curve_hat(curve: ZeroCurve, k: int, bp: float) -> ZeroCurve:
    from .utils import yearfrac

    s = bp / 10000.0
    dates = pd.to_datetime(curve.knot_dates)
    dfs = np.exp(curve.knot_log_dfs)

    taus = np.array([yearfrac(curve.val_date, d, curve.zero_day_count) for d in dates], dtype=float)
    zeros = -np.log(dfs) / taus

    b_k = hat_basis(taus, k)
    shifts = np.array([b_k(t) for t in taus], dtype=float) * s

    zeros_shifted = zeros + shifts
    dfs_shifted = np.exp(-zeros_shifted * taus)
    logdfs_shifted = np.log(dfs_shifted)

    return ZeroCurve(curve.val_date, curve.knot_dates.copy(), logdfs_shifted, curve.zero_day_count)


def compute_krd_hat(curve: ZeroCurve, portfolio: pd.DataFrame, val_date: pd.Timestamp, settle: pd.Timestamp, bp: float = 1.0):
    base_total = price_portfolio_vectorized(curve, portfolio, val_date, settle)["dirty"].sum()

    bucket_pnl = []
    for k in range(len(curve.knot_dates)):
        scurve = key_rate_curve_hat(curve, k, bp)
        shocked_total = price_portfolio_vectorized(scurve, portfolio, val_date, settle)["dirty"].sum()
        bucket_pnl.append(shocked_total - base_total)

    bucket_pnl = np.array(bucket_pnl, dtype=float)

    par_curve = curve_from_shifted_zeros(curve, parallel_shift_bp(bp))
    par_total = price_portfolio_vectorized(par_curve, portfolio, val_date, settle)["dirty"].sum()
    par_dv01 = par_total - base_total

    return bucket_pnl, par_dv01


# ---- Spread risk ----

def portfolio_spread_dv01(curve: ZeroCurve, corp_portfolio: pd.DataFrame, val_date: pd.Timestamp, settle: pd.Timestamp) -> float:
    base = price_corporate_portfolio_vectorized(curve, corp_portfolio, val_date, settle)["dirty"].sum()
    bumped = corp_portfolio.copy()
    bumped["spread"] = bumped["spread"] + 1 / 10000.0
    shocked = price_corporate_portfolio_vectorized(curve, bumped, val_date, settle)["dirty"].sum()
    return shocked - base


def compute_spread_dv01_per_bond(curve: ZeroCurve, corp_portfolio: pd.DataFrame, val_date: pd.Timestamp, settle: pd.Timestamp) -> pd.DataFrame:
    base = price_corporate_portfolio_vectorized(curve, corp_portfolio, val_date, settle)[["bond_id", "dirty"]].rename(columns={"dirty": "base"})
    bumped = corp_portfolio.copy()
    bumped["spread"] = bumped["spread"] + 1 / 10000.0
    shocked = price_corporate_portfolio_vectorized(curve, bumped, val_date, settle)[["bond_id", "dirty"]].rename(columns={"dirty": "shocked"})
    tmp = base.merge(shocked, on="bond_id")
    tmp["spread_dv01"] = tmp["shocked"] - tmp["base"]
    return tmp
