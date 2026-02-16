from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Optional, Dict

from scipy.optimize import brentq

from .utils import yearfrac, semiannual_coupon_dates


@dataclass(frozen=True)
class ZeroCurve:
    """
    Discount curve represented by bootstrapped knot discount factors,
    interpolated linearly in log discount factor space.

    - Within knot range: log-linear interpolation on DF.
    - Short-end extrapolation: flat cc zero implied by first knot.
    - Long-end extrapolation: NOT allowed (raises).
    """
    val_date: pd.Timestamp
    knot_dates: np.ndarray          # dtype datetime64[ns]
    knot_log_dfs: np.ndarray        # log(D)
    zero_day_count: str = "ACT/365"

    def _to_datetime64(self, dates: Iterable[pd.Timestamp]) -> np.ndarray:
        return np.array([pd.Timestamp(d).to_datetime64() for d in dates], dtype="datetime64[ns]")

    def df(self, dates: Iterable[pd.Timestamp]) -> np.ndarray:
        dates_list = [pd.Timestamp(d) for d in dates]
        x = self._to_datetime64(dates_list).astype("datetime64[ns]").astype("int64")
        kx = self.knot_dates.astype("datetime64[ns]").astype("int64")
        kv = self.knot_log_dfs

        x_max = x.max()
        k_min, k_max = kx.min(), kx.max()

        if x_max > k_max:
            raise ValueError("Requested date beyond curve knot range (no long-end extrapolation).")

        # first knot implied flat cc zero for short-end extrapolation
        first_date = pd.Timestamp(pd.to_datetime(self.knot_dates[0]))
        df1 = float(np.exp(kv[0]))
        tau1 = yearfrac(self.val_date, first_date, self.zero_day_count)
        if tau1 <= 0:
            raise ValueError("First knot must be after valuation date.")
        z1 = -np.log(df1) / tau1

        out = np.empty_like(x, dtype=float)

        mask_short = x < k_min
        if np.any(mask_short):
            idxs = np.where(mask_short)[0]
            taus = np.array([yearfrac(self.val_date, dates_list[i], self.zero_day_count) for i in idxs], dtype=float)
            if np.any(taus < 0):
                raise ValueError("Requested date before valuation date.")
            out[mask_short] = np.exp(-z1 * taus)

        mask_in = ~mask_short
        if np.any(mask_in):
            log_df = np.interp(x[mask_in], kx, kv)
            out[mask_in] = np.exp(log_df)

        return out

    def zero_rate_cc(self, dates: Iterable[pd.Timestamp]) -> np.ndarray:
        dates_list = [pd.Timestamp(d) for d in dates]
        dfs = self.df(dates_list)

        taus = np.array([yearfrac(self.val_date, d, self.zero_day_count) for d in dates_list], dtype=float)
        if np.any(taus <= 0):
            raise ValueError("Non-positive tau encountered in zero rate computation.")

        return -np.log(dfs) / taus


def bootstrap_curve_from_bills_notes(
    market: pd.DataFrame,
    val_date: pd.Timestamp,
    zero_day_count: str = "ACT/365",
    enforce_monotone_df: bool = True,
) -> ZeroCurve:
    """
    Bootstrap a discount curve from:
      - Treasury bills quoted as simple yields (money-market)
      - Treasury notes quoted as par yields (semiannual)

    Interpolation is log-linear in discount factors.

    Notes are bootstrapped via a 1D root solve for log D(T), using:
      - knot interpolation for dates <= last knot
      - unknown-endpoint interpolation between (last_knot, logD_last) and (T, logD_T)
        for coupon dates in (last_knot, T].

    Returns
    -------
    ZeroCurve
    """
    market = market.sort_values("maturity").reset_index(drop=True)
    val_date = pd.Timestamp(val_date)

    knot_logdf: Dict[pd.Timestamp, float] = {}

    def _sorted_knots():
        dates = sorted(knot_logdf.keys())
        logdfs = np.array([knot_logdf[d] for d in dates], dtype=float)
        kx = np.array(
            [d.to_datetime64().astype("datetime64[ns]").astype("int64") for d in dates],
            dtype=np.int64,
        )
        return dates, logdfs, kx

    def current_curve() -> Optional[ZeroCurve]:
        dates, logdfs, _ = _sorted_knots()
        if len(dates) < 2:
            return None
        kd = np.array([d.to_datetime64() for d in dates], dtype="datetime64[ns]")
        return ZeroCurve(val_date, kd, logdfs, zero_day_count)

    # STEP A: Bills
    for _, row in market[market["type"] == "bill"].iterrows():
        T = pd.Timestamp(row["maturity"])
        y = float(row["quote"])
        dc = str(row.get("day_count", "ACT/360"))

        tau = yearfrac(val_date, T, dc)
        if tau <= 0:
            raise ValueError("Bill maturity must be after valuation date.")

        df_T = 1.0 / (1.0 + y * tau)
        if not (0.0 < df_T < 1.5):
            raise ValueError("Bill DF out of bounds.")

        knot_logdf[T] = float(np.log(df_T))

        if enforce_monotone_df:
            _, logdfs, _ = _sorted_knots()
            if np.any(np.diff(np.exp(logdfs)) > 1e-10):
                raise ValueError("Non-monotone DF detected in bills.")

    if len(knot_logdf) < 2:
        raise ValueError("Need at least two bills before notes.")

    # STEP B: Notes (root solve)
    for _, row in market[market["type"] == "note"].iterrows():
        T = pd.Timestamp(row["maturity"])
        c = float(row["quote"])
        freq = int(row.get("coupon_freq", 2))
        if freq != 2:
            raise NotImplementedError("Only semiannual supported in this phase.")

        curve = current_curve()
        if curve is None:
            raise ValueError("Curve not initialized.")

        cpn_dates = semiannual_coupon_dates(val_date, T)
        coupon = c / freq

        dates, logdfs, kx = _sorted_knots()
        kv = logdfs

        tL = dates[-1]
        if tL >= T:
            raise ValueError("Note maturity must exceed last knot.")

        logDF_L = knot_logdf[tL]
        tL_int = tL.to_datetime64().astype("datetime64[ns]").astype("int64")
        T_int = T.to_datetime64().astype("datetime64[ns]").astype("int64")

        def df_with_unknown_endpoint(date: pd.Timestamp, logDF_T: float) -> float:
            x = date.to_datetime64().astype("datetime64[ns]").astype("int64")

            # short-end extrapolation (between val_date and first knot)
            if x < kx.min():
                first_date = dates[0]
                df1 = float(np.exp(kv[0]))
                tau1 = yearfrac(val_date, first_date, zero_day_count)
                z1 = -np.log(df1) / tau1

                tau_x = yearfrac(val_date, date, zero_day_count)
                return float(np.exp(-z1 * tau_x))

            # within known knots
            if x <= tL_int:
                logDF = float(np.interp(x, kx, kv))
                return float(np.exp(logDF))

            # between (tL, T]
            if x <= T_int:
                w = (x - tL_int) / (T_int - tL_int)
                logDF = (1 - w) * logDF_L + w * logDF_T
                return float(np.exp(logDF))

            raise ValueError("Date beyond maturity during bootstrap.")

        def par_residual(logDF_T: float) -> float:
            pv = 0.0
            for d in cpn_dates:
                pv += coupon * df_with_unknown_endpoint(d, logDF_T)
            pv += df_with_unknown_endpoint(T, logDF_T)
            return pv - 1.0

        a = np.log(1e-6)
        b = min(logDF_L, -1e-12)

        fa, fb = par_residual(a), par_residual(b)
        if fa * fb > 0:
            raise ValueError("Root not bracketed — inconsistent market data.")

        logDF_T = brentq(par_residual, a, b, maxiter=300, xtol=1e-14)
        knot_logdf[T] = float(logDF_T)

        if enforce_monotone_df:
            _, logdfs2, _ = _sorted_knots()
            if np.any(np.diff(np.exp(logdfs2)) > 1e-10):
                raise ValueError("Non-monotone DF after adding note.")

    final_dates, final_logdfs, _ = _sorted_knots()
    kd = np.array([d.to_datetime64() for d in final_dates], dtype="datetime64[ns]")
    return ZeroCurve(val_date, kd, final_logdfs, zero_day_count)


def curve_qc_report(curve: ZeroCurve) -> pd.DataFrame:
    dates = pd.to_datetime(curve.knot_dates)
    dfs = np.exp(curve.knot_log_dfs)
    taus = np.array([yearfrac(curve.val_date, pd.Timestamp(d), curve.zero_day_count) for d in dates], dtype=float)
    zeros = -np.log(dfs) / taus

    return pd.DataFrame(
        {
            "date": dates,
            "tau": taus,
            "df": dfs,
            "zero_cc": zeros,
            "df_positive": dfs > 0,
            "df_monotone": np.r_[True, np.diff(dfs) <= 1e-10],
        }
    )


def shocked_curve_parallel(curve: ZeroCurve, shift_bp: float) -> ZeroCurve:
    """Parallel shift in continuously-compounded zero rates by shift_bp."""
    shift = shift_bp / 10000.0

    dates = pd.to_datetime(curve.knot_dates)
    dfs = np.exp(curve.knot_log_dfs)
    taus = np.array([yearfrac(curve.val_date, d, curve.zero_day_count) for d in dates], dtype=float)
    zeros = -np.log(dfs) / taus

    zeros_shifted = zeros + shift
    dfs_shifted = np.exp(-zeros_shifted * taus)
    logdfs_shifted = np.log(dfs_shifted)

    return ZeroCurve(curve.val_date, curve.knot_dates.copy(), logdfs_shifted, curve.zero_day_count)


def curve_from_shifted_zeros(curve: ZeroCurve, shift_func) -> ZeroCurve:
    """Build a new curve by shifting cc zeros z(t) by shift_func(tau) (decimal)."""
    dates = pd.to_datetime(curve.knot_dates)
    dfs = np.exp(curve.knot_log_dfs)

    taus = np.array([yearfrac(curve.val_date, d, curve.zero_day_count) for d in dates], dtype=float)
    zeros = -np.log(dfs) / taus

    shifts = np.array([shift_func(t) for t in taus], dtype=float)
    zeros_shifted = zeros + shifts

    dfs_shifted = np.exp(-zeros_shifted * taus)
    logdfs_shifted = np.log(dfs_shifted)

    return ZeroCurve(curve.val_date, curve.knot_dates.copy(), logdfs_shifted, curve.zero_day_count)


def parallel_shift_bp(bp: float):
    s = bp / 10000.0
    return lambda tau: s


def steepener_shift_bp(bp: float, pivot: float = 2.0, long: float = 10.0):
    A = bp / 10000.0

    def f(tau: float) -> float:
        if tau <= pivot:
            return +A
        if tau >= long:
            return -A
        w = (tau - pivot) / (long - pivot)
        return (1 - w) * (+A) + w * (-A)

    return f


def flattener_shift_bp(bp: float, pivot: float = 2.0, long: float = 10.0):
    A = bp / 10000.0

    def f(tau: float) -> float:
        if tau <= pivot:
            return -A
        if tau >= long:
            return +A
        w = (tau - pivot) / (long - pivot)
        return (1 - w) * (-A) + w * (+A)

    return f
