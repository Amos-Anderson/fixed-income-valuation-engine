from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple

from .curves import (
    ZeroCurve,
    curve_from_shifted_zeros,
    parallel_shift_bp,
    steepener_shift_bp,
    flattener_shift_bp,
)
from .portfolio import price_portfolio_vectorized, price_corporate_portfolio_vectorized


def run_rate_scenarios(
    curve: ZeroCurve,
    portfolio: pd.DataFrame,
    val_date: pd.Timestamp,
    settle: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = price_portfolio_vectorized(curve, portfolio, val_date, settle)[["bond_id", "dirty"]].rename(columns={"dirty": "base"})

    scenarios = {
        "PAR_-50bp": curve_from_shifted_zeros(curve, parallel_shift_bp(-50)),
        "PAR_-25bp": curve_from_shifted_zeros(curve, parallel_shift_bp(-25)),
        "PAR_+25bp": curve_from_shifted_zeros(curve, parallel_shift_bp(+25)),
        "PAR_+50bp": curve_from_shifted_zeros(curve, parallel_shift_bp(+50)),
        "STEEPENER_25bp": curve_from_shifted_zeros(curve, steepener_shift_bp(25)),
        "FLATTENER_25bp": curve_from_shifted_zeros(curve, flattener_shift_bp(25)),
    }

    per_bond = base.copy()
    for name, scurve in scenarios.items():
        px = price_portfolio_vectorized(scurve, portfolio, val_date, settle)[["bond_id", "dirty"]].rename(columns={"dirty": name})
        per_bond = per_bond.merge(px, on="bond_id", how="left")
        per_bond[name + "_PnL"] = per_bond[name] - per_bond["base"]

    pnl_cols = [c for c in per_bond.columns if c.endswith("_PnL")]
    summary = pd.DataFrame({"scenario": pnl_cols, "total_pnl_per_100_notional": [per_bond[c].sum() for c in pnl_cols]})

    return per_bond, summary


def run_spread_scenarios(
    curve: ZeroCurve,
    corp_portfolio: pd.DataFrame,
    val_date: pd.Timestamp,
    settle: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = price_corporate_portfolio_vectorized(curve, corp_portfolio, val_date, settle)[["bond_id", "dirty"]].rename(columns={"dirty": "base"})

    scenarios = {"SPR_+25bp": 25 / 10000.0, "SPR_+100bp": 100 / 10000.0}

    per_bond = base.copy()
    for name, bump in scenarios.items():
        shocked_port = corp_portfolio.copy()
        shocked_port["spread"] = shocked_port["spread"] + bump

        shocked = price_corporate_portfolio_vectorized(curve, shocked_port, val_date, settle)[["bond_id", "dirty"]].rename(columns={"dirty": name})
        per_bond = per_bond.merge(shocked, on="bond_id")
        per_bond[name + "_PnL"] = per_bond[name] - per_bond["base"]

    summary = pd.DataFrame(
        {"scenario": [c for c in per_bond.columns if c.endswith("_PnL")],
         "total_pnl_per_100_notional": [per_bond[c].sum() for c in per_bond.columns if c.endswith("_PnL")]}
    )
    return per_bond, summary


def run_combined_rate_spread_scenarios(
    base_curve: ZeroCurve,
    corp_portfolio: pd.DataFrame,
    val_date: pd.Timestamp,
    settle: pd.Timestamp,
) -> pd.DataFrame:
    base_prices = price_corporate_portfolio_vectorized(base_curve, corp_portfolio, val_date, settle)
    base_total = base_prices["dirty"].sum()

    rate_shocks = [-50, -25, 0, 25, 50]
    spread_shocks = [0, 25, 100]

    rows = []
    for r_bp in rate_shocks:
        scurve = curve_from_shifted_zeros(base_curve, parallel_shift_bp(r_bp))

        for s_bp in spread_shocks:
            shocked_port = corp_portfolio.copy()
            shocked_port["spread"] = shocked_port["spread"] + s_bp / 10000.0

            shocked_prices = price_corporate_portfolio_vectorized(scurve, shocked_port, val_date, settle)
            shocked_total = shocked_prices["dirty"].sum()

            rows.append(
                {
                    "rate_shock_bp": r_bp,
                    "spread_shock_bp": s_bp,
                    "total_dirty_base": base_total,
                    "total_dirty_shocked": shocked_total,
                    "total_pnl_per_100_notional": shocked_total - base_total,
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["rate_shock_bp", "spread_shock_bp"]).reset_index(drop=True)
