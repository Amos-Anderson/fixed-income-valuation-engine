import numpy as np
import pandas as pd
import pytest

from fixed_income_engine.curves import (
    bootstrap_curve_from_bills_notes,
    curve_from_shifted_zeros,
    parallel_shift_bp,
)
from fixed_income_engine.portfolio import (
    price_portfolio_vectorized,
    price_corporate_portfolio_vectorized,
)
from fixed_income_engine.risk import (
    compute_portfolio_dv01,
    compute_portfolio_convexity,
    duration_from_dv01,
    compute_krd_hat,
    portfolio_spread_dv01,
    compute_spread_dv01_per_bond,
)
from fixed_income_engine.scenarios import run_combined_rate_spread_scenarios


@pytest.fixture(scope="module")
def val_date():
    return pd.Timestamp("2026-02-13")


@pytest.fixture(scope="module")
def settle():
    # Use a non-coupon day to keep accrued > 0 for most names
    return pd.Timestamp("2026-02-16")


@pytest.fixture(scope="module")
def market_df():
    market = pd.DataFrame(
        [
            {"type": "bill", "maturity": pd.Timestamp("2026-02-20"), "quote": 0.0525, "day_count": "ACT/360"},
            {"type": "bill", "maturity": pd.Timestamp("2026-03-13"), "quote": 0.0520, "day_count": "ACT/360"},
            {"type": "bill", "maturity": pd.Timestamp("2026-05-15"), "quote": 0.0515, "day_count": "ACT/360"},
            {"type": "bill", "maturity": pd.Timestamp("2026-08-14"), "quote": 0.0505, "day_count": "ACT/360"},
            {"type": "bill", "maturity": pd.Timestamp("2027-02-12"), "quote": 0.0485, "day_count": "ACT/360"},
            {"type": "note", "maturity": pd.Timestamp("2028-02-15"), "quote": 0.0450, "coupon_freq": 2, "day_count": "30/360"},
            {"type": "note", "maturity": pd.Timestamp("2031-02-15"), "quote": 0.0430, "coupon_freq": 2, "day_count": "30/360"},
            {"type": "note", "maturity": pd.Timestamp("2036-02-15"), "quote": 0.0425, "coupon_freq": 2, "day_count": "30/360"},
        ]
    )
    return market.sort_values("maturity").reset_index(drop=True)


@pytest.fixture(scope="module")
def curve(val_date, market_df):
    return bootstrap_curve_from_bills_notes(market_df, val_date, zero_day_count="ACT/365", enforce_monotone_df=True)


@pytest.fixture(scope="module")
def portfolio_df(val_date):
    """
    Deterministic mini-portfolio (10 names) to test DV01/convexity/KRD.
    """
    maturities = [
        pd.Timestamp("2027-02-15"),
        pd.Timestamp("2028-02-15"),
        pd.Timestamp("2029-02-15"),
        pd.Timestamp("2030-02-15"),
        pd.Timestamp("2031-02-15"),
        pd.Timestamp("2032-02-15"),
        pd.Timestamp("2033-02-15"),
        pd.Timestamp("2034-02-15"),
        pd.Timestamp("2035-02-15"),
        pd.Timestamp("2036-02-15"),
    ]
    coupons = [0.04, 0.045, 0.05, 0.055, 0.06, 0.035, 0.065, 0.07, 0.03, 0.075]
    dcs = ["30/360"] * 9 + ["ACT/365"]

    df = pd.DataFrame(
        {
            "bond_id": [f"BOND_{i:03d}" for i in range(len(maturities))],
            "maturity": maturities,
            "coupon_rate": coupons,
            "freq": [2] * len(maturities),
            "day_count": dcs,
            "face": [100.0] * len(maturities),
        }
    )
    return df


@pytest.fixture(scope="module")
def corp_portfolio_df(portfolio_df):
    """
    Add deterministic spreads.
    """
    out = portfolio_df.copy()
    spreads_bp = [50, 80, 120, 150, 200, 60, 90, 110, 140, 170]
    out["spread"] = np.array(spreads_bp, dtype=float) / 10000.0
    return out


def test_portfolio_pricing_outputs(curve, portfolio_df, val_date, settle):
    priced = price_portfolio_vectorized(curve, portfolio_df, val_date, settle)
    assert {"bond_id", "dirty", "clean", "accrued", "flags"}.issubset(priced.columns)
    assert priced["dirty"].notna().all()
    assert priced["clean"].notna().all()
    assert np.isfinite(priced["dirty"]).all()


def test_rate_dv01_sign_sanity(curve, portfolio_df, val_date, settle):
    """
    For a standard positive-duration bond portfolio:
    +1bp rate shock => dirty price should go down => DV01 typically negative.
    """
    dv01 = compute_portfolio_dv01(curve, portfolio_df, val_date, settle)
    assert dv01["dv01"].mean() < 0.0, "Average DV01 should be negative for long-only bond portfolio"


def test_convexity_positive_sanity(curve, portfolio_df, val_date, settle):
    conv = compute_portfolio_convexity(curve, portfolio_df, val_date, settle)
    # Most vanilla bonds should have positive convexity; allow small noise
    assert (conv["convexity"] > -1e-6).mean() > 0.8, "Convexity should be mostly positive"


def test_duration_from_dv01_consistency(curve, portfolio_df, val_date, settle):
    dv01 = compute_portfolio_dv01(curve, portfolio_df, val_date, settle)
    dur = duration_from_dv01(dv01)
    assert dur["mod_duration"].notna().all()
    assert (dur["mod_duration"] > 0.0).mean() > 0.8, "Most durations should be positive"


def test_krd_hat_reconciles_parallel(curve, portfolio_df, val_date, settle):
    """
    Key rate bucket sum should approximate parallel DV01 (same bp),
    because hat basis shocks form a partition of unity over the grid (approximately).
    """
    bucket_pnl, par_dv01 = compute_krd_hat(curve, portfolio_df, val_date, settle, bp=1.0)
    diff = float(bucket_pnl.sum() - par_dv01)
    assert abs(diff) < 1e-3, f"KRD bucket sum should reconcile to parallel DV01 (diff={diff})"


def test_corporate_spread_dv01_negative(curve, corp_portfolio_df, val_date, settle):
    """
    +1bp spread bump => price down => spread DV01 negative.
    """
    psdv01 = portfolio_spread_dv01(curve, corp_portfolio_df, val_date, settle)
    assert psdv01 < 0.0, "Portfolio spread DV01 should be negative"


def test_spread_dv01_per_bond_matches_portfolio_total(curve, corp_portfolio_df, val_date, settle):
    per_bond = compute_spread_dv01_per_bond(curve, corp_portfolio_df, val_date, settle)
    total = per_bond["spread_dv01"].sum()
    psdv01 = portfolio_spread_dv01(curve, corp_portfolio_df, val_date, settle)
    assert abs(total - psdv01) < 1e-8, "Sum of per-bond spread DV01 should match portfolio spread DV01"


def test_combined_rate_spread_scenario_grid_monotone(curve, corp_portfolio_df, val_date, settle):
    """
    Sanity: higher rate shocks and higher spread shocks should worsen PnL (more negative),
    at least along monotone directions. This is not a theorem but should hold for a typical long-only book.
    """
    combo = run_combined_rate_spread_scenarios(curve, corp_portfolio_df, val_date, settle)
    pivot = combo.pivot(index="rate_shock_bp", columns="spread_shock_bp", values="total_pnl_per_100_notional")

    # For fixed rate shock, PnL should generally decrease as spread shock increases
    for r in pivot.index:
        row = pivot.loc[r].values
        assert row[0] >= row[-1] - 1e-8, "PnL should not improve when spreads widen"

    # For fixed spread shock, PnL should generally decrease as rates rise
    for s in pivot.columns:
        col = pivot[s].values
        assert col[0] >= col[-1] - 1e-8, "PnL should not improve when rates rise"
