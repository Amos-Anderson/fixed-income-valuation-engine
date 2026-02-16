import numpy as np
import pandas as pd
import pytest


from fixed_income_engine.curves import (
    bootstrap_curve_from_bills_notes,
    curve_qc_report,
)


@pytest.fixture(scope="module")
def val_date():
    return pd.Timestamp("2026-02-13")


@pytest.fixture(scope="module")
def market_df():
    # Same structure you used in the notebook
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


def test_curve_knots_increasing(curve):
    knot_dates = pd.to_datetime(curve.knot_dates)
    assert knot_dates.is_monotonic_increasing, "Knot dates must be strictly increasing"


def test_curve_discount_factors_positive_and_monotone(curve):
    dfs = np.exp(curve.knot_log_dfs)
    assert np.all(dfs > 0.0), "All discount factors must be positive"
    # monotone decreasing (allow tiny numeric noise)
    assert np.all(np.diff(dfs) <= 1e-10), "Discount factors should be non-increasing across knots"


def test_curve_qc_report_flags(curve):
    qc = curve_qc_report(curve)
    assert qc["df_positive"].all()
    assert qc["df_monotone"].all()
    assert np.isfinite(qc["zero_cc"]).all()


def test_df_short_end_extrapolation_between_val_and_first_knot(curve, val_date):
    """
    Your ZeroCurve.df implements short-end extrapolation (val_date -> first knot)
    using flat cc zero implied by the first knot.

    This test ensures:
    - requesting a date prior to first knot but >= val_date works
    - DF is close to 1 and <= DF(first_knot)
    """
    first_knot = pd.Timestamp(pd.to_datetime(curve.knot_dates[0]))
    t = val_date + pd.Timedelta(days=1)

    df_t = curve.df([t])[0]
    df_first = curve.df([first_knot])[0]

    assert 0.999 < df_t <= 1.0, "Very short-end DF should be close to 1"
    assert df_t >= df_first - 1e-12, "Earlier date should have DF >= DF(first knot)"


def test_df_raises_on_long_end_extrapolation(curve):
    """
    By design: no long-end extrapolation beyond last knot.
    """
    last_knot = pd.Timestamp(pd.to_datetime(curve.knot_dates[-1]))
    t = last_knot + pd.DateOffset(days=1)
    with pytest.raises(ValueError):
        _ = curve.df([t])


