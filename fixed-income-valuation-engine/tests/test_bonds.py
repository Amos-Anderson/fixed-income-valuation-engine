import numpy as np
import pandas as pd
import pytest

from fixed_income_engine.curves import bootstrap_curve_from_bills_notes
from fixed_income_engine.bonds import (
    Bond,
    CorporateBond,
    price_bond_dirty_clean,
    price_corporate_bond_dirty_clean,
)
from fixed_income_engine.utils import settlement_date


@pytest.fixture(scope="module")
def val_date():
    return pd.Timestamp("2026-02-13")


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
def bond():
    return Bond(
        bond_id="CORP_5Y_5PCT",
        maturity=pd.Timestamp("2031-02-15"),
        coupon_rate=0.05,
        freq=2,
        day_count="30/360",
        face=100.0,
    )


def test_dirty_clean_identity(curve, val_date, bond):
    settle = settlement_date(val_date, 2)
    dirty, clean, ai = price_bond_dirty_clean(curve, bond, val_date, settle)
    assert np.isfinite(dirty) and np.isfinite(clean) and np.isfinite(ai)
    assert abs((dirty - ai) - clean) < 1e-10, "clean must equal dirty - accrued"
    assert dirty >= clean - 1e-12


def test_accrued_interest_zero_on_coupon_date(curve, val_date, bond):
    """
    With your maturity-anchored schedule and coupon cycle at Feb 15 / Aug 15,
    settlement on Feb 15 should give AI ~ 0.
    """
    settle = pd.Timestamp("2026-02-15")
    dirty, clean, ai = price_bond_dirty_clean(curve, bond, val_date, settle)
    assert abs(ai) < 1e-10, "Accrued interest should be ~0 on coupon date"
    assert abs(dirty - clean) < 1e-10, "Dirty and clean should match on coupon date"


def test_accrued_interest_positive_day_after_coupon_date(curve, val_date, bond):
    settle = pd.Timestamp("2026-02-16")
    dirty, clean, ai = price_bond_dirty_clean(curve, bond, val_date, settle)
    assert ai > 0.0, "Accrued should be positive right after coupon date"
    assert dirty >= clean - 1e-12


def test_corporate_spread_monotonicity(curve, val_date):
    """
    Increasing spread should decrease price (all else equal).
    """
    settle = pd.Timestamp("2026-02-16")

    base = CorporateBond(
        bond_id="CORP_TEST",
        maturity=pd.Timestamp("2031-02-15"),
        coupon_rate=0.05,
        freq=2,
        day_count="30/360",
        face=100.0,
        spread=0.0,
    )

    px0, _, _ = price_corporate_bond_dirty_clean(curve, base, val_date, settle)
    px50, _, _ = price_corporate_bond_dirty_clean(curve, CorporateBond(**{**base.__dict__, "spread": 50 / 10000.0}), val_date, settle)
    px100, _, _ = price_corporate_bond_dirty_clean(curve, CorporateBond(**{**base.__dict__, "spread": 100 / 10000.0}), val_date, settle)

    assert px0 > px50 > px100, "Price should decrease as spread increases"
