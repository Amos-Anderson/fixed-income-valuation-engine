## Testing & Validation

This repository includes a pytest suite under `tests/` that validates the valuation engine at three levels:

1. curve construction (bootstrap + interpolation behavior)
2. instrument pricing (dirty/clean/accrued consistency, spread monotonicity)
3. risk & scenario analytics (DV01, convexity, KRD reconciliation, rate/spread grid sanity)

Run all tests from repo root:

```bash
pytest -q
```

### What the tests validate

#### 1) Curve tests (`tests/test_curve.py`)

These tests validate the **structural integrity** and **numerical sanity** of the bootstrapped curve:

* **Knot ordering**: knot dates are strictly increasing.
* **Discount factor positivity**: all $D(t) > 0$.
* **Monotonicity across knots**: $D(t)$ is non-increasing across maturity knots (allowing tiny numerical tolerance).
  This is a basic no-arbitrage sanity check under “normal” positive-rate environments.
* **Short-end extrapolation behavior**: requests between `val_date` and the first knot are supported using a flat continuously-compounded zero rate implied by the first knot.
  That is, for $t \in [t_0, t_1)$:
  $$
  D(t) = \exp(-z_1 , \tau(t_0,t)),
  \quad
  z_1 = -\frac{\ln D(t_1)}{\tau(t_0,t_1)}.
  $$
* **No long-end extrapolation**: requests beyond the last knot raise by design.
  This prevents silently making up discount factors beyond the curve construction horizon.

**What these curve tests do *not* prove**

* They do not guarantee arbitrage-free behavior between knots beyond monotone $D(t)$ at knots.
* They do not validate market convention correctness (e.g., bill quoting conventions, settlement assumptions, day count edge cases).
* They do not validate robustness to negative rate regimes (the monotonicity check can legitimately fail if rates are negative).

---

#### 2) Bond pricing tests (`tests/test_bonds.py`)

These tests validate the **accounting identities** and **basic pricing monotonicity**:

* **Dirty/Clean/Accrued identity**:
  $$
  P_{\text{clean}} = P_{\text{dirty}} - AI.
  $$
* **Coupon-date behavior**: accrued interest should be approximately zero when settlement is exactly on a coupon date (given the schedule logic).
* **Post-coupon behavior**: accrued interest becomes positive immediately after the coupon date.
* **Corporate spread monotonicity**: increasing spread reduces price (all else equal), consistent with:
  $$
  D_{\text{corp}}(t) = D(t),\exp(-s,\tau),
  \quad s \uparrow \Rightarrow P \downarrow.
  $$

**What these bond tests do *not* prove**

* The schedule logic is maturity-anchored (issue date is not modeled). This is acceptable for a demo engine, but it is not how production bond systems build schedules.
* Settlement is simplified (calendar $T+2$) unless you later replace it with a business-day calendar.
* The corporate spread model is a **flat spread-on-discounting** approximation (not a hazard-rate model, not a full credit curve).

---

#### 3) Risk & scenario tests (`tests/test_risk.py`)

These tests validate that risk analytics are **directionally correct**, **internally consistent**, and **reconcilable**:

* **Rate DV01 sign sanity**: for a typical long-only positive-duration portfolio, a +1bp parallel rate shock should decrease price, so DV01 should be negative on average.
* **Convexity sanity**: convexity is mostly positive for vanilla fixed coupons.
* **Duration-from-DV01 consistency**:
  $$
  D_{\text{mod}} \approx -\frac{\Delta P}{P,\Delta y}.
  $$
* **Key Rate Duration reconciliation**: bucketed KRD “hat basis” shocks approximately reconcile to the parallel DV01:
  $$
  \sum_k \text{KRD}*k \approx \text{DV01}*{\text{parallel}}.
  $$
  This checks that the local basis shocks behave like a partition of unity over the knot grid (up to discretization error).
* **Spread DV01 sign sanity**: a +1bp spread bump reduces corporate prices, so spread DV01 should be negative.
* **Spread DV01 aggregation**: sum of per-bond spread DV01 matches portfolio spread DV01.
* **Combined rate/spread scenario grid sanity**: widening spreads or rising rates should not improve P&L for a plain long-only credit book (directional check, not a theorem).

**What these risk tests do *not* prove**

* They do not guarantee accurate second-order (convexity) matching for large shocks.
* They do not validate interpolation sensitivity (e.g., what happens with sparse knots, irregular knot spacing).
* They do not validate hedging performance or replication error against real market quotes.
* They assume “plain vanilla” economics; structured books may violate monotone P&L assumptions.

---

### Modeling assumptions you should understand before using this

This engine is intentionally transparent and educational. Its most important simplifying assumptions are:

* **Schedule generation** is maturity-anchored (issue date not modeled).
* **Settlement** is simplified ($T+2$ calendar days by default).
* **Curve interpolation** is log-linear on discount factors.
* **Short-end extrapolation** (between `val_date` and first knot) is allowed using a flat cc zero rate implied by the first knot.
* **No long-end extrapolation** beyond the last knot.
* **Corporate spread** is implemented as a constant exponential spread factor applied to all cashflows (flat spread model).
