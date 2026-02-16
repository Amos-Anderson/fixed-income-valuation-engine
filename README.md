# Fixed Income Valuation & Risk Engine

A modular, production-style **fixed income valuation engine** for:

* Zero curve bootstrapping (Bills + Notes)
* Clean & dirty bond pricing
* Accrued interest under multiple day count conventions
* Portfolio valuation (vectorized)
* Interest rate risk (DV01, duration, convexity)
* Key Rate Duration (bucketed exposure)
* Spread risk (corporate bonds)
* Combined rate + spread scenario analysis

---

## 1. Project Objective

This engine demonstrates how institutional bond valuation systems are built from first principles:

* Bootstrapped arbitrage-free discount curves
* Settlement-adjusted discounting
* Cashflow expansion logic
* Vectorized portfolio valuation
* Sensitivity-based risk analytics
* Multi-factor scenario stress testing

The goal is to bridge academic fixed income theory with **real desk-level pricing and risk implementation**.

---

## 2. Architecture

```
fixed-income-valuation-engine/
│
├── fixed_income_engine/
│   ├── curves.py        # Zero curve & bootstrapping
│   ├── bonds.py         # Bond definitions & pricing
│   ├── portfolio.py     # Vectorized portfolio valuation
│   ├── risk.py          # DV01, duration, convexity, KRD
│   ├── scenarios.py     # Rate & spread stress testing
│   ├── utils.py         # Day counts, schedules, helpers
|   └── README.md        # Important information regarding the modules in this folder
│
├── notebooks/
│   ├── 01_curve_bootstrap.ipynb
│   ├── 02_bond_pricing_engine.ipynb
│   ├── 03_risk_measures.ipynb
│   ├── 04_scenario_analysis.ipynb 
│   ├── 05_portfolio_analytics.ipynb
│   └── full_workflow_notebook_fixed_income_engine.ipynb
│
├── tests/
│   ├── test_curve.py
│   ├── test_bonds.py
│   ├── test_risk.py
│
├── requirements.txt
├── LICENSE.txt
└── README.md
```

---

## 3. Curve Construction

The engine bootstraps a zero curve from:

* Treasury Bills (simple yield quotes)
* Treasury Notes (par coupon quotes)

Discount factors are interpolated **log-linearly**:

$$
\log D(t) = \text{linear interpolation}
$$

Zero rates are continuously compounded:

$$
D(t) = e^{-z(t)\tau}
$$

Features:

* Monotonic discount factor enforcement
* Settlement-adjusted discounting
* Short-end flat-zero extrapolation
* No long-end extrapolation (explicit risk control)

---

## 4. Bond Pricing Engine

For each bond:

### Dirty Price

$$
P_{\text{dirty}} = \sum_i CF_i \cdot \frac{D(T_i)}{D(s)}
$$

where

* $s$ = settlement date
* $T_i$ = payment date

### Clean Price

$$
P_{\text{clean}} = P_{\text{dirty}} - \text{Accrued Interest}
$$

### Accrued Interest

Supports:

* ACT/365
* ACT/360
* 30/360 (US)

Settlement conventions implemented (T+2 baseline).

---

## 5. Portfolio Valuation

The portfolio engine:

* Expands all cashflows
* Maps discount factors vectorized
* Applies settlement adjustment
* Aggregates PV per bond
* Computes clean & dirty prices
* Applies QC flags

This avoids nested Python loops and reflects real desk implementations.

---

## 6. Risk Measures

### Parallel DV01

$$
\text{DV01} = P(z + 1bp) - P(z)
$$

### Modified Duration

$$
D_{mod} = -\frac{\text{DV01}}{P \cdot 0.0001}
$$

### Convexity (finite difference)

$$
C = \frac{P_{up} + P_{down} - 2P}{P \cdot h^2}
$$

### Key Rate Duration (KRD)

Implemented using **hat basis functions** to localize zero-rate shocks.

Reconciliation:

$$
\sum_k \text{KRD}_k \approx \text{Parallel DV01}
$$

---

## 7. Corporate Spread Risk

Corporate bonds priced using constant spread model:

$$
D_{corp}(t) = \frac{D(t)}{D(s)} \cdot e^{-s \cdot \tau(s,t)}
$$

Spread risk metrics:

* Spread DV01
* Spread duration
* Spread scenario P&L

---

## 8. Scenario Analysis

Supported stress tests:

* Parallel rate shocks (±25bp, ±50bp)
* Steepener shocks
* Flattener shocks
* Spread shocks (+25bp, +100bp)
* Combined rate + spread grid

Outputs:

* Per-bond P&L
* Portfolio P&L summary
* Scenario pivot heatmaps

---

## 9. Example Portfolio Output

The engine produces:

* Market value
* Parallel DV01
* Spread DV01
* Bucketed key rate exposure
* Top risk contributors
* Scenario P&L matrix

Example:

```
Total Market Value: 1907.77
Parallel DV01: -1.03
Spread DV01: -0.95
```

Bucket concentration typically observed at longer maturities, as expected from duration structure.

---

## 10. Installation

### Option 1 — pip

```bash
pip install -r requirements.txt
```

### Option 2 — conda

```bash
conda create -n fixed_income python=3.11
conda activate fixed_income
pip install -r requirements.txt
```

---

## 11. Running Tests

```bash
pytest
```

All tests validate:

* Curve monotonicity
* Par repricing consistency
* Dirty vs clean pricing logic
* DV01 directionality
* KRD reconciliation

---

## 12. Why This Project Matters

This engine mirrors functionality found in:

* Investment Grade bond valuation desks
* Market risk systems
* Model validation frameworks
* Asset management risk platforms

It demonstrates:

* Financial mathematics rigor
* Numerical stability
* Sensitivity-based risk design
* Portfolio-scale vectorization
* Clear modular architecture

---

## 13. Future Extensions

* OIS discounting
* Multi-curve framework
* Credit curve bootstrapping
* Stochastic rate models
* Liquidity and funding adjustments
* Performance optimization (Numba)

---

## License

MIT License.

---
