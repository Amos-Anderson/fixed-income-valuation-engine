"""
Fixed Income Valuation Engine

Production-style modules:
- curves: yield curve bootstrapping + curve object
- bonds: bond objects + single-bond pricing + schedule/accrual
- portfolio: portfolio cashflow expansion + vectorized valuation
- risk: DV01/convexity/duration/KRD/spread risk
- scenarios: rate/spread scenario runners
- utils: day count + schedule helpers

Notebook layer should import from this package.
"""
