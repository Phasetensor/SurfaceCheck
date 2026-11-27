# SurfaceCheck

SurfaceCheck is PhaseTensor's sixth scientific Python library and the third in its Quantitative Finance Suite. It audits European option quotes and fitted volatility surfaces under explicit forward, carry and deterministic discounting assumptions. The development version is 0.1.0.

A quote can violate convexity at its midpoint while admitting a coherent price inside its spread. SurfaceCheck reports both findings, the strikes involved and the adjustments needed on the finite grid.

```python
from surfacecheck.grid import repair

result = repair(
    strikes=[80, 95, 100, 120],
    bids=[21, 12, 11, 0],
    asks=[25, 16, 15, 4],
    forward=100,
    discount=1,
)
assert result["status"] == "spread_feasible"
print(result["violated_strikes"], result["adjustments"])
```

## Reproduce the research branch

Use Python 3.10 or later. The reference CI environment uses Python 3.12.

```sh
git clone --branch scientific-validation https://github.com/Phasetensor/SurfaceCheck
cd SurfaceCheck
python -m pip install '.[validation]'
coverage run -m pytest
coverage json
python benchmarks/coverage_gate.py
python benchmarks/cases.py
```

Examples, evidence standards and assumptions appear in the [methodology](docs/methodology.md), [API guide](docs/api.md), [benchmark guide](docs/benchmarks.md) and [traceability matrix](docs/traceability.md). See the [release checklist](docs/release.md) for publication status. The core dependencies are NumPy and SciPy. Optional integrations select temporal records with VintageCheck and construct discount and forward inputs with CurveMetric.

## Domain

Prices are per unit in one currency, annual volatility is a decimal and maturity is an explicit year fraction. The underlying forward, strike and discount factor are positive. Evidence from finite grids applies to the sampled domain. The SSVI certificate applies to the restricted family and interpolated ATM term structure specified in the methodology. Execution, trading strategies, regulatory certification and banking model approval are outside the project scope.

Licensed under [Apache 2.0](LICENSE).
