# Reproducible experiments

Run `python benchmarks/cases.py` after installing the validation extra. The complete run writes thirteen figures, metrics.json and three validation report formats into results/. Runtime and peak RSS cover this Python process including plotting. All quotes are synthetic. Seed 47 controls disturbances.

| Study | Design | Independent comparison or failure case |
|---|---|---|
| A | Construct a restricted SSVI surface, introduce ATM and later-expiry shocks, evaluate strike and calendar grids | Classify 40 clean Black slices and 40 with a forced monotonicity violation on random nonuniform strikes; precision/recall refer only to this balanced construction |
| B | Fit raw SVI to noisy variance and SSVI to clean term data, refit 20 perturbations | Independent finite differences, known synthetic smile, limited near-ATM fit and weak directions |
| C | Perturb a Black call node, repair inside spread, differentiate before/after | Analytic lognormal density, finite-support mass and moment, constant-volatility Dupire recovery |
| D | Reconstruct a timestamped option/rate/spot/yield snapshot using released companions, rebuild a discount curve, bump its rate by 10 bp | Future price 19 must be excluded, eligible call is 9; report both inverted volatilities and the selected release fingerprint |

Interpolation compares spline, PCHIP and linear interpolation on convex call nodes. Negative slope increments expose between-node convexity violations. Wing rules agree at the final knot but diverge at |k|=2. Reported density and Dupire errors depend on support; the local-volatility metric covers strikes 80–120. Synthetic detection scores cannot establish market precision or recall.

The historical study also fits a nine-strike chain both at the decision cutoff and with a deliberately leaked future ATM revision. Its output records the change in fitted variance error and finite-strike feasibility, alongside a parallel rate perturbation using the same eligible premiums. The three-strike feasibility reference in test_references.py enumerates intersections of all possible triples of active bounding planes independently of the production optimizer.
