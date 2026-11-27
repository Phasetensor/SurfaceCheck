# Scientific methodology

## Prices and evidence

The input is a European option on a positive underlying with integrable terminal value, deterministic discounting and a specified forward. The Black formula uses total standard deviation σ√T. Calls lie in [D max(F−K,0), DF], and puts in [D max(K−F,0), DK]. The upper endpoint is a volatility limit. Intrinsic prices can arise from the precision of quoted or rounded prices. The inversion returns a boundary status. A finite interior solution with price uncertainty comparable to time value or saturation distance receives a poor-conditioning status. Vega is sensitivity per unit annual volatility.

Price the out-of-the-money leg using logarithmic normal probabilities and an exponential difference. Recover the other leg by parity. At the money, use the error function. Bracketing in total standard deviation separates root scaling from maturity. Repricing is checked in price units. Brent, bisection and bracketed Newton share these bounds. The configurable price tolerance must be chosen for the quote units.

For matched contracts the parity interval is [call bid − put ask, call ask − put bid]. A difference outside this interval contradicts the supplied deterministic forward and funding inputs. Trade size, borrowing constraints, fees, synchrony and executable underlying/forward quotes still matter to an execution claim.

## Strike and calendar constraints

Secant slopes on each actual strike interval must lie in [−D,0] and increase with strike. These conditions and price bounds are linear inequalities. Single-slice repair minimizes one half of the sum of weighted squared midpoint adjustments subject to the inequalities and bid-ask boxes. A linear program tests feasibility before quadratic optimization. Recheck the candidate against every inequality. Joint repair uses prices C/(DF) and strikes K/F on a shared normalized grid and imposes nondecreasing normalized prices across expiries.

For deterministic proportional carry, assume S(T)/F(T) is the same true martingale observed at different maturities under a common measure. Its normalized call expectation increases by convex order. At fixed k = log(K/F), Black normalized call price increases with total variance. Comparing corresponding w = σ²T is therefore legitimate. Deterministic curves alone do not establish the martingale assumption for arbitrary dividends or payoffs. For unequal grids, compare on overlapping support; separate shared-node interval violations from those conditional on linear interpolation. The pairwise absence of an interval violation is weaker than joint interval feasibility.

## Parametric smiles and wings

Raw SVI uses (a,b,ρ,m,η). The variance minimum is a+bη√(1−ρ²). During calibration the optimizer varies this positive floor, b, ρ, m and η. The Jacobian and reported covariance refer to those fitted coordinates. The asymptotic slopes in |k| are b(1−ρ) and b(1+ρ). Lee relates limsup wing slopes to positive and inverse critical moments at a fixed expiry under the forward distribution. The universal upper bound is 2 under the applicable moment formula. Finite secants do not identify critical moments.

For a strictly positive twice-differentiable total variance, the implied strike density has sign given by g=(1−kw′/(2w))²−(w′²/4)(1/w+1/4)+w″/2. The right call tail must also vanish (the Gatheral–Jacquier condition d+(k) tending to minus infinity). Sampling g can locate a violation, while a positive finite sample cannot exclude inter-node or tail violations. Atoms require a measure-valued derivative.

Surface SVI uses constant ρ and φ(θ)=s/√(θ(1+θ)). Its ATM variance is θ. Here θφ=s√(θ/(1+θ)) increases and its logarithmic elasticity with respect to θ is 1/[2(1+θ)]. This lies between zero and one, satisfying the Gatheral–Jacquier calendar derivative bounds for constant ρ when θ is nondecreasing. Since θφ<s and θφ²≤s², sufficient butterfly inequalities are s(1+|ρ|)<4 and s²(1+|ρ|)≤4. The surface API checks the shape restriction and the nondecreasing piecewise-linear θ schedule, including the origin. The certificate covers all real k over the supplied time horizon under the normalized martingale assumptions. A free fit to a set of θ values alone supplies no maturity ordering.

## Calibration and risk

Raw variance, annual volatility and currency price residuals define different objectives. To first order, transforming variance residuals to volatility divides by 2σT, and transforming volatility to price multiplies by vega. Weights multiply squared residuals; inverse error variances correspond to independent Gaussian measurement errors in the chosen units. Multi-start optimization and convergence messages expose numerical outcomes. Small residuals can accompany poorly identified parameters.

For independent known observation standard deviations, whiten the unweighted residual Jacobian once and compute its SVD. The covariance in a full-column-rank local linearization is V diag(1/s²) Vᵀ. Supply unit errors to an already whitened Jacobian. Rank deficiency leaves unidentified directions; covariance is withheld. Constrained optima on a boundary, correlated errors and nonlinear regimes require additional analysis. Seeded quote perturbation and refitting provide a second view of estimation sensitivity.

## Density, local variance and history

Under an absolutely continuous terminal law, strike density is C_KK/D. The interior three-point second difference handles unequal spacings, with reduced accuracy on strongly asymmetric grids. Integrate signed densities on the interior support by trapezoids. Missing tail mass, discretization and violations can all change the mass and first moment. Atoms are outside the density routine.

For deterministic instantaneous rates r and proportional yield q, Dupire local variance at fixed strike is 2[C_T+(r−q)K C_K+q C]/[K² C_KK]. Use a shared strike axis, interior time points and positive, sufficiently large denominators. Negative numerators remain visible. Derivative instability is amplified by short time intervals, narrow strike spacing and tail gamma. Flat Black volatility provides an independent recovery target.

Historical changes use the common log-forward strike and year-fraction grid. Align inside observed strike support, flag missing support and require a complete panel for PCA. PCA centers absolute, relative or variance changes and applies unscaled SVD; component sign is arbitrary. Restrict the training prefix before using factors at a historical decision. Stress scenarios trigger new finite-grid checks. Nonuniform observation intervals are period changes, not daily changes. Carry, time basis, units, currency and corporate-action conventions must be reconciled before combining external snapshots.

## Literature and existing software

- Black and Scholes (1973), *The Pricing of Options and Corporate Liabilities*, Journal of Political Economy 81, 637–654. [Publisher](https://doi.org/10.1086/260062)
- Merton (1973), *Theory of Rational Option Pricing*, Bell Journal of Economics and Management Science 4, 141–183. [Publisher](https://doi.org/10.2307/3003143)
- Black (1976), *The pricing of commodity contracts*, Journal of Financial Economics 3, 167–179. [Publisher](https://doi.org/10.1016/0304-405X(76)90024-6)
- Breeden and Litzenberger (1978), *Prices of State-Contingent Claims Implicit in Option Prices*, Journal of Business 51, 621–651. [Publisher](https://doi.org/10.1086/296025)
- Dupire (1994), *Pricing with a smile*, Risk 7(1), 18–20. [Original article](https://www.risk.net/sites/default/files/import_unmanaged/risk.net/data/Pay_per_view/risk/technical/1994/risk_0194_volatility.pdf)
- Lee (2004), *The Moment Formula for Implied Volatility at Extreme Strikes*, Mathematical Finance 14, 469–480. [Author manuscript](https://math.uchicago.edu/~rogerlee/moment.pdf)
- Gatheral and Jacquier (2014), *Arbitrage-free SVI volatility surfaces*, Quantitative Finance 14, 59–71. [Author preprint](https://arxiv.org/abs/1204.0646)
- Cohen, Reisinger and Wang, *Detecting and repairing arbitrage in traded option prices*, develop a minimal-perturbation LP. SurfaceCheck uses an interval-constrained weighted quadratic objective. [Oxford manuscript](https://ora.ox.ac.uk/objects/uuid%3A77110176-4528-42c1-85eb-2f78b94514e8)

QuantLib supplies volatility structures, smile sections, regularization and a Black implementation suitable as an external reference. Vollib provides specialized IV inversion based on LetsBeRational. SciPy supplies numerical primitives without instrument or timestamp semantics. Public SVI calibration examples already exist. The workflow addressed here ties availability timestamps, forward construction, interval feasibility and uncertainty to the same validation record. This is an integration objective; the comparison supplies no exhaustive uniqueness or speed claim. [QuantLib](https://quantlib-python-docs.readthedocs.io/en/latest/termstructures.html) · [vollib](https://vollib.org/) · [SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.html) · [SVI calibration example](https://github.com/wangys96/SVI-Volatility-Surface-Calibration)

## Known limitations

American exercise, stochastic discounting, discrete cash dividends, negative forwards, normal volatilities and exotic payoffs require other conventions or models. Bid and ask premiums must already be normalized per unit of underlying. Forecasting, trading and execution are outside scope. Dense grid optimization is intended for research-sized panels. No global raw-SVI certificate, continuous-surface repair, eSSVI fit, general covariance estimator, production local-volatility solver or proprietary historical dataset is supplied.

## Time and unit conventions

Timezone-aware quote timestamps are normalized to UTC. The quote year-fraction check uses elapsed UTC seconds divided by the fixed day-count denominator. For date-based CurveMetric workflows the adapter instead uses the explicit valuation and expiration dates. Resolve a venue's expiration clock before combining intraday and date-based data. Convert percent volatilities to decimals, basis points to rates and contract premiums to per-unit cash premiums upstream.
