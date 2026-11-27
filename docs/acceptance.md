# Scientific acceptance for SurfaceCheck 0.1.0

The scientific reference environment uses CPython 3.12, QuantLib 1.44, CurveMetric 1.5.0 and VintageCheck 0.4.0. The companion packages were installed from PyPI. Core compatibility jobs separately exercise pricing, shape restrictions, calibration, repair, history and reporting on the interpreter matrix. Optional reference-package support is verified on 3.12.

| Acceptance criterion | Observed evidence |
|---|---|
| Test suite and coverage | 65 tests, 97.96% of statements, 94.27% of branches in the scientific reference run |
| Static arbitrage detection | 40 clean and 40 deliberately contaminated synthetic slices; 40 true positives and zero false positives, false negatives or unknowns |
| Calibration | Weighted total-variance RMSE 0.00022207 for noisy SVI and 2.54e-18 for synthetic SSVI |
| Conditioning | Sparse-smile Jacobian condition number approximately 78,109 |
| Density | Mass 0.999793 on finite support; first-moment error −0.04212 on a forward of 100 |
| Local volatility | Maximum absolute recovery error 3.475e-5 in the specified strike interval |
| Repair | Maximum constraint residual 3.55e-15; maximum premium adjustment 0.30321 inside the supplied intervals |
| Temporal leakage | Decision-time ATM-node IV 22.00%; including a later revision increased it to 27.17% and violated strike constraints |
| Packaging | Wheel and sdist passed strict Twine validation and independent clean installations |
| Scientific deliverables | Four synthetic studies, thirteen reproducible figures and JSON, CSV and HTML output |

The full financial scope, assumptions, external comparisons and tests for all fifteen areas are listed in [methodology](methodology.md) and [traceability](traceability.md). The detection sample measures sensitivity to the injected contamination. Results do not establish performance on a proprietary historical market dataset. The analytic SSVI result is restricted to its documented family. Continuous repair, global raw-SVI certification, extended SSVI, stochastic rates, American and exotic options and production local-volatility calibration are outside the approved scope.

[Scientific reference run](https://github.com/Phasetensor/SurfaceCheck/actions/runs/46019927549). Release publication, cross-registry digests and installation from the public index require separate post-publication evidence.

Verified revision `2aecbb5eb6bf255db8cd899af1ac4ef5da02188f`. [CI evidence](https://github.com/Phasetensor/SurfaceCheck/actions/runs/46020728675).
