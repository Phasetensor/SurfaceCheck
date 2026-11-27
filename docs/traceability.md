# Implementation and acceptance evidence

| Area | Acceptance check | Evidence |
|---|---|---|
| 01 Market | Reject malformed snapshots, crossed markets, duplicate eligible contracts; interval parity | test_market |
| 02 IV | Reprice all three solvers; differentiate boundaries, poor conditioning and failures | test_inversion, test_edges, test_quantlib, test_black_properties |
| 03 Coordinates | Invert log-moneyness and total variance with explicit F,T | test_grid, test_exact_density |
| 04 Butterfly | Actual strike spacing, upper/lower bounds, signed discrete and sampled smooth checks | test_grid, test_svi, study A |
| 05 Calendar | Shared normalized martingale, overlap, interpolated versus observed breaches | test_calendar, test_panel, study A |
| 06 SVI | Three objectives, multi-start and derivative comparisons | test_svi, study B |
| 07 SSVI | Known synthetic fit, ATM equality and restricted-family conditions | test_ssvi_and_svd, test_history_alignment_and_certificate |
| 08 Interpolation | Reproduce convex-knot counterexample and enforce support | test_grid, interpolation metrics |
| 09 Wings | Compare observed secants, analytic SVI slopes and extrapolated prices | test_svi, test_wings_stresses, tail_variances |
| 10 Density | Signed density against lognormal; finite-support mass and moment | test_exact_density, test_density_dupire, study C |
| 11 Local volatility | Constant-volatility recovery under nonzero carry; flag zero denominators | test_density_dupire, study C |
| 12 Stability | Compare whitened covariance with full-rank reference, expose deficient rank and perturbed parameters | test_ssvi_and_svd, study B |
| 13 Repair | Independent residual checks, infeasible boxes, analytical joint optimum | test_grid, test_panel, study C |
| 14 History | Common grid, missing support, ATM/skew, centered PCA, factor shocks and temporal eligibility | test_history_alignment_and_certificate, test_wings_stresses, study D |
| 15 Reporting | Preserve undefined evidence, fingerprint inputs, escape HTML and emit parseable JSON/CSV | test_history_report, four case reports combined in validation.json |

Every reported result is scoped to the instrument, observations, model family and numerical tolerances. Unknown or numerical-failure outcomes require inspection. Scientific review should also consider the failure cases and unsupported regimes in the methodology.
