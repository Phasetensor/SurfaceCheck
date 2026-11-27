# Python API

| Module | Entry points | Inputs and outputs |
|---|---|---|
| market | Quote, snapshot, parity | Aware timestamps, ACT/365F or ACT/360, absolute premiums; a parity evidence category or eligible contracts |
| pricing | black, implied_vol | F, K, T, D and decimal volatility or premium; discounted price or Inversion with residual and vega |
| grid | coordinates, interpolate, calendar, repair | Strictly ordered grids; coordinate transforms, conditional evidence and finite-grid repair records |
| models | SVI, SSVI, fit_svi, fit_ssvi, ssvi_surface | Raw model parameters or supplied ATM variances; fit residuals, convergence and restricted-family conditions |
| models | stability, propagation | Residual Jacobian and observation errors, or a callable refit and independent perturbations |
| diagnostics | curvature, density, local_variance, wing_slopes | Signed interior derivatives, finite support, integrated mass, first moment, denominator mask or tail secants |
| surface | from_variance, svi_density, extrapolated_wings, stress, repair_panel | Inverse coordinates, analytic density, wing rule, stressed grid or normalized joint repair |
| diagnostics | align_smiles, historical_summary, historical_pca | As-of ordered panels; support gaps, ATM/skew history, period changes, parameter paths, centered factors |
| adapters | historical_snapshot, forward_inputs | VintageCheck releases and operator cutoff; CurveMetric conventions, spot and continuous dividend yield |
| reporting | Report.render | Valuation, source, conventions, named evidence and inputs; JSON, CSV or escaped HTML |

`check_slice` preserves the original strike-shape API when the forward is unavailable. Use `repair` to include forward-dependent price bounds. Quote selection rejects unresolved competing quotes; revision selection belongs upstream. Array-based numerical entry points expect inputs in the same conventions. The integration example checks conventions before extracting arrays.
