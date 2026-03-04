# Poster Workplan (preliminary, reproducible)

## Current status (implemented)

- Deterministic script 2 event processing path only.
- OT-aware matrix labels are now saved in `comparison_metrics` for each parameter.
- `alpha` uses absolute uncertainty (not fractional denominator).
- `piEN`, `piEE` use `|piE| = sqrt(piEN^2 + piEE^2)` as denominator.
- Unphysical flux events are flagged and can be excluded.
- Poster-ready summary tables and text are produced by part 3.

## What to run

1) Build base caches (script 1)
- Keep your existing mass-bin/force-reload behavior.

2) Compute OT + matrix metrics + minimal caches (script 2)
- Run once for quick check, then forced rerun for final deterministic output.

3) Build final figures + poster tables (script 3)
- Outputs include:
  - `global_uncertainty_comparison.png`
  - `param_diagnostics_*.png`
  - `excluded_unphysical_flux_events.csv`
  - `poster_matrix_rows.csv`
  - `poster_matrix_counts_by_parameter.csv`
  - `poster_parameter_ratio_summary_constrained.csv`
  - `poster_key_messages.txt`

## Suggested poster language

- “Preliminary comparison uses an OT-gated constrained definition for MCMC: 
  constrained if fractional uncertainty <= 1/3 and OT(prior→posterior) > threshold.”
- “Reported constrained comparison focuses on matrix cells A1+A2 (both Fisher and MCMC constrained).”
- “Events with unphysical flux posteriors (`Fs < 0` or `FB > 1`) were excluded and listed for rerun.”

## Remaining TODOs

- Validate OT threshold choice (currently 0.04) with sensitivity test.
- Optional: include prior in Fisher posterior covariance (separate methodological variant).
- Rerun flagged events with corrected priors.
