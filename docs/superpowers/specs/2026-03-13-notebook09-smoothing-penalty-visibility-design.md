# Notebook 09 Smoothing Penalty Visibility Design

**Date:** 2026-03-13

## Goal

Make `smoothing_penalty_total` visibly appear in the final saved reviewer-facing verdict table of `notebooks/09_3period_pricing_extension.ipynb` with a minimal patch.

## Scope

In scope:

- `notebooks/09_3period_pricing_extension.ipynb`

Out of scope:

- helper refactors unless strictly required
- any modeling logic changes
- benchmark framework changes
- any other notebook

## Design

Patch only the existing verdict-table cell so `verdict_df` includes a visible `smoothing_penalty_total` column taken directly from the existing scenario-matrix payload.

If the field is already available in `scenario_matrix_df`, expose it directly in the `verdict_df` selection and keep all current reviewer-facing fields intact.

If the field is not present in the payload, the only acceptable fallback is a tiny follow-up display below the verdict table with `scenario_label` and `smoothing_penalty_total`.

## Verification

- re-execute the notebook in place with `jupyter nbconvert --to notebook --execute --inplace`
- verify from the saved notebook JSON that at least one embedded output contains:
  - `smoothing_penalty_total`
  - `dynamic_horizon_revenue`
  - `static_horizon_revenue`
  - `static_vs_dynamic_delta`
  - `verdict`
