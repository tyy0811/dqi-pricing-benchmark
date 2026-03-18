# Appendix And Resource Presentation Cleanup Design

Date: 2026-03-13

## Goal

Clean up appendix and resource presentation so the reviewer-facing notebooks
match the repo's updated story, without changing any underlying alpha,
coherent, or Qiskit algorithms.

This phase is intentionally narrow:
- reporting and filtering cleanup only
- no algorithmic changes
- no broad schema churn
- no new claims about full Qiskit parity

## Helper Contracts

Files:
- `src/dqi_pipeline.py`
- `src/resources.py`

### `src/dqi_pipeline.py`

Add `has_complete_alpha_metrics(report)`.

Required contract:
- Pure reviewer-facing row gate only.
- Returns boolean only.
- Must not score, repair, infer, or enrich rows.
- Used only to support filtering and report cleanliness.

Required fields for completeness:
- `alpha_mode`
- `execution_mode`
- `status`
- `best_F`
- normalized success probability
- `top1_regret`

Optional field:
- distribution metric

Success-probability precedence contract:
1. `success_prob`
2. `postselection_success`
3. `decoder_success_rate`

Completeness rule:
- Missing key is incomplete.
- `None` is incomplete.
- `NaN` is incomplete.
- Optional distribution metric is never required for completeness.

### `src/resources.py`

Add:
- `resource_validation_scope = "analytic_full_pipeline_plus_qiskit_structural_subset"`
- `summarize_resource_scope(report)`

Required contract:
- Additive only.
- Do not rename existing keys.
- Keep the current resource structure stable unless a small additive field is
  required.
- `summarize_resource_scope(report)` should produce a compact reviewer-facing
  summary of:
  - analytic full-pipeline estimates
  - Qiskit structural subset counts
  - the fact that these are distinct validation scopes

### Helper Acceptance Checks

- `has_complete_alpha_metrics(report)` is a strict gate, not a permissive
  validator.
- Success-probability precedence is implemented identically wherever used.
- `None` and `NaN` both count as incomplete.
- `resources.py` changes are additive only.

## Resource And Quality-vs-Cost Notebook Boundary

Files:
- `notebooks/06_resource_analysis.ipynb`
- `notebooks/12_quality_vs_cost.ipynb`

### Notebook 06

Required presentation:
- Clearly separate:
  - analytic full-pipeline estimates
  - Qiskit structural subset counts
- Remove wording that could imply the Qiskit structural subset validates the
  full decode-inclusive pipeline.
- Present the resource-scope distinction explicitly in markdown and/or summary
  table form.

### Notebook 12

Required defaults:
- exact-reference mainline rows first
- mixture mode first
- `bp1` and `bruteforce` where available
- WHT off by default
- coherent off by default

Required ordering rule:
- Even when exploratory rows are later enabled, the default sort/order must
  present exact-reference mainline rows first.

Required boundary:
- This is filter/order cleanup only.
- No ranking-logic or algorithmic changes.

### Notebook Boundary Acceptance Checks

- Notebook 06 visibly distinguishes analytic full-pipeline estimates from
  Qiskit structural subset counts.
- Notebook 12 defaults to exact-reference + mixture and orders those rows
  first.
- No reviewer-facing notebook suggests unsupported coherent or Qiskit claims.

## Appendix Notebook Structure

Files:
- repurpose `notebooks/10_alpha_finite_size_validation.ipynb` into the
  reviewer-facing appendix role
- optionally preserve the current detailed finite-size notebook under a
  diagnostic/internal name only if it still has internal value

### Notebook 10

Required role:
- Compact reviewer-facing appendix.
- Not a deep validation notebook.

Required row gate:
- Use `has_complete_alpha_metrics(report)` to keep complete rows only.
- Rows failing the completeness gate are omitted from the main appendix table.
- Do not show incomplete rows with placeholders in the reviewer-facing table.

Required compact appendix fields:
- `alpha_mode`
- `execution_mode`
- `best_F`
- normalized success probability
- `top1_regret`
- optional distribution metric
- `status`

Canonical appendix column order:
1. `alpha_mode`
2. `execution_mode`
3. `best_F`
4. `success_probability`
5. `top1_regret`
6. `distribution_metric`
7. `status`

Narrative boundary:
- Keep alpha/coherent detail out of the main application-facing story when key
  fields are incomplete or `NaN`.
- If a detailed finite-size validation notebook is preserved, it is
  diagnostic/internal and not part of the main reviewer-facing flow.

### Appendix Acceptance Checks

- Notebook 10 shows complete rows only.
- Incomplete rows are omitted, not placeholder-filled.
- Column order is stable and reviewer-facing.
- Notebook 10 does not overclaim coherent completeness or application
  relevance.

## Verification And Testing Scope

Files, only as needed:
- targeted tests for the new helper(s)
- notebook smoke tests for Notebooks 06, 10, and 12

Required tests:
- `has_complete_alpha_metrics(report)` returns `True` for complete compact
  appendix rows
- `has_complete_alpha_metrics(report)` returns `False` for missing, `None`, or
  `NaN` values in required fields
- success-probability precedence is respected
- `summarize_resource_scope(report)` reflects the additive resource-scope
  distinction
- Notebook 06 separates analytic full-pipeline estimates from Qiskit
  structural subset counts
- Notebook 12 defaults to exact-reference + mixture and presents those rows
  first
- Notebook 10 uses the completeness gate and excludes incomplete rows from the
  reviewer-facing appendix table

### Verification Acceptance Checks

- Tests verify reporting gates, labels, ordering, and notebook messaging only.
- No new tests assert algorithm changes.
- No new tests imply full coherent or Qiskit parity.

## Non-Goals

- No algorithmic changes to alpha/coherent/Qiskit paths.
- No broad report-schema rewrite.
- No new full-parity claims for Qiskit.
- No attempt to promote incomplete coherent/alpha rows into the main
  reviewer-facing story.

## Compatibility Constraints

- Keep existing resource/report keys stable.
- Prefer additive helpers and additive notebook cleanup.
- Preserve internal diagnostic detail only if it remains useful, but keep the
  reviewer-facing notebook order clean.

