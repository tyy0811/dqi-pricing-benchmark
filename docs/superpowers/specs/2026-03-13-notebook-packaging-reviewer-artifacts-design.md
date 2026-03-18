# Notebook Packaging Reviewer Artifacts Design

**Date:** 2026-03-13

## Goal

Package the strongest reviewer-facing claims without adding new notebooks:

- re-freeze Notebook 07 so saved outputs match the tiny validated subset case
- turn Notebook 09 into a reusable mini-benchmark artifact with CSV export and a verdict table
- add one short reviewer summary markdown file

## Scope

In scope:

- `notebooks/07_qiskit_structural_parity.ipynb`
- `notebooks/09_3period_pricing_extension.ipynb`
- minimal helper support already used by Notebook 09’s scenario matrix
- one exported CSV under `results/tables/`
- one reviewer summary markdown file under `docs/`

Out of scope:

- new notebooks
- new modeling logic
- benchmark framework rewrites
- full DQI parity claims beyond the tiny validated subset case

## Notebook 07 Packaging

Notebook 07 should be split into two clearly labeled reviewer-facing sections.

### Section 1: Validated tiny-case subset comparison

This becomes the primary saved evidence.

Required content:

- exact tiny-case parameter block
- top-8 NumPy subset probabilities
- top-8 Qiskit subset probabilities
- side-by-side labeled syndrome comparison
- raw TVD
- optional bit-reversed/remapped TVD for diagnosis only
- one explicit sentence stating that near-zero TVD applies only to this tiny case

Allowed claim:

- validated subset parity on the tiny hand-checkable case only

### Section 2: Family-S structural/resource context

This remains in the notebook, but is explicitly secondary.

Required content:

- family-S artifact identifier and dimensions
- existing family-S output if present
- resource counts
- explicit scope text that this section is structural/resource context only and not a general parity validation result

Allowed claim:

- structural subset parity/resource evidence only

## Notebook 09 Packaging

Notebook 09 keeps its current scenario matrix logic and adds only artifact packaging.

Required additions:

- export the scenario matrix to `results/tables/3period_scenario_matrix.csv`
- end the notebook with a compact verdict table

Verdict table columns:

- `scenario_label`
- `dynamic_revenue`
- `static_revenue`
- `delta`
- `inventory_used`
- `package_changed_across_periods`
- `verdict`

Verdict derivation:

- use only existing row data
- use simple deterministic labels
- no new modeling logic

## Reviewer Summary Markdown

Create `docs/BMW_reviewer_summary.md` as a short reviewer-facing overview.

Include only:

1. pricing pipeline works
2. exact-reference / ILP-derived encoding beats WHT
3. decoder robustness matters
4. 3-period extension shows positive dynamic-vs-static uplift

Required scope note:

- Qiskit evidence is structural subset parity/resource evidence, not full decode-inclusive DQI parity

## Verification

Verification should cover:

- notebook smoke/contract tests for the new packaging language and cells
- helper tests for verdict/export data if helper code changes
- in-place notebook execution so saved outputs reflect the intended reviewer-facing evidence

## Git Note

This workspace is not a git repository root, so the spec can be saved locally but cannot be committed from the current directory.
