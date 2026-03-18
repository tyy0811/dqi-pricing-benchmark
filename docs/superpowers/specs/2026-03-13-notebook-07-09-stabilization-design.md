# Notebook 07/09 Stabilization Design

**Date:** 2026-03-13

## Goal

Stabilize the two newest reviewer-facing claims without adding new notebooks:

- Notebook 07 should become a parity-first subset diagnostic surface for the Qiskit path, while remaining honest if parity is not established.
- Notebook 09 should replace a single anecdotal 3-period example with a compact reviewer-facing scenario matrix.

## Scope

In scope:

- `notebooks/07_qiskit_structural_parity.ipynb`
- `notebooks/09_3period_pricing_extension.ipynb`
- small supporting changes in the Qiskit subset helpers and multiperiod helpers if they improve correctness or testability
- minimal, high-value tests for the new diagnostic surfaces

Out of scope:

- any new notebook
- full end-to-end DQI parity
- decode/uncompute support in Notebook 07
- a broad benchmark framework rewrite

## Design Rules

### Notebook 07

Notebook 07 is parity-first but not parity-required: perform a targeted diagnostic pass to resolve ordering, marginal, and basis mismatches; if meaningful subset agreement is not recovered, keep the notebook as structural/resource evidence only and make the non-parity status explicit with a tiny diagnostic comparison table.

Primary target:

- attempt to recover meaningful subset agreement by debugging register ordering, syndrome marginal extraction, final-Hadamard targeting, and NumPy/Qiskit basis-convention mismatches

Acceptable landing state:

- if the targeted pass does not materially reduce the mismatch, the notebook ships as structural/resource evidence only
- the mismatch must be visible rather than hidden
- no notebook language may imply successful subset parity validation unless supported by the diagnostic outputs

Required outputs after rerun:

- tiny hand-checkable case
- top-8 probabilities from the NumPy subset reference
- top-8 probabilities from the Qiskit subset path
- side-by-side comparison on explicit syndrome labels
- raw TVD
- optional diagnostic remapped/permuted TVD if it helps isolate ordering issues
- `qubit_count`
- `transpiled_depth`
- `cx_count`
- `gate_counts`

Implementation intent:

- add only a thin diagnostic layer around existing subset APIs
- patch only demonstrable ordering or marginal bugs
- do not silently “fix” mismatches with post-hoc relabeling

### Notebook 09

Notebook 09 should become a compact reviewer-facing scenario matrix instead of a single anecdote.

Scenario selection rule:

Select scenarios to maximize regime coverage and interpretability, while ensuring the table includes both positive-uplift and near-zero-uplift cases; do not cherry-pick rows to mostly maximize `static_vs_dynamic_delta`.

Required matrix coverage:

- demand shift: low / medium / high
- inventory regime: at least 2 levels
- smoothing penalty: `0` and one nonzero value
- target table size: roughly 6-9 rows

Required table fields:

- scenario label
- `dynamic_horizon_revenue`
- `static_horizon_revenue`
- `static_vs_dynamic_delta`
- remaining inventory and/or usage summary
- whether package choice changes across periods

Interpretation rule:

- show the compact table first
- keep any follow-up interpretation to one short reviewer-facing cell

## Test Strategy

Add only small, high-value tests.

Notebook 07 / Qiskit subset path:

- tiny-case diagnostic test that would catch ordering or marginalization bugs
- normalization check for the subset distribution
- resource key contract check for the subset report

Notebook 09 / multiperiod support:

- small helper or contract test for scenario matrix generation
- verify required fields/columns are present
- verify the matrix size stays within the intended small range

## Acceptance Criteria

### Path A: Notebook 07 becomes parity evidence

Accept if:

- a real ordering or marginal bug is identified and fixed
- the tiny-case diagnostic shows materially improved subset agreement
- notebook wording can honestly present subset parity evidence

### Path B: Notebook 07 remains resource evidence only

Accept if:

- the targeted diagnostic pass is completed
- the notebook shows the tiny-case side-by-side probability table
- the notebook explicitly states parity is not established
- the notebook is framed as structural/resource evidence only
- no misleading parity language remains

Both paths remain subset-only and exclude decode/uncompute.

## Constraints

- minimal diff only
- keep both notebooks in place
- no benchmark.py rewrite unless strictly required
- no scope creep into full DQI parity
- preserve honest scientific framing

## Git Note

This workspace does not appear to be a git repository root, so this spec can be saved locally but cannot be committed from the current directory structure.
