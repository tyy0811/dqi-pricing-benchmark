# Notebook 08 Pairing and Stage E Follow-up Design

## Scope and Non-Goals

This change set is limited to three follow-up items after Stage E hardening:

1. Fix Notebook 08 so it uses canonical `comparison_key` pairing when available.
2. Guarantee that `quality_cost_duplicate_audit` writes a schema-stable empty artifact.
3. Trace Stage C and Stage D missing-quality-field blockers and patch `scripts/make_report.py` only if the missing fields already exist upstream in raw source rows.

Non-goals:

- redesigning Notebook 08 output structure
- redesigning Stage E contracts
- inventing or synthesizing missing Stage C/D quality fields
- changing Stage D fixed-baseline behavior

## Locked Notebook 08 Pairing Contract

`notebooks/08_paired_encoding_comparison.ipynb` uses `comparison_key` as the authoritative pairing identity when that column is present in the input dataframe.

In the canonical path:

- `pair_keys == ["comparison_key"]` exactly
- legacy inferred key construction is skipped
- completed-row uniqueness is checked using normalized status:
  - count completed WHT rows where `run_status_norm == "completed"` and encoding is WHT-family
  - count completed ILP rows where `run_status_norm == "completed"` and encoding is ILP-family
- there must be at most one completed WHT row and at most one completed ILP row per `comparison_key`
- if that uniqueness check fails, the notebook does not repair or collapse rows; it routes to the unavailable path with a concise reason

Legacy fallback is allowed only when `comparison_key` is absent. In that fallback path, `slot_id` is excluded from pairing candidates.

The notebook will keep an internal-only dataframe named `pairing_debug_summary_df` containing:

- unique `comparison_key`
- WHT row count
- ILP row count
- matched pair count
- duplicate condition count when canonical uniqueness fails

This dataframe is for tests and runtime checks only and is not rendered as a normal notebook output.

## Duplicate-Audit Empty-Schema Contract

`quality_cost_duplicate_audit` must always be written with a fixed declared schema, even when the table has zero rows.

The schema is declared in code, not inferred from runtime rows. Zero-row output must still include the full expected column set so downstream readers and tests see a stable artifact header.

## Stage C/D Investigation Boundary

Stage C/D work starts as read/trace only.

For Stage C `decoder_study` rows:

- inspect raw source rows before report projection
- determine whether `retained_energy_eta` already exists upstream

For Stage D `alpha_validation` rows:

- inspect raw source rows before report projection
- determine whether `topk_regret`, `decision_distortion`, `spearman_rho_F_G`, and `retained_energy_eta` already exist upstream

Decision rule:

- if those fields already exist upstream, patch `scripts/make_report.py` to preserve them into `metrics_master`, which will allow canonical Stage E admission when the rest of the row is valid
- if those fields do not exist upstream, do not synthesize them; record the blocker as missing experiment output

## Test Contract

Add or update tests for:

- Notebook 08 canonical pairing: fail when valid upstream `comparison_key` pairs exist but the notebook still renders the placeholder
- Notebook 08 duplicate-completed canonical pair handling: fail if the notebook silently merges duplicate completed rows
- `quality_cost_duplicate_audit` zero-row schema preservation
- Stage C/D projection only if raw-source trace proves the fields already exist upstream

Existing Stage D fixed-baseline tests remain unchanged.

## Verification / Rerun Plan

After implementation:

1. rerun `python scripts/make_report.py --results-root results --matrices matrix_a,matrix_b,matrix_c`
2. execute `notebooks/08_paired_encoding_comparison.ipynb`
3. verify that the paired WHT-vs-ILP delta table renders when valid `comparison_key` pairs exist
4. verify that `quality_cost_duplicate_audit` is schema-shaped when empty
5. if a Stage C/D projection patch was made, re-execute notebooks 11 and 12 and inspect whether Stage C or Stage D comparable slices are newly populated

## Regression Boundaries

- Stage E schema and canonical builder behavior remain unchanged
- Stage D baseline selection and selected-baseline invariants remain unchanged
- Notebook 08 output stays focused on the paired result table or a concise unavailable reason
