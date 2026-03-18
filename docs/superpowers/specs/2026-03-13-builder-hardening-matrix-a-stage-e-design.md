# Builder Hardening for Matrix A and Stage E

## Purpose

Unify `scripts/make_report.py` with the canonical Stage E reporting contract, add a builder-owned Matrix A `comparison_key`, and make admission/dedup behavior artifact-visible through required audit outputs.

## Locked Constraints

- Persisted artifacts must not export `run_status="skipped"`.
- Planned/skipped/unsupported internal states normalize to `run_status="not_applicable"` at artifact write.
- Stage E unavailable rows must be produced through the canonical builder contract and must include `missing_required_fields`, `missing_quality_fields`, and `missing_resource_fields`.
- Builder-owned estimated-comparable rows must carry `resource_comparability_basis`; otherwise they route to canonical unavailable output.
- Audit artifacts are required outputs. Failure to generate or write any audit fails the build.
- Stage D fixed-baseline behavior is unchanged and protected by existing tests.

## Identity Keys

- `run_key`: unique per persisted run row.
- `comparison_key`: canonical Matrix A paired-comparison identity; excludes `encoding_norm` so WHT and ILP can pair.
- `counterpart_key`: canonical coherent-vs-mixture identity for Stage D.
- `stage_e_admission_key`: sink-specific dedup/admission identity for `quality_cost_master`.

## Pipeline and Ownership

`metrics_master` remains the builder-owned source of truth for all reporting rows. After existing master row assembly, and before writing `metrics_master` or building downstream report tables, `scripts/make_report.py` computes and persists canonical normalized identity columns on `metrics_master`: `matrix_norm`, `stage_norm`, `family_norm`, `encoding_norm`, `decoder_norm`, `alpha_mode_norm`, and `execution_mode_norm`.

In that same pass, `scripts/make_report.py` computes a canonical Matrix A `comparison_key` from the scientific identity used to pair WHT and ILP rows. `comparison_key` intentionally excludes `encoding_norm` and is populated for rows in the Matrix A paired-comparison domain; it may be null outside that domain. Stage D continues to use its existing `counterpart_key` unchanged.

Stage E export does not classify or define identity locally. Instead, `scripts/make_report.py` passes builder-owned rows from `metrics_master` into the canonical Stage E builder in `src/resources_compare.py`, and `quality_cost_master` / `quality_cost_unavailable` project the normalized identity columns and `comparison_key` forward from `metrics_master` rather than recomputing them.

Stage D code paths, baseline selection, and selected-baseline invariants remain unchanged except for continuing to consume the same normalized identity columns already present on `metrics_master`.

After Stage E tables are built, `scripts/make_report.py` validates `quality_cost_master` and `quality_cost_unavailable` against `src/report_schema.py` before artifact write. Schema-invalid outputs fail the build.

## Stage E Behavior and Artifacts

Raw internal row states can still exist during assembly, but artifact-write paths must not export `run_status="skipped"`. For persisted outputs, planned, skipped, and unsupported cases normalize to `run_status="not_applicable"`, while `status_reason_code` and `status_reason` preserve the original distinction.

Stage E unavailable rows must come from the canonical builder contract, including required `missing_required_fields`, `missing_quality_fields`, and `missing_resource_fields` lists. `scripts/make_report.py` does not maintain a private unavailable-row shape.

Builder-owned candidate rows passed into Stage E must carry `resource_comparability_basis` whenever resources are `estimated_comparable`. Missing basis routes rows into canonical unavailable output instead of dropping them.

`quality_cost_admission_audit` must include at least:

- `run_key`
- `comparison_key` when present
- `counterpart_key` when present
- `stage_e_admission_key`
- `matrix_norm`
- `stage_norm`
- `family_norm`
- `encoding_norm`
- `decoder_norm`
- `alpha_mode_norm`
- `execution_mode_norm`
- admission outcome fields
- canonical unavailable reason fields when rejected

`quality_cost_duplicate_audit` must include:

- duplicate group key
- winner row id or `run_id`
- loser row ids
- ranking inputs used to select the winner
- final disposition for each loser

`matrix_a_pairing_audit` must expose the fields needed to inspect `comparison_key` construction and whether a Matrix A row is pairable across WHT and ILP.

Audit artifact generation is required. If any audit cannot be generated or written, the build fails rather than publishing partial Stage E artifacts.

## Required Audits

- `matrix_a_pairing_audit`
- `quality_cost_admission_audit`
- `quality_cost_duplicate_audit`

## Validation

- Schema validation runs before artifact publish.
- Invalid Stage E outputs fail the build.
- Missing required audits fail the build.

## Rollout

1. Replace private Stage E unavailable-row path with canonical builder output.
2. Add artifact-write status normalization.
3. Add normalized identity columns and `comparison_key`.
4. Add required audit generation.
5. Add schema validation and fail-fast behavior.
6. Regenerate artifacts and execute notebooks 08/11/12.
