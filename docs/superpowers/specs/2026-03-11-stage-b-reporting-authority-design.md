# Stage B Reporting Authority Design

## Scope
Create a manifest-driven reporting authority layer so notebooks read one canonical run/metric/resource stack instead of inferring semantics from ad hoc per-notebook artifacts.

This stage introduces:
- `scripts/make_report.py`
- canonical outputs under `results/`
- notebook helper loaders for manifest/master tables
- notebook rewiring (reader-first), with minimum markdown churn

## Goals
1. Establish one authoritative intended run grid (`run_manifest.json`).
2. Build one canonical metrics table (`metrics_master`) and one canonical resources table (`resources_master`) keyed by manifest slots.
3. Regenerate matrix compatibility views (`matrix_a/b/c_summary.csv`, `matrix_a/b/c_report.json`) as pure projections from masters.
4. Distinguish clearly between:
   - planned but not executed,
   - observed and failed,
   - out-of-matrix display slices.

## Non-Goals
- No synthetic master-table rows for full outside-the-matrix Cartesian combinations.
- No broad notebook narrative rewrites.
- No additional benchmark execution logic.

## Canonical Authority Model

### Planned-slot authority
`results/run_manifest.json` is authoritative for intended slots.

Manifest slot primary key:
- `slot_id` (deterministic hash of normalized identity key)

Manifest identity fields:
- `slot_id`
- `stage`
- `matrix`
- `family`
- `instance_id`
- `encoding`
- `decoder`
- `alpha_mode`
- `execution_mode`
- `seed`

Manifest metadata fields:
- `report_schema_version`
- `manifest_version`
- `created_at_utc`
- `source_commit`
- `source_manifest_path`

### Master tables
- `results/metrics_master.parquet` (or JSON fallback)
- `results/resources_master.json` (or parquet+json sidecar if needed)

Both are one row per manifest-defined `slot_id` only.

No rows are added for out-of-matrix combinations.

## Canonical Status Contract

### Slot execution status
- `run_status ∈ {completed, failed, skipped}`
- `observed_run ∈ {true, false}`
- `status_reason_code`
- `status_reason`
- `run_error`
- `in_experiment_matrix` (always true in master rows)

For any manifest-defined slot with no observed run:
- `run_status = "skipped"`
- `observed_run = false`
- `status_reason_code = "planned_not_executed"` by default
- or `status_reason_code = "intentionally_skipped"` only with explicit skip evidence.

### Metric availability semantics
Per metric `M`:
- `availability_M = "available"` => `M` may be non-null
- `availability_M in {"failed_upstream", "not_applicable"}` => `M` must be null
- flattened `availability_M` must equal `metric_availability[M]`

Display-only state:
- `not_in_experiment_matrix` is notebook-display logic when a requested slice is absent from manifest; not a stored synthetic row in masters.

## Duplicate Observed-Run Resolution
If multiple observed rows map to one slot:
1. exact normalized identity match including canonical seed
2. prefer `completed` over non-completed
3. newest `run_timestamp_utc`
4. fallback file mtime
5. final deterministic fallback: lexicographically smallest `run_id`

All conflicting `run_id`s are preserved in provenance fields:
- `duplicate_run_ids`
- `duplicate_observation_count`

## Seed Normalization and Slot Hashing
Seed canonicalization rules:
- missing seed -> `NULL`
- numeric seed -> canonical integer string
- hash input includes sentinel `seed=<NULL>` for null seed

`slot_id` hash input:
`stage|matrix|family|instance_id|encoding|decoder|alpha_mode|execution_mode|seed=<normalized>`

## Row-level Provenance Contract

### metrics_master row provenance
- `observed_metrics_path`
- `observed_report_path`
- `producer`
- `source_matrix_manifest`
- `source_row_index`
- `ingested_at_utc`
- `observed_at_utc`
- `normalization_version`

### resources_master additional provenance
- `resource_scope`
- `resource_key`
- `resource_confidence`
- `resource_status`
- `resource_reason_code`
- `resource_model_version`

## Compatibility View Contract
Regenerated compatibility outputs are strict projections from master tables + manifest metadata and must not contain information absent from masters:
- `results/tables/matrix_a_summary.csv`
- `results/tables/matrix_b_summary.csv`
- `results/tables/matrix_c_summary.csv`
- `results/reports/matrix_a_report.json`
- `results/reports/matrix_b_report.json`
- `results/reports/matrix_c_report.json`

## `scripts/make_report.py` Responsibilities
1. Build manifest slot grid from matrix definitions/config and existing matrix manifests.
2. Read observed source summaries/reports for requested matrices.
3. Normalize identities + seed and assign `slot_id`.
4. Join observed rows to manifest slots.
5. Apply status mapping + availability normalization.
6. Resolve duplicates using the locked tie-break policy.
7. Emit canonical master outputs.
8. Emit compatibility matrix views from masters.

## Hard Fail vs Soft Warning

### Hard-fail
- Missing core inputs (manifest/grid definitions + required observed source summaries/reports for requested matrices)
- Duplicate `slot_id` in manifest grid generation
- Malformed identity fields for planned slots

### Soft warnings
- Duplicate observed runs collapsed by tie-break policy
- Rows with parse issues converted to `failed` + `status_reason_code="parse_error"`

## Notebook Integration Strategy
1. Add `_helpers.py` loaders:
   - `load_run_manifest(results_root)`
   - `load_metrics_master(results_root)`
   - `load_resources_master(results_root)`
   - status/availability normalization helpers for display
2. Rewrite notebook 06 first to reader mode (no local artifact generation).
3. Switch notebooks 11 and 12 to master loaders.
4. Switch notebooks 08, 09, 10 to master loaders last, preserving existing reviewer-facing panels.

## Acceptance Criteria
- `python scripts/make_report.py` produces:
  - `results/run_manifest.json`
  - `results/metrics_master.parquet` (or JSON fallback)
  - `results/resources_master.json`
- Notebooks no longer generate authoritative metrics/resources locally.
- Notebook 06 stops treating `small_instance/default_instance` as authoritative inputs.
- `not_in_experiment_matrix` is displayed from manifest-absence logic, not synthetic master rows.
- Matrix A/B/C compatibility views are regenerated from masters.
