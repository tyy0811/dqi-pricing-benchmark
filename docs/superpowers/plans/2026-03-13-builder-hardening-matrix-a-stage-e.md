# Builder Hardening for Matrix A and Stage E Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `scripts/make_report.py`'s private Stage E export behavior with the canonical builder path, persist builder-owned Matrix A comparison identities in `metrics_master`, and fail fast on invalid Stage E artifacts or missing audits.

**Architecture:** `scripts/make_report.py` remains the report orchestrator, but Stage E normalization/admission/dedup comes from `src/resources_compare.py`. Canonical normalized identity columns and `comparison_key` are computed once on `metrics_master`, then projected into Stage E outputs and audits. Stage D baseline logic and `counterpart_key` remain unchanged.

**Tech Stack:** Python, pandas, pytest, existing `src.report_schema` and `src.resources_compare` helpers

---

## Chunk 1: Lock the failing tests

### Task 1: Add Stage E regression tests for canonical export behavior

**Files:**
- Modify: `tests/test_make_report_stage_e_partition.py`
- Modify: `tests/test_make_report_stage_e_end_to_end.py`
- Modify: `tests/test_make_report_stage_e_identity.py`
- Create: `tests/test_make_report_stage_e_audits.py`

- [ ] **Step 1: Write failing tests for persisted Stage E status normalization**

Add assertions that exported `quality_cost_unavailable` rows never contain `run_status == "skipped"` and that planned/skipped/unsupported exported rows are `not_applicable`.

- [ ] **Step 2: Run targeted tests to verify they fail for the current implementation**

Run: `pytest tests/test_make_report_stage_e_partition.py tests/test_make_report_stage_e_end_to_end.py tests/test_make_report_stage_e_identity.py -v`
Expected: FAIL because the current private builder still exports `skipped`, lacks `comparison_key` projection, and does not emit required audits.

- [ ] **Step 3: Write failing tests for `metrics_master` identity columns and `comparison_key`**

Add assertions that `metrics_master` contains `matrix_norm`, `stage_norm`, `family_norm`, `encoding_norm`, `decoder_norm`, `alpha_mode_norm`, `execution_mode_norm`, and `comparison_key`, and that valid Matrix A WHT/ILP rows share a common `comparison_key`.

- [ ] **Step 4: Write failing tests for required audit artifacts**

Add a new test module asserting `matrix_a_pairing_audit`, `quality_cost_admission_audit`, and `quality_cost_duplicate_audit` are written and contain the required reconciliation and duplicate-determinism fields.

- [ ] **Step 5: Run the expanded Stage E test subset and verify it fails for the expected reasons**

Run: `pytest tests/test_make_report_stage_e_partition.py tests/test_make_report_stage_e_end_to_end.py tests/test_make_report_stage_e_identity.py tests/test_make_report_stage_e_audits.py -v`
Expected: FAIL on missing status normalization, missing projected identity columns, and missing audit artifacts.

## Chunk 2: Switch `make_report.py` to canonical Stage E outputs

### Task 2: Replace the private Stage E builder path and persist builder-owned identities

**Files:**
- Modify: `scripts/make_report.py`
- Modify: `src/resources_compare.py`
- Test: `tests/test_make_report_stage_e_partition.py`
- Test: `tests/test_make_report_stage_e_identity.py`

- [ ] **Step 1: Add a `metrics_master` identity enrichment pass**

Implement one helper in `scripts/make_report.py` that computes `run_key`, the normalized identity columns, and Matrix A `comparison_key` without changing Stage D `counterpart_key` behavior.

- [ ] **Step 2: Run the identity-focused tests and verify they still fail before Stage E export changes**

Run: `pytest tests/test_make_report_stage_e_identity.py -v`
Expected: FAIL until the new columns are persisted and exposed by `run_make_report()`.

- [ ] **Step 3: Replace `_build_quality_cost_tables()` with the canonical Stage E builder path**

Update `scripts/make_report.py` to pass enriched `metrics_master` records to `src.resources_compare.build_stage_e_quality_cost_tables()`. Remove private Stage E classification/dedup logic from the report layer.

- [ ] **Step 4: Extend Stage E row projection to preserve builder-owned identity fields**

Update `src/resources_compare.py` projection helpers so `quality_cost_master` and `quality_cost_unavailable` carry the normalized identity columns, `comparison_key`, `run_key`, and a report-owned `stage_e_admission_key`.

- [ ] **Step 5: Run the Stage E builder and projection tests**

Run: `pytest tests/test_stage_e_quality_cost_builder.py tests/test_resources_compare_stage_e.py tests/test_make_report_stage_e_partition.py tests/test_make_report_stage_e_identity.py -v`
Expected: PASS for canonical builder behavior and projected identity columns; FAIL only if remaining audit/schema work is incomplete.

## Chunk 3: Enforce schema-valid output and required audits

### Task 3: Add fail-fast validation and required debug artifacts

**Files:**
- Modify: `scripts/make_report.py`
- Modify: `src/report_schema.py` only if required for already-approved projection fields
- Create: `tests/test_make_report_stage_e_audits.py`

- [ ] **Step 1: Write the failing audit and schema validation tests**

Cover:
- build failure when `quality_cost_unavailable` rows violate schema
- build failure when a required audit cannot be generated or written
- audit columns for admission reconciliation and duplicate determinism

- [ ] **Step 2: Run the audit/schema test subset and verify red**

Run: `pytest tests/test_make_report_stage_e_audits.py tests/test_make_report_stage_e_partition.py -v`
Expected: FAIL because schema validation is not yet enforced and audits are not yet written.

- [ ] **Step 3: Implement artifact-write status normalization and schema validation**

Normalize exported Stage E statuses to schema-compatible values before writing and call `validate_quality_cost_master()` / `validate_quality_cost_unavailable()` before artifact publication.

- [ ] **Step 4: Implement required audit generation**

Write:
- `results/matrix_a_pairing_audit.(json|parquet as supported by existing helper behavior)`
- `results/quality_cost_admission_audit.(json|parquet as supported by existing helper behavior)`
- `results/quality_cost_duplicate_audit.(json|parquet as supported by existing helper behavior)`

Make audit generation fatal if any required artifact cannot be produced.

- [ ] **Step 5: Run the Stage E audit/schema tests and verify green**

Run: `pytest tests/test_make_report_stage_e_audits.py tests/test_make_report_stage_e_partition.py tests/test_make_report_stage_e_end_to_end.py -v`
Expected: PASS

## Chunk 4: Guard Stage D and finish verification

### Task 4: Prove Stage D is unchanged and run notebook-adjacent smoke coverage

**Files:**
- Modify: `scripts/make_report.py` only if Stage D regressions appear
- Test: `tests/test_make_report_stage_d_baseline_selector.py`
- Test: `tests/test_notebook08_paired_encoding_smoke.py`
- Test: `tests/test_notebook11_stage_e_smoke.py`
- Test: `tests/test_notebook12_stage_e_smoke.py`

- [ ] **Step 1: Run Stage D contract tests unchanged**

Run: `pytest tests/test_make_report_stage_d_baseline_selector.py -v`
Expected: PASS with no test edits required.

- [ ] **Step 2: Run notebook smoke tests for the impacted readers**

Run: `pytest tests/test_notebook08_paired_encoding_smoke.py tests/test_notebook11_stage_e_smoke.py tests/test_notebook12_stage_e_smoke.py -v`
Expected: PASS

- [ ] **Step 3: Run the full targeted regression set**

Run: `pytest tests/test_stage_e_quality_cost_builder.py tests/test_resources_compare_stage_e.py tests/test_make_report_stage_e_partition.py tests/test_make_report_stage_e_identity.py tests/test_make_report_stage_e_end_to_end.py tests/test_make_report_stage_e_audits.py tests/test_make_report_stage_d_baseline_selector.py tests/test_notebook08_paired_encoding_smoke.py tests/test_notebook11_stage_e_smoke.py tests/test_notebook12_stage_e_smoke.py -v`
Expected: PASS

- [ ] **Step 4: Regenerate report artifacts and verify notebook-facing outputs**

Run: `pytest tests/test_make_report_stage_e_end_to_end.py -v`
Expected: PASS with schema-valid Stage E artifacts, projected identity columns, and required audits present.

- [ ] **Step 5: Document any residual gaps**

If notebooks `08/11/12` cannot be executed directly in this environment, record that limitation and rely on the smoke tests plus regenerated artifacts as verification evidence.
