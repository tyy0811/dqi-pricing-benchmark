# P6/P7 Recovery And Matrix C Design

## Context

The branch already contains the first scaling slice for P6/P7:

- P6/P7 frozen pricing instances and paired artifacts exist.
- Structured boundary handling is implemented.
- `run_status` is the canonical status field.
- Matrix A canonical outputs already include:
  - P6 completed rows
  - P7 explicit `unavailable` rows with `boundary_reason="qubit_count_exceeded"`

The branch is not yet integration-safe because the full test suite still fails. The current failure mode is notebook/runtime contract drift, not scaling logic. The next design therefore separates recovery into two strict phases:

1. Restore notebook/runtime contract compatibility until `python3 -m pytest tests/ -x` is green.
2. Only then resume Matrix C P-family integration for P6/P7 through the canonical path.

## Goals

- Recover branch integration safety without regenerating unrelated artifacts.
- Preserve the existing canonical boundary-row schema and write path.
- Add P6/P7 Matrix C canonical coverage only after tests are green.
- Leave frozen P3/P4/P5 values unchanged.

## Non-Goals

- No Notebook 08 expansion in this work.
- No Notebook 13 creation in this work.
- No new pricing instances or paired artifacts.
- No reintroduction of `status` as a canonical field.
- No ad hoc canonical writes outside the existing report pipeline.

## Approach Options

### Option 1: Two-phase recovery, then Matrix C integration

Phase 1 restores notebook JSON/code-cell contracts only. Phase 2 extends Matrix C P-family execution through `run_matrix_c(...)` and rebuilds canonical outputs through `scripts/make_report.py`.

Pros:

- Clean debugging boundary between notebook contract drift and scaling execution work.
- Preserves one authoritative Matrix C pricing-family path.
- Minimizes accidental canonical-output churn during recovery.

Cons:

- Slightly slower overall because canonical regeneration is deferred until after test recovery.

### Option 2: Interleave notebook fixes and Matrix C generation

Repair notebook contracts while also adding new Matrix C outputs and regenerating canonical artifacts.

Pros:

- Potentially faster if all failures are tightly coupled.

Cons:

- Blurs causality.
- Makes it harder to isolate whether failures come from notebook contracts or new data generation.

### Option 3: Patch tests to fit current notebook drift

Keep notebook structure as-is and weaken tests.

Pros:

- Lowest immediate code churn.

Cons:

- Weakens established notebook contracts.
- Makes the branch less trustworthy and harder to maintain.

## Recommendation

Use Option 1.

Phase 1 should be a strict notebook-interface repair pass with no canonical regeneration. Phase 2 should then extend Matrix C P-family rows through `run_matrix_c(...)` and the existing `make_report` rebuild path. This preserves the repo’s current architecture:

- one authoritative Matrix C pricing-family path
- one canonical write path
- one ingestion path through `make_report`
- one boundary-row contract shared with Matrix A

## Phase 1: Notebook/Runtime Contract Restoration

### Scope

Restore notebook JSON/code-cell contract expectations only. Do not regenerate canonical metrics, reports, or notebook outputs unless a failing test explicitly requires persisted executed outputs.

### Primary Target

`notebooks/06_resource_analysis.ipynb`

Restore the exact literal marker and contract expected by the notebook runtime tests:

`PANEL_SAME = "Structure vs decode (same-model authoritative)"`

Also restore any missing placeholder/unavailable-panel behavior expected when inputs such as `resources_master` or `metrics_master` are absent.

### Secondary Audit

Check only source-level contract expectations for:

- `notebooks/10_alpha_finite_size_validation.ipynb`
- `notebooks/12_quality_vs_cost.ipynb`

Allowed edits in this phase:

- restoring exact marker strings
- restoring expected helper-cell variables/constants
- restoring missing fallback text/panels for absent data
- restoring notebook source structure parsed by tests

Disallowed edits in this phase:

- changing notebook scientific claims
- changing canonical metrics/report generation
- broad notebook re-execution
- regenerating results just to satisfy source-contract tests

### Validation Order

1. Run the currently failing notebook/runtime test first.
2. Run notebook-runtime and notebook-helper suites.
3. Run the full suite:
   - `python3 -m pytest tests/ -x`

### Acceptance

- `python3 -m pytest tests/ -x` passes.
- Notebook marker expectations are restored.
- No new canonical metrics generation is introduced just to satisfy notebook tests.
- Existing P3/P4/P5 values remain unchanged.

## Phase 2: Matrix C P6/P7 Canonical Integration

### Authoritative Path

Extend only the pricing-family branch inside:

`src/matrix_runner.py::run_matrix_c(...)`

Do not create a second P-family Matrix C producer. In particular, do not make `src/decoder_study_runner.py` an alternative canonical pricing-family Matrix C path unless forced by an existing ingestion/test contract.

### Canonical Contract

Preserve the Matrix A boundary-row semantics exactly:

- canonical status field: `run_status`
- allowed statuses only:
  - `completed`
  - `unavailable`
  - `failed`
- `attempted_execution=False` only for pre-execution policy skips
- `attempted_execution=True` if execution was actually entered

Preserve grouping keys on every Matrix C pricing-family row, including boundary rows:

- `run_id`
- `instance_id`
- `encoding`
- `decoder`
- `alpha_mode`
- `execution_mode`
- `ell`

Preserve the optional fields already introduced:

- `attempted_execution`
- `boundary_reason`
- `error_type`
- `error_message`
- `estimated_qubits`
- `estimated_memory_gb`
- `wall_clock_s`
- `peak_memory_mb`

### Execution Semantics

- Extend pricing-family execution only.
- Leave S-family semantics unchanged apart from already-supported optional columns.
- If P7 completes, write completed rows normally.
- If P7 is rejected by policy or fails, emit explicit canonical rows with:
  - `run_status="unavailable"` or `run_status="failed"`
  - structured boundary metadata
- Never silently omit attempted or preflight-rejected P7 configurations.

### Canonical Rebuild

After Matrix C generation, rebuild canonical outputs only through:

`scripts/make_report.py`

Do not write directly into `results/metrics_master.json` or any other canonical artifact.

### Verification

After regeneration, load `results/metrics_master.json` and summarize:

`{instance_id: {completed: N, unavailable: M, failed: K, not_applicable: M2}}`

Acceptance means:

- P6 and P7 have pricing-family Matrix C rows in canonical outputs.
- Those rows are either:
  - `run_status="completed"`, or
  - explicit boundary rows with `run_status in {"unavailable", "failed"}`
- P7 failed or preflight-rejected configurations are represented explicitly, never omitted.
- P3/P4/P5 values remain unchanged.
- Full test suite still passes after regeneration:
  - `python3 -m pytest tests/ -x`

## Testing Strategy

### Phase 1

- notebook/runtime smoke tests
- notebook helper/hygiene tests
- full suite gate

### Phase 2

- targeted Matrix C runner/report tests for P6/P7
- canonical-master presence and status-count verification
- full suite gate after regeneration

## Risks And Mitigations

### Risk: Notebook contract repair accidentally broadens notebook behavior

Mitigation:

- Restrict edits to exact source-level markers, helper variables, and missing-data placeholders required by tests.

### Risk: Matrix C P6/P7 rows duplicate or conflict with existing Stage C raw rows

Mitigation:

- Keep `run_matrix_c(...)` as the only authoritative pricing-family Matrix C generator.
- Preserve grouping keys and canonical rebuild through `make_report` only.

### Risk: P7 failures disappear during canonical rebuild

Mitigation:

- Ensure every attempted or preflight-rejected configuration produces a row.
- Verify status counts from `results/metrics_master.json` after rebuild.

## Deliverables

1. Minimal notebook/runtime contract fixes restoring full test green status.
2. Matrix C P6/P7 canonical integration using the existing boundary-row semantics.
3. A short implementation summary reporting:
   - what notebook/runtime contract issue was fixed
   - whether full `pytest -x` is green
   - whether Matrix C P6/P7 rows were added to the canonical master
   - final per-instance status summary
