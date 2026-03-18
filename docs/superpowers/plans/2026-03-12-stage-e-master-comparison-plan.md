# Stage E Master Comparison Layer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single Stage E comparison layer that produces `quality_cost_master` (completed/comparable only) and `quality_cost_unavailable` (explicit unavailable rows), then rewire notebooks 11/12 to consume those authoritative outputs while staying thin.

**Architecture:** Reuse `scripts/make_report.py` as the single writer and introduce focused `src/` helpers for faithfulness normalization, paired-encoding validation scope checks, and resource comparability normalization. Build Stage E outputs after existing `metrics_master/resources_master`, enforce deterministic duplicate resolution with `manifest_slot`, and keep notebooks as reader-only layers.

**Tech Stack:** Python 3.11, pandas, pytest, Jupyter notebooks, existing repo reporting pipeline.

---

## Chunk 1: Stage E Core Schema + Utilities

### Task 1: Add Stage E faithfulness utility module

**Files:**
- Create: `src/faithfulness.py`
- Test: `tests/test_faithfulness_stage_e.py`

- [ ] **Step 1: Write failing tests for faithfulness normalization/guardrails**

```python
# tests/test_faithfulness_stage_e.py

def test_faithfulness_uses_existing_values_when_present():
    ...

def test_faithfulness_does_not_compute_without_verified_provenance():
    ...

def test_surrogate_gap_computation_when_inputs_available():
    ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_faithfulness_stage_e.py -v`  
Expected: FAIL (module/functions missing)

- [ ] **Step 3: Implement minimal module**

Implement in `src/faithfulness.py`:
- `normalize_faithfulness_fields(row: dict) -> dict`
- `can_compute_faithfulness(row: dict) -> tuple[bool, str|None]`
- `compute_surrogate_best_gap(row: dict) -> float|None`
- never coerce unavailable values to zero

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_faithfulness_stage_e.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_faithfulness_stage_e.py src/faithfulness.py
git commit -m "feat(stage-e): add faithfulness normalization utilities"
```

### Task 2: Add paired-encoding validator with explicit opt-in scope

**Files:**
- Create: `src/verify_encoding.py`
- Test: `tests/test_verify_encoding_stage_e.py`

- [ ] **Step 1: Write failing tests for scoped validation behavior**

```python
# tests/test_verify_encoding_stage_e.py

def test_validation_applies_only_when_expects_flag_true():
    ...

def test_validation_passes_for_joinable_paired_artifact():
    ...

def test_validation_failure_returns_machine_reason_code():
    ...
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_verify_encoding_stage_e.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement minimal validator**

Implement in `src/verify_encoding.py`:
- constant `PAIR_VALIDATION_FLAG = "expects_paired_encoding_validation"`
- `should_validate_paired_encoding(row: dict) -> bool`
- `validate_paired_encoding_row(row: dict, paired_artifact: dict|None) -> tuple[bool, str|None, str|None]`
- required outputs: `(ok, unavailable_reason_code, unavailable_reason)`

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_verify_encoding_stage_e.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_verify_encoding_stage_e.py src/verify_encoding.py
git commit -m "feat(stage-e): add scoped paired-encoding validation"
```

### Task 3: Evolve resources normalization for Stage E comparability

**Files:**
- Modify: `src/resources_compare.py`
- Test: `tests/test_resources_compare_stage_e.py`

- [ ] **Step 1: Write failing tests for comparability status mapping**

```python
# tests/test_resources_compare_stage_e.py

def test_normalize_resources_emits_qubits_gates_depth_fields():
    ...

def test_estimated_comparable_requires_comparability_basis():
    ...

def test_non_comparable_resource_status_flagged():
    ...
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_resources_compare_stage_e.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement resource normalization helpers**

Add/modify helpers:
- `normalize_resource_row_for_stage_e(row: dict) -> dict`
- accepted statuses: `{"comparable", "estimated_comparable"}`
- if `estimated_comparable`, require basis in:
  - `same_estimator`, `same_calibration`, `same_method_version`

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_resources_compare_stage_e.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_resources_compare_stage_e.py src/resources_compare.py
git commit -m "feat(stage-e): normalize resource comparability schema"
```

---

## Chunk 2: `make_report.py` Stage E Builder

### Task 4: Add manifest_slot requirement and canonical seed fallback

**Files:**
- Modify: `scripts/make_report.py`
- Test: `tests/test_make_report_stage_e_identity.py`

- [ ] **Step 1: Write failing identity tests**

```python
# tests/test_make_report_stage_e_identity.py

def test_manifest_slot_required_in_run_manifest_rows():
    ...

def test_canonical_trial_seed_precedence():
    # canonical_trial_seed ?? trial_seed ?? seed
    ...
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_make_report_stage_e_identity.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement in `make_report.py`**

- Ensure `manifest_slot` emitted in `run_manifest` rows
- Add deterministic seed precedence normalization before Stage E keying

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_make_report_stage_e_identity.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_make_report_stage_e_identity.py scripts/make_report.py
git commit -m "feat(stage-e): enforce manifest_slot and seed canonicalization"
```

### Task 5: Implement Stage E partition outputs

**Files:**
- Modify: `scripts/make_report.py`
- Test: `tests/test_make_report_stage_e_partition.py`

- [ ] **Step 1: Write failing partition tests**

```python
# tests/test_make_report_stage_e_partition.py

def test_completed_comparable_rows_go_to_quality_cost_master_only():
    ...

def test_unavailable_rows_preserved_with_class_and_reason_fields():
    ...

def test_quality_cost_files_written_json_and_parquet_or_fallback_json():
    ...
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_make_report_stage_e_partition.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement Stage E build step**

In `make_report.py`:
- Build Stage E rows after master metrics/resources
- Output:
  - `results/quality_cost_master.parquet|json`
  - `results/quality_cost_unavailable.parquet|json`
- Explicit columns for unavailable:
  - `unavailable_class`, `unavailable_reason_code`, `unavailable_reason`
- Preserve provenance fields in both outputs

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_make_report_stage_e_partition.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_make_report_stage_e_partition.py scripts/make_report.py
git commit -m "feat(stage-e): add quality_cost master/unavailable outputs"
```

### Task 6: Add duplicate-key deterministic winner + unavailable routing

**Files:**
- Modify: `scripts/make_report.py`
- Test: `tests/test_make_report_stage_e_duplicates.py`

- [ ] **Step 1: Write failing duplicate tests**

```python
# tests/test_make_report_stage_e_duplicates.py

def test_duplicate_key_definition_is_locked():
    # instance_id, encoding, decoder, alpha_mode, execution_mode
    ...

def test_duplicate_winner_tiebreak_order():
    # completed > verified provenance > earlier manifest_slot > lexical run_id
    ...

def test_duplicate_losers_go_to_unavailable_with_duplicate_key_class():
    ...
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_make_report_stage_e_duplicates.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement duplicate resolver logic**

Implement deterministic resolver and loser routing to unavailable with stable reason code.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_make_report_stage_e_duplicates.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_make_report_stage_e_duplicates.py scripts/make_report.py
git commit -m "feat(stage-e): deterministic duplicate resolution and routing"
```

---

## Chunk 3: Notebook Integration (Thin Reader Contract)

### Task 7: Add helper loaders for Stage E files

**Files:**
- Modify: `notebooks/_helpers.py`
- Test: `tests/test_notebook_helpers_stage_e.py`

- [ ] **Step 1: Write failing helper tests**

```python
# tests/test_notebook_helpers_stage_e.py

def test_load_quality_cost_master_prefers_parquet_fallback_json():
    ...

def test_load_quality_cost_unavailable_prefers_parquet_fallback_json():
    ...
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_notebook_helpers_stage_e.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement helper functions**

Add loaders:
- `load_quality_cost_master(results_root)`
- `load_quality_cost_unavailable(results_root)`

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_notebook_helpers_stage_e.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook_helpers_stage_e.py notebooks/_helpers.py
git commit -m "feat(stage-e): add notebook loaders for quality_cost outputs"
```

### Task 8: Update notebook 11 source precedence + placeholder contract

**Files:**
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`
- Test: `tests/test_notebook11_stage_e_smoke.py`

- [ ] **Step 1: Write failing notebook smoke test**

```python
# tests/test_notebook11_stage_e_smoke.py

def test_notebook11_missing_or_empty_quality_cost_master_renders_placeholder():
    ...
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_notebook11_stage_e_smoke.py -v`  
Expected: FAIL

- [ ] **Step 3: Patch notebook logic (minimal churn)**

- Comparable panels from `quality_cost_master` only
- Transparency card from `quality_cost_unavailable` if present
- Status/availability from `metrics_master/run_manifest` only
- Explicit empty/missing-file placeholders

- [ ] **Step 4: Execute notebook**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/11_decision_faithfulness_dashboard.ipynb`  
Expected: executes with outputs; no hard crash on missing/empty Stage E comparable data

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook11_stage_e_smoke.py notebooks/11_decision_faithfulness_dashboard.ipynb
git commit -m "feat(stage-e): rewire notebook 11 to quality_cost precedence"
```

### Task 9: Update notebook 12 to use `quality_cost_master` only for analysis rows

**Files:**
- Modify: `notebooks/12_quality_vs_cost.ipynb`
- Test: `tests/test_notebook12_stage_e_smoke.py`

- [ ] **Step 1: Write failing notebook smoke + thinness guard test**

```python
# tests/test_notebook12_stage_e_smoke.py

def test_notebook12_reads_quality_cost_master_for_pareto():
    ...

def test_notebook12_handles_empty_or_missing_quality_cost_master():
    ...
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_notebook12_stage_e_smoke.py -v`  
Expected: FAIL

- [ ] **Step 3: Patch notebook logic**

- Use `quality_cost_master` for Pareto and comparison slices
- Keep cells free of identity-changing merge/join/groupby
- Optional unavailable count card from `quality_cost_unavailable`

- [ ] **Step 4: Execute notebook**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb`  
Expected: PASS with reviewer-facing outputs

- [ ] **Step 5: Commit**

```bash
git add tests/test_notebook12_stage_e_smoke.py notebooks/12_quality_vs_cost.ipynb
git commit -m "feat(stage-e): rewire notebook 12 to quality_cost_master"
```

---

## Chunk 4: End-to-End Verification and Reporting

### Task 10: End-to-end report generation check

**Files:**
- Modify: `tests/test_make_report_stage_e_end_to_end.py`

- [ ] **Step 1: Write end-to-end failing test**

```python
# tests/test_make_report_stage_e_end_to_end.py

def test_make_report_emits_stage_e_outputs_and_preserves_unavailable_auditability():
    ...
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_make_report_stage_e_end_to_end.py -v`  
Expected: FAIL

- [ ] **Step 3: Finalize report output metadata**

Ensure return payload from `run_make_report(...)` includes Stage E output paths and row counts.

- [ ] **Step 4: Run full Stage E test set**

Run:
`pytest tests/test_faithfulness_stage_e.py tests/test_verify_encoding_stage_e.py tests/test_resources_compare_stage_e.py tests/test_make_report_stage_e_identity.py tests/test_make_report_stage_e_partition.py tests/test_make_report_stage_e_duplicates.py tests/test_notebook_helpers_stage_e.py tests/test_notebook11_stage_e_smoke.py tests/test_notebook12_stage_e_smoke.py tests/test_make_report_stage_e_end_to_end.py -v`

Expected: all PASS

- [ ] **Step 5: Regenerate artifacts and execute notebooks**

Run:
- `python scripts/make_report.py --results-root results --matrices matrix_a,matrix_b,matrix_c`
- `jupyter nbconvert --to notebook --execute --inplace notebooks/11_decision_faithfulness_dashboard.ipynb`
- `jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb`

Expected: Stage E outputs exist and notebooks render from canonical artifacts.

- [ ] **Step 6: Commit**

```bash
git add scripts/make_report.py results/quality_cost_master.json results/quality_cost_unavailable.json notebooks/11_decision_faithfulness_dashboard.ipynb notebooks/12_quality_vs_cost.ipynb tests
git commit -m "feat(stage-e): finalize master comparison layer and notebook integration"
```

---

## Notes for Executor

- Follow TDD strictly per task.
- Keep notebook markdown churn minimal.
- Do not move join logic into notebooks.
- If parquet write fails in environment, ensure JSON fallback exists and stale parquet is removed.
- If repository has no `.git` directory, skip commit steps and document skipped commit commands in handoff.

## Review chunks

- `## Chunk 1`: utility modules and tests
- `## Chunk 2`: report builder + identity/duplicate contracts
- `## Chunk 3`: notebook integration + smoke tests
- `## Chunk 4`: end-to-end verification

