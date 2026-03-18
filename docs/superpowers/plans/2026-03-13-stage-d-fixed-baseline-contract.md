# Stage D Fixed-Baseline Contract Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic fixed-priority Stage D baseline selection upstream and enforce Notebook 10 as a strict fixed-baseline reader over `counterpart_key` pairs.

**Architecture:** Extend `scripts/make_report.py` to compute normalized Stage D candidate baselines, validate counterpart uniqueness, select one global baseline via fixed-priority policy, and persist selector metadata on all Stage D rows. Keep notebook logic reader-only: it consumes selector metadata, filters selected baseline rows, pairs only by `counterpart_key`, and reports explicit unavailable/error states without notebook-side repair.

**Tech Stack:** Python 3.11, pandas, Jupyter notebooks (`.ipynb` JSON), pytest.

**Spec:** `docs/superpowers/specs/2026-03-13-stage-d-fixed-baseline-contract-design.md`

---

## File Structure Map

- Modify: `scripts/make_report.py`
  - Add Stage D baseline candidate counting, duplicate completed counterpart guardrails, fixed-priority selector, and metadata projection onto Stage D rows.
- Modify: `src/report_schema.py`
  - Add (or tighten) optional/required field validators for persisted Stage D baseline metadata semantics if enforced centrally.
- Modify: `notebooks/10_alpha_finite_size_validation.ipynb`
  - Remove dynamic key inference; enforce required upstream metadata and `counterpart_key`-only pairing.
- Modify: `tests/test_make_report.py`
  - Extend existing Stage D tests for fixed-priority selection metadata and explicit no-baseline semantics.
- Create: `tests/test_make_report_stage_d_baseline_selector.py`
  - Focused selector and invariant tests (preferred/fallback/no-selection, mismatch errors, duplicate completed counterparts).
- Modify: `tests/test_notebook10_alpha_validation_smoke.py`
  - Replace old dynamic pairing contract checks with fixed-baseline reader contract checks.
- Create: `tests/test_notebook10_stage_d_fixed_baseline_runtime.py`
  - Runtime-style notebook cell contract tests for metadata required, selected-baseline filtering, counterpart counts, and loud failure paths.
- Modify: `tests/test_notebook_runtime_smoke_missing_data.py`
  - Add missing-metadata `baseline_metadata_unavailable` behavior checks.

---

## Chunk 1: Upstream Selector Contract Tests (TDD First)

### Task 1: Add failing tests for fixed-priority baseline policy and metadata

**Files:**
- Create: `tests/test_make_report_stage_d_baseline_selector.py`
- Modify: `tests/test_make_report.py`

- [ ] **Step 1: Write failing tests for preferred/fallback/no-selection policy**

```python
def test_selector_prefers_wht_exact_oracle_when_threshold_met():
    ...

def test_selector_falls_back_to_ilp_exact_oracle_when_preferred_below_threshold():
    ...

def test_selector_emits_no_baseline_meets_threshold_with_explicit_null_fields():
    ...
```

- [ ] **Step 2: Write failing tests for required persisted metadata on all Stage D rows**

```python
def test_stage_d_rows_persist_selector_fields_on_all_rows():
    ...
```

- [ ] **Step 3: Write failing invariant tests**

```python
def test_duplicate_completed_counterpart_rows_fail_before_selection():
    ...

def test_selected_baseline_mismatch_raises_contract_error():
    ...

def test_multiple_selected_baselines_raises_contract_error():
    ...
```

- [ ] **Step 4: Run tests to confirm failures**

Run: `pytest tests/test_make_report_stage_d_baseline_selector.py tests/test_make_report.py -k "baseline or alpha_validation" -v`  
Expected: FAIL (missing selector implementation/metadata/invariants)

- [ ] **Step 5: Commit failing-test scaffold**

```bash
git add tests/test_make_report_stage_d_baseline_selector.py tests/test_make_report.py
git commit -m "test(stage-d): define fixed-priority baseline selector contract"
```

---

## Chunk 2: Implement Upstream Stage D Baseline Selector

### Task 2: Implement counterpart guardrails, candidate counting, and fixed-priority selection

**Files:**
- Modify: `scripts/make_report.py`
- Modify: `src/report_schema.py`

- [ ] **Step 1: Add helper(s) in `make_report.py` for normalized Stage D candidate extraction**

```python
def _stage_d_candidate_rows(master_df: pd.DataFrame) -> pd.DataFrame:
    ...  # matrix_b + alpha_validation + family C + normalized fields
```

- [ ] **Step 2: Implement strict counterpart validation before selection**

```python
def _validate_stage_d_counterpart_uniqueness(df: pd.DataFrame) -> None:
    ...  # raise on >1 completed row per (counterpart_key, execution_mode_norm)
```

- [ ] **Step 3: Implement candidate pair counting by `counterpart_key`**

```python
def _count_stage_d_candidate_pairs(df: pd.DataFrame, encoding_norm: str, decoder_norm: str) -> int:
    ...  # exactly one completed mixture + one completed coherent per counterpart_key
```

- [ ] **Step 4: Implement fixed-priority selector with explicit reason-code enum**

```python
def _select_stage_d_baseline(..., min_pairs: int = 8) -> dict[str, Any]:
    ...  # preferred/fallback/no-selection only
```

- [ ] **Step 5: Persist selector metadata on all Stage D rows and set `is_stage_d_selected_baseline`**

```python
def _apply_stage_d_baseline_metadata(master_df: pd.DataFrame, selector: dict[str, Any]) -> pd.DataFrame:
    ...
```

- [ ] **Step 6: Implement global invariant checks**

```python
def _assert_single_global_stage_d_baseline(df: pd.DataFrame) -> None:
    ...  # selected_baseline_mismatch / multiple_selected_baselines
```

- [ ] **Step 7: Wire selector flow into report build path after Stage D rows/keys exist**

Run placement: in `run_make_report` path where Stage D rows are already normalized and before write-out.

- [ ] **Step 8: Add/adjust schema checks (if strict-required in validators)**

`report_schema.py` updates should validate explicit no-selection values and boolean consistency for `is_stage_d_selected_baseline` when Stage D baseline metadata fields are present.

- [ ] **Step 9: Run targeted tests**

Run: `pytest tests/test_make_report_stage_d_baseline_selector.py tests/test_make_report.py -k "baseline or alpha_validation" -v`  
Expected: PASS

- [ ] **Step 10: Commit implementation**

```bash
git add scripts/make_report.py src/report_schema.py tests/test_make_report_stage_d_baseline_selector.py tests/test_make_report.py
git commit -m "feat(stage-d): add fixed-priority baseline selector and metadata contract"
```

---

## Chunk 3: Notebook 10 Reader-Only Contract Enforcement

### Task 3: Remove notebook-side baseline inference and enforce `counterpart_key` pairing

**Files:**
- Modify: `notebooks/10_alpha_finite_size_validation.ipynb`
- Modify: `tests/test_notebook10_alpha_validation_smoke.py`
- Create: `tests/test_notebook10_stage_d_fixed_baseline_runtime.py`
- Modify: `tests/test_notebook_runtime_smoke_missing_data.py`

- [ ] **Step 1: Update smoke tests first (failing) for new Notebook 10 contract**

```python
def test_notebook10_requires_stage_d_baseline_metadata_and_counterpart_key():
    ...

def test_notebook10_has_no_dynamic_intersection_pairing_logic():
    ...

def test_notebook10_filters_selected_baseline_before_pairing():
    ...
```

- [ ] **Step 2: Add runtime tests for explicit failure/unavailable states**

```python
def test_missing_baseline_metadata_renders_baseline_metadata_unavailable():
    ...

def test_duplicate_completed_counterpart_rows_raise_contract_error():
    ...

def test_main_conclusion_uses_valid_pair_counts_and_reason_breakdowns_only():
    ...
```

- [ ] **Step 3: Run tests to confirm failure before notebook edits**

Run: `pytest tests/test_notebook10_alpha_validation_smoke.py tests/test_notebook10_stage_d_fixed_baseline_runtime.py tests/test_notebook_runtime_smoke_missing_data.py -k "baseline or counterpart" -v`  
Expected: FAIL

- [ ] **Step 4: Edit Notebook 10 cells to enforce reader-only baseline flow**

Required notebook logic changes:
- require all baseline metadata fields + `counterpart_key`
- if missing: render unavailable panel with reason `baseline_metadata_unavailable` and stop main Stage D path
- filter Stage D rows by `is_stage_d_selected_baseline == True` before pairing
- restrict `execution_mode_norm` to coherent/mixture only
- pair only on `counterpart_key`
- classify incomplete keys into coherent/mixture absent/non-completed buckets
- compute deltas per metric using precomputed-or-recompute rule with explicit per-pair reason codes
- ensure optional all-baseline block is labeled `diagnostic_only` and excluded from main conclusion

- [ ] **Step 5: Run notebook contract tests**

Run: `pytest tests/test_notebook10_alpha_validation_smoke.py tests/test_notebook10_stage_d_fixed_baseline_runtime.py tests/test_notebook_runtime_smoke_missing_data.py -k "baseline or counterpart" -v`  
Expected: PASS

- [ ] **Step 6: Commit notebook + tests**

```bash
git add notebooks/10_alpha_finite_size_validation.ipynb tests/test_notebook10_alpha_validation_smoke.py tests/test_notebook10_stage_d_fixed_baseline_runtime.py tests/test_notebook_runtime_smoke_missing_data.py
git commit -m "feat(notebook10): enforce fixed-baseline counterpart-key reader contract"
```

---

## Chunk 4: Integration Verification and Artifact Regeneration

### Task 4: Verify end-to-end behavior and regenerate artifacts

**Files:**
- Generated: `results/metrics_master.*`
- Generated: `results/quality_cost_master.*`
- Generated: `results/quality_cost_unavailable.*`
- Execution targets: `notebooks/08_paired_encoding_comparison.ipynb`, `notebooks/10_alpha_finite_size_validation.ipynb`, `notebooks/11_decision_faithfulness_dashboard.ipynb`, `notebooks/12_quality_vs_cost.ipynb`

- [ ] **Step 1: Run full relevant test suite slice**

Run:
`pytest tests/test_make_report_stage_d_baseline_selector.py tests/test_make_report.py tests/test_notebook10_alpha_validation_smoke.py tests/test_notebook10_stage_d_fixed_baseline_runtime.py tests/test_notebook_runtime_smoke_missing_data.py tests/test_notebook08_paired_encoding_smoke.py tests/test_notebook11_stage_e_smoke.py tests/test_notebook12_stage_e_smoke.py -v`

Expected: PASS

- [ ] **Step 2: Regenerate report artifacts**

Run: `python scripts/make_report.py --results-root results --matrices matrix_a,matrix_b,matrix_c`  
Expected: writes refreshed `metrics_master`, `quality_cost_master`, `quality_cost_unavailable`

- [ ] **Step 3: Execute notebooks 08/10/11/12 in-place**

Run:
`jupyter nbconvert --to notebook --execute --inplace notebooks/08_paired_encoding_comparison.ipynb`

`jupyter nbconvert --to notebook --execute --inplace notebooks/10_alpha_finite_size_validation.ipynb`

`jupyter nbconvert --to notebook --execute --inplace notebooks/11_decision_faithfulness_dashboard.ipynb`

`jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb`

Expected: successful execution with non-placeholder baseline metadata panel for Stage D when baseline exists; explicit unavailable context otherwise.

- [ ] **Step 4: Sanity-check produced Stage D selector columns in `metrics_master`**

Run (example):
`python - <<'PY'\nimport pandas as pd\ndf=pd.read_parquet('results/metrics_master.parquet')\ns=df[(df['matrix']=='matrix_b')&(df['stage']=='alpha_validation')&(df['family']=='C')]\nprint(s[['baseline_selector_policy','baseline_encoding_norm','baseline_decoder_norm','baseline_min_pairs','baseline_pair_count','best_available_pair_count','baseline_selected_reason','baseline_fallback_used','is_stage_d_selected_baseline']].drop_duplicates())\nPY`

Expected: exactly one global selector state for Stage D artifact.

- [ ] **Step 5: Commit integration updates (if notebooks/results are tracked per repo policy)**

```bash
git add scripts/make_report.py src/report_schema.py notebooks/10_alpha_finite_size_validation.ipynb tests/test_make_report_stage_d_baseline_selector.py tests/test_make_report.py tests/test_notebook10_alpha_validation_smoke.py tests/test_notebook10_stage_d_fixed_baseline_runtime.py tests/test_notebook_runtime_smoke_missing_data.py
git commit -m "feat(stage-d): ship fixed-priority baseline selection and notebook reader contract"
```

---

Plan complete and saved to `docs/superpowers/plans/2026-03-13-stage-d-fixed-baseline-contract.md`. Ready to execute?
