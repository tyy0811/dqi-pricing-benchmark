# Notebook Reader Layer Cleanup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize notebook reader bootstrap/data-compatibility behavior so notebooks `06/08/09/10/11/12` use one root resolver, one loader policy, one status bridge, and uniform panel-level placeholders for missing/empty optional artifacts.

**Architecture:** Keep scientific analysis logic in notebooks, but move compatibility concerns into `notebooks/_helpers.py`. Add frozen helper contracts (`repo_root`, loader policy wrapper, run-status materialization, optional-column guards, placeholder schema helpers), then rewire notebook setup and guarded field access to call those helpers consistently.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, Jupyter notebooks, pytest.

**Spec:** `docs/superpowers/specs/2026-03-12-notebook-reader-layer-cleanup-design.md`

---

## File Structure Map

- Modify: `notebooks/_helpers.py`
  - Add/freeze root, loader-policy, status-materialization, guarded-column, placeholder contracts.
- Modify: `notebooks/06_resource_analysis.ipynb`
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
- Modify: `notebooks/09_decoder_phase_diagram.ipynb`
- Modify: `notebooks/10_alpha_finite_size_validation.ipynb`
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`
- Modify: `notebooks/12_quality_vs_cost.ipynb`
  - Replace bootstrap/path logic, use frozen helpers, preserve scientific flow.
- Create: `tests/test_notebook_reader_hygiene.py`
  - Lint-style checks for notebook bootstrap/status hygiene rules.
- Modify: `tests/test_notebook_support.py`
  - Helper contract tests (`normalize_run_status`, loader policy return shapes, repo_root failure shape).
- Modify: `tests/test_notebook_helpers_stage_e.py`
  - Status normalization and loader-policy invariants for Stage E helper paths.
- Modify: `tests/test_notebook_runtime_smoke_missing_data.py`
  - Keep panel-level placeholder behavior green under optional/missing artifacts.

---

## Chunk 1: Helper Contract Hardening

### Task 1: Freeze `repo_root()` markers, traversal order, and failure diagnostics

**Files:**
- Modify: `notebooks/_helpers.py`
- Test: `tests/test_notebook_support.py`

- [ ] **Step 1: Write failing tests for root resolution contract**

```python
def test_repo_root_resolves_repo_with_required_markers():
    ...

def test_repo_root_error_lists_markers_and_visited_candidates():
    ...
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/test_notebook_support.py -k "repo_root" -v`
Expected: FAIL (missing frozen behavior)

- [ ] **Step 3: Implement minimal `repo_root()` contract**

Implement in `notebooks/_helpers.py`:
- required markers: `src/`, `notebooks/`
- optional markers: `pyproject.toml`, `.git`
- candidate order: `cwd`, helper dir, then parent chains (de-duplicated)
- loud `RuntimeError` with markers + visited candidates + per-candidate checks

- [ ] **Step 4: Re-run targeted tests**

Run: `pytest tests/test_notebook_support.py -k "repo_root" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/_helpers.py tests/test_notebook_support.py
git commit -m "feat(notebooks): freeze repo_root resolver contract"
```

### Task 2: Introduce one shared loader-policy helper and unify loader wrappers

**Files:**
- Modify: `notebooks/_helpers.py`
- Test: `tests/test_notebook_support.py`
- Test: `tests/test_notebook_helpers_stage_e.py`

- [ ] **Step 1: Write failing tests for loader policy invariants**

```python
def test_optional_table_loader_returns_empty_dataframe_not_none():
    ...

def test_optional_json_loader_returns_empty_dict_not_none():
    ...

def test_required_loader_raises_with_consistent_message():
    ...
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/test_notebook_support.py tests/test_notebook_helpers_stage_e.py -k "load_" -v`
Expected: FAIL for new policy checks

- [ ] **Step 3: Implement shared loader policy helper and wire all `load_*` wrappers through it**

Implement in `notebooks/_helpers.py`:
- one internal helper handling `required`, missing behavior, default-empty shape, message formatting
- ensure:
  - table loaders => `pd.DataFrame()` when optional missing/unreadable
  - JSON/report loaders => `{}` when optional missing/unreadable
  - no implicit `None`

- [ ] **Step 4: Re-run loader tests**

Run: `pytest tests/test_notebook_support.py tests/test_notebook_helpers_stage_e.py -k "load_" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/_helpers.py tests/test_notebook_support.py tests/test_notebook_helpers_stage_e.py
git commit -m "refactor(notebooks): centralize loader required/optional policy"
```

### Task 3: Freeze status normalization + DataFrame materialization helper

**Files:**
- Modify: `notebooks/_helpers.py`
- Test: `tests/test_notebook_support.py`
- Test: `tests/test_notebook_runtime_smoke_missing_data.py`

- [ ] **Step 1: Write failing tests for status mapping + null safety + materialization**

```python
def test_normalize_run_status_maps_skipped_and_unsupported_to_not_applicable():
    ...

def test_materialize_run_status_norm_handles_missing_run_status_column():
    ...
```

- [ ] **Step 2: Run targeted tests to confirm failure**

Run: `pytest tests/test_notebook_support.py tests/test_notebook_runtime_smoke_missing_data.py -k "normalize_run_status or run_status_norm" -v`
Expected: FAIL for new expectations

- [ ] **Step 3: Implement minimal status contract**

Implement in `notebooks/_helpers.py`:
- `normalize_run_status(value_or_series)` with frozen mapping
- one helper to materialize `run_status_norm` once per DataFrame with missing-column fallback

- [ ] **Step 4: Re-run status tests**

Run: `pytest tests/test_notebook_support.py tests/test_notebook_runtime_smoke_missing_data.py -k "normalize_run_status or run_status_norm" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/_helpers.py tests/test_notebook_support.py tests/test_notebook_runtime_smoke_missing_data.py
git commit -m "feat(notebooks): add canonical run_status_norm materialization"
```

### Task 4: Freeze optional-column guard helpers + placeholder schema helper

**Files:**
- Modify: `notebooks/_helpers.py`
- Test: `tests/test_notebook_support.py`

- [ ] **Step 1: Write failing tests for guarded helper contracts**

```python
def test_guard_optional_columns_materializes_numeric_and_object_fields():
    ...

def test_placeholder_table_uses_frozen_schema_columns():
    ...
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/test_notebook_support.py -k "guard or placeholder" -v`
Expected: FAIL

- [ ] **Step 3: Implement helper-level guards/accessors**

Implement in `notebooks/_helpers.py`:
- frozen optional-column materialization/coercion helper(s)
- frozen placeholder/unavailable helper schema (`panel`, `status`, `reason`, optional `details`)

- [ ] **Step 4: Re-run tests**

Run: `pytest tests/test_notebook_support.py -k "guard or placeholder" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/_helpers.py tests/test_notebook_support.py
git commit -m "feat(notebooks): freeze guarded-column and placeholder helper contracts"
```

---

## Chunk 2: Notebook Bootstrap and Status Usage Rewire

### Task 5: Replace bootstrap/root logic in notebooks 06/08/09/10/11/12

**Files:**
- Modify: `notebooks/06_resource_analysis.ipynb`
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
- Modify: `notebooks/09_decoder_phase_diagram.ipynb`
- Modify: `notebooks/10_alpha_finite_size_validation.ipynb`
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`
- Modify: `notebooks/12_quality_vs_cost.ipynb`

- [ ] **Step 1: Write/adjust hygiene tests first**

Create skeleton tests that fail on legacy patterns:

```python
assert "sys.path.insert(" not in source
assert "Path.cwd().resolve()" not in bootstrap_cell
assert "ROOT = repo_root()" in bootstrap_cell
```

- [ ] **Step 2: Run hygiene tests and confirm failure**

Run: `pytest tests/test_notebook_reader_hygiene.py -v`
Expected: FAIL

- [ ] **Step 3: Update notebook bootstrap cells minimally**

- remove `import sys` where no longer needed
- import `repo_root` from helpers
- set `ROOT = repo_root()`
- keep other notebook structure unchanged

- [ ] **Step 4: Re-run hygiene tests**

Run: `pytest tests/test_notebook_reader_hygiene.py -k "bootstrap" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/06_resource_analysis.ipynb notebooks/08_paired_encoding_comparison.ipynb notebooks/09_decoder_phase_diagram.ipynb notebooks/10_alpha_finite_size_validation.ipynb notebooks/11_decision_faithfulness_dashboard.ipynb notebooks/12_quality_vs_cost.ipynb tests/test_notebook_reader_hygiene.py
git commit -m "refactor(notebooks): standardize bootstrap on repo_root helper"
```

### Task 6: Replace notebook-local status derivation/filter drift with helper materialization

**Files:**
- Modify: `notebooks/06_resource_analysis.ipynb`
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
- Modify: `notebooks/09_decoder_phase_diagram.ipynb`
- Modify: `notebooks/10_alpha_finite_size_validation.ipynb`
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`
- Modify: `notebooks/12_quality_vs_cost.ipynb`
- Test: `tests/test_notebook_reader_hygiene.py`

- [ ] **Step 1: Add failing hygiene test for raw-status filtering drift**

```python
def test_notebooks_filter_on_run_status_norm_not_raw_literals():
    ...
```

- [ ] **Step 2: Run test and confirm failure**

Run: `pytest tests/test_notebook_reader_hygiene.py -k "status" -v`
Expected: FAIL

- [ ] **Step 3: Rewire status handling per notebook**

- materialize `run_status_norm` via helper once per frame
- use normalized status in filters/buckets where possible
- keep raw `run_status` only in display tables

- [ ] **Step 4: Re-run hygiene status tests**

Run: `pytest tests/test_notebook_reader_hygiene.py -k "status" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/06_resource_analysis.ipynb notebooks/08_paired_encoding_comparison.ipynb notebooks/09_decoder_phase_diagram.ipynb notebooks/10_alpha_finite_size_validation.ipynb notebooks/11_decision_faithfulness_dashboard.ipynb notebooks/12_quality_vs_cost.ipynb tests/test_notebook_reader_hygiene.py
git commit -m "refactor(notebooks): normalize status handling via helper contract"
```

---

## Chunk 3: Guarded Fields and Panel-Level Placeholder Behavior

### Task 7: Apply helper guard/accessor APIs for protected optional fields in all six notebooks

**Files:**
- Modify: `notebooks/06_resource_analysis.ipynb`
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
- Modify: `notebooks/09_decoder_phase_diagram.ipynb`
- Modify: `notebooks/10_alpha_finite_size_validation.ipynb`
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`
- Modify: `notebooks/12_quality_vs_cost.ipynb`
- Test: `tests/test_notebook_runtime_smoke_missing_data.py`

- [ ] **Step 1: Add/adjust failing runtime-smoke assertions for protected fields**

```python
assert "metric_availability" in focus.columns
assert "coherent_mode_record" in focus.columns
# and delta/structure/resource guards as applicable
```

- [ ] **Step 2: Run targeted smoke tests and confirm failure**

Run: `pytest tests/test_notebook_runtime_smoke_missing_data.py -k "notebook06 or notebook08 or notebook09 or notebook10 or notebook11 or notebook12" -v`
Expected: FAIL

- [ ] **Step 3: Replace notebook-local optional column guards with helper APIs**

Use helper-level guard/accessor contracts before plotting/filtering for:
- `metric_availability`
- `coherent_mode_record`
- coherent-vs-mixture delta fields
- `structure_*` decoder-study fields
- Stage E resource fields used in quality-vs-cost panels

- [ ] **Step 4: Re-run smoke tests**

Run: `pytest tests/test_notebook_runtime_smoke_missing_data.py -k "notebook06 or notebook08 or notebook09 or notebook10 or notebook11 or notebook12" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/06_resource_analysis.ipynb notebooks/08_paired_encoding_comparison.ipynb notebooks/09_decoder_phase_diagram.ipynb notebooks/10_alpha_finite_size_validation.ipynb notebooks/11_decision_faithfulness_dashboard.ipynb notebooks/12_quality_vs_cost.ipynb tests/test_notebook_runtime_smoke_missing_data.py
git commit -m "refactor(notebooks): enforce shared guarded-field access patterns"
```

### Task 8: Enforce panel-level placeholder contract and Stage E source boundary

**Files:**
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`
- Modify: `notebooks/12_quality_vs_cost.ipynb`
- Modify: `tests/test_notebook11_stage_e_smoke.py`
- Modify: `tests/test_notebook12_stage_e_smoke.py`

- [ ] **Step 1: Add failing assertions for Stage E source boundary + placeholders**

```python
assert "quality_cost_master" in src
assert "missing or empty" in src
assert "load_metrics_master" not in comparable-panel paths
```

- [ ] **Step 2: Run Stage E notebook smoke tests and confirm failure**

Run: `pytest tests/test_notebook11_stage_e_smoke.py tests/test_notebook12_stage_e_smoke.py -v`
Expected: FAIL on new checks

- [ ] **Step 3: Update notebook cells minimally to satisfy source-boundary rule**

- comparable panels read only `quality_cost_master`
- unavailable transparency reads `quality_cost_unavailable`
- optional-missing behavior emits panel-level placeholders

- [ ] **Step 4: Re-run Stage E smoke tests**

Run: `pytest tests/test_notebook11_stage_e_smoke.py tests/test_notebook12_stage_e_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/11_decision_faithfulness_dashboard.ipynb notebooks/12_quality_vs_cost.ipynb tests/test_notebook11_stage_e_smoke.py tests/test_notebook12_stage_e_smoke.py
git commit -m "fix(notebooks): enforce Stage E comparable-source and placeholder rules"
```

---

## Chunk 4: Integrated Verification + Targeted Notebook Execution

### Task 9: Run full targeted test suite for helper contracts and notebook smoke coverage

**Files:**
- Test only

- [ ] **Step 1: Run consolidated pytest command**

Run:
`pytest tests/test_notebook_support.py tests/test_notebook_helpers_stage_e.py tests/test_notebook_reader_hygiene.py tests/test_notebook_runtime_smoke_missing_data.py tests/test_notebook11_stage_e_smoke.py tests/test_notebook12_stage_e_smoke.py -v`

Expected: PASS

- [ ] **Step 2: Fix any failing tests and rerun until green**

Run: same command above
Expected: PASS

- [ ] **Step 3: Commit verification fixes (if any)**

```bash
git add notebooks/_helpers.py notebooks/06_resource_analysis.ipynb notebooks/08_paired_encoding_comparison.ipynb notebooks/09_decoder_phase_diagram.ipynb notebooks/10_alpha_finite_size_validation.ipynb notebooks/11_decision_faithfulness_dashboard.ipynb notebooks/12_quality_vs_cost.ipynb tests/test_notebook_support.py tests/test_notebook_helpers_stage_e.py tests/test_notebook_reader_hygiene.py tests/test_notebook_runtime_smoke_missing_data.py tests/test_notebook11_stage_e_smoke.py tests/test_notebook12_stage_e_smoke.py
git commit -m "test(notebooks): finalize reader-layer cleanup verification"
```

### Task 10: Execute targeted notebook smoke run (`06, 08, 09, 10, 11, 12`)

**Files:**
- Modify in place (executed notebooks):
  - `notebooks/06_resource_analysis.ipynb`
  - `notebooks/08_paired_encoding_comparison.ipynb`
  - `notebooks/09_decoder_phase_diagram.ipynb`
  - `notebooks/10_alpha_finite_size_validation.ipynb`
  - `notebooks/11_decision_faithfulness_dashboard.ipynb`
  - `notebooks/12_quality_vs_cost.ipynb`

- [ ] **Step 1: Execute notebooks individually for debuggable failures**

Run:
- `jupyter nbconvert --to notebook --execute --inplace notebooks/06_resource_analysis.ipynb`
- `jupyter nbconvert --to notebook --execute --inplace notebooks/08_paired_encoding_comparison.ipynb`
- `jupyter nbconvert --to notebook --execute --inplace notebooks/09_decoder_phase_diagram.ipynb`
- `jupyter nbconvert --to notebook --execute --inplace notebooks/10_alpha_finite_size_validation.ipynb`
- `jupyter nbconvert --to notebook --execute --inplace notebooks/11_decision_faithfulness_dashboard.ipynb`
- `jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb`

Expected: all execute without hard aborts from optional artifact absence

- [ ] **Step 2: Verify panel-level placeholder behavior remains visible in outputs**

Run quick checks (source/output inspections as needed) for unavailable tables/placeholder plots.

- [ ] **Step 3: Final commit**

```bash
git add notebooks/06_resource_analysis.ipynb notebooks/08_paired_encoding_comparison.ipynb notebooks/09_decoder_phase_diagram.ipynb notebooks/10_alpha_finite_size_validation.ipynb notebooks/11_decision_faithfulness_dashboard.ipynb notebooks/12_quality_vs_cost.ipynb
git commit -m "chore(notebooks): execute reader-layer cleaned analysis notebooks"
```

---

## Plan Review Notes

- Apply plan-document review loop per chunk when reviewer tooling is available.
- If reviewer tooling is unavailable in this harness, perform manual self-review against spec before execution handoff.

---

Plan complete and saved to `docs/superpowers/plans/2026-03-12-notebook-reader-layer-cleanup.md`. Ready to execute?
