# Notebook 06 Reader-Only Panel Separation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Notebook 06 into a strictly reader-only analysis notebook with explicit same-model vs cross-model panel separation and deterministic placeholder/failure contracts.

**Architecture:** Keep notebook section flow intact while tightening only reader contracts. Use canonical artifacts (`resources_master`, `metrics_master`, `run_manifest`), freeze aligned identity + duplicate behavior, enforce main-panel same-model-only semantics (`gain_over_bp1`), and move `gain_over_bruteforce` into a separately labeled diagnostic panel.

**Tech Stack:** Python 3.11, pandas, matplotlib, Jupyter notebooks, pytest.

**Spec:** `docs/superpowers/specs/2026-03-12-notebook-06-reader-only-panel-separation-design.md`

---

## File Structure Map

- Modify: `notebooks/06_resource_analysis.ipynb`
  - Tighten panel logic and markdown labels.
  - Enforce no writer behavior and strict panel contracts.
- Modify: `tests/test_notebook_runtime_smoke_missing_data.py`
  - Add panel-level behavior tests for same-model missing metric and duplicate aligned identity.
- Modify: `tests/test_notebook_reader_hygiene.py`
  - Add Notebook 06 checks for no artifact-write behavior and same-model/cross-model separation guards.

---

## Chunk 1: Contract Tests First (TDD)

### Task 1: Add failing smoke test for strict same-model main panel behavior

**Files:**
- Modify: `tests/test_notebook_runtime_smoke_missing_data.py`

- [ ] **Step 1: Write failing test**

Add a new test that executes the Notebook 06 structure panel cell with aligned completed rows where:
- `gain_over_bp1` is fully absent/non-usable
- `gain_over_bruteforce` is present

Expected:
- main panel renders placeholder with reason contract equivalent to `non_usable_metric_values`
- no fallback to main-panel plotting using `gain_over_bruteforce`

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_notebook_runtime_smoke_missing_data.py -k "notebook06 and same_model" -v`
Expected: FAIL

- [ ] **Step 3: Commit test-only change**

```bash
git add tests/test_notebook_runtime_smoke_missing_data.py
git commit -m "test(notebook06): require strict same-model panel without fallback"
```

### Task 2: Add failing smoke test for duplicate aligned identity handling

**Files:**
- Modify: `tests/test_notebook_runtime_smoke_missing_data.py`

- [ ] **Step 1: Write failing test**

Inject duplicate aligned identities into either resource or metric source after key resolution (`slot_id` or fallback identity).

Expected:
- panel placeholder renders
- no auto-dedup/winner selection

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_notebook_runtime_smoke_missing_data.py -k "notebook06 and duplicate" -v`
Expected: FAIL

- [ ] **Step 3: Commit test-only change**

```bash
git add tests/test_notebook_runtime_smoke_missing_data.py
git commit -m "test(notebook06): enforce duplicate aligned identity placeholder behavior"
```

### Task 3: Add failing hygiene checks for reader-only and panel separation

**Files:**
- Modify: `tests/test_notebook_reader_hygiene.py`

- [ ] **Step 1: Write failing checks**

Add Notebook 06 source checks ensuring:
- no writer calls/paths (`resources_scaling.json`, `export_resources_json`, notebook-authored artifact outputs)
- no main-panel fallback from `gain_over_bp1` to `gain_over_bruteforce`
- presence of explicit cross-model diagnostic label text

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_notebook_reader_hygiene.py -k "notebook06" -v`
Expected: FAIL

- [ ] **Step 3: Commit test-only change**

```bash
git add tests/test_notebook_reader_hygiene.py
git commit -m "test(notebook06): add reader-only and panel-separation hygiene checks"
```

---

## Chunk 2: Notebook 06 Implementation

### Task 4: Freeze alignment identity and duplicate failure behavior in Notebook 06 panel logic

**Files:**
- Modify: `notebooks/06_resource_analysis.ipynb`

- [ ] **Step 1: Implement resolved identity contract**

In structure-vs-decode panel cell:
- keep `slot_id` as primary key when shared
- fallback key minimum: `instance_id, encoding, decoder, alpha_mode, execution_mode`
- include `canonical_trial_seed` only when shared

- [ ] **Step 2: Implement deterministic duplicate failure**

After key resolution:
- if duplicates in either source over resolved identity, render placeholder with duplicate-identity reason
- do not auto-deduplicate

- [ ] **Step 3: Run targeted tests**

Run:
- `pytest tests/test_notebook_runtime_smoke_missing_data.py -k "notebook06 and duplicate" -v`
- `pytest tests/test_notebook_reader_hygiene.py -k "notebook06" -v`

Expected: PASS for implemented parts

- [ ] **Step 4: Commit**

```bash
git add notebooks/06_resource_analysis.ipynb tests/test_notebook_runtime_smoke_missing_data.py tests/test_notebook_reader_hygiene.py
git commit -m "refactor(notebook06): freeze aligned identity and duplicate failure contract"
```

### Task 5: Enforce strict same-model main panel and separate cross-model diagnostic panel

**Files:**
- Modify: `notebooks/06_resource_analysis.ipynb`

- [ ] **Step 1: Remove fallback logic in main panel**

Main panel must:
- use only `gain_over_bp1`
- render placeholder when non-usable/absent
- never fall back to `gain_over_bruteforce`

- [ ] **Step 2: Add separate cross-model diagnostic panel**

Add distinct panel for `gain_over_bruteforce` with explicit title/subtitle text:
- `cross-model`
- `non-authoritative`
- `not used for claims`

- [ ] **Step 3: Run targeted tests**

Run:
- `pytest tests/test_notebook_runtime_smoke_missing_data.py -k "notebook06 and same_model" -v`
- `pytest tests/test_notebook_reader_hygiene.py -k "notebook06" -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add notebooks/06_resource_analysis.ipynb tests/test_notebook_runtime_smoke_missing_data.py tests/test_notebook_reader_hygiene.py
git commit -m "refactor(notebook06): split same-model authoritative and cross-model diagnostic panels"
```

### Task 6: Normalize placeholder reason-code contract in Notebook 06

**Files:**
- Modify: `notebooks/06_resource_analysis.ipynb`

- [ ] **Step 1: Implement closed reason mapping in panel placeholders**

Ensure panel placeholder reasons map to enum:
- `missing_artifact`
- `empty_aligned_set`
- `missing_required_fields`
- `non_usable_metric_values`
- `duplicate_aligned_identity`

- [ ] **Step 2: Run focused smoke tests**

Run: `pytest tests/test_notebook_runtime_smoke_missing_data.py -k "notebook06" -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add notebooks/06_resource_analysis.ipynb tests/test_notebook_runtime_smoke_missing_data.py
git commit -m "refactor(notebook06): freeze placeholder reason-code contract"
```

---

## Chunk 3: Integrated Verification

### Task 7: Run full targeted test suite

**Files:**
- Test only

- [ ] **Step 1: Run consolidated tests**

Run:
`pytest tests/test_notebook_support.py tests/test_notebook_reader_hygiene.py tests/test_notebook_runtime_smoke_missing_data.py tests/test_notebook11_stage_e_smoke.py tests/test_notebook12_stage_e_smoke.py -v`

Expected: PASS

- [ ] **Step 2: Fix regressions and rerun until green**

Run: same command
Expected: PASS

- [ ] **Step 3: Commit verification fixes (if any)**

```bash
git add notebooks/06_resource_analysis.ipynb tests/test_notebook_reader_hygiene.py tests/test_notebook_runtime_smoke_missing_data.py
git commit -m "test(notebook06): finalize reader-only panel-separation verification"
```

### Task 8: Execute Notebook 06 and verify panel outputs

**Files:**
- Modify in place: `notebooks/06_resource_analysis.ipynb`

- [ ] **Step 1: Execute notebook**

Run:
`jupyter nbconvert --to notebook --execute --inplace notebooks/06_resource_analysis.ipynb`

Expected:
- notebook executes
- main panel shows placeholder when same-model metric unusable
- cross-model panel is separately rendered and explicitly diagnostic

- [ ] **Step 2: Commit executed notebook artifact**

```bash
git add notebooks/06_resource_analysis.ipynb
git commit -m "chore(notebook06): execute reader-only panel-separated resource analysis"
```

---

Plan complete and saved to `docs/superpowers/plans/2026-03-12-notebook-06-reader-only-panel-separation.md`. Ready to execute?
