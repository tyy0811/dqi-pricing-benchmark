# Notebook Packaging Reviewer Artifacts Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-freeze Notebook 07 around the tiny validated case, package Notebook 09 as a reusable mini-benchmark artifact, and add one short reviewer-facing markdown summary.

**Architecture:** Keep both notebooks in place and make only packaging-oriented edits. Reuse the existing scenario-matrix helper for Notebook 09, adding only minimal verdict/export support where needed, and execute both notebooks in place so saved outputs match the intended reviewer-facing claims.

**Tech Stack:** Python, pandas, Jupyter notebooks, existing multiperiod helpers, pytest, markdown

---

## Chunk 1: Add minimal helper and test support

### Task 1: Add the smallest verdict/export surface for Notebook 09

**Files:**
- Modify: `src/classical_baselines_multiperiod.py`
- Modify: `tests/test_classical_baselines_multiperiod.py`
- Modify: `tests/test_notebook09_3period_pricing_extension_smoke.py`

- [ ] **Step 1: Write failing tests for verdict/export-facing row fields**

Run: `pytest tests/test_classical_baselines_multiperiod.py tests/test_notebook09_3period_pricing_extension_smoke.py -k 'scenario_matrix or notebook09' -v`
Expected: FAIL on missing export/verdict surface

- [ ] **Step 2: Implement the minimal helper additions**

Required:

- keep current scenario matrix logic
- add any missing row fields needed for the verdict table
- avoid new modeling logic

- [ ] **Step 3: Run the focused tests again**

Run: `pytest tests/test_classical_baselines_multiperiod.py tests/test_notebook09_3period_pricing_extension_smoke.py -k 'scenario_matrix or notebook09' -v`
Expected: PASS

## Chunk 2: Patch notebooks for reviewer-facing packaging

### Task 2: Reorganize Notebook 07 and Notebook 09

**Files:**
- Modify: `notebooks/07_qiskit_structural_parity.ipynb`
- Modify: `notebooks/09_3period_pricing_extension.ipynb`
- Modify: `tests/test_notebook07_qiskit_structural_parity_smoke.py`

- [ ] **Step 1: Update Notebook 07 structure**

Required sections:

- validated tiny-case subset comparison first
- family-S structural/resource context second

- [ ] **Step 2: Update Notebook 09 structure**

Required additions:

- scenario matrix CSV export
- final compact verdict table

- [ ] **Step 3: Run notebook smoke tests**

Run: `pytest tests/test_notebook07_qiskit_structural_parity_smoke.py tests/test_notebook09_3period_pricing_extension_smoke.py -v`
Expected: PASS

## Chunk 3: Re-execute notebooks and add reviewer summary

### Task 3: Save outputs and add final reviewer artifact

**Files:**
- Create: `docs/BMW_reviewer_summary.md`
- Create: `results/tables/3period_scenario_matrix.csv`
- Modify: `notebooks/07_qiskit_structural_parity.ipynb`
- Modify: `notebooks/09_3period_pricing_extension.ipynb`

- [ ] **Step 1: Execute Notebook 07 in place**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/07_qiskit_structural_parity.ipynb`
Expected: notebook saved with refreshed outputs

- [ ] **Step 2: Execute Notebook 09 in place**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/09_3period_pricing_extension.ipynb`
Expected: notebook saved with refreshed outputs and CSV export present

- [ ] **Step 3: Add `docs/BMW_reviewer_summary.md`**

Content:

- four reviewer-facing claims only
- explicit Qiskit scope note
- links to supporting notebooks

- [ ] **Step 4: Run final focused verification**

Run:

```bash
pytest \
  tests/test_qiskit.py \
  tests/test_classical_baselines_multiperiod.py \
  tests/test_notebook07_qiskit_structural_parity_smoke.py \
  tests/test_notebook09_3period_pricing_extension_smoke.py \
  -v
```

Expected: PASS
