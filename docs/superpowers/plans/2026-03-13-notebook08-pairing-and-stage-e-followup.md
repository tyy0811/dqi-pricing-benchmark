# Notebook 08 Pairing and Stage E Follow-up Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Notebook 08 to consume canonical `comparison_key`, guarantee schema-stable empty duplicate-audit artifacts, and patch Stage C/D projection only if raw upstream fields already exist.

**Architecture:** Notebook 08 gets a narrow reader-layer fix that prefers canonical `comparison_key` and keeps debug state internal. The report layer gets a fixed-schema empty duplicate-audit artifact. Stage C/D projection changes are gated by read/trace evidence from raw source rows rather than assumptions.

**Tech Stack:** Python, pandas, pytest, Jupyter notebooks, existing report artifacts

---

## Chunk 1: Trace and lock failing tests

### Task 1: Trace Stage C/D raw-source fields before any projection edits

**Files:**
- Inspect: `scripts/make_report.py`
- Inspect: raw result files under `results/matrix_b/runs/` and `results/matrix_c/runs/`

- [ ] **Step 1: Trace Stage C raw rows for `retained_energy_eta`**

Run a read-only inspection over Stage C raw source rows used by `scripts/make_report.py`.

- [ ] **Step 2: Trace Stage D raw rows for `topk_regret`, `decision_distortion`, `spearman_rho_F_G`, and `retained_energy_eta`**

Run a read-only inspection over Stage D raw source rows used by `scripts/make_report.py`.

- [ ] **Step 3: Record whether these are projection bugs or upstream data gaps**

Do not change production code in this task.

### Task 2: Add failing tests for Notebook 08 canonical pairing and empty duplicate-audit schema

**Files:**
- Modify: `tests/test_notebook08_paired_encoding_smoke.py`
- Modify: `tests/test_make_report_stage_e_audits.py`
- Create or modify any focused runtime/smoke test needed for notebook 08 execution behavior

- [ ] **Step 1: Write the failing Notebook 08 test**

Add a test that provides valid upstream `comparison_key` pairs and fails if notebook 08 still renders the paired placeholder.

- [ ] **Step 2: Write the failing duplicate-audit schema test**

Add a test that fails if `quality_cost_duplicate_audit` is empty and loses its declared columns.

- [ ] **Step 3: Run the targeted test subset to verify red**

Run: `pytest tests/test_notebook08_paired_encoding_smoke.py tests/test_make_report_stage_e_audits.py -v`
Expected: FAIL because notebook 08 still uses legacy key inference and the empty duplicate audit currently loses schema columns.

## Chunk 2: Implement the notebook and audit fixes

### Task 3: Patch Notebook 08 canonical pairing

**Files:**
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
- Test: `tests/test_notebook08_paired_encoding_smoke.py`

- [ ] **Step 1: Implement canonical `comparison_key` pairing**

In the canonical path:
- use `pair_keys = ["comparison_key"]`
- skip legacy pair-key construction
- validate at most one completed WHT row and one completed ILP row per `comparison_key`
- keep `pairing_debug_summary_df` internal only

- [ ] **Step 2: Remove `slot_id` from legacy fallback candidates**

Fallback is only allowed when `comparison_key` is absent.

- [ ] **Step 3: Run the notebook 08 test to verify green**

Run: `pytest tests/test_notebook08_paired_encoding_smoke.py -v`
Expected: PASS

### Task 4: Guarantee schema-stable empty duplicate-audit output

**Files:**
- Modify: `scripts/make_report.py`
- Test: `tests/test_make_report_stage_e_audits.py`

- [ ] **Step 1: Add a fixed declared duplicate-audit schema in code**

Write the schema once and use it even for zero-row artifacts.

- [ ] **Step 2: Run the duplicate-audit test to verify green**

Run: `pytest tests/test_make_report_stage_e_audits.py -v`
Expected: PASS

## Chunk 3: Patch projection only if the trace proves it is needed

### Task 5: Stage C/D projection patch, conditional on upstream field presence

**Files:**
- Modify only if needed: `scripts/make_report.py`
- Add tests only if needed: a focused `tests/test_make_report_*.py` module covering the preserved fields

- [ ] **Step 1: If Stage C raw rows already contain `retained_energy_eta`, write the failing projection test**

Run a targeted test first and confirm it fails before production edits.

- [ ] **Step 2: If Stage D raw rows already contain required faithfulness/regret fields, write the failing projection test**

Run a targeted test first and confirm it fails before production edits.

- [ ] **Step 3: Implement the minimal projection fix**

Carry the existing raw fields into `metrics_master` only; do not synthesize missing values.

- [ ] **Step 4: Run the projection tests to verify green**

Run only the added focused tests.

- [ ] **Step 5: If the raw fields do not exist upstream, stop here**

Do not patch production code. Record the exact missing fields and stages for the final report.

## Chunk 4: Verification

### Task 6: Rerun artifacts and notebooks

**Files:**
- Verify: `results/`
- Verify: `notebooks/08_paired_encoding_comparison.ipynb`
- Verify if projection changed: `notebooks/11_decision_faithfulness_dashboard.ipynb`
- Verify if projection changed: `notebooks/12_quality_vs_cost.ipynb`

- [ ] **Step 1: Regenerate reporting artifacts**

Run: `python scripts/make_report.py --results-root results --matrices matrix_a,matrix_b,matrix_c`
Expected: exit 0

- [ ] **Step 2: Execute notebook 08**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/08_paired_encoding_comparison.ipynb`
Expected: exit 0 and paired table populated when valid upstream pairs exist

- [ ] **Step 3: If projection changed, execute notebooks 11 and 12**

Run:
- `jupyter nbconvert --to notebook --execute --inplace notebooks/11_decision_faithfulness_dashboard.ipynb`
- `jupyter nbconvert --to notebook --execute --inplace notebooks/12_quality_vs_cost.ipynb`

- [ ] **Step 4: Run the targeted regression set**

Run:
`pytest tests/test_notebook08_paired_encoding_smoke.py tests/test_make_report_stage_e_audits.py -v`

If projection changed, include the new projection tests.

- [ ] **Step 5: Summarize the outcome**

State:
- whether notebook 08 now renders the paired delta table
- whether Stage C/D blockers were projection bugs or genuine upstream metric gaps
