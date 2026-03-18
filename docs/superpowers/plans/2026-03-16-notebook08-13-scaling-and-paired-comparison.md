# Notebook 08 and 13 Scaling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand Notebook 08 to use the current canonical paired-comparison slice dynamically and add Notebook 13 as a reviewer-facing scaling notebook for pricing instances P3 through P7.

**Architecture:** Keep both notebooks as thin readers over frozen artifacts. Notebook 08 derives its paired slice directly from `results/metrics_master.json` with an explicit pairing audit and duplicate-safe key policy; Notebook 13 separates frozen instance metadata, artifact-based encoding diagnostics, canonical execution coverage, and analytic resource estimation into distinct sections. Prefer existing notebook/helper patterns and add only the minimum new notebook-local logic required.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, Jupyter notebooks, pytest, jupyter nbconvert.

**Spec:** `docs/superpowers/specs/2026-03-16-notebook08-13-scaling-and-paired-comparison-design.md`

**Environment note:** The current workspace snapshot is not a git worktree, so commit steps should be skipped or adapted unless execution moves into a git-backed checkout.

---

## File Structure Map

- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
  - Replace hard-coded pricing-family pairing assumptions with canonical dynamic pairing, add pairing audit output, and update verdict text/output persistence.
- Create: `notebooks/13_scaling_analysis.ipynb`
  - Add reviewer-facing scaling sections for instance summary, encoding diagnostics, decoder scaling, analytic resources, timing/boundary analysis, and verdicts.
- Modify: `tests/test_notebook08_paired_encoding_smoke.py`
  - Add smoke assertions for the stricter pairing contract and explicit P7 exclusion note.
- Modify: `tests/test_notebook_runtime_smoke_missing_data.py`
  - Add runtime-oriented Notebook 08 checks for fallback-key rejection/audit behavior and Notebook 13 missing-data coverage behavior.
- Create: `tests/test_notebook13_scaling_analysis_smoke.py`
  - Add static notebook-shape checks for section headers, provenance notes, canonical `m=15` plotting policy, and explicit boundary language.
- Modify: `results/plots/`
  - Store Notebook 13 generated scaling/resource plots here during execution.

---

## Chunk 1: Lock Notebook 08 Canonical Pairing Contract

### Task 1: Extend Notebook 08 smoke tests for the approved pairing policy

**Files:**
- Modify: `tests/test_notebook08_paired_encoding_smoke.py`

- [ ] **Step 1: Add failing smoke assertions for the canonical pairing rules**

Add tests that assert Notebook 08 source now includes:

- canonical completed pricing-family mixture filtering
- strict `comparison_key` acceptance criteria
- explicit fallback-reason reporting
- pairing audit output fields
- explicit P7 exclusion note text

- [ ] **Step 2: Run the targeted smoke tests to confirm failure**

Run: `python3 -m pytest tests/test_notebook08_paired_encoding_smoke.py -x -v`
Expected: FAIL because the notebook source does not yet include the stricter pairing/audit contract.

- [ ] **Step 3: Update the runtime smoke file with pairing-ambiguity scenarios**

Add targeted tests in `tests/test_notebook_runtime_smoke_missing_data.py` that execute the Notebook 08 pairing cells under:

- duplicated `comparison_key` values
- partially missing `comparison_key` values
- valid fallback composite-key pairing
- no completed matched P7 rows

The tests should assert that the notebook reports the fallback reason, surfaces an unavailable/audit table on duplicate ambiguity, and does not fabricate matched pairs.

- [ ] **Step 4: Run the new Notebook 08 runtime tests to confirm failure**

Run: `python3 -m pytest tests/test_notebook_runtime_smoke_missing_data.py -k "notebook08 and (comparison_key or pairing or p7)" -x -v`
Expected: FAIL because the current notebook does not yet implement the stricter audit/fallback behavior.

### Task 2: Implement dynamic canonical pairing in Notebook 08

**Files:**
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`

- [ ] **Step 1: Read the current pairing, summary, delta, and verdict cells end to end**

Confirm where the notebook currently:

- filters `metrics_master`
- chooses `pair_keys`
- builds `matched_keys`
- computes `delta_df`
- renders verdict markdown

- [ ] **Step 2: Update the candidate-slice cell to enforce the canonical filter**

Implement in `notebooks/08_paired_encoding_comparison.ipynb`:

- pricing-family-only filter
- normalized mixture execution filter
- `run_status_norm == "completed"` as the only paired benchmark eligibility status
- removal of any hard-coded pricing-instance list assumptions

- [ ] **Step 3: Replace the pairing cell with the approved key strategy**

Implement:

- strict `comparison_key` acceptance only when present, non-null, and unique within the candidate slice
- explicit fallback to the composite key using only available columns from:
  - `instance_id`
  - `decoder`
  - `alpha_mode`
  - `execution_mode_norm`
  - `ell`
  - `stage`
  - `family`
  - `canonical_trial_seed`
- duplicate detection on each encoding side before merge
- explicit fallback reason output when `comparison_key` is rejected

- [ ] **Step 4: Add the pairing-audit panel**

Render a compact DataFrame or equivalent audit output containing:

- completed pricing rows considered
- candidate WHT rows
- candidate ILP rows
- matched pairs
- dropped unmatched WHT rows
- dropped unmatched ILP rows
- active key strategy
- fallback reason when applicable

- [ ] **Step 5: Recompute dynamic tables and verdict text from the canonical paired slice**

Update the notebook so the saved outputs derive dynamically from the paired rows:

- matched-record count
- summary statistics
- paired delta table
- reviewer verdict language
- explicit note that P7 is excluded if it has zero matched completed rows

- [ ] **Step 6: Run the targeted Notebook 08 smoke and runtime tests**

Run: `python3 -m pytest tests/test_notebook08_paired_encoding_smoke.py tests/test_notebook_runtime_smoke_missing_data.py -k "notebook08" -x -v`
Expected: PASS

---

## Chunk 2: Add Reviewer-Facing Notebook 13

### Task 3: Add Notebook 13 smoke tests before implementation

**Files:**
- Create: `tests/test_notebook13_scaling_analysis_smoke.py`

- [ ] **Step 1: Write failing smoke tests for notebook structure and narrative contract**

Add source or markdown-level tests asserting that Notebook 13 contains:

- all required section headers
- explicit provenance language separating canonical execution rows from artifact-based encoding diagnostics
- canonical `m=15` plotting policy
- diagnostic-only P7 `m=20` note path
- explicit tractability-boundary and P7 boundary language
- `results/plots/` save paths

- [ ] **Step 2: Run the new Notebook 13 smoke tests to confirm failure**

Run: `python3 -m pytest tests/test_notebook13_scaling_analysis_smoke.py -x -v`
Expected: FAIL because Notebook 13 does not exist yet.

### Task 4: Add Notebook 13 runtime tests for missing-data and placeholder behavior

**Files:**
- Modify: `tests/test_notebook_runtime_smoke_missing_data.py`

- [ ] **Step 1: Add failing runtime tests for Notebook 13 section coverage**

Add tests that execute selected Notebook 13 cells under thin or partial fixture data and assert:

- Section 1 keeps frozen CP-SAT values and only fills missing fields
- Section 2 can render canonical `m=15` diagnostics without DQI completion rows
- Sections 3 and 5 distinguish completed rows from unavailable or not-applicable slices
- boundary-only P7 rows are rendered explicitly
- wall-clock placeholders show `—` instead of `NaN`

- [ ] **Step 2: Run the targeted Notebook 13 runtime tests to confirm failure**

Run: `python3 -m pytest tests/test_notebook_runtime_smoke_missing_data.py -k "notebook13" -x -v`
Expected: FAIL because Notebook 13 cells and contracts are not implemented yet.

### Task 5: Create Notebook 13 using existing reader patterns

**Files:**
- Create: `notebooks/13_scaling_analysis.ipynb`
- Reference: `notebooks/06_resource_analysis.ipynb`
- Reference: `notebooks/08_paired_encoding_comparison.ipynb`
- Reference: `notebooks/09_decoder_phase_diagram.ipynb`
- Reference: `notebooks/_helpers.py`
- Reference: `src/resources.py`

- [ ] **Step 1: Scaffold notebook bootstrap and imports**

Create Notebook 13 with:

- `repo_root()`-based bootstrap
- standard pandas/numpy/matplotlib imports
- access to `results/metrics_master.json`
- access to pricing-family frozen JSONs
- access to `src/resources.py` utilities through established notebook patterns

- [ ] **Step 2: Implement Section 1 instance-summary logic**

Build the P3 through P7 table using frozen instance metadata first, then deterministic lightweight fallback recomputation for only missing CP-SAT fields, never overwriting frozen values. Add a short interpretation paragraph and provenance note when fallback is used.

- [ ] **Step 3: Implement Section 2 encoding-diagnostic logic**

Build the `wht_truncated` and `ilp_derived` table from frozen paired or truth-table diagnostics at canonical `m=15`, add table-only `P7 (m=20, diagnostic only)` support if available, generate and save `rho vs n` and `eta vs n` plots to `results/plots/`, and add a short interpretation paragraph.

- [ ] **Step 4: Implement Section 3 decoder-scaling logic**

Use canonical completed pricing-family rows only, prefer `ell=2` and fall back to `ell=1` when necessary, label fallbacks explicitly, render readable placeholders for missing wall-clock values, and add a coverage note plus interpretation paragraph.

- [ ] **Step 5: Implement Section 4 analytic resource-scaling logic**

Use analytic estimation only for P3 through P7 at reference `ell=2`, generate and save resource-scaling plot(s) to `results/plots/`, and label P7 analytic-only if no executed DQI validation exists.

- [ ] **Step 6: Implement Section 5 timing and boundary analysis**

Render completed timing summaries and explicit boundary rows from the canonical master without dropping unavailable or failed rows. Make sure P7 boundary-only cases stay visible and semantically distinct from not-applicable slices.

- [ ] **Step 7: Implement Section 6 verdict lines and Section 7 frozen-output behavior**

Derive all verdict lines from the measured tables in the notebook, keep the language cautious, and ensure plots/tables are persisted in the saved notebook outputs.

- [ ] **Step 8: Run the Notebook 13 smoke and runtime tests**

Run: `python3 -m pytest tests/test_notebook13_scaling_analysis_smoke.py tests/test_notebook_runtime_smoke_missing_data.py -k "notebook13" -x -v`
Expected: PASS

---

## Chunk 3: Execute Notebooks and Verify End-to-End Outputs

### Task 6: Execute Notebook 08 with frozen outputs

**Files:**
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`

- [ ] **Step 1: Execute Notebook 08 end to end**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/08_paired_encoding_comparison.ipynb`
Expected: command exits successfully and the notebook is saved with persisted outputs and no cell errors.

- [ ] **Step 2: Sanity-check the saved notebook outputs**

Verify from the saved notebook JSON or output text that:

- matched-record count is updated from canonical data
- P6 appears if matched completed rows exist
- P7 paired-comparison exclusion text appears when appropriate
- no placeholder “run manually” text remains

### Task 7: Execute Notebook 13 with frozen outputs and verify plots

**Files:**
- Modify: `notebooks/13_scaling_analysis.ipynb`
- Modify: `results/plots/`

- [ ] **Step 1: Execute Notebook 13 end to end**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/13_scaling_analysis.ipynb`
Expected: command exits successfully and the notebook is saved with persisted outputs and no cell errors.

- [ ] **Step 2: Verify expected plots exist under `results/plots/`**

Check that the notebook wrote at least:

- `rho vs n`
- `eta vs n`
- resource scaling plot(s)

Use the actual filenames produced by the notebook and confirm they are present on disk.

### Task 8: Run final regression and summarize outcomes

**Files:**
- Modify: none

- [ ] **Step 1: Run the full test suite**

Run: `python3 -m pytest tests/ -x`
Expected: PASS

- [ ] **Step 2: Prepare the required completion summary**

Report:

- Notebook 08 matched-record count
- whether P6 was included
- whether P7 was excluded from paired comparison
- whether Notebook 13 includes P7 encoding/resource diagnostics
- whether Notebook 13 boundary analysis includes explicit P7 rows
- final test result

- [ ] **Step 3: If running in a git-backed checkout, commit in small logical units**

Suggested commit boundaries:

- tests for Notebook 08 pairing contract
- Notebook 08 implementation
- tests for Notebook 13
- Notebook 13 implementation
- executed notebooks and plot artifacts

If git is unavailable in the execution environment, record that limitation in the handoff instead of fabricating commit output.
