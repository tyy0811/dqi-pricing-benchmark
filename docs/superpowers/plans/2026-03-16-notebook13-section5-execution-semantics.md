# Notebook 13 Section 5 Execution Semantics Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Notebook 13 Section 5 so it reports the canonical executed pricing scaling slice from execution/boundary semantics rather than conflating it with Stage E comparability semantics, and align Section 6 verdict wording to match.

**Architecture:** Keep the change notebook-local unless a direct test helper addition is unavoidable. Section 5 will filter `metrics_master_df` to one explicit execution slice, split completed timing evidence from explicit boundary evidence and generic unavailable rows, and add prose that states this is an execution-semantics section rather than a Stage E comparability section. Section 6 will read the resulting Section 5 outputs and downgrade verdict strength to a validated executed coverage boundary unless strong boundary metadata proves more.

**Tech Stack:** Python 3.11, Jupyter notebooks, pandas, pytest, nbconvert.

**Spec:** `docs/superpowers/specs/2026-03-16-notebook13-section5-execution-semantics-design.md`

**Environment note:** The current workspace snapshot is not a git worktree, so commit steps should be skipped or adapted unless execution moves into a git-backed checkout.

---

## File Structure Map

- Modify: `notebooks/13_scaling_analysis.ipynb`
  - Refactor Section 5 to use the canonical execution slice and separate timing evidence from explicit boundary evidence and generic unavailable coverage rows; align Section 6 verdict 4 wording.
- Modify: `tests/test_notebook_runtime_smoke_missing_data.py`
  - Add runtime checks for Section 5 slice discipline, boundary split behavior, and Section 6 downgraded claim wording.
- Modify: `tests/test_notebook13_scaling_analysis_smoke.py`
  - Add static assertions for the explicit execution-vs-Stage-E sentence and coverage-boundary wording if needed.

## Chunk 1: Lock The Section 5 Semantic Contract In Tests

### Task 1: Add failing runtime tests for the canonical execution slice and boundary split

**Files:**
- Modify: `tests/test_notebook_runtime_smoke_missing_data.py`

- [ ] **Step 1: Write failing tests for Section 5 canonical slice discipline**

Add tests that execute the Section 5 cell and assert the completed timing table excludes:

- `oracle`
- `ilp_derived`
- `wht_truncated`

and includes only:

- `family == "P"`
- `execution_mode == "mixture"`
- `encoding == "wht_truncated_pricing"`
- `decoder in {"bp1", "bruteforce"}`

- [ ] **Step 2: Write failing tests for the explicit boundary split**

Add tests asserting the Section 5 output distinguishes:

- explicit boundary evidence with real boundary metadata
- generic unavailable rows without strong provenance

and does not present generic unavailable rows as tractability evidence.

- [ ] **Step 3: Write a failing test for Section 6 verdict 4 wording**

Assert that verdict 4 uses coverage-boundary language rather than tractability-boundary language when Section 5 lacks strong boundary metadata.

- [ ] **Step 4: Run the targeted tests to confirm failure**

Run: `python3 -m pytest tests/test_notebook_runtime_smoke_missing_data.py tests/test_notebook13_scaling_analysis_smoke.py -k "notebook13 and (section5 or section6)" -x -v`
Expected: FAIL because Notebook 13 currently mixes semantic layers and overstates verdict 4.

## Chunk 2: Refactor Notebook 13 Section 5 And Section 6

### Task 2: Implement the canonical execution slice and split displays in Section 5

**Files:**
- Modify: `notebooks/13_scaling_analysis.ipynb`

- [ ] **Step 1: Add the explicit semantic distinction sentence near the top of Section 5**

State that:

- Section 5 reports the executed pricing scaling slice
- it is not the Stage E comparability slice used for reviewer-facing quality-vs-cost panels

- [ ] **Step 2: Filter Section 5 to the approved canonical execution slice**

Implement the main Section 5 slice using only rows matching:

- `family == "P"`
- `execution_mode == "mixture"`
- `encoding == "wht_truncated_pricing"`
- `decoder in {"bp1", "bruteforce"}`

Keep the recorded per-row `ell` values.

- [ ] **Step 3: Build the completed timing table from execution-layer completed rows only**

Use canonical completed rows with timing/memory evidence from the selected slice. Keep the table focused on executed runtime evidence and avoid notebook-only heuristics outside the documented slice.

- [ ] **Step 4: Split boundary evidence into two tables or panels**

Render:

- explicit boundary evidence rows with meaningful execution metadata
- generic unavailable rows without strong boundary provenance

Use execution-layer fields only (`attempted_execution`, `boundary_reason`, `estimated_qubits`, `estimated_memory_gb`, `run_status`, etc.) and do not pull Stage E unavailable classes into the main claim.

- [ ] **Step 5: Add prose that matches metadata strength**

If strong boundary metadata is sparse, describe the result as a validated executed coverage boundary for the executed `mixture` / `wht_truncated_pricing` backend rather than a universal tractability boundary.

### Task 3: Align Section 6 verdict 4 with Section 5 semantics

**Files:**
- Modify: `notebooks/13_scaling_analysis.ipynb`

- [ ] **Step 1: Recompute verdict 4 from Section 5 outputs**

Use the new Section 5 evidence split to determine whether verdict 4 can claim:

- explicit execution/resource boundary records

or only:

- validated executed coverage boundary

- [ ] **Step 2: Remove tractability overclaiming**

Ensure verdict 4 never treats generic unavailability or Stage E-style missing comparability as proof of tractability failure.

## Chunk 3: Re-Execute And Verify

### Task 4: Run targeted tests and execute the notebook

**Files:**
- Modify: `notebooks/13_scaling_analysis.ipynb`

- [ ] **Step 1: Run the targeted Notebook 13 tests**

Run: `python3 -m pytest tests/test_notebook_runtime_smoke_missing_data.py tests/test_notebook13_scaling_analysis_smoke.py -k "notebook13 and (section5 or section6)" -x -v`
Expected: PASS

- [ ] **Step 2: Execute Notebook 13 end to end**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/13_scaling_analysis.ipynb`
Expected: command exits successfully and the notebook is saved with frozen outputs.

- [ ] **Step 3: Manually verify Section 5 postconditions from the saved notebook**

Confirm:

- no rows outside the approved canonical slice
- no `oracle`
- no `ilp_derived`
- no `wht_truncated`
- prose distinguishes execution boundary from Stage E comparability filtering
- explicit boundary evidence is visually separated from generic unavailable coverage rows

- [ ] **Step 4: Verify Section 6 verdict 4 wording**

Confirm verdict 4 matches the downgraded execution-semantics claim.

- [ ] **Step 5: Run the full required test suite**

Run: `python3 -m pytest tests/ -x`
Expected: PASS

---

Plan complete and saved to `docs/superpowers/plans/2026-03-16-notebook13-section5-execution-semantics.md`. Ready to execute?

