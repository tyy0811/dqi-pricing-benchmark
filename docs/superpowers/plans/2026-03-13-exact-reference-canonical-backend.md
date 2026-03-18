# Exact-Reference Canonical Backend Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make exact-reference / ILP-derived encoding the canonical reviewer-facing pricing backend while preserving WHT support as an exploratory path and keeping payload compatibility intact.

**Architecture:** Add reviewer-facing verdict and recommendation semantics on top of the existing paired-comparison payload, tag WHT artifacts with machine-readable exploratory metadata, and have `src/benchmark.py` emit a mandatory recommendation block on the default pricing path. Update Notebook 08 to read paired-comparison outputs only and present the canonical backend story, while Notebook 11 becomes a view layer that prefers benchmark-emitted recommendation metadata over local verdict logic.

**Tech Stack:** Python 3.11, pytest, Jupyter notebooks, Markdown/JSON notebook cells.

**Spec:** `docs/superpowers/specs/2026-03-13-exact-reference-canonical-backend-design.md`

---

## File Structure Map

- Modify: `src/encoding_compare.py`
  - Add canonical reviewer-facing exact-reference constant, conservative verdict helper, additive summary block, and compatibility-preserving reviewer-facing labels.
- Modify: `src/reduction.py`
  - Add additive exploratory metadata fields to WHT artifact payloads.
- Modify: `src/benchmark.py`
  - Run paired comparison by default on the default pricing path, add `run_wht_ablation`, emit mandatory `encoding_backend_recommendation`, and keep exact-reference-first ordering in reviewer-facing summaries only.
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
  - Reframe as the main reviewer-facing encoding notebook, consuming existing paired outputs only and presenting exact-reference-first verdict tables with neutral unavailable handling.
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`
  - Add `include_exploratory_wht = False`, prefer benchmark recommendation metadata, and keep dashboard behavior downstream of benchmark verdict logic.

---

## Chunk 1: Verdict Semantics and Exploratory Metadata

### Task 1: Add conservative paired-comparison verdict semantics without breaking existing payload names

**Files:**
- Modify: `src/encoding_compare.py`
- Test: `tests/test_encoding_compare.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write failing tests for the new verdict and summary contract**

Add targeted tests covering:

```python
def test_paired_encoding_verdict_uses_conservative_dominance_without_best_sampled_f():
    ...

def test_paired_encoding_summary_defaults_reviewer_verdict_to_ilp_exact_on_tie():
    ...
```

Assertions should verify:
- decisive-set comparison excludes `best_sampled_F`
- formal verdict returns only `mainline_exact_reference`, `tie`, or `exploratory_only`
- reviewer-facing recommendation defaults to `ilp_derived_exact` on `tie`
- legacy payload fields such as `backend_b_name == "ilp_exact_reference"` remain valid unless implementation proves unavoidable

- [ ] **Step 2: Run the targeted tests to confirm failure**

Run: `pytest tests/test_encoding_compare.py tests/test_benchmark.py -k "paired_encoding_verdict or reviewer_verdict or ilp_derived_exact" -v`
Expected: FAIL because the verdict helper and summary block do not exist yet.

- [ ] **Step 3: Implement the minimal additive verdict/reporting layer**

Implement in `src/encoding_compare.py`:
- a top-level exact-reference constant for reviewer-facing use
- `paired_encoding_verdict(delta_metrics: dict) -> str`
- conservative dominance over existing decision-quality metrics, excluding `best_sampled_F`
- additive `summary = {"winner": ..., "metrics_used": [...], "reviewer_verdict": ...}`
- reviewer-facing label normalization to `ilp_derived_exact` in new additive fields only
- preserve legacy payload keys/values unless change is required for display correctness

- [ ] **Step 4: Re-run the targeted tests**

Run: `pytest tests/test_encoding_compare.py tests/test_benchmark.py -k "paired_encoding_verdict or reviewer_verdict or ilp_derived_exact" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/encoding_compare.py tests/test_encoding_compare.py tests/test_benchmark.py
git commit -m "feat(encoding): add conservative exact-reference verdict semantics"
```

### Task 2: Tag WHT artifacts as exploratory with machine-readable metadata

**Files:**
- Modify: `src/reduction.py`
- Test: `tests/test_reduction.py`

- [ ] **Step 1: Write failing tests for exploratory artifact metadata**

Add tests similar to:

```python
def test_wht_surrogate_artifact_has_exploratory_story_metadata():
    ...
```

Assertions should verify:
- `story_role == "exploratory"`
- `benchmark_status == "non_mainline_surrogate"`
- existing artifact keys remain present

- [ ] **Step 2: Run the targeted reduction test**

Run: `pytest tests/test_reduction.py -k "story_role or benchmark_status" -v`
Expected: FAIL because the additive metadata does not exist yet.

- [ ] **Step 3: Implement additive artifact metadata**

Implement in `src/reduction.py`:
- add the two exploratory metadata fields only for the WHT surrogate artifact builder
- preserve existing artifact schema and numeric payloads

- [ ] **Step 4: Re-run the targeted reduction test**

Run: `pytest tests/test_reduction.py -k "story_role or benchmark_status" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/reduction.py tests/test_reduction.py
git commit -m "feat(reduction): tag WHT artifacts as exploratory"
```

---

## Chunk 2: Benchmark Recommendation Emission and Default Pricing Path

### Task 3: Emit mandatory `encoding_backend_recommendation` on the default pricing path

**Files:**
- Modify: `src/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write failing tests for recommendation semantics and fallback**

Add tests covering:

```python
def test_benchmark_emits_encoding_backend_recommendation_when_comparison_available():
    ...

def test_benchmark_emits_unavailable_recommendation_when_paired_comparison_missing():
    ...
```

Assertions should verify:
- `encoding_backend_recommendation` always exists in reviewer-facing benchmark outputs
- it distinguishes:
  - formal verdict
  - default reviewer-facing backend
  - recommendation basis
  - status
  - reason
- no winner is claimed when paired comparison is unavailable or partial

- [ ] **Step 2: Run the targeted benchmark tests to confirm failure**

Run: `pytest tests/test_benchmark.py -k "encoding_backend_recommendation or paired_comparison_missing" -v`
Expected: FAIL because the recommendation block does not exist yet.

- [ ] **Step 3: Implement the minimal benchmark recommendation layer**

Implement in `src/benchmark.py`:
- add `run_wht_ablation: bool = True`
- make paired encoding comparison run by default only on the default pricing path
- add `results["encoding_backend_recommendation"]`
- ensure recommendation behavior is:
  - available when paired comparison is present and complete
  - unavailable or not_applicable with a short reason otherwise
- keep ordering and promotion as display/reporting semantics rather than raw payload reshaping

- [ ] **Step 4: Re-run the targeted benchmark tests**

Run: `pytest tests/test_benchmark.py -k "encoding_backend_recommendation or paired_comparison_missing" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/benchmark.py tests/test_benchmark.py
git commit -m "feat(benchmark): emit exact-reference backend recommendation metadata"
```

### Task 4: Keep exact-reference first in summaries only when both backends exist

**Files:**
- Modify: `src/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write a failing test for display-only ordering**

Add a test similar to:

```python
def test_reviewer_summary_puts_exact_reference_first_only_when_both_backends_exist():
    ...
```

Assertions should verify:
- exact-reference appears first in reviewer-facing summaries only when both backends exist
- no misleading exact-reference-first ordering is emitted when paired comparison was not run
- internal payload storage semantics are unchanged

- [ ] **Step 2: Run the targeted ordering test**

Run: `pytest tests/test_benchmark.py -k "exact_reference_first or reviewer_summary" -v`
Expected: FAIL because the default ordering and fallback handling are not fully implemented yet.

- [ ] **Step 3: Update reviewer-facing summary assembly**

Implement in `src/benchmark.py`:
- exact-reference-first display ordering for paired-comparison summaries
- neutral unavailable status note for non-pricing/default-missing cases
- no reviewer-facing recommendation text when comparison is absent or partial

- [ ] **Step 4: Re-run the targeted ordering test**

Run: `pytest tests/test_benchmark.py -k "exact_reference_first or reviewer_summary" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/benchmark.py tests/test_benchmark.py
git commit -m "refactor(benchmark): keep exact-reference-first ordering in reviewer-facing summaries"
```

---

## Chunk 3: Notebook Consumption and Reviewer-Facing Framing

### Task 5: Reframe Notebook 08 as the canonical encoding notebook without recomputing metrics

**Files:**
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
- Test: `tests/test_notebook08_paired_encoding_smoke.py`
- Test: `tests/test_notebook_runtime_smoke_missing_data.py`

- [ ] **Step 1: Add or update failing notebook smoke expectations**

Add/update tests so they verify:
- Notebook 08 frames the opening question around exact-reference versus WHT decision quality
- Notebook 08 reads existing paired-comparison outputs rather than recomputing benchmark metrics
- Notebook 08 includes a first-page summary table with the required columns
- Notebook 08 shows a neutral unavailable message and no winner when paired comparison is missing or partial

- [ ] **Step 2: Run the targeted notebook tests to confirm failure**

Run: `pytest tests/test_notebook08_paired_encoding_smoke.py tests/test_notebook_runtime_smoke_missing_data.py -k "paired_encoding_comparison or Notebook 08" -v`
Expected: FAIL because the notebook has the old framing and no unavailable/recommendation semantics.

- [ ] **Step 3: Edit Notebook 08 cells minimally**

Update `notebooks/08_paired_encoding_comparison.ipynb` to:
- keep notebook logic reader-only
- lead with the reviewer question about exact-reference improving decision quality over WHT
- build the first-page summary table from existing paired outputs and recommendation metadata
- show exact-reference first only when both backends exist
- end with markdown verdict text only when paired comparison is present and complete
- otherwise show a neutral unavailable status note with no winner

- [ ] **Step 4: Re-run the targeted notebook tests**

Run: `pytest tests/test_notebook08_paired_encoding_smoke.py tests/test_notebook_runtime_smoke_missing_data.py -k "paired_encoding_comparison or Notebook 08" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/08_paired_encoding_comparison.ipynb tests/test_notebook08_paired_encoding_smoke.py tests/test_notebook_runtime_smoke_missing_data.py
git commit -m "docs(notebook08): make exact-reference the canonical reviewer-facing encoding view"
```

### Task 6: Make Notebook 11 consume benchmark recommendation metadata first

**Files:**
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`
- Test: `tests/test_notebook_runtime_smoke_missing_data.py`

- [ ] **Step 1: Add or update failing tests for Notebook 11 defaults**

Add/update tests verifying:
- notebook defines `include_exploratory_wht = False`
- notebook prefers `results["encoding_backend_recommendation"]` when available
- notebook classifies WHT as exploratory from artifact metadata first

- [ ] **Step 2: Run the targeted notebook 11 tests**

Run: `pytest tests/test_notebook_runtime_smoke_missing_data.py -k "decision_faithfulness_dashboard or exploratory_wht or encoding_backend_recommendation" -v`
Expected: FAIL because the current notebook uses older local logic and lacks the toggle.

- [ ] **Step 3: Edit Notebook 11 cells minimally**

Update `notebooks/11_decision_faithfulness_dashboard.ipynb` to:
- add `include_exploratory_wht = False`
- prefer benchmark-emitted recommendation metadata over local verdict logic
- treat WHT as exploratory via artifact metadata first, with fallback only if metadata is absent
- keep Notebook 11 as a dashboard/view layer rather than a source of canonical verdict logic

- [ ] **Step 4: Re-run the targeted notebook 11 tests**

Run: `pytest tests/test_notebook_runtime_smoke_missing_data.py -k "decision_faithfulness_dashboard or exploratory_wht or encoding_backend_recommendation" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/11_decision_faithfulness_dashboard.ipynb tests/test_notebook_runtime_smoke_missing_data.py
git commit -m "refactor(notebook11): prefer benchmark recommendation metadata and hide exploratory WHT by default"
```

---

## Chunk 4: Full Verification and Scope Check

### Task 7: Run focused verification for the scoped phase

**Files:**
- Modify: `src/encoding_compare.py`
- Modify: `src/reduction.py`
- Modify: `src/benchmark.py`
- Modify: `notebooks/08_paired_encoding_comparison.ipynb`
- Modify: `notebooks/11_decision_faithfulness_dashboard.ipynb`

- [ ] **Step 1: Run focused test coverage**

Run: `python -m pytest tests/test_encoding_compare.py tests/test_reduction.py tests/test_benchmark.py tests/test_notebook08_paired_encoding_smoke.py tests/test_notebook_runtime_smoke_missing_data.py -v`
Expected: PASS

- [ ] **Step 2: Review final diffs for scope discipline**

Run: `git diff -- src/encoding_compare.py src/reduction.py src/benchmark.py notebooks/08_paired_encoding_comparison.ipynb notebooks/11_decision_faithfulness_dashboard.ipynb`
Expected: Only additive metadata, recommendation semantics, default pricing-path behavior, and notebook framing changes.

- [ ] **Step 3: Commit the scoped phase**

```bash
git add src/encoding_compare.py src/reduction.py src/benchmark.py notebooks/08_paired_encoding_comparison.ipynb notebooks/11_decision_faithfulness_dashboard.ipynb
git commit -m "feat: promote exact-reference as the canonical reviewer-facing backend"
```

