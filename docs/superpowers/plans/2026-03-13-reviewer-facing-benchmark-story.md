# Reviewer-Facing Benchmark Story Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the repo's default reviewer-facing story so docs and benchmark summaries center the mixture-mode mainline and exact-reference / ILP-derived encoding without changing the underlying benchmark payload schema.

**Architecture:** Keep existing result payload generation and legacy aliases stable, then add a thin reporting adapter inside `src/benchmark.py` that controls reviewer-facing ordering, labels, and appendix filtering. Update `README.md` and `Theoretical_Framework.md` to distinguish mainline benchmark path, exploratory WHT path, coherent appendix material, and structural-only Qiskit validation.

**Tech Stack:** Python 3.11, pytest, Markdown documentation.

**Spec:** `docs/superpowers/specs/2026-03-13-reviewer-facing-benchmark-story-design.md`

---

## File Structure Map

- Modify: `README.md`
  - Rewrite pitch, add mainline vs exploratory framing, add scope note, reorder reviewer-facing notebook list.
- Modify: `Theoretical_Framework.md`
  - Reframe maturity levels into mainline benchmark path, exploratory WHT path, and structural Qiskit subset validation.
- Modify: `src/benchmark.py`
  - Add display/reporting adapter helpers, add appendix flags, update summary ordering, and isolate coherent appendix rendering policy.
- Modify: `tests/test_benchmark.py`
  - Add or adjust tests for reviewer-facing ordering, omission of incomplete coherent appendix rows, and payload-key compatibility.

---

## Chunk 1: Documentation Reframing

### Task 1: Rewrite README narrative and notebook ordering

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README sections that define repo pitch, notebook suite, and disclaimers**

Run: `sed -n '1,220p' README.md`
Expected: Current pitch centers WHT/max-XORSAT/DQI without the new mainline vs exploratory framing.

- [ ] **Step 2: Edit the README to match the approved Phase 1 story**

Implement in `README.md`:
- replace the one-line pitch with the approved reviewer-facing sentence
- add a short `Mainline vs exploratory` section
- add a scope note with:
  - mainline = mixture execution on pricing artifacts with exact-reference / ILP-derived encoding
  - exploratory = WHT-truncated surrogate artifacts
  - appendix only = coherent / alpha finite-size analyses on tiny supported instances
  - structural only = Qiskit subset checks, not decode-inclusive end-to-end parity
- reorder reviewer-facing notebooks to:
  - 04 pricing pipeline
  - 08 paired encoding comparison
  - 05 decoder comparison
  - 06 resource analysis
  - 07 qiskit structural parity
  - 09 3-period pricing extension
  - 10 alpha appendix
  - 03 paper-faithful DQI appendix
- use `ilp_derived_exact` as the canonical reviewer-facing label

- [ ] **Step 3: Review the README diff for accidental claims**

Run: `git diff -- README.md`
Expected: Only wording/order changes; no invented results or new algorithm claims.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: reframe reviewer-facing benchmark story in README"
```

### Task 2: Reframe the theoretical framework around maturity and scope

**Files:**
- Modify: `Theoretical_Framework.md`

- [ ] **Step 1: Locate the introduction and any Qiskit/coherent parity language**

Run: `rg -n "Qiskit|coherent|path|pipeline|parity|surrogate" Theoretical_Framework.md`
Expected: Existing text presents multiple paths too symmetrically and overstates parity risk.

- [ ] **Step 2: Edit the framework to use the approved path hierarchy**

Implement in `Theoretical_Framework.md`:
- describe:
  - `Mainline benchmark path`
  - `Exploratory surrogate path (WHT)`
  - `Structural Qiskit subset validation`
- make clear that:
  - exact-reference / ILP-derived pricing artifacts are the mainline reviewer-facing path
  - WHT is exploratory rather than the main claim path
  - coherent is appendix-only for tiny supported instances
  - current Qiskit support is an implemented structural subset, not decode-inclusive end-to-end parity

- [ ] **Step 3: Review wording for symmetry and overclaiming**

Run: `git diff -- Theoretical_Framework.md`
Expected: The new headings and intro language are intentionally asymmetric and avoid end-to-end Qiskit parity claims.

- [ ] **Step 4: Commit**

```bash
git add Theoretical_Framework.md
git commit -m "docs: scope theoretical framework to mainline, exploratory, and structural paths"
```

---

## Chunk 2: Reporting Adapter in `src/benchmark.py`

### Task 3: Add appendix flags and centralized reviewer-facing render helpers

**Files:**
- Modify: `src/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write failing tests for the new reporting controls**

Add tests in `tests/test_benchmark.py` for:

```python
def test_default_reviewer_headline_excludes_coherent_appendix():
    ...

def test_incomplete_coherent_appendix_row_is_not_rendered():
    ...
```

The tests should assert:
- default reviewer-facing summary excludes coherent rows
- coherent appendix rows require enablement plus complete non-NaN metrics

- [ ] **Step 2: Run the targeted tests to confirm failure**

Run: `pytest tests/test_benchmark.py -k "coherent_appendix or reviewer_headline" -v`
Expected: FAIL because the reporting adapter and centralized gate do not exist yet.

- [ ] **Step 3: Add the minimal reporting adapter and explicit coherent gate**

Implement in `src/benchmark.py`:
- extend `run_benchmark(...)` signature with:
  - `include_alpha_ablation: bool = False`
  - `include_coherent_appendix: bool = False`
- keep existing result payload keys intact unless unavoidable
- add small internal reviewer-facing helpers, including:
  - one mapping payload keys to reviewer-facing labels
  - one ordering helper for headline summaries
  - `should_render_coherent_appendix(...) -> bool`
- define the coherent appendix gate once:
  - appendix flag enabled
  - coherent result exists
  - required metrics are present
  - required metrics are non-NaN

- [ ] **Step 4: Re-run the targeted tests**

Run: `pytest tests/test_benchmark.py -k "coherent_appendix or reviewer_headline" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/benchmark.py tests/test_benchmark.py
git commit -m "feat(benchmark): add reviewer-facing appendix gating helpers"
```

### Task 4: Reorder default headline summaries and isolate appendix blocks

**Files:**
- Modify: `src/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write a failing test for default reviewer-facing order**

Add a test similar to:

```python
def test_reviewer_facing_summary_order_is_mainline_then_bp1_then_ilp_exact():
    ...
```

The test should assert the default display order is:
1. `dqi_paper_mixture`
2. `dqi_paper_mixture_bp1`
3. paired encoding comparison with reviewer-facing label `ilp_derived_exact`

- [ ] **Step 2: Run the targeted order test**

Run: `pytest tests/test_benchmark.py -k "summary_order or ilp_derived_exact" -v`
Expected: FAIL because current benchmark summaries still center uniform/heuristic/coherent inline.

- [ ] **Step 3: Update benchmark summary assembly and print paths**

Implement in `src/benchmark.py`:
- change comments/docstrings so coherent is appendix-only
- update any default summary builder, CLI print path, and comparison-table path to use the reporting adapter
- keep default headline summaries limited to:
  - `dqi_paper_mixture`
  - `dqi_paper_mixture_bp1`
  - paired encoding comparison
- move coherent rows out of the main headline comparison table into an appendix block
- render alpha/coherent appendix blocks only when enabled and complete
- use `ilp_derived_exact` in reviewer-facing labels without broadly renaming stable payload keys

- [ ] **Step 4: Re-run targeted summary tests**

Run: `pytest tests/test_benchmark.py -k "summary_order or ilp_derived_exact or coherent_appendix" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/benchmark.py tests/test_benchmark.py
git commit -m "refactor(benchmark): reorder reviewer-facing summaries around mainline mixture path"
```

---

## Chunk 3: Compatibility and Regression Coverage

### Task 5: Verify payload-key stability and backward-compatible defaults

**Files:**
- Modify: `tests/test_benchmark.py`

- [ ] **Step 1: Add a failing compatibility test before changing assertions**

Add a test similar to:

```python
def test_reporting_layer_changes_do_not_remove_existing_payload_keys():
    ...
```

It should assert that after the reporting-layer change:
- core payload keys still exist
- `quality_vs_cost` still exists
- legacy aliases still exist
- default `run_benchmark()` execution remains valid without enabling appendix flags

- [ ] **Step 2: Run the compatibility test**

Run: `pytest tests/test_benchmark.py -k "payload_keys or compatibility" -v`
Expected: FAIL if current assertions do not cover the new boundary or if implementation drift removed keys.

- [ ] **Step 3: Make minimal test updates needed to lock the Phase 1 boundary**

Update `tests/test_benchmark.py` so it checks:
- payload-key stability
- reviewer-facing order contract
- appendix omission contract
- no algorithmic expectations changed

- [ ] **Step 4: Run the benchmark test module**

Run: `pytest tests/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmark.py
git commit -m "test: lock reviewer-facing reporting boundary and payload compatibility"
```

### Task 6: Run final verification for the scoped files

**Files:**
- Modify: `README.md`
- Modify: `Theoretical_Framework.md`
- Modify: `src/benchmark.py`
- Modify: `tests/test_benchmark.py`

- [ ] **Step 1: Run formatting-neutral verification on the changed files**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 2: Review the final diffs for scope discipline**

Run: `git diff -- README.md Theoretical_Framework.md src/benchmark.py tests/test_benchmark.py`
Expected: Only Phase 1 doc/reporting changes; no new algorithms and no broad payload churn.

- [ ] **Step 3: Commit the final scoped pass**

```bash
git add README.md Theoretical_Framework.md src/benchmark.py tests/test_benchmark.py
git commit -m "feat: shift reviewer-facing benchmark story to mainline mixture and exact encoding"
```

